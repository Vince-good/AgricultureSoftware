"""可移植计算图的序列化与 ONNX 转换。

构建期（工作站，装了 onnx）：把 MobileNetV3 的 .onnx 转成单文件 `.hgraph.npz`。
运行期（边缘设备，只要 numpy）：直接加载 `.hgraph.npz` 交给 NumpyGraphBackend。

单文件设计是为了田间部署方便 —— 拷一个文件到 SD 卡就行，不用管目录结构。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .backends.base import InputSpec
from .backends.numpy_graph import Graph, Node

GRAPH_KEY = "__graph_json__"


# --------------------------------------------------------------------------
# 序列化
# --------------------------------------------------------------------------

def save_graph(graph: Graph, path: Path | str) -> Path:
    path = Path(path)
    payload: Dict[str, np.ndarray] = {}
    for name, arr in graph.tensors.items():
        payload[name] = np.ascontiguousarray(arr)
    meta = {
        "format": "heyan-graph",
        "format_version": 1,
        "nodes": [{"op": n.op, "inputs": n.inputs, "outputs": n.outputs, "attrs": n.attrs}
                  for n in graph.nodes],
        "graph_inputs": graph.graph_inputs,
        "graph_outputs": graph.graph_outputs,
        "input_spec": graph.input_spec.__dict__ if graph.input_spec else None,
    }
    payload[GRAPH_KEY] = np.array(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(path, **payload)
    return path


def load_graph_file(path: Path | str) -> Graph:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if GRAPH_KEY not in data.files:
            raise ValueError(f"{path} 不是 heyan 图文件（缺 {GRAPH_KEY}）")
        meta = json.loads(str(data[GRAPH_KEY]))
        tensors = {k: data[k] for k in data.files if k != GRAPH_KEY}
    spec_raw = meta.get("input_spec")
    spec = InputSpec(**spec_raw) if spec_raw else None
    if spec is not None:
        spec.shape = tuple(spec.shape)
    nodes = [Node(op=n["op"], inputs=list(n["inputs"]), outputs=list(n["outputs"]),
                   attrs=dict(n.get("attrs") or {})) for n in meta["nodes"]]
    return Graph(nodes=nodes, graph_inputs=list(meta["graph_inputs"]),
                 graph_outputs=list(meta["graph_outputs"]), tensors=tensors, input_spec=spec)


def load_graph(path: Path | str) -> Graph:
    """按扩展名自动选择加载方式：.hgraph.npz 走零依赖路径，.onnx 需要 onnx 包。"""
    path = Path(path)
    if path.suffix == ".onnx":
        return graph_from_onnx(path)
    if path.is_dir():
        for cand in sorted(path.glob("*.hgraph.npz")) + sorted(path.glob("*.npz")):
            return load_graph_file(cand)
        raise FileNotFoundError(f"目录里找不到图文件: {path}")
    return load_graph_file(path)


# --------------------------------------------------------------------------
# ONNX → Graph
# --------------------------------------------------------------------------

_ONNX_ELEM_TYPE = {
    1: np.float32, 2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16,
    6: np.int32, 7: np.int64, 9: bool, 10: np.float16, 11: np.float64,
    12: np.uint32, 13: np.uint64,
}

_NP_TO_ONNX_DTYPE = {v: k for k, v in _ONNX_ELEM_TYPE.items() if v is not bool}


def _attr_value(attr) -> Any:
    from onnx import AttributeProto, numpy_helper

    t = attr.type
    if t == AttributeProto.FLOAT:
        return float(attr.f)
    if t == AttributeProto.INT:
        return int(attr.i)
    if t == AttributeProto.STRING:
        return attr.s.decode("utf-8") if isinstance(attr.s, bytes) else str(attr.s)
    if t == AttributeProto.FLOATS:
        return [float(x) for x in attr.floats]
    if t == AttributeProto.INTS:
        return [int(x) for x in attr.ints]
    if t == AttributeProto.STRINGS:
        return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in attr.strings]
    if t == AttributeProto.TENSOR:
        return numpy_helper.to_array(attr.t)
    raise NotImplementedError(f"不支持的属性类型: {t}")


def _input_spec_from_onnx(vi) -> InputSpec:
    tt = vi.type.tensor_type
    elem = _ONNX_ELEM_TYPE.get(int(tt.elem_type), np.float32)
    shape: List[int] = []
    for d in tt.shape.dim:
        shape.append(int(d.dim_value) if d.HasField("dim_value") and d.dim_value > 0 else -1)
    dtype_name = np.dtype(elem).name
    scale, zp = 1.0, 0
    for qp in getattr(tt, "quantization_parameter", []) or []:
        try:
            scale = float(qp.scale)
            zp = int(float(qp.zero_point))
        except Exception:  # pragma: no cover - 结构不齐时忽略
            pass
    return InputSpec(name=vi.name, shape=tuple(shape), dtype=dtype_name, scale=scale, zero_point=zp)


def graph_from_onnx(path: Path | str, fold_batchnorm: bool = True) -> Graph:
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "读取 .onnx 需要安装 onnx 包（pip install onnx）。"
            "边缘设备上请改用构建期导出的 .hgraph.npz。"
        ) from exc

    model = onnx.load(str(path))
    g = model.graph
    tensors: Dict[str, np.ndarray] = {init.name: numpy_helper.to_array(init) for init in g.initializer}
    init_names = set(tensors)
    nodes = [
        Node(op=n.op_type, inputs=[i for i in n.input], outputs=list(n.output),
             attrs={a.name: _attr_value(a) for a in n.attribute})
        for n in g.node
    ]
    graph = Graph(
        nodes=nodes,
        graph_inputs=[i.name for i in g.input if i.name not in init_names],
        graph_outputs=[o.name for o in g.output],
        tensors=tensors,
        input_spec=_input_spec_from_onnx(g.input[0]) if g.input else None,
    )
    if fold_batchnorm:
        graph = fold_batchnorm_into_conv(graph)
    return graph


def fold_batchnorm_into_conv(graph: Graph) -> Graph:
    """把 Conv→BatchNormalization 折叠成带 bias 的 Conv。

    MobileNetV3 里几乎每个卷积后面都跟着 BN，折叠后节点数减少约三分之一，
    在纯 NumPy 后端上能省下相当可观的一次全张量读写。
    """
    producer: Dict[str, Node] = {}
    for n in graph.nodes:
        for o in n.outputs:
            producer[o] = n

    new_nodes: List[Node] = []
    skip: set = set()
    extra_tensors: Dict[str, np.ndarray] = {}

    for node in graph.nodes:
        if id(node) in skip or node.op != "BatchNormalization":
            new_nodes.append(node)
            continue
        src = producer.get(node.inputs[0])
        if src is None or src.op != "Conv" or any(i in src.inputs[1:] and i not in graph.tensors
                                                  for i in src.inputs[1:]):
            new_nodes.append(node)
            continue

        w = graph.tensors.get(src.inputs[1])
        if w is None:
            new_nodes.append(node)
            continue
        scale = graph.tensors.get(node.inputs[1])
        bias = graph.tensors.get(node.inputs[2])
        mean = graph.tensors.get(node.inputs[3])
        var = graph.tensors.get(node.inputs[4])
        if scale is None or bias is None or mean is None or var is None:
            new_nodes.append(node)
            continue

        eps = float(node.attrs.get("epsilon", 1e-5))
        inv = scale / np.sqrt(var + eps)
        w_name = f"{src.outputs[0]}__folded_w"
        b_name = f"{src.outputs[0]}__folded_b"
        extra_tensors[w_name] = (w * inv.reshape(1, -1, 1, 1)).astype(np.float32)
        old_b = graph.tensors.get(src.inputs[2]) if len(src.inputs) > 2 else None
        old_b = np.zeros_like(bias) if old_b is None else old_b
        extra_tensors[b_name] = ((old_b - mean) * inv + bias).astype(np.float32)

        folded = Node(op="Conv", inputs=[src.inputs[0], w_name, b_name],
                      outputs=[node.outputs[0]], attrs=dict(src.attrs))
        new_nodes.append(folded)
        skip.add(id(src))

    if not extra_tensors:
        return graph

    kept = [n for n in new_nodes if id(n) not in skip]
    tensors = dict(graph.tensors)
    tensors.update(extra_tensors)
    used = {i for n in kept for i in n.inputs}
    tensors = {k: v for k, v in tensors.items() if k in used}
    return Graph(nodes=kept, graph_inputs=graph.graph_inputs, graph_outputs=graph.graph_outputs,
                 tensors=tensors, input_spec=graph.input_spec)


def graph_stats(graph: Graph) -> Dict[str, Any]:
    from collections import Counter

    ops = Counter(n.op for n in graph.nodes)
    params = int(sum(int(np.prod(v.shape)) for v in graph.tensors.values()))
    bytes_ = int(sum(v.nbytes for v in graph.tensors.values()))
    return {
        "nodes": len(graph.nodes),
        "ops": dict(ops),
        "tensors": len(graph.tensors),
        "params": params,
        "param_mb": round(bytes_ / (1024 * 1024), 3),
        "dtypes": sorted({str(v.dtype) for v in graph.tensors.values()}),
    }
