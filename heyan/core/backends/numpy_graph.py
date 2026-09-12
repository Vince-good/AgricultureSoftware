"""纯 NumPy 计算图执行器（零依赖回退后端）。

为什么要有这个东西：300 元级开发板、老版本 Android 的 Termux、某些国产 Linux
板卡上，onnxruntime 的 wheel 经常装不上（glibc 太旧、没有 manylinux 包）。
方案文档要求"识别核心与交互界面解耦，便于后续移植到不同硬件平台"，
所以核心必须有一条只依赖 numpy 的执行路径。

图的序列化格式见 `heyan/core/portable.py`：构建期在有 onnx 的工作站上把
MobileNetV3 导出成 .onnx，同时转成 graph.json + tensors.npz；边缘设备只读后者。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import Backend, InputSpec


@dataclass
class Node:
    op: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Graph:
    nodes: List[Node]
    graph_inputs: List[str]
    graph_outputs: List[str]
    tensors: Dict[str, np.ndarray] = field(default_factory=dict)
    input_spec: Optional[InputSpec] = None

    def num_classes(self) -> int:
        for t in reversed(self.tensors.values()):
            if t.ndim == 1 and t.size > 1:
                return int(t.size)
        return 0


# --------------------------------------------------------------------------
# 算子实现
# --------------------------------------------------------------------------

def _as_list(v: Any, default: Sequence[int]) -> List[int]:
    if v is None:
        return list(default)
    if isinstance(v, (int, float)):
        return [int(v)]
    return [int(x) for x in v]


def _pair(v: Any, default: int = 1) -> Tuple[int, int]:
    lst = _as_list(v, [default])
    if len(lst) == 1:
        return lst[0], lst[0]
    return lst[0], lst[1]


def _pad_nchw(x: np.ndarray, pt: int, pl: int, pb: int, pr: int) -> np.ndarray:
    if not (pt or pl or pb or pr):
        return x
    return np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), mode="constant", constant_values=0)


def conv2d(x: np.ndarray, w: np.ndarray, b: Optional[np.ndarray], attrs: Dict[str, Any]) -> np.ndarray:
    sh, sw = _pair(attrs.get("strides"), 1)
    dh, dw = _pair(attrs.get("dilations"), 1)
    kh, kw = w.shape[2], w.shape[3]
    group = int(attrs.get("group", 1) or 1)
    pads = _as_list(attrs.get("pads"), [0, 0, 0, 0])
    if len(pads) == 4:
        pt, pl, pb, pr = pads
    else:
        pt = pl = pb = pr = pads[0] if pads else 0
    auto_pad = str(attrs.get("auto_pad", "NOTSET"))

    n, cin, h, wd = x.shape
    kout, cin_g = w.shape[0], w.shape[1]

    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        oh = int(np.ceil(h / sh)); ow = int(np.ceil(wd / sw))
        pad_h = max((oh - 1) * sh + dh * (kh - 1) + 1 - h, 0)
        pad_w = max((ow - 1) * sw + dw * (kw - 1) + 1 - wd, 0)
        if auto_pad == "SAME_UPPER":
            pt, pb = pad_h // 2, pad_h - pad_h // 2
            pl, pr = pad_w // 2, pad_w - pad_w // 2
        else:
            pt, pb = pad_h - pad_h // 2, pad_h // 2
            pl, pr = pad_w - pad_w // 2, pad_w // 2

    xp = _pad_nchw(x, pt, pl, pb, pr)
    oh = (xp.shape[2] - dh * (kh - 1) - 1) // sh + 1
    ow = (xp.shape[3] - dw * (kw - 1) - 1) // sw + 1

    # 1x1 且无分组：直接一次大矩阵乘，这是 MobileNetV3 里占比最高的算子
    if kh == 1 and kw == 1 and group == 1 and sh == 1 and sw == 1 and dh == 1 and dw == 1:
        out = np.matmul(w.reshape(kout, cin), xp.reshape(n, cin, oh * ow))
        out = out.reshape(n, kout, oh, ow)
        if b is not None:
            out = out + b.reshape(1, -1, 1, 1)
        return out.astype(np.float32, copy=False)

    # 深度可分离卷积的深度部分：9 次广播乘加，避免逐通道循环
    if group == cin == kout:
        out = np.zeros((n, kout, oh, ow), dtype=np.float32)
        for i in range(kh):
            for j in range(kw):
                y0 = i * dh; x0 = j * dw
                patch = xp[:, :, y0:y0 + oh * sh:sh, x0:x0 + ow * sw:sw]
                if patch.shape[2:] != (oh, ow):  # 边界不足时补齐
                    tmp = np.zeros((n, cin, oh, ow), dtype=patch.dtype)
                    tmp[:, :, :patch.shape[2], :patch.shape[3]] = patch
                    patch = tmp
                out += patch * w[:, 0, i, j].reshape(1, -1, 1, 1)
        if b is not None:
            out += b.reshape(1, -1, 1, 1)
        return out

    # 通用分组卷积：按 kernel 位置做 (Kg,Cg) @ (N,Cg,OH*OW) 的批量矩阵乘
    cg = cin // group
    kg = kout // group
    out = np.zeros((n, kout, oh, ow), dtype=np.float32)
    for g in range(group):
        xg = xp[:, g * cg:(g + 1) * cg]
        wg = w[g * kg:(g + 1) * kg]
        acc = np.zeros((n, kg, oh * ow), dtype=np.float32)
        for i in range(kh):
            for j in range(kw):
                y0 = i * dh; x0 = j * dw
                patch = xg[:, :, y0:y0 + oh * sh:sh, x0:x0 + ow * sw:sw]
                if patch.shape[2:] != (oh, ow):
                    tmp = np.zeros((n, cg, oh, ow), dtype=patch.dtype)
                    tmp[:, :, :patch.shape[2], :patch.shape[3]] = patch
                    patch = tmp
                acc += np.matmul(wg[:, :, i, j], patch.reshape(n, cg, oh * ow))
        out[:, g * kg:(g + 1) * kg] = acc.reshape(n, kg, oh, ow)
    if b is not None:
        out += b.reshape(1, -1, 1, 1)
    return out


def batch_norm(x: np.ndarray, scale: np.ndarray, bias: np.ndarray, mean: np.ndarray,
               var: np.ndarray, eps: float) -> np.ndarray:
    shape = (1, -1) + (1,) * (x.ndim - 2)
    y = (x - mean.reshape(shape)) / np.sqrt(var.reshape(shape) + eps)
    return (y * scale.reshape(shape) + bias.reshape(shape)).astype(np.float32, copy=False)


def _pool(x: np.ndarray, kh: int, kw: int, sh: int, sw: int, pads: Sequence[int],
          mode: str, count_include_pad: bool = True, ceil_mode: bool = False) -> np.ndarray:
    pt, pl = (pads[0], pads[1]) if len(pads) >= 2 else (0, 0)
    pb, pr = (pads[2], pads[3]) if len(pads) >= 4 else (pt, pl)
    n, c, h, w = x.shape
    xp = _pad_nchw(x.astype(np.float32), pt, pl, pb, pr) if (pt or pl or pb or pr) else x.astype(np.float32)
    if ceil_mode:
        oh = int(np.ceil((xp.shape[2] - kh) / sh + 1)); ow = int(np.ceil((xp.shape[3] - kw) / sw + 1))
    else:
        oh = (xp.shape[2] - kh) // sh + 1; ow = (xp.shape[3] - kw) // sw + 1
    init = -np.inf if mode == "max" else 0.0
    out = np.full((n, c, oh, ow), init, dtype=np.float32)
    cnt = np.zeros((n, c, oh, ow), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            y0 = i; x0 = j
            patch = xp[:, :, y0:y0 + oh * sh:sh, x0:x0 + ow * sw:sw]
            if patch.shape[2:] != (oh, ow):
                tmp = np.full((n, c, oh, ow), init, dtype=np.float32)
                tmp[:, :, :patch.shape[2], :patch.shape[3]] = patch
                patch = tmp
            if mode == "max":
                out = np.maximum(out, patch)
            else:
                out += patch
                cnt += 1.0
    if mode == "avg":
        if count_include_pad:
            denom = float(kh * kw)
        else:
            denom = np.maximum(cnt, 1.0)
        out = out / denom
    return out.astype(np.float32, copy=False)


def gemm(a: np.ndarray, b: np.ndarray, c: Optional[np.ndarray], attrs: Dict[str, Any]) -> np.ndarray:
    alpha = float(attrs.get("alpha", 1.0)); beta = float(attrs.get("beta", 1.0))
    if int(attrs.get("transA", 0)):
        a = a.T
    if int(attrs.get("transB", 0)):
        b = b.T
    out = alpha * np.matmul(a.astype(np.float32), b.astype(np.float32))
    if c is not None:
        out = out + beta * c.astype(np.float32)
    return out.astype(np.float32, copy=False)


def _clip(x: np.ndarray, lo: Optional[np.ndarray], hi: Optional[np.ndarray]) -> np.ndarray:
    out = x
    if lo is not None:
        out = np.maximum(out, float(np.asarray(lo).reshape(-1)[0]))
    if hi is not None:
        out = np.minimum(out, float(np.asarray(hi).reshape(-1)[0]))
    return out.astype(np.float32, copy=False)


class Interpreter:
    """按拓扑序（ONNX 节点顺序即拓扑序）执行 Graph。"""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def run(self, feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
        vals: Dict[str, np.ndarray] = dict(self.graph.tensors)
        vals.update(feed)
        for node in self.graph.nodes:
            ins = [vals.get(name) for name in node.inputs]
            outs = self._exec(node, ins)
            if not isinstance(outs, (list, tuple)):
                outs = [outs]
            for name, value in zip(node.outputs, outs):
                if name:
                    vals[name] = value
        missing = [o for o in self.graph.graph_outputs if o not in vals]
        if missing:
            raise RuntimeError(f"图执行后缺少输出: {missing}")
        return [vals[o] for o in self.graph.graph_outputs]

    # -- 单节点分派 --
    def _exec(self, node: Node, ins: List[Optional[np.ndarray]]) -> Any:
        op = node.op
        a = node.attrs

        if op == "Conv":
            return conv2d(ins[0], ins[1], ins[2] if len(ins) > 2 else None, a)
        if op == "BatchNormalization":
            return batch_norm(ins[0], ins[1], ins[2], ins[3], ins[4], float(a.get("epsilon", 1e-5)))
        if op in ("Relu", "ReLU"):
            return np.maximum(ins[0], 0.0).astype(np.float32, copy=False)
        if op == "Sigmoid":
            return (1.0 / (1.0 + np.exp(-ins[0].astype(np.float32)))).astype(np.float32)
        if op == "Tanh":
            return np.tanh(ins[0].astype(np.float32)).astype(np.float32)
        if op == "Exp":
            return np.exp(ins[0].astype(np.float32)).astype(np.float32)
        if op == "Sqrt":
            return np.sqrt(ins[0].astype(np.float32)).astype(np.float32)
        if op == "Clip":
            lo = ins[1] if len(ins) > 1 else None
            hi = ins[2] if len(ins) > 2 else None
            if lo is None and "min" in a:
                lo = np.float32(a["min"])
            if hi is None and "max" in a:
                hi = np.float32(a["max"])
            return _clip(ins[0], lo, hi)
        if op == "HardSigmoid":
            alpha = float(a.get("alpha", 0.2)); beta = float(a.get("beta", 0.5))
            return _clip(ins[0].astype(np.float32) * alpha + beta, 0.0, 1.0)
        if op in ("Hardswish", "HardSwish"):
            x = ins[0].astype(np.float32)
            return (x * _clip(x + 3.0, 0.0, 6.0) / 6.0).astype(np.float32)
        if op == "Mul":
            return (ins[0] * ins[1]).astype(np.float32, copy=False)
        if op == "Add":
            return (ins[0] + ins[1]).astype(np.float32, copy=False)
        if op == "Sub":
            return (ins[0] - ins[1]).astype(np.float32, copy=False)
        if op == "Div":
            return (ins[0] / ins[1]).astype(np.float32, copy=False)
        if op == "Pow":
            return np.power(ins[0].astype(np.float32), ins[1].astype(np.float32)).astype(np.float32)
        if op == "GlobalAveragePool":
            axes = tuple(range(2, ins[0].ndim))
            return np.mean(ins[0].astype(np.float32), axis=axes, keepdims=True).astype(np.float32)
        if op == "AveragePool":
            kh, kw = _pair(a.get("kernel_shape"), 1)
            sh, sw = _pair(a.get("strides"), 1)
            return _pool(ins[0], kh, kw, sh, sw, _as_list(a.get("pads"), [0, 0, 0, 0]), "avg",
                         bool(a.get("count_include_pad", 1)), bool(a.get("ceil_mode", 0)))
        if op == "MaxPool":
            kh, kw = _pair(a.get("kernel_shape"), 1)
            sh, sw = _pair(a.get("strides"), 1)
            return _pool(ins[0], kh, kw, sh, sw, _as_list(a.get("pads"), [0, 0, 0, 0]), "max",
                         ceil_mode=bool(a.get("ceil_mode", 0)))
        if op == "Gemm":
            return gemm(ins[0], ins[1], ins[2] if len(ins) > 2 else None, a)
        if op == "MatMul":
            return np.matmul(ins[0].astype(np.float32), ins[1].astype(np.float32)).astype(np.float32)
        if op == "Reshape":
            shape = [int(v) for v in np.asarray(ins[1]).reshape(-1)]
            return ins[0].reshape(shape).astype(np.float32, copy=False)
        if op == "Flatten":
            axis = int(a.get("axis", 1))
            shp = ins[0].shape
            left = int(np.prod(shp[:axis])) if axis > 0 else 1
            return ins[0].reshape(left, -1).astype(np.float32, copy=False)
        if op == "Squeeze":
            axes = _as_list(a.get("axes"), [])
            if not axes and len(ins) > 1 and ins[1] is not None:
                axes = [int(v) for v in np.asarray(ins[1]).reshape(-1)]
            out = ins[0]
            for ax in sorted(axes, reverse=True):
                out = np.squeeze(out, axis=ax)
            return out
        if op == "Unsqueeze":
            axes = _as_list(a.get("axes"), [])
            if not axes and len(ins) > 1 and ins[1] is not None:
                axes = [int(v) for v in np.asarray(ins[1]).reshape(-1)]
            out = ins[0]
            for ax in sorted(axes):
                out = np.expand_dims(out, axis=ax)
            return out
        if op == "Transpose":
            perm = a.get("perm")
            return np.ascontiguousarray(np.transpose(ins[0], perm))
        if op == "Concat":
            axis = int(a.get("axis", 0))
            return np.concatenate([i for i in ins if i is not None], axis=axis)
        if op == "Identity":
            return ins[0]
        if op == "Cast":
            to = a.get("to", 1)
            return ins[0].astype(_ONNX_DTYPE.get(int(to), np.float32))
        if op == "QuantizeLinear":
            scale = float(np.asarray(ins[1]).reshape(-1)[0])
            zp = np.asarray(ins[2]).reshape(-1)[0] if len(ins) > 2 and ins[2] is not None else 0
            q = np.rint(ins[0].astype(np.float32) / scale) + zp
            return np.clip(q, 0, 255).astype(np.uint8)
        if op == "DequantizeLinear":
            scale = np.asarray(ins[1]).astype(np.float32)
            zp = np.asarray(ins[2]).astype(np.float32) if len(ins) > 2 and ins[2] is not None else 0.0
            return ((ins[0].astype(np.float32) - zp) * scale).astype(np.float32)
        if op == "Shape":
            return np.asarray(ins[0].shape, dtype=np.int64)
        if op == "Gather":
            axis = int(a.get("axis", 0))
            return np.take(ins[0], np.asarray(ins[1]).astype(np.int64), axis=axis)
        if op == "ReduceMean":
            axes = _as_list(a.get("axes"), [])
            if not axes and len(ins) > 1 and ins[1] is not None:
                axes = [int(v) for v in np.asarray(ins[1]).reshape(-1)]
            keep = bool(a.get("keepdims", 1))
            return np.mean(ins[0].astype(np.float32), axis=tuple(axes) or None, keepdims=keep)
        if op == "Constant":
            if "value" in a:
                return np.asarray(a["value"])
            return np.asarray(a.get("value_float", 0.0), dtype=np.float32)
        if op == "Dropout":
            return ins[0]
        raise NotImplementedError(f"NumPy 后端暂不支持算子: {op}")


_ONNX_DTYPE = {
    1: np.float32, 2: np.uint8, 3: np.int8, 5: np.int16, 6: np.int32, 7: np.int64,
    9: bool, 10: np.float16, 11: np.float64, 12: np.uint32, 13: np.uint64,
}


class NumpyGraphBackend(Backend):
    """只依赖 numpy 的后端。加载 graph.json + tensors.npz，或（装了 onnx 时）直接读 .onnx。"""

    name = "numpy"
    requires = ("numpy",)

    def _load(self) -> None:
        from ..portable import load_graph

        self.graph = load_graph(self.model_path)
        self._interp = Interpreter(self.graph)
        in_name = self.graph.graph_inputs[0]
        shape = (1, 3, 224, 224)
        if self.graph.input_spec is not None:
            self._input_spec = self.graph.input_spec
        else:
            self._input_spec = InputSpec(name=in_name, shape=shape, dtype="float32")

    def run(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = self._interp.run({self.graph.graph_inputs[0]: x})
        return np.asarray(out[0], dtype=np.float32)

    def info(self) -> Dict[str, Any]:
        base = super().info()
        base["nodes"] = len(self.graph.nodes)
        base["ops"] = sorted({n.op for n in self.graph.nodes})
        return base
