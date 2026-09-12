"""ONNX 导出 + 可移植图导出。

固定 batch=1、固定 224x224 输入，不留动态维度。边缘设备上动态 shape 会挡掉
一部分图优化，也让纯 NumPy 后端的实现复杂度陡增，没有任何收益。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..config import INPUT_SIZE
from ..core.portable import graph_from_onnx, graph_stats, save_graph
from .model import DEFAULT_ARCH, build_model

OPSET = 17


def load_checkpoint(checkpoint: Path | str, arch: str = DEFAULT_ARCH,
                    num_classes: Optional[int] = None,
                    low_rank_head: int = 0) -> tuple[torch.nn.Module, Dict[str, Any], List[str]]:
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"{checkpoint} 不是 heyan 训练产出的检查点")
    class_ids: List[str] = list(ckpt.get("class_ids") or [])
    n_cls = int(num_classes or len(class_ids) or 0)
    if n_cls <= 0:
        raise ValueError("无法确定类别数，检查点里缺 class_ids")
    model = build_model(arch, n_cls, pretrained=True, low_rank_head=low_rank_head)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt, class_ids


def export_module_onnx(model: torch.nn.Module, out_path: Path | str,
                       input_size: int = INPUT_SIZE, opset: int = OPSET) -> Dict[str, Any]:
    """导出任意内存中的模型对象。

    剪枝后的网络结构与 `build_model()` 的产物不同，无法再从检查点重建，
    所以流水线里走这条路径。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval()

    dummy = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
    kwargs: Dict[str, Any] = dict(
        input_names=["input"], output_names=["logits"], export_params=True,
        opset_version=opset, do_constant_folding=True,
    )
    try:
        torch.onnx.export(model, (dummy,), str(out_path), dynamo=False, **kwargs)
    except TypeError:  # 老版本 torch 没有 dynamo 参数
        torch.onnx.export(model, (dummy,), str(out_path), **kwargs)

    import onnx

    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model, full_check=False)
    ops = sorted({n.op_type for n in onnx_model.graph.node})
    size_mb = round(out_path.stat().st_size / (1024 * 1024), 3)
    return {
        "path": str(out_path),
        "input_size": input_size,
        "opset": opset,
        "size_mb": size_mb,
        "nodes": len(onnx_model.graph.node),
        "ops": ops,
    }


def export_onnx(checkpoint: Path | str, out_path: Path | str, arch: str = DEFAULT_ARCH,
                num_classes: Optional[int] = None, input_size: int = INPUT_SIZE,
                opset: int = OPSET, low_rank_head: int = 0) -> Dict[str, Any]:
    model, ckpt, class_ids = load_checkpoint(checkpoint, arch, num_classes, low_rank_head)
    info = export_module_onnx(model, out_path, input_size=input_size, opset=opset)
    info.update({
        "arch": arch,
        "num_classes": len(class_ids) or int(num_classes or 0),
        "class_ids": class_ids,
        "checkpoint": str(checkpoint),
        "train_metrics": ckpt.get("metrics", {}),
    })
    Path(info["path"]).with_suffix(".export.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[export] ONNX -> {info['path']} ({info['size_mb']} MB, {info['nodes']} 节点)")
    return info


def export_portable(onnx_path: Path | str, out_path: Path | str,
                    fold_batchnorm: bool = True) -> Dict[str, Any]:
    """把 ONNX 转成纯 NumPy 后端可读的单文件图，供没有 onnxruntime 的设备使用。"""
    onnx_path = Path(onnx_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    graph = graph_from_onnx(onnx_path, fold_batchnorm=fold_batchnorm)
    save_graph(graph, out_path)
    stats = graph_stats(graph)
    stats["path"] = str(out_path)
    stats["size_mb"] = round(out_path.stat().st_size / (1024 * 1024), 3)
    print(f"[export] 可移植图 -> {out_path} ({stats['size_mb']} MB, {stats['nodes']} 节点, "
          f"{stats['params']:,} 参数)")
    return stats


def export_torchscript(checkpoint: Path | str, out_path: Path | str, arch: str = DEFAULT_ARCH,
                       num_classes: Optional[int] = None, input_size: int = INPUT_SIZE) -> Path:
    """开发机对拍用；边缘设备不用这个。"""
    model, _, _ = load_checkpoint(checkpoint, arch, num_classes)
    model.eval()
    traced = torch.jit.trace(model, torch.zeros(1, 3, input_size, input_size))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    print(f"[export] TorchScript -> {out_path}")
    return out_path
