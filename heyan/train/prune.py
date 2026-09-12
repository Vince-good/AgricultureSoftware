"""模型压缩：低秩分解 + 稀疏度分析（文档 3.2(2) 的"网络剪枝"路径）。

为什么用低秩分解而不是直接删通道：MobileNetV3 的通道数是 NAS 搜出来的，
随手删掉几个通道会破坏 block 间的宽度比例，恢复训练成本很高，
而且 torchvision 的结构不支持动态改通道数。

低秩分解针对的是真正吃体积的部分 —— 1x1 逐点卷积。把 W(out,in) 分解成
U(out,r)·V(r,in)，当 r 远小于 min(in,out) 时参数量线性下降，精度损失可控，
分解后只需短暂微调即可恢复。这是可以直接落到 ONNX 上的结构性压缩。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class CompressionReport:
    params_before: int = 0
    params_after: int = 0
    size_mb_before: float = 0.0
    size_mb_after: float = 0.0
    layers_compressed: int = 0
    layers: List[Dict[str, Any]] = field(default_factory=list)
    method: str = "low_rank_factorization"
    energy: float = 0.95

    @property
    def reduction(self) -> float:
        if self.params_before == 0:
            return 0.0
        return round(1.0 - self.params_after / self.params_before, 4)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["reduction"] = self.reduction
        return d


def _mb(params: int) -> float:
    return round(params * 4 / (1024 * 1024), 3)


def count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _rank_for_energy(singular: np.ndarray, energy: float) -> int:
    total = float((singular ** 2).sum())
    if total <= 0:
        return 1
    cum = np.cumsum(singular ** 2) / total
    r = int(np.searchsorted(cum, energy) + 1)
    return max(1, min(r, len(singular)))


def _find_pointwise_convs(model: nn.Module, min_channels: int) -> List[Tuple[nn.Module, str, nn.Conv2d]]:
    """定位所有可分解的 1x1 卷积，返回 (父模块, 属性名, 卷积层)。"""
    found: List[Tuple[nn.Module, str, nn.Conv2d]] = []
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Conv2d) and child.kernel_size == (1, 1) \
                    and child.groups == 1 and child.dilation == (1, 1):
                if min(child.in_channels, child.out_channels) >= min_channels:
                    found.append((parent, name, child))
    return found


def low_rank_compress(model: nn.Module, energy: float = 0.95, min_channels: int = 128,
                      min_gain: float = 0.15, verbose: bool = True) -> CompressionReport:
    """就地替换符合条件的 1x1 卷积为两级低秩卷积。"""
    report = CompressionReport(params_before=count(model), size_mb_before=_mb(count(model)),
                               energy=energy)
    targets = _find_pointwise_convs(model, min_channels)
    if verbose:
        print(f"[prune] 候选 1x1 卷积 {len(targets)} 个（min_channels={min_channels}）")

    for parent, name, conv in targets:
        w = conv.weight.detach().cpu().numpy().reshape(conv.out_channels, conv.in_channels)
        try:
            u, s, vt = np.linalg.svd(w, full_matrices=False)
        except np.linalg.LinAlgError:  # pragma: no cover
            continue
        rank = _rank_for_energy(s, energy)
        before = conv.in_channels * conv.out_channels
        after = conv.in_channels * rank + rank * conv.out_channels
        gain = (before - after) / max(before, 1)
        if gain < min_gain or rank >= min(conv.in_channels, conv.out_channels):
            report.layers.append({"name": name, "shape": [conv.out_channels, conv.in_channels],
                                  "rank": rank, "gain": round(gain, 4), "applied": False,
                                  "reason": "gain_below_threshold"})
            continue

        u_r = (u[:, :rank] * s[:rank]).astype(np.float32)
        v_r = vt[:rank, :].astype(np.float32)

        conv_a = nn.Conv2d(conv.in_channels, rank, kernel_size=1, stride=1,
                           padding=0, dilation=1, groups=1, bias=False)
        conv_b = nn.Conv2d(rank, conv.out_channels, kernel_size=1, stride=conv.stride[0],
                           padding=conv.padding[0], dilation=1, groups=1,
                           bias=conv.bias is not None)
        with torch.no_grad():
            conv_a.weight.copy_(torch.from_numpy(v_r.reshape(rank, conv.in_channels, 1, 1)))
            conv_b.weight.copy_(torch.from_numpy(u_r.reshape(conv.out_channels, rank, 1, 1)))
            if conv.bias is not None:
                conv_b.bias.copy_(conv.bias.detach())
        conv_a.weight.requires_grad = conv.weight.requires_grad
        conv_b.weight.requires_grad = conv.weight.requires_grad
        if conv_b.bias is not None:
            conv_b.bias.requires_grad = conv.weight.requires_grad

        setattr(parent, name, nn.Sequential(conv_a, conv_b))
        report.layers_compressed += 1
        report.layers.append({"name": name, "shape": [conv.out_channels, conv.in_channels],
                              "rank": rank, "gain": round(gain, 4), "applied": True})
        if verbose:
            print(f"[prune]   {name}: {conv.out_channels}x{conv.in_channels} -> rank {rank} "
                  f"(params {before:,} -> {after:,}, -{gain:.1%})")

    report.params_after = count(model)
    report.size_mb_after = _mb(report.params_after)
    if verbose:
        print(f"[prune] 压缩 {report.layers_compressed} 层，参数 "
              f"{report.params_before:,} -> {report.params_after:,} "
              f"(-{report.reduction:.1%}), FP32 {report.size_mb_before}MB -> {report.size_mb_after}MB")
    return report


def sparsity_report(model: nn.Module) -> Dict[str, Any]:
    """统计各层权重分布，用于判断还有多少剪枝空间。"""
    rows: List[Dict[str, Any]] = []
    total = 0
    near_zero = 0
    for name, p in model.named_parameters():
        if p.ndim < 2:
            continue
        arr = p.detach().cpu().numpy()
        n = arr.size
        threshold = float(np.percentile(np.abs(arr), 10))
        rows.append({
            "name": name,
            "shape": list(arr.shape),
            "params": int(n),
            "l1_mean": round(float(np.abs(arr).mean()), 6),
            "energy_p99": round(float(np.percentile(np.abs(arr), 99)), 6),
            "pct_below_p10": round(float((np.abs(arr) <= threshold).mean()), 4),
        })
        total += n
        near_zero += int((np.abs(arr) <= threshold).sum())
    rows.sort(key=lambda r: -r["params"])
    return {"layers": rows[:20], "total_params": total,
            "low_magnitude_ratio": round(near_zero / max(total, 1), 4)}


def magnitude_prune_(model: nn.Module, sparsity: float = 0.3) -> Dict[str, Any]:
    """全局非结构化幅值剪枝（就地置零）。

    注意：置零本身不会让 ONNX 文件变小，它的作用是配合微调找出真正冗余的连接，
    为后续低秩分解提供依据。默认不在导出流水线里启用。
    """
    masks: Dict[str, torch.Tensor] = {}
    all_vals: List[np.ndarray] = []
    for name, p in model.named_parameters():
        if p.ndim >= 2:
            all_vals.append(p.detach().abs().cpu().numpy().reshape(-1))
    if not all_vals:
        return {"sparsity": 0.0, "threshold": 0.0}
    flat = np.concatenate(all_vals)
    threshold = float(np.percentile(flat, sparsity * 100.0))
    pruned = 0
    total = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim < 2:
                continue
            mask = (p.detach().abs() > threshold).to(p.dtype)
            p.mul_(mask)
            masks[name] = mask
            pruned += int((mask == 0).sum())
            total += int(mask.numel())
    return {"sparsity": round(pruned / max(total, 1), 4), "threshold": threshold,
            "pruned": pruned, "total": total}
