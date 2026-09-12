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


SE_BLOCK_TYPES = ("SqueezeExcitation", "SEBlock", "SqueezeExcite", "SEModule")


def _is_se_block(module: nn.Module) -> bool:
    """SE 门控本身已是极低秩瓶颈（实测常近 rank 1）。

    再对它做低秩分解省不下几个参数，却很容易毁掉通道注意力，所以整块跳过。"""
    return type(module).__name__ in SE_BLOCK_TYPES


def _qualified_names(model: nn.Module) -> Dict[int, str]:
    return {id(m): n for n, m in model.named_modules()}


def _find_pointwise_convs(model: nn.Module, min_channels: int,
                          skip_se: bool = True) -> List[Tuple[nn.Module, str, nn.Conv2d]]:
    """定位所有可分解的 1x1 卷积，返回 (父模块, 属性名, 卷积层)。"""
    found: List[Tuple[nn.Module, str, nn.Conv2d]] = []
    for parent in model.modules():
        if skip_se and _is_se_block(parent):
            continue
        for name, child in parent.named_children():
            if isinstance(child, nn.Conv2d) and child.kernel_size == (1, 1) \
                    and child.groups == 1 and child.dilation == (1, 1):
                if min(child.in_channels, child.out_channels) >= min_channels:
                    found.append((parent, name, child))
    return found


def _find_linears(model: nn.Module, min_features: int,
                  skip_se: bool = True) -> List[Tuple[nn.Module, str, nn.Linear]]:
    """定位可分解的全连接层。

    MobileNetV3-Small 的参数大头其实在 classifier 的第一层 Linear（576->1024，
    约占全模型 38%）。只盯着 1x1 卷积会完全错过它，剪枝也就形同虚设。"""
    found: List[Tuple[nn.Module, str, nn.Linear]] = []
    if min_features <= 0:
        return found
    for parent in model.modules():
        if skip_se and _is_se_block(parent):
            continue
        for name, child in parent.named_children():
            if isinstance(child, nn.Linear) \
                    and min(child.in_features, child.out_features) >= min_features:
                found.append((parent, name, child))
    return found


def _split_pair_conv(conv: nn.Conv2d, rank: int,
                     v_r: np.ndarray, u_r: np.ndarray) -> nn.Sequential:
    a = nn.Conv2d(conv.in_channels, rank, kernel_size=1, stride=1,
                  padding=0, dilation=1, groups=1, bias=False)
    b = nn.Conv2d(rank, conv.out_channels, kernel_size=1, stride=conv.stride[0],
                  padding=conv.padding[0], dilation=1, groups=1,
                  bias=conv.bias is not None)
    with torch.no_grad():
        a.weight.copy_(torch.from_numpy(v_r.reshape(rank, conv.in_channels, 1, 1)))
        b.weight.copy_(torch.from_numpy(u_r.reshape(conv.out_channels, rank, 1, 1)))
        if conv.bias is not None:
            b.bias.copy_(conv.bias.detach())
    return nn.Sequential(a, b)


def _split_pair_linear(fc: nn.Linear, rank: int,
                       v_r: np.ndarray, u_r: np.ndarray) -> nn.Sequential:
    a = nn.Linear(fc.in_features, rank, bias=False)
    b = nn.Linear(rank, fc.out_features, bias=fc.bias is not None)
    with torch.no_grad():
        a.weight.copy_(torch.from_numpy(v_r.reshape(rank, fc.in_features)))
        b.weight.copy_(torch.from_numpy(u_r.reshape(fc.out_features, rank)))
        if fc.bias is not None:
            b.bias.copy_(fc.bias.detach())
    return nn.Sequential(a, b)


def low_rank_compress(model: nn.Module, energy: float = 0.85, min_channels: int = 24,
                      min_gain: float = 0.15, verbose: bool = True,
                      min_linear: int = 256, skip_se: bool = True) -> CompressionReport:
    """就地把符合条件的 1x1 卷积 / 全连接层换成两级低秩结构。

    energy 是保留的奇异值能量占比：调得越低秩越小、省得越多、精度风险越大。
    MobileNetV3-Small 权重接近满秩，energy=0.95 基本压不动（实测 -0.0%），
    0.85 才有约 13% 的真实收益，因此默认取 0.85，并由恢复训练 + 回退兜底精度。"""
    report = CompressionReport(params_before=count(model), size_mb_before=_mb(count(model)),
                               energy=energy)
    qnames = _qualified_names(model)

    conv_targets = _find_pointwise_convs(model, min_channels, skip_se=skip_se)
    lin_targets = _find_linears(model, min_linear, skip_se=skip_se)
    if verbose:
        print(f"[prune] 候选 1x1 卷积 {len(conv_targets)} 个（min_channels={min_channels}）, "
              f"全连接 {len(lin_targets)} 个（min_features={min_linear}）, energy={energy}")

    jobs: List[Tuple[str, nn.Module, str, nn.Module, int, int, np.ndarray]] = []
    for parent, name, conv in conv_targets:
        jobs.append((qnames.get(id(conv), name), parent, name, conv,
                     conv.out_channels, conv.in_channels,
                     conv.weight.detach().cpu().numpy().reshape(conv.out_channels,
                                                                conv.in_channels)))
    for parent, name, fc in lin_targets:
        jobs.append((qnames.get(id(fc), name), parent, name, fc,
                     fc.out_features, fc.in_features,
                     fc.weight.detach().cpu().numpy()))

    for qname, parent, name, layer, out_dim, in_dim, w in jobs:
        try:
            u, s, vt = np.linalg.svd(w, full_matrices=False)
        except np.linalg.LinAlgError:  # pragma: no cover
            continue
        rank = _rank_for_energy(s, energy)
        before = in_dim * out_dim
        after = in_dim * rank + rank * out_dim
        gain = (before - after) / max(before, 1)
        if gain < min_gain or rank >= min(in_dim, out_dim):
            report.layers.append({"name": qname, "shape": [out_dim, in_dim],
                                  "rank": rank, "gain": round(gain, 4), "applied": False,
                                  "reason": "gain_below_threshold"})
            continue

        u_r = (u[:, :rank] * s[:rank]).astype(np.float32)
        v_r = vt[:rank, :].astype(np.float32)
        pair = (_split_pair_conv(layer, rank, v_r, u_r) if isinstance(layer, nn.Conv2d)
                else _split_pair_linear(layer, rank, v_r, u_r))
        for p in pair.parameters():
            p.requires_grad = layer.weight.requires_grad

        setattr(parent, name, pair)
        report.layers_compressed += 1
        report.layers.append({"name": qname, "shape": [out_dim, in_dim],
                              "rank": rank, "gain": round(gain, 4), "applied": True})
        if verbose:
            print(f"[prune]   {qname}: {out_dim}x{in_dim} -> rank {rank} "
                  f"(params {before:,} -> {after:,}, -{gain:.1%})")

    report.params_after = count(model)
    report.size_mb_after = _mb(report.params_after)
    if verbose:
        if report.layers_compressed == 0:
            print(f"[prune] 没有层满足条件（min_gain={min_gain}）：该架构权重接近满秩，"
                  f"低秩分解无收益，跳过压缩")
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
