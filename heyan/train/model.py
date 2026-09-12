"""模型构建与参数高效微调策略。

文档 3.2(1) 的选型结论是 MobileNetV3；同时把 ShuffleNet V2 与 EfficientNet-B0
也放进来，作为"轻量化模型选型参考表"里可实测对比的候选（见 docs/TECH_ROUTE.md）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# 候选架构：体积/算力从小到大大致递增
ARCHS: Dict[str, Dict[str, Any]] = {
    "mobilenet_v3_small": {
        "factory": "mobilenet_v3_small",
        "weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
        "feature_dim": 576,
        "reported_params_m": 2.5,
        "note": "文档首选：精度—速度—体积最均衡",
    },
    "mobilenet_v3_large": {
        "factory": "mobilenet_v3_large",
        "weights": "MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        "feature_dim": 960,
        "reported_params_m": 5.5,
        "note": "作为知识蒸馏的教师模型",
    },
    "shufflenet_v2_x0_5": {
        "factory": "shufflenet_v2_x0_5",
        "weights": "ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1",
        "feature_dim": 1024,
        "reported_params_m": 1.4,
        "note": "算力最紧张时的备选",
    },
    "shufflenet_v2_x1_0": {
        "factory": "shufflenet_v2_x1_0",
        "weights": "ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1",
        "feature_dim": 1024,
        "reported_params_m": 2.3,
        "note": "ShuffleNet 精度更高的档位",
    },
    "efficientnet_b0": {
        "factory": "efficientnet_b0",
        "weights": "EfficientNet_B0_Weights.IMAGENET1K_V1",
        "feature_dim": 1280,
        "reported_params_m": 5.3,
        "note": "EfficientNet-Lite 的公开替代，体积偏大",
    },
}

DEFAULT_ARCH = "mobilenet_v3_small"


@dataclass
class ModelInfo:
    arch: str
    num_classes: int
    params: int
    trainable_params: int
    size_mb_fp32: float
    strategy: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_model(arch: str = DEFAULT_ARCH, num_classes: int = 14, pretrained: bool = True,
                dropout: float = 0.2, low_rank_head: int = 0) -> nn.Module:
    """构建带 ImageNet 预训练权重的骨干，并替换分类头。"""
    import torchvision.models as tvm

    if arch not in ARCHS:
        raise KeyError(f"未知架构 {arch}，可选: {sorted(ARCHS)}")
    spec = ARCHS[arch]
    factory: Callable = getattr(tvm, spec["factory"])
    if pretrained:
        weights = _resolve_weights(tvm, spec["weights"])
        model = factory(weights=weights)
    else:
        model = factory(weights=None)

    if dropout and dropout > 0:
        _apply_dropout(model, dropout)

    feature_dim = spec["feature_dim"]
    head = _build_head(model, feature_dim, num_classes, low_rank_head)
    _attach_head(model, head)
    model._heyan_feature_dim = feature_dim  # type: ignore[attr-defined]
    model._heyan_arch = arch  # type: ignore[attr-defined]
    return model


def _resolve_weights(tvm, dotted: str):
    module_name, attr = dotted.rsplit(".", 1)
    enum = getattr(tvm, module_name)
    return getattr(enum, attr)


def _apply_dropout(model: nn.Module, p: float) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = p


def _build_head(model: nn.Module, feature_dim: int, num_classes: int, low_rank: int) -> nn.Module:
    """重建分类头。low_rank>0 时用低秩瓶颈（LoRA 思路），只训练极少量参数。"""
    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Sequential) and isinstance(classifier[-1], nn.Linear):
        head = nn.Sequential(*list(classifier)[:-1])
        in_dim = classifier[-1].in_features
    else:
        head = nn.Identity()
        in_dim = feature_dim
    if low_rank and low_rank > 0:
        head.add_module("low_rank_down", nn.Linear(in_dim, low_rank))
        head.add_module("low_rank_act", nn.Hardswish(inplace=True))
        head.add_module("low_rank_up", nn.Linear(low_rank, num_classes))
        nn.init.zeros_(head.low_rank_up.weight)  # type: ignore[attr-defined]
        nn.init.zeros_(head.low_rank_up.bias)  # type: ignore[attr-defined]
    else:
        head.add_module("fc", nn.Linear(in_dim, num_classes))
    return head


def _attach_head(model: nn.Module, head: nn.Module) -> None:
    if hasattr(model, "classifier"):
        model.classifier = head
    elif hasattr(model, "fc"):
        model.fc = head
    elif hasattr(model, "heads"):
        model.heads = head
    else:  # pragma: no cover
        raise AttributeError("无法定位分类头，请为该架构补充 _attach_head 分支")


# --------------------------------------------------------------------------
# 微调策略（文档 3.2(3)：小样本条件下的参数高效微调）
# --------------------------------------------------------------------------

STRATEGIES = ("linear_probe", "bitfit", "partial", "full", "recover", "qat")


def apply_strategy(model: nn.Module, strategy: str, unfreeze_blocks: int = 3) -> List[str]:
    """冻结/解冻参数，返回被解冻的参数组名列表。"""
    if strategy not in STRATEGIES:
        raise KeyError(f"未知微调策略 {strategy}，可选: {STRATEGIES}")

    for p in model.parameters():
        p.requires_grad = False

    unfrozen: List[str] = []
    head = getattr(model, "classifier", None) or getattr(model, "fc", None) or getattr(model, "heads", None)

    if strategy in ("linear_probe", "partial", "full", "recover"):
        for name, p in (head.named_parameters() if head is not None else []):
            p.requires_grad = True
            unfrozen.append(f"head.{name}")

    if strategy == "qat":
        # 量化感知训练：全网权重都要能微调，否则模型没法适应 8-bit 网格。
        # 学习率由调用方压到很低（backbone 3e-5 量级），小样本下不至于洗掉预训练知识。
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                for p in module.parameters():
                    p.requires_grad = True
                unfrozen.append(name)
        return sorted(set(unfrozen))

    if strategy == "bitfit":
        # 只更新偏置与 BN 统计量：小样本下最不容易过拟合
        for name, p in model.named_parameters():
            if name.endswith(".bias"):
                p.requires_grad = True
                unfrozen.append(name)
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                for p in module.parameters():
                    p.requires_grad = True
        return sorted(set(unfrozen))

    if strategy == "recover":
        # 剪枝恢复训练：低秩分解产出的因子都是 1x1 卷积，只解冻这些新参数与 BN 统计量，
        # 骨干其余部分保持冻结，避免在几十张田间照片上把预训练知识洗掉。
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1) and module.groups == 1:
                for p in module.parameters():
                    p.requires_grad = True
                unfrozen.append(name)
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                for p in module.parameters():
                    p.requires_grad = True
        return sorted(set(unfrozen))

    if strategy in ("partial", "full"):
        features = getattr(model, "features", None)
        if features is not None and strategy == "partial":
            total = len(features)
            start = max(0, total - unfreeze_blocks)
            for i in range(start, total):
                for name, p in features[i].named_parameters():
                    p.requires_grad = True
                    unfrozen.append(f"features.{i}.{name}")
        elif strategy == "full":
            for name, p in model.named_parameters():
                p.requires_grad = True
                unfrozen.append(name)

    return sorted(set(unfrozen))


def param_groups(model: nn.Module, head_lr: float, backbone_lr: float) -> List[Dict[str, Any]]:
    """分类头用大学习率，预训练骨干用小学习率（迁移学习常规做法）。"""
    head_names = set()
    head = getattr(model, "classifier", None) or getattr(model, "fc", None) or getattr(model, "heads", None)
    if head is not None:
        head_names = {id(p) for p in head.parameters()}
    head_ps = [p for p in model.parameters() if id(p) in head_names and p.requires_grad]
    back_ps = [p for p in model.parameters() if id(p) not in head_names and p.requires_grad]
    groups = []
    if head_ps:
        groups.append({"params": head_ps, "lr": head_lr})
    if back_ps:
        groups.append({"params": back_ps, "lr": backbone_lr})
    return groups


def model_info(model: nn.Module, arch: str, num_classes: int, strategy: str) -> ModelInfo:
    total, trainable = count_params(model)
    return ModelInfo(
        arch=arch,
        num_classes=num_classes,
        params=total,
        trainable_params=trainable,
        size_mb_fp32=round(total * 4 / (1024 * 1024), 3),
        strategy=strategy,
    )


def macs_estimate(model: nn.Module, input_size: int = 224) -> int:
    """粗略统计乘加数，用于选型参考表里的算力列。"""
    total = 0
    hooks = []

    def hook(module, inp, out):
        nonlocal total
        if isinstance(module, nn.Conv2d):
            oh, ow = out.shape[-2:]
            total += module.in_channels * module.out_channels * module.kernel_size[0] * \
                module.kernel_size[1] * oh * ow // (module.groups or 1)
        elif isinstance(module, nn.Linear):
            total += module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        model(torch.zeros(1, 3, input_size, input_size))
    for h in hooks:
        h.remove()
    return int(total)
