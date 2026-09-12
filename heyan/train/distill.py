"""知识蒸馏（文档 3.2(2)：Hinton 等，2015）。

教师用 MobileNetV3-Large，学生用 MobileNetV3-Small。
在只有约 100 张田间照片的条件下，蒸馏的软标签比 one-hot 标签携带多得多的信息
（"这张图有 60% 像稻瘟病、30% 像胡麻叶斑病"），是缓解小样本过拟合最划算的一招。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from .finetune import TrainConfig, TrainResult, set_seed, train
from .model import DEFAULT_ARCH, build_model

TEACHER_ARCH = "mobilenet_v3_large"
STUDENT_ARCH = "mobilenet_v3_small"


@dataclass
class DistillConfig:
    teacher_arch: str = TEACHER_ARCH
    student_arch: str = STUDENT_ARCH
    temperature: float = 4.0
    alpha: float = 0.5          # 蒸馏损失权重，0.5 表示软标签与硬标签各占一半
    teacher_epochs: int = 14
    student_epochs: int = 12
    teacher_strategy: str = "partial"
    student_strategy: str = "partial"


def load_teacher(checkpoint: Path | str, arch: str = TEACHER_ARCH, num_classes: int = 14,
                 low_rank_head: int = 0) -> nn.Module:
    model = build_model(arch, num_classes, pretrained=True, low_rank_head=low_rank_head)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def train_teacher(train_ds, val_ds, class_ids: List[str], num_classes: int,
                  dcfg: DistillConfig, out_dir: Path | str, seed: int = 42,
                  verbose: bool = True, base_cfg: Optional[TrainConfig] = None) -> TrainResult:
    cfg = TrainConfig(**{**base_cfg.to_dict(), **{
        "arch": dcfg.teacher_arch, "num_classes": num_classes,
        "strategy": dcfg.teacher_strategy, "epochs": dcfg.teacher_epochs, "seed": seed,
        "out_dir": str(Path(out_dir) / "teacher"), "tag": "teacher", "label_smoothing": 0.1,
    }}) if base_cfg is not None else TrainConfig(
        arch=dcfg.teacher_arch, num_classes=num_classes, strategy=dcfg.teacher_strategy,
        epochs=dcfg.teacher_epochs, seed=seed, out_dir=str(Path(out_dir) / "teacher"),
        tag="teacher", label_smoothing=0.1,
    )
    return train(train_ds, val_ds, class_ids, cfg, teacher=None, verbose=verbose)


def distill(train_ds, val_ds, class_ids: List[str], teacher_ckpt: Path | str,
            num_classes: int, dcfg: DistillConfig, out_dir: Path | str,
            seed: int = 42, verbose: bool = True,
            base_cfg: Optional[TrainConfig] = None) -> TrainResult:
    set_seed(seed)
    teacher = load_teacher(teacher_ckpt, dcfg.teacher_arch, num_classes)
    cfg = base_cfg or TrainConfig()
    cfg.arch = dcfg.student_arch
    cfg.num_classes = num_classes
    cfg.strategy = dcfg.student_strategy
    cfg.epochs = dcfg.student_epochs
    cfg.distill_temp = dcfg.temperature
    cfg.distill_alpha = dcfg.alpha
    cfg.seed = seed
    cfg.out_dir = str(Path(out_dir) / "student")
    cfg.tag = "student_distilled"
    if verbose:
        print(f"[distill] teacher={dcfg.teacher_arch} student={dcfg.student_arch} "
              f"T={dcfg.temperature} alpha={dcfg.alpha}")
    result = train(train_ds, val_ds, class_ids, cfg, teacher=teacher, verbose=verbose)
    result.config["distillation"] = {
        "teacher_arch": dcfg.teacher_arch,
        "teacher_checkpoint": str(teacher_ckpt),
        "temperature": dcfg.temperature,
        "alpha": dcfg.alpha,
    }
    return result
