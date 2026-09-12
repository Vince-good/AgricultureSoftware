"""微调训练。

文档 3.2(3) 的结论是：小样本下必须走"预训练 + 微调 + 数据增强"，从零训练没有出路。
这里把三种策略都实现了，方便在同一份数据上横向对比：
  linear_probe  只训分类头，最稳但上限低
  bitfit        只训偏置和 BN，参数高效微调的极简版
  partial       解冻骨干最后 N 个 block，小样本下的推荐档
  full          全量微调，数据多了才用，否则很容易过拟合到 100 张田间照片上
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..eval.metrics import summarize
from .model import apply_strategy, build_model, count_params, model_info, param_groups


@dataclass
class TrainConfig:
    arch: str = "mobilenet_v3_small"
    num_classes: int = 14
    strategy: str = "partial"
    unfreeze_blocks: int = 4
    low_rank_head: int = 0
    epochs: int = 12
    batch_size: int = 32
    head_lr: float = 3e-3
    backbone_lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: float = 1.0
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.0
    class_weights: bool = True
    balanced: bool = True
    grad_clip: float = 5.0
    patience: int = 5
    seed: int = 42
    threads: int = 0
    distill_temp: float = 4.0
    distill_alpha: float = 0.5
    out_dir: Optional[str] = None
    tag: str = "finetune"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainResult:
    checkpoint: Path
    history: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    best_epoch: int = 0
    elapsed_s: float = 0.0
    model: Optional[nn.Module] = None


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _cosine_lr(step: int, total: int, base: float, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return base * (step + 1) / float(warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def _class_weight_vector(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.mean() / counts
    w = np.clip(w, 0.3, 3.0)  # 极端不平衡时也别让某一类权重爆掉
    return torch.tensor(w, dtype=torch.float32)


def _mixup(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, class_ids: List[str],
             device: torch.device) -> Dict[str, Any]:
    model.eval()
    all_logits: List[np.ndarray] = []
    all_labels: List[int] = []
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += float(loss_fn(logits, y)) * y.size(0)
        total_n += int(y.size(0))
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.extend(y.detach().cpu().tolist())
    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, len(class_ids)))
    labels_np = np.asarray(all_labels, dtype=np.int64)
    out = summarize(logits_np, labels_np, class_ids)
    out["loss"] = round(total_loss / max(total_n, 1), 4)
    out["logits_shape"] = list(logits_np.shape)
    return out


def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device):
    """给量化前的 FP32 基线评估用。"""
    model.eval()
    xs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            xs.append(model(x.to(device)).detach().cpu().numpy())
            ys.extend(y.tolist())
    return (np.concatenate(xs, axis=0) if xs else np.zeros((0, 1))), np.asarray(ys, dtype=np.int64)


def train(train_ds, val_ds, class_ids: List[str], cfg: TrainConfig,
          teacher: Optional[nn.Module] = None, device: Optional[torch.device] = None,
          verbose: bool = True, init_model: Optional[nn.Module] = None) -> TrainResult:
    """teacher 不为空时自动开启知识蒸馏（文档 3.2(2)）。

    init_model 用于剪枝后的恢复训练：此时网络结构已被低秩分解改过，
    不能再按 arch 重新构建，必须接着传进来的模型继续训。
    """
    set_seed(cfg.seed)
    threads = cfg.threads or max(1, min(8, os.cpu_count() or 1))
    torch.set_num_threads(threads)
    device = device or torch.device("cpu")

    if init_model is not None:
        model = init_model
    else:
        model = build_model(cfg.arch, cfg.num_classes, pretrained=True,
                            low_rank_head=cfg.low_rank_head)
    unfrozen = apply_strategy(model, cfg.strategy, cfg.unfreeze_blocks)
    model.to(device)
    if teacher is not None:
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False

    total, trainable = count_params(model)
    if verbose:
        print(f"[finetune] arch={cfg.arch} strategy={cfg.strategy} "
              f"params={total:,} trainable={trainable:,} ({trainable / max(total,1):.1%}) "
              f"unfrozen_groups={len(unfrozen)}")

    # 小样本田间数据各类数量常差几倍，用类别均衡重采样拉平每类的每轮迭代次数
    sampler = None
    if cfg.balanced and hasattr(train_ds, "labels"):
        from .data import ClassBalancedSampler
        sampler = ClassBalancedSampler(train_ds.labels, seed=cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=(sampler is None),
                              sampler=sampler, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    groups = param_groups(model, cfg.head_lr, cfg.backbone_lr)
    if not groups:
        raise RuntimeError("没有任何可训练参数，请检查 strategy 设置")
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    weight_vec = _class_weight_vector(train_ds.labels, cfg.num_classes) if cfg.class_weights else None
    if weight_vec is not None:
        weight_vec = weight_vec.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight_vec, label_smoothing=cfg.label_smoothing)
    kd_fn = nn.KLDivLoss(reduction="batchmean")

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    out_dir = Path(cfg.out_dir) if cfg.out_dir else Path("artifacts/runs") / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"

    history: List[Dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    no_improve = 0
    t_start = time.time()
    step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, n = 0.0, 0
        t_ep = time.time()
        for x, y in train_loader:
            lr_scale_groups = optimizer.param_groups
            base_lrs = [g.get("initial_lr", g["lr"]) for g in lr_scale_groups]
            for g, base in zip(lr_scale_groups, base_lrs):
                g.setdefault("initial_lr", base)
                g["lr"] = _cosine_lr(step, total_steps, base, warmup_steps)

            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            x, ya, yb, lam = _mixup(x, y, cfg.mixup_alpha)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            if lam >= 1.0:
                loss = loss_fn(logits, ya)
            else:
                loss = lam * loss_fn(logits, ya) + (1 - lam) * loss_fn(logits, yb)

            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(x)
                T = cfg.distill_temp
                kd = kd_fn(torch.log_softmax(logits / T, dim=1),
                           torch.softmax(t_logits / T, dim=1)) * (T * T)
                loss = (1 - cfg.distill_alpha) * loss + cfg.distill_alpha * kd

            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                         cfg.grad_clip)
            optimizer.step()
            running += float(loss.detach()) * y.size(0)
            n += int(y.size(0))
            step += 1

        val = evaluate(model, val_loader, class_ids, device)
        record = {
            "epoch": epoch,
            "train_loss": round(running / max(n, 1), 4),
            "val_loss": val["loss"],
            "val_top1": val["top1"],
            "val_top3": val["top3"],
            "val_macro_f1": val["macro_f1"],
            "lr": round(optimizer.param_groups[-1]["lr"], 6),
            "epoch_sec": round(time.time() - t_ep, 1),
        }
        history.append(record)
        if verbose:
            print(f"[finetune] epoch {epoch:>3}/{cfg.epochs} "
                  f"train_loss={record['train_loss']:.4f} val_loss={record['val_loss']:.4f} "
                  f"top1={record['val_top1']:.3f} top3={record['val_top3']:.3f} "
                  f"f1={record['val_macro_f1']:.3f} ({record['epoch_sec']}s)")

        score = val["top1"] * 0.6 + val["macro_f1"] * 0.4
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict(),
                        "class_ids": list(class_ids), "epoch": epoch,
                        "metrics": {k: v for k, v in val.items() if k != "confusion_matrix"},
                        "model_info": model_info(model, cfg.arch, cfg.num_classes,
                                                 cfg.strategy).to_dict()},
                       ckpt_path)
        else:
            no_improve += 1
            if cfg.patience > 0 and no_improve >= cfg.patience:
                if verbose:
                    print(f"[finetune] 早停于 epoch {epoch}（best={best_epoch}）")
                break

    best = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["state_dict"])
    final_metrics = evaluate(model, val_loader, class_ids, device)
    elapsed = time.time() - t_start

    payload = {
        "config": cfg.to_dict(),
        "model_info": model_info(model, cfg.arch, cfg.num_classes, cfg.strategy).to_dict(),
        "best_epoch": best_epoch,
        "elapsed_s": round(elapsed, 1),
        "threads": threads,
        "distilled": teacher is not None,
        "history": history,
        "metrics": {k: v for k, v in final_metrics.items() if k != "confusion_matrix"},
        "confusion_matrix": final_metrics["confusion_matrix"],
        "class_ids": list(class_ids),
        "train_stats": getattr(train_ds, "stats", None),
    }
    (out_dir / "history.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return TrainResult(checkpoint=ckpt_path, history=history, metrics=final_metrics,
                       config=cfg.to_dict(), best_epoch=best_epoch, elapsed_s=elapsed, model=model)
