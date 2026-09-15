#!/usr/bin/env python3
"""田间真实照片小样本微调：冻结 MobileNetV3 主干、只训分类头，合成+真实混合防遗忘。

为什么不直接跑 `heyan.cli build` 的完整流水线：
  那条路带教师蒸馏 + 低秩剪枝 + QAT，一次几十分钟，而且剪枝会改网络结构，
  与"骨干一个字节不动、只动分类头"的诉求冲突。真实照片每类只有二三十张，
  全量微调必然过拟合到这几张图上，所以这里走最保守的一档：

    1. 合成回放集（artifacts/data/demo_dataset）与田间真实集（artifacts/data/field）
       都按完整 taxonomy 的 class_id 定位标签下标，两边拼起来不会错位；
    2. 两个域各自分层切 train/val：混合验证集管早停，另留两份单域验证集做遗忘体检
       （老类别在合成回放上的 top1 训前训后各测一次，掉了就是灾难性遗忘）；
    3. 从上一版**未剪枝**检查点热启动：骨干与分类头前两层照搬，fc 按 class_id 对齐行号，
       新增类别保留随机初始化（零初始化的行在 softmax 里是死区，收敛更慢）；
    4. linear_probe 冻结骨干只训分类头；真实照片走完整增强（随机旋转/翻转/亮度/
       色温/遮挡/JPEG），合成图走更强一档（供给无限，增强坏了不心疼）；
    5. 导出 FP32 ONNX + 可移植图，再 QAT 量化成 INT8（纯 PTQ 在本模型上实测掉 15~47
       个百分点，兑现不了 <=3% 的合同），最后打新版 bundle 替换 artifacts/bundles 旧包。

QAT 阶段会把骨干以 3e-5 量级的学习率解冻做量化适配，那是"适应 8-bit 网格"的独立步骤，
和上面第 4 步的头-only 微调不是一回事；不想要它就用 --no-qat 走 PTQ 降级档。

用法：
    python tools/finetune_field.py                    # 全流程：微调 + 量化 + 打包 v1.1.0
    python tools/finetune_field.py --dry-run          # 只看数据体检与热启动对齐报告
    python tools/finetune_field.py --no-qat           # 无训练条件下的 PTQ 降级档
    python tools/finetune_field.py --epochs 20 --head-lr 1e-3 --no-bundle
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from heyan.classes import load_taxonomy  # noqa: E402
from heyan.config import (ACCURACY_DROP_MAX, DEFAULT_ARCH, INPUT_SIZE, LATENCY_MAX_S,
                          MODEL_SIZE_MAX_MB, PATHS, QUANT_CALIBRATION_SAMPLES, QUANT_QAT_EPOCHS,
                          QUANT_QAT_LR)  # noqa: E402
from heyan.core.bundle import FP32_NAME, INT8_NAME  # noqa: E402
from heyan.train import data as train_data  # noqa: E402
from heyan.train import export_onnx, finetune, pipeline as pipe, qat, quantize  # noqa: E402
from heyan.train.augment import AugmentConfig  # noqa: E402
from heyan.train.model import build_model  # noqa: E402

DEFAULT_SYNTH = REPO_ROOT / "artifacts" / "data" / "demo_dataset"
DEFAULT_FIELD = REPO_ROOT / "artifacts" / "data" / "field"
PORTABLE_NAME = "model_portable.hgraph.npz"
REPORT_NAME = "build_report.json"
BENCH_NAME = "benchmark.json"
STATE_NAME = "stage_state.json"


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------

class MixedDataset(Dataset):
    """把多个 LeafDataset 拼成一个，各部分保留自己的增强强度。

    真实照片与合成图的增强档位不同（真实照片少而贵，合成图多而贱），但训练回路只认
    一个 dataset 对象，所以在这里做拼接；labels / raw_array 都按偏移转发，
    类别均衡采样器与量化校准集拿到的接口和单个 LeafDataset 完全一致。
    """

    def __init__(self, parts: Sequence[train_data.LeafDataset], names: Sequence[str]) -> None:
        self.parts = list(parts)
        self.names = list(names)
        self._offsets: List[int] = []
        acc = 0
        for p in self.parts:
            self._offsets.append(acc)
            acc += len(p)
        self._total = acc

    def __len__(self) -> int:
        return self._total

    def _locate(self, idx: int) -> Tuple[train_data.LeafDataset, int]:
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        for i in range(len(self.parts) - 1, -1, -1):
            if idx >= self._offsets[i]:
                return self.parts[i], idx - self._offsets[i]
        raise IndexError(idx)  # pragma: no cover

    def __getitem__(self, idx: int):
        part, j = self._locate(idx)
        return part[j]

    @property
    def labels(self) -> List[int]:
        out: List[int] = []
        for p in self.parts:
            out.extend(p.labels)
        return out

    def raw_array(self, idx: int) -> np.ndarray:
        part, j = self._locate(idx)
        return part.raw_array(j)

    def domain_of(self, idx: int) -> str:
        part, _ = self._locate(idx)
        return self.names[self.parts.index(part)]


def _augment(strong: bool, seed: Optional[int]) -> AugmentConfig:
    """真实照片用标准档；合成图反正供给无限，几何扰动再狠一点也无妨。"""
    if strong:
        return AugmentConfig(rotate_deg=30.0, scale_range=(0.70, 1.30), shift_range=0.15,
                             brightness=(0.60, 1.50), occlusion_p=0.35, jpeg_p=0.6,
                             seed=seed)
    return AugmentConfig(seed=seed)


def scan(args, class_ids: List[str]) -> Dict[str, Any]:
    """扫描两个数据源并各自分层切分，返回训练/验证/单域验证集与体检报告。"""
    synth_samples, synth_ids = train_data.discover_samples(args.synthetic_dir, class_ids)
    field_disc = train_data.discover_aliased_samples(args.field_dir, class_ids, strict=True)
    field_samples = field_disc.samples

    synth_train, synth_val = train_data.split_samples(synth_samples, args.val_ratio, args.seed)
    field_train, field_val = train_data.split_samples(field_samples, args.val_ratio, args.seed)

    def mk(samples, aug, seed):
        return train_data.LeafDataset(samples, class_ids, augment=aug, seed=seed,
                                      cache=args.cache)

    train_ds = MixedDataset(
        [mk(synth_train, _augment(True, args.seed), args.seed),
         mk(field_train, _augment(False, args.seed + 1), args.seed + 1)],
        ["synthetic", "field"])
    val_ds = MixedDataset(
        [mk(synth_val, None, None), mk(field_val, None, None)],
        ["synthetic", "field"])
    # 单域验证集：遗忘体检用，不参与早停
    carried = set(field_disc.dir_map.values())
    val_synth_old = mk([s for s in synth_val if class_ids[s[1]] not in carried], None, None)
    val_field = mk(field_val, None, None)

    report = {
        "class_ids": class_ids,
        "synthetic": {"root": str(args.synthetic_dir), "train": len(synth_train),
                      "val": len(synth_val),
                      "per_class": train_data.stats(synth_samples, class_ids).to_dict()},
        "field": {**field_disc.to_dict(), "train": len(field_train), "val": len(field_val)},
        "mixed_train": len(train_ds),
        "mixed_val": len(val_ds),
        "val_synth_old": len(val_synth_old),
        "val_field": len(val_field),
    }
    return {"train_ds": train_ds, "val_ds": val_ds, "val_synth_old": val_synth_old,
            "val_field": val_field, "report": report}


# --------------------------------------------------------------------------
# 热启动
# --------------------------------------------------------------------------

def default_warm_start(class_ids: Sequence[str]) -> Optional[Path]:
    """挑最近一个"类别是本次子集"的未剪枝学生检查点作为热启动源。"""
    cands = sorted((REPO_ROOT / "artifacts" / "runs").glob("*/student/best.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    wanted = set(class_ids)
    for p in cands:
        try:
            ck = torch.load(str(p), map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001 - 坏检查点直接跳过
            continue
        old = set(ck.get("class_ids") or [])
        if old and old < wanted:
            return p
    return None


def warm_start_head(model: torch.nn.Module, ckpt_path: Path,
                    class_ids: Sequence[str]) -> Dict[str, Any]:
    """把旧检查点的骨干与分类头搬进 17 类模型，fc 行号按 class_id 对齐。

    旧检查点的 class_ids 是字母序、和新 taxonomy 的下标不是一套，直接 load 会把
    "水稻稻瘟病"的权重行塞给"花生叶斑病"，错得无声无息。所以 fc 逐行按名字搬，
    新增类别保留 build_model 的随机初始化。
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    old_ids: List[str] = list(ckpt.get("class_ids") or [])
    sd: Dict[str, torch.Tensor] = ckpt["state_dict"]
    unknown = [c for c in old_ids if c not in class_ids]
    if unknown:
        raise ValueError(f"热启动检查点含未知类别 {unknown}，无法对齐分类头")

    body = {k: v for k, v in sd.items() if not k.startswith("classifier.fc.")}
    msg = model.load_state_dict(body, strict=False)
    fc = model.classifier.fc
    old_w, old_b = sd["classifier.fc.weight"], sd["classifier.fc.bias"]
    if old_w.shape[1] != fc.weight.shape[1]:
        raise ValueError(f"热启动头维度不匹配: 旧 {old_w.shape} vs 新 {tuple(fc.weight.shape)}")
    with torch.no_grad():
        for i, cid in enumerate(old_ids):
            j = class_ids.index(cid)
            fc.weight[j].copy_(old_w[i])
            fc.bias[j].copy_(old_b[i])
    return {
        "checkpoint": str(ckpt_path),
        "old_class_ids": old_ids,
        "carried_rows": len(old_ids),
        "new_rows": [c for c in class_ids if c not in old_ids],
        "missing_keys": list(msg.missing_keys),
        "unexpected_keys": list(msg.unexpected_keys),
        "old_metrics": {k: v for k, v in (ckpt.get("metrics") or {}).items()
                        if k != "confusion_matrix"},
    }


# --------------------------------------------------------------------------
# 域内评估：遗忘体检
# --------------------------------------------------------------------------

def _loader(ds, batch_size: int) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _top1(model, ds, class_ids: List[str], device, batch_size: int) -> Dict[str, Any]:
    if len(ds) == 0:
        return {"top1": None, "macro_f1": None, "n": 0}
    m = finetune.evaluate(model, _loader(ds, batch_size), class_ids, device)
    m["n"] = len(ds)
    return m


def _load_state(work: Path) -> Dict[str, Any]:
    p = work / STATE_NAME
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 状态文件坏了就当成没有，大不了重测一遍
        return {}


def _save_state(work: Path, key: str, value: Any) -> None:
    """把已完成阶段的结论落盘：CPU 上微调一轮十分钟，后段崩了不该逼着重跑。"""
    work.mkdir(parents=True, exist_ok=True)
    data = _load_state(work)
    data[key] = value
    (work / STATE_NAME).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _load_previous_run(work: Path) -> Any:
    """从已有 run 目录复原微调结果，供 --resume-run 跳过训练直接走导出/量化/打包。"""
    ckpt = work / "head" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"找不到可复用的检查点: {ckpt}")
    hist = work / "head" / "history.json"
    payload = json.loads(hist.read_text(encoding="utf-8")) if hist.exists() else {}
    metrics = dict(payload.get("metrics") or {})
    if payload.get("confusion_matrix") is not None:
        metrics["confusion_matrix"] = payload["confusion_matrix"]
    return SimpleNamespace(checkpoint=ckpt, history=payload.get("history") or [],
                           metrics=metrics, config=payload.get("config") or {},
                           best_epoch=payload.get("best_epoch", 0),
                           elapsed_s=payload.get("elapsed_s", 0.0), model=None)


def _previous_bundle(model_id: str, exclude: Optional[Path] = None) -> Optional[Path]:
    """artifacts/bundles 下最近一个 bundle：新包要从它那儿继承语音包。

    必须把"正在重建的这个包"排除掉：重跑同一版本号时它的 mtime 最新，会被选成
    自己的上一个包，而 build_bundle 又已经把它 rmtree 过，于是语音包继承到 0 个
    wav，几百条话术全要从头渲染。
    """
    root = PATHS.root / "bundles"
    skip = str(Path(exclude).resolve()).lower() if exclude else None
    cands = [d for d in root.glob(f"{model_id}-v*")
             if d.is_dir() and str(d.resolve()).lower() != skip]
    return max(cands, key=lambda d: d.stat().st_mtime) if cands else None


def carry_voicepack(prev: Optional[Path], bundle) -> Dict[str, Any]:
    """把旧包的语音包整目录搬到新包，attach_voicepack 便只增量渲染新增类别的话术。

    语音包是几百个 wav，全量重渲染要几分钟且和模型精度无关；玉米三个新类别的
    话术旧包里没有，正是需要补的那部分。
    """
    if prev is None:
        return {"carried_from": None, "wav_files": 0}
    src = prev / "voicepack"
    if not src.is_dir():
        return {"carried_from": None, "wav_files": 0}
    shutil.copytree(src, bundle.voicepack_dir, dirs_exist_ok=True)
    return {"carried_from": str(src),
            "wav_files": sum(1 for _ in bundle.voicepack_dir.rglob("*.wav"))}


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="田间真实照片小样本微调：冻结 MobileNetV3 主干，只训分类头",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--synthetic-dir", type=Path, default=DEFAULT_SYNTH,
                   help="合成回放集（ImageFolder，目录名即 class_id）")
    p.add_argument("--field-dir", type=Path, default=DEFAULT_FIELD,
                   help="田间真实集（目录名过 label_aliases.json 映射成 class_id）")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arch", default=DEFAULT_ARCH)
    p.add_argument("--strategy", default="linear_probe",
                   choices=["linear_probe", "bitfit", "partial", "full"],
                   help="linear_probe=只训分类头（默认，最抗过拟合）")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--no-cache", action="store_true", help="不把图片解码结果常驻内存")
    p.add_argument("--warm-start", type=Path, default=None, help="指定热启动检查点")
    p.add_argument("--no-warm-start", action="store_true", help="只用 ImageNet 骨干 + 随机头")
    p.add_argument("--resume-run", type=Path, default=None,
                   help="复用已有 run 目录的 head/best.pt，跳过微调直接走导出/量化/打包")
    p.add_argument("--forget-tolerance", type=float, default=0.02,
                   help="老类别在合成回放上的 top1 允许回落的上限")
    # 量化与打包
    p.add_argument("--no-qat", action="store_true", help="退回纯 PTQ（本模型上掉点很大）")
    p.add_argument("--qat-epochs", type=int, default=QUANT_QAT_EPOCHS)
    p.add_argument("--qat-lr", type=float, default=QUANT_QAT_LR)
    p.add_argument("--model-id", default="heyan-mnv3s-int8")
    p.add_argument("--version", default="1.1.0")
    p.add_argument("--region", default="粤东西北")
    p.add_argument("--no-bundle", action="store_true", help="只微调，不导出/不量化/不打包")
    p.add_argument("--benchmark-rounds", type=int, default=20)
    p.add_argument("--device-ram-mb", type=int, default=None)
    p.add_argument("--device-cost-cny", type=float, default=None)
    p.add_argument("--dry-run", action="store_true", help="只做数据体检与热启动对齐，不训练")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    args.cache = not args.no_cache
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    finetune.set_seed(args.seed)
    taxonomy = load_taxonomy()
    class_ids: List[str] = taxonomy.ids
    device = torch.device(args.device)
    work = (Path(args.resume_run) if args.resume_run else
            PATHS.root / "runs" / f"{args.model_id}-field-{time.strftime('%Y%m%d-%H%M%S')}")
    state = _load_state(work)

    # ---------------- 数据 ----------------
    print(f"[field-ft] taxonomy {len(class_ids)} 类，工作目录 {work}")
    disc = scan(args, class_ids)
    rep: Dict[str, Any] = disc["report"]
    train_ds, val_ds = disc["train_ds"], disc["val_ds"]
    val_field, val_synth_old = disc["val_field"], disc["val_synth_old"]

    fld = rep["field"]
    print("[field-ft] 别名映射: "
          + ", ".join(f"{k} -> {v}" for k, v in sorted(fld["dir_map"].items())))
    got = {k: v for k, v in fld["per_class"].items() if v}
    print(f"[field-ft] 田间真实 {fld['total']} 张，每类 {got}")
    syn = rep["synthetic"]
    print(f"[field-ft] 合成回放 {syn['train'] + syn['val']} 张 / {len(class_ids)} 类")
    print(f"[field-ft] 混合训练集 {rep['mixed_train']}（合成 {syn['train']} + 真实 {fld['train']}），"
          f"混合验证集 {rep['mixed_val']}")
    print(f"[field-ft] 遗忘体检集：老类别合成验证 {rep['val_synth_old']}，田间验证 {rep['val_field']}")
    if fld["total"] == 0:
        print("[field-ft] 田间集为空，先用 tools/ingest_field_samples.py 归档照片", file=sys.stderr)
        return 2

    # ---------------- 模型与热启动 ----------------
    model = build_model(args.arch, len(class_ids), pretrained=True)
    warm: Optional[Dict[str, Any]] = state.get("warm_start")
    resumed: Optional[Any] = None
    if args.resume_run:
        resumed = _load_previous_run(work)
        model.load_state_dict(torch.load(str(resumed.checkpoint), map_location="cpu",
                                         weights_only=False)["state_dict"])
        print(f"[field-ft] 复用已有微调结果 {resumed.checkpoint}"
              f"（best_epoch {resumed.best_epoch}）")
    else:
        ckpt = args.warm_start
        if ckpt is None and not args.no_warm_start:
            ckpt = default_warm_start(class_ids)
        if ckpt is not None:
            warm = warm_start_head(model, Path(ckpt), class_ids)
            print(f"[field-ft] 热启动 {warm['checkpoint']}")
            print(f"[field-ft]   搬运 {warm['carried_rows']} 行分类头权重，"
                  f"新增随机初始化 {warm['new_rows']}")
            # fc 的两个张量是故意从 body 里剔掉再逐行搬的，出现在 missing_keys 里属于正常
            extra_missing = [k for k in warm["missing_keys"]
                             if not k.startswith("classifier.fc.")]
            if extra_missing:
                print(f"[field-ft]   警告：检查点缺 {len(extra_missing)} 个张量"
                      f"（{extra_missing[:3]}...），这些层退回 ImageNet 权重")
            if warm["unexpected_keys"]:
                print(f"[field-ft]   警告：检查点多出 {warm['unexpected_keys'][:3]}，已忽略")
        else:
            print("[field-ft] 无热启动检查点，分类头随机初始化")
    model.to(device)

    # ---------------- 训练前基线 ----------------
    if resumed is not None and state.get("baseline"):
        base = state["baseline"]
        base_field, base_old, base_mixed = base["field"], base["old_classes"], base["mixed"]
        print(f"[field-ft] 复用训练前基线：田间 {base_field['top1']} | "
              f"老类别 {base_old['top1']} | 混合 {base_mixed['top1']}")
    else:
        print("[field-ft] 测训练前基线 ...")
        base_field = _top1(model, val_field, class_ids, device, args.batch_size)
        base_old = _top1(model, val_synth_old, class_ids, device, args.batch_size)
        base_mixed = _top1(model, val_ds, class_ids, device, args.batch_size)
        _save_state(work, "warm_start", warm)
        _save_state(work, "baseline", {"field": base_field, "old_classes": base_old,
                                       "mixed": base_mixed})
    print(f"[field-ft]   田间 top1 {base_field['top1']} | 老类别 top1 {base_old['top1']} | "
          f"混合 top1 {base_mixed['top1']}")

    if args.dry_run:
        work.mkdir(parents=True, exist_ok=True)
        (work / "dry_run.json").write_text(
            json.dumps({"report": rep, "warm_start": warm,
                        "baseline": {"field": base_field, "old_classes": base_old,
                                     "mixed": base_mixed}},
                       ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"[field-ft] dry-run 报告 -> {work / 'dry_run.json'}")
        return 0

    # ---------------- 微调（骨干冻结） ----------------
    work.mkdir(parents=True, exist_ok=True)
    if resumed is not None:
        res = resumed
    else:
        cfg = finetune.TrainConfig(
            arch=args.arch, num_classes=len(class_ids), strategy=args.strategy,
            epochs=args.epochs, batch_size=args.batch_size,
            head_lr=args.head_lr,
            backbone_lr=0.0,  # 骨干全程冻结；写 0 是为了把意图落进报告，不是靠它省算力
            weight_decay=args.weight_decay, label_smoothing=args.label_smoothing,
            balanced=True, class_weights=True, patience=args.patience,
            seed=args.seed, threads=args.threads,
            out_dir=str(work / "head"), tag="field-head")
        res = finetune.train(train_ds, val_ds, class_ids, cfg, device=device, init_model=model)
        model = res.model if res.model is not None else model

    if resumed is not None and state.get("domain_eval"):
        domain_eval = state["domain_eval"]
        after_field = domain_eval["field"]["after"]
        after_old = domain_eval["old_classes_synthetic"]["after"]
        after_mixed = domain_eval["mixed_val"]["after"]
        forgetting = domain_eval["old_classes_synthetic"]["forgetting_top1"]
    else:
        after_field = _top1(model, val_field, class_ids, device, args.batch_size)
        after_old = _top1(model, val_synth_old, class_ids, device, args.batch_size)
        after_mixed = _top1(model, val_ds, class_ids, device, args.batch_size)
        forgetting = (round(base_old["top1"] - after_old["top1"], 4)
                      if base_old["top1"] is not None and after_old["top1"] is not None
                      else None)
        domain_eval = {
            "field": {"before": base_field, "after": after_field},
            "old_classes_synthetic": {"before": base_old, "after": after_old,
                                      "forgetting_top1": forgetting,
                                      "tolerance": args.forget_tolerance,
                                      "ok": bool(forgetting is not None
                                                 and forgetting <= args.forget_tolerance)},
            "mixed_val": {"before": base_mixed, "after": after_mixed},
        }
        _save_state(work, "domain_eval", domain_eval)
    print(f"[field-ft] 田间 top1 {base_field['top1']} -> {after_field['top1']}")
    print(f"[field-ft] 老类别 top1 {base_old['top1']} -> {after_old['top1']} "
          f"(遗忘 {forgetting}，容忍 {args.forget_tolerance})")
    print(f"[field-ft] 混合验证 top1 {base_mixed['top1']} -> {after_mixed['top1']}，"
          f"macro_f1 {after_mixed['macro_f1']}，best_epoch {res.best_epoch}")
    if forgetting is not None and forgetting > args.forget_tolerance:
        print(f"[field-ft] 警告：老类别掉了 {forgetting}，超过容忍线。"
              f"建议加大合成回放比重或改用 --strategy bitfit", file=sys.stderr)

    if args.no_bundle:
        print(f"[field-ft] 检查点 -> {res.checkpoint}（--no-bundle，跳过导出与打包）")
        return 0

    # ---------------- 导出 ONNX ----------------
    fp32_path = work / FP32_NAME
    onnx_info = export_onnx.export_module_onnx(model, fp32_path, input_size=INPUT_SIZE)
    onnx_info.update({"arch": args.arch, "num_classes": len(class_ids), "class_ids": class_ids})
    portable_path: Path = work / PORTABLE_NAME
    pstats: Dict[str, Any] = {}
    try:
        pstats = export_onnx.export_portable(fp32_path, portable_path)
    except Exception as exc:  # 纯 NumPy 后端是加分项，缺 onnx 包时不该中断交付
        print(f"[field-ft] 可移植图导出失败（不影响主路径）: {type(exc).__name__}: {exc}")
        portable_path = Path("")
        pstats = {"error": f"{type(exc).__name__}: {exc}"}

    # ---------------- INT8 量化 ----------------
    int8_path = work / INT8_NAME
    calib = train_data.calibration_arrays(
        train_ds, min(QUANT_CALIBRATION_SAMPLES, len(train_ds)), seed=args.seed)
    val_arrays = [val_ds.raw_array(i) for i in range(len(val_ds))]
    limits = {"model_mb": MODEL_SIZE_MAX_MB, "latency_s": LATENCY_MAX_S}
    if not args.no_qat:
        base_cfg = finetune.TrainConfig(
            arch=args.arch, num_classes=len(class_ids), batch_size=args.batch_size,
            seed=args.seed, threads=args.threads, label_smoothing=0.05,
            out_dir=str(work / "qat"), tag="qat")
        qrep = qat.qat_quantize_and_verify(
            model=model, fp32_path=fp32_path, int8_path=int8_path, calib_arrays=calib,
            val_arrays=val_arrays, val_labels=val_ds.labels, class_ids=class_ids,
            train_ds=train_ds, val_ds=val_ds, base_cfg=base_cfg, work=work / "qat",
            epochs=args.qat_epochs, batch_size=args.batch_size, lr=args.qat_lr,
            threads=1, ptq_fallback=True)
        # QAT 用适配过 8-bit 网格的权重覆盖重导了 FP32，可移植图必须跟着重导
        if str(portable_path):
            try:
                pstats = export_onnx.export_portable(fp32_path, portable_path)
            except Exception as exc:
                print(f"[field-ft] 可移植图重导失败: {type(exc).__name__}: {exc}")
    else:
        qrep = quantize.quantize_and_verify(
            fp32_path, int8_path, calib, val_arrays, val_ds.labels, class_ids, threads=1,
            max_model_mb=limits["model_mb"], max_latency_s=limits["latency_s"])
    qd = qrep.to_dict()
    print(f"[field-ft] 量化 {qd.get('method')}: FP32 top1 {(qd.get('metrics_fp32') or {}).get('top1')}"
          f" -> INT8 {(qd.get('metrics_int8') or {}).get('top1')}"
          f"（掉 {qd.get('accuracy_drop_top1')}，上限 {ACCURACY_DROP_MAX}）")

    # ---------------- 打包新 bundle ----------------
    pcfg = pipe.PipelineConfig(
        arch=args.arch, model_id=args.model_id, version=args.version, region=args.region,
        batch_size=args.batch_size, seed=args.seed, threads=args.threads,
        quantize=True, qat=not args.no_qat, qat_epochs=args.qat_epochs, qat_lr=args.qat_lr,
        benchmark_rounds=args.benchmark_rounds, benchmark_threads=1,
        device_ram_mb=args.device_ram_mb, device_cost_cny=args.device_cost_cny,
        limits=limits,
        notes=(f"MobileNetV3-Small 田间小样本微调（{args.strategy} 冻结骨干）+ 合成回放防遗忘 "
               f"+ INT8 {'QAT' if not args.no_qat else 'PTQ'}"))
    stages: Dict[str, Any] = {
        "dataset": rep,
        "train": {
            "student": {
                "metrics": {k: v for k, v in res.metrics.items() if k != "confusion_matrix"},
                "confusion_matrix": res.metrics.get("confusion_matrix"),
                "best_epoch": res.best_epoch, "elapsed_s": res.elapsed_s,
                "config": res.config, "history": res.history,
                "checkpoint": str(res.checkpoint),
            },
            "warm_start": warm,
            "mode": "field_finetune_head_only",
        },
        "domain_eval": domain_eval,
        "export": {"onnx": onnx_info, "portable": pstats},
        "quantize": qd,
    }
    prev = _previous_bundle(args.model_id,
                            exclude=PATHS.root / "bundles" / f"{args.model_id}-v{args.version}")
    bundle = pipe.build_bundle(pcfg, taxonomy, fp32_path, portable_path, int8_path, stages, work)
    stages["voicepack"] = {"carry": carry_voicepack(prev, bundle),
                           **pipe.attach_voicepack(pcfg, bundle)}

    # ---------------- 端到端验收 ----------------
    from heyan.eval.benchmark import benchmark_bundle, format_report, save_report

    # 延迟/内存验收要用真实图片跑端到端，随机噪声测不出预处理那一段的开销
    probe_paths = ([p for p, _ in val_field.samples[:8]]
                   or [p for p, _ in val_synth_old.samples[:8]])
    bench = benchmark_bundle(bundle.root, images=probe_paths, rounds=args.benchmark_rounds,
                             threads=1, device_ram_mb=args.device_ram_mb,
                             device_cost_cny=args.device_cost_cny)
    bench["budgets"]["limits"] = pcfg.resolve_limits()
    stages["benchmark"] = bench
    save_report(bench, bundle.root / BENCH_NAME)
    (bundle.root / "benchmark.txt").write_text(format_report(bench), encoding="utf-8")

    bundle.manifest.metrics.update({
        "latency_p50_ms": bench["latency_ms"]["total_ms"]["p50"],
        "latency_p95_ms": bench["latency_ms"]["total_ms"]["p95"],
        "runtime_memory_delta_mb": bench["memory_mb"]["runtime_delta_mb"],
        "accuracy_contract_ok": pipe.quant_ok(stages),
        "budget_ok": bool(bench["budget_ok"] and pipe.quant_ok(stages)),
        "field_top1": after_field["top1"],
        "old_class_forgetting_top1": forgetting,
    })
    bundle.manifest.budgets["verified"] = bench["budgets"]
    bundle.manifest.extra["domain_eval"] = domain_eval
    bundle.write_metadata()

    report = {
        "schema_version": "1.0",
        "model_id": args.model_id,
        "version": args.version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(res.elapsed_s, 1),
        "work_dir": str(work),
        "bundle_dir": str(bundle.root),
        "config": pcfg.to_dict(),
        "stages": stages,
        "budget_ok": bool(bench["budget_ok"] and pipe.quant_ok(stages)),
        "summary": pipe.summarize(stages, bundle, bench),
    }
    (bundle.root / REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (work / REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print()
    print(format_report(bench))
    print()
    for line in report["summary"]:
        print(f"[field-ft] {line}")
    print(f"[field-ft] 新 bundle -> {bundle.root}")
    print(f"[field-ft] 报告 -> {bundle.root / REPORT_NAME}")
    print("[field-ft] 重启服务后生效：python -m heyan.cli serve --host 127.0.0.1 --port 8080")
    return 0 if report["budget_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
