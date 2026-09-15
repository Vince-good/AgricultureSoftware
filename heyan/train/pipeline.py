"""一键构建流水线：从田间照片到可部署的边缘模型包。

对应文档 3.2 节梳理出的完整技术路径，串成一条命令：

    数据集 → 教师模型微调(MobileNetV3-Large)
           → 知识蒸馏得到学生(MobileNetV3-Small)
           → 低秩剪枝 + 恢复训练
           → 导出 ONNX(FP32) + 可移植图(纯 NumPy 后端用)
           → 8-bit 静态量化(真实图片校准)
           → 体积/精度/延迟/内存四项预算验收
           → 打包 bundle（含 labels、advisory、离线语音包）

任何一步的实测数字都会落进 `build_report.json`，并写进 bundle 的 manifest，
这样设备上不用重跑基准就能知道自己跑的是"验收通过"还是"降级可用"的模型。

这一层只在构建机上运行，边缘设备完全不需要 torch。
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..advice import load_advisory
from ..classes import Taxonomy, load_taxonomy, taxonomy_from_ids
from ..config import (ACCURACY_DROP_MAX, DEFAULT_ARCH, DEFAULT_TEACHER_ARCH, IMAGE_MEAN,
                      IMAGE_STD, INPUT_SIZE, LATENCY_MAX_S, MODEL_SIZE_MAX_MB, PATHS,
                      QUANT_CALIBRATION_SAMPLES, QUANT_QAT_EPOCHS, QUANT_QAT_LR,
                      QUANT_WEIGHT_TYPE, RESIZE_SIZE, RUNTIME_MEMORY_MAX_MB)
from ..core.bundle import ADVISORY_NAME, FP32_NAME, INT8_NAME, LABELS_NAME, MANIFEST_NAME
from ..core.bundle import VOICEPACK_DIR, BundleManifest, ModelBundle
from . import augment, data, distill, export_onnx, finetune, prune, qat, quantize

PORTABLE_NAME = "model_portable.hgraph.npz"
REPORT_NAME = "build_report.json"
BENCH_NAME = "benchmark.json"


@dataclass
class PipelineConfig:
    # 数据
    data_dir: Optional[str] = None          # ImageFolder 布局
    labels_csv: Optional[str] = None        # 或 filename,class_id 两列的 csv
    image_root: Optional[str] = None
    val_ratio: float = 0.2
    cache_images: bool = True

    # 模型与训练
    arch: str = DEFAULT_ARCH
    teacher_arch: str = DEFAULT_TEACHER_ARCH
    model_id: str = "heyan-mnv3s-int8"
    version: str = "1.0.0"
    epochs: int = 12
    teacher_epochs: int = 14
    recover_epochs: int = 4
    batch_size: int = 16
    strategy: str = "partial"
    unfreeze_blocks: int = 4
    head_lr: float = 3e-3
    backbone_lr: float = 3e-4
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.0
    seed: int = 42
    threads: int = 0
    patience: int = 4

    # 压缩
    distill: bool = True
    temperature: float = 4.0
    alpha: float = 0.5
    do_prune: bool = True
    # MobileNetV3-Small 权重接近满秩：energy=0.95 实测压不动（-0.0%），
    # 0.85 才有约 13% 的真实收益，精度由恢复训练 + 超阈值自动回退兜底。
    prune_energy: float = 0.85
    prune_min_channels: int = 24
    prune_min_gain: float = 0.15
    prune_min_linear: int = 256
    quantize: bool = True
    per_channel: bool = True
    calibration_samples: int = QUANT_CALIBRATION_SAMPLES
    # 量化主路径：QAT。纯 PTQ 在 MobileNetV3-Small 上实测掉 15~47 个百分点，
    # 兑现不了文档 ≤3% 的承诺，只保留 --no-qat 作为无训练条件下的降级开关。
    qat: bool = True
    qat_epochs: int = QUANT_QAT_EPOCHS
    qat_lr: float = QUANT_QAT_LR
    qat_ptq_fallback: bool = True

    # 产物
    out_root: Optional[str] = None
    region: str = "粤东西北"
    notes: str = ""
    clean: bool = True
    pack_zip: bool = False
    benchmark_rounds: int = 20
    benchmark_threads: int = 1
    voice_languages: Tuple[str, ...] = ("zh",)
    build_voicepack: bool = True
    device_ram_mb: Optional[int] = None
    device_cost_cny: Optional[float] = None
    limits: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["voice_languages"] = list(self.voice_languages)
        return d

    @property
    def root(self) -> Path:
        return Path(self.out_root) if self.out_root else PATHS.root

    def resolve_limits(self) -> Dict[str, float]:
        lim = {
            "model_mb": MODEL_SIZE_MAX_MB,
            "latency_s": LATENCY_MAX_S,
            "memory_mb": RUNTIME_MEMORY_MAX_MB,
        }
        lim.update(self.limits or {})
        return lim


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------

def resolve_dataset(cfg: PipelineConfig) -> Tuple[List[Tuple[Path, int]], List[str]]:
    if cfg.labels_csv:
        root = Path(cfg.image_root) if cfg.image_root else Path(cfg.labels_csv).parent
        return data.read_label_csv(cfg.labels_csv, root)
    if cfg.data_dir:
        return data.discover_samples(cfg.data_dir)
    raise ValueError("必须指定 data_dir（ImageFolder 布局）或 labels_csv（filename,class_id）")


def build_datasets(cfg: PipelineConfig, samples: Sequence[Tuple[Path, int]],
                   class_ids: Sequence[str]):
    train_samples, val_samples = data.split_samples(samples, cfg.val_ratio, cfg.seed)
    aug_cfg = augment.AugmentConfig()
    train_ds = data.LeafDataset(train_samples, list(class_ids), augment=aug_cfg,
                                seed=cfg.seed, cache=cfg.cache_images)
    val_ds = data.LeafDataset(val_samples, list(class_ids), augment=None, cache=cfg.cache_images)
    stats = data.stats(samples, class_ids)
    print(f"[pipeline] 数据集 {stats.total} 张 / {len(class_ids)} 类，"
          f"训练 {len(train_ds)} 验证 {len(val_ds)}，每类 {stats.min_per_class}~"
          f"{stats.max_per_class} 张（{'小样本' if stats.few_shot else '常规'}）")
    return train_ds, val_ds, train_samples, val_samples, stats


def _base_train_cfg(cfg: PipelineConfig, *, epochs: int, tag: str, out_dir: Path,
                    strategy: Optional[str] = None, arch: Optional[str] = None,
                    lr_scale: float = 1.0) -> finetune.TrainConfig:
    return finetune.TrainConfig(
        arch=arch or cfg.arch, num_classes=0,  # num_classes 由调用方填
        strategy=strategy or cfg.strategy, unfreeze_blocks=cfg.unfreeze_blocks,
        epochs=epochs, batch_size=cfg.batch_size,
        head_lr=cfg.head_lr * lr_scale, backbone_lr=cfg.backbone_lr * lr_scale,
        label_smoothing=cfg.label_smoothing, mixup_alpha=cfg.mixup_alpha,
        patience=cfg.patience, seed=cfg.seed, threads=cfg.threads,
        distill_temp=cfg.temperature, distill_alpha=cfg.alpha,
        out_dir=str(out_dir), tag=tag,
    )


# --------------------------------------------------------------------------
# 各阶段
# --------------------------------------------------------------------------

def stage_train(cfg: PipelineConfig, train_ds, val_ds, class_ids: List[str],
                work: Path) -> Tuple[Any, Dict[str, Any]]:
    """教师 + 蒸馏，或直接微调。返回 (学生模型, 阶段报告)。"""
    n = len(class_ids)
    report: Dict[str, Any] = {"num_classes": n, "distilled": False}
    base = _base_train_cfg(cfg, epochs=cfg.epochs, tag="student", out_dir=work / "student")
    base.num_classes = n

    if cfg.distill:
        dcfg = distill.DistillConfig(
            teacher_arch=cfg.teacher_arch, student_arch=cfg.arch,
            temperature=cfg.temperature, alpha=cfg.alpha,
            teacher_epochs=cfg.teacher_epochs, student_epochs=cfg.epochs,
            teacher_strategy=cfg.strategy, student_strategy=cfg.strategy,
        )
        t_cfg = _base_train_cfg(cfg, epochs=cfg.teacher_epochs, tag="teacher",
                                out_dir=work / "teacher", arch=cfg.teacher_arch)
        t_cfg.num_classes = n
        t_res = distill.train_teacher(train_ds, val_ds, class_ids, n, dcfg, work,
                                      seed=cfg.seed, base_cfg=t_cfg)
        report["teacher"] = {
            "arch": cfg.teacher_arch,
            "checkpoint": str(t_res.checkpoint),
            "best_epoch": t_res.best_epoch,
            "elapsed_s": round(t_res.elapsed_s, 1),
            "metrics": {k: v for k, v in t_res.metrics.items() if k != "confusion_matrix"},
        }
        s_res = distill.distill(train_ds, val_ds, class_ids, t_res.checkpoint, n, dcfg, work,
                                seed=cfg.seed, base_cfg=base)
        report["distilled"] = True
        report["distillation"] = {"temperature": cfg.temperature, "alpha": cfg.alpha}
    else:
        s_res = finetune.train(train_ds, val_ds, class_ids, base)

    report["student"] = {
        "arch": cfg.arch,
        "checkpoint": str(s_res.checkpoint),
        "best_epoch": s_res.best_epoch,
        "elapsed_s": round(s_res.elapsed_s, 1),
        "model_info": s_res.config.get("model_info"),
        "metrics": {k: v for k, v in s_res.metrics.items() if k != "confusion_matrix"},
        "history": s_res.history,
    }
    return s_res.model, report


def stage_prune(cfg: PipelineConfig, model, train_ds, val_ds, class_ids: List[str],
                work: Path) -> Tuple[Any, Dict[str, Any]]:
    """低秩剪枝 + 恢复训练。剪枝后若精度反而下降，就退回剪枝前的模型。"""
    import torch

    n = len(class_ids)
    report: Dict[str, Any] = {"enabled": cfg.do_prune}
    if not cfg.do_prune:
        return model, report

    from .model import count_params

    params_before, _ = count_params(model)
    comp = prune.low_rank_compress(model, energy=cfg.prune_energy,
                                   min_channels=cfg.prune_min_channels,
                                   min_gain=cfg.prune_min_gain,
                                   min_linear=cfg.prune_min_linear)
    params_after, _ = count_params(model)
    report["compression"] = comp.to_dict()
    report["params_before"] = int(params_before)
    report["params_after"] = int(params_after)

    if comp.layers_compressed == 0:
        print("[pipeline] 没有层满足剪枝条件，跳过恢复训练")
        report["recovered"] = False
        return model, report

    # 剪枝前基线，用来判断恢复训练是否成功
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    device = torch.device("cpu")
    baseline = finetune.evaluate(model, val_loader, class_ids, device)
    report["metrics_before_recover"] = {k: v for k, v in baseline.items()
                                        if k != "confusion_matrix"}

    rec_cfg = _base_train_cfg(cfg, epochs=cfg.recover_epochs, tag="recover",
                              out_dir=work / "recover", strategy="recover", lr_scale=0.4)
    rec_cfg.num_classes = n
    rec_res = finetune.train(train_ds, val_ds, class_ids, rec_cfg, init_model=model,
                             verbose=True)
    recovered = {k: v for k, v in rec_res.metrics.items() if k != "confusion_matrix"}
    report["metrics_after_recover"] = recovered
    report["recover_epochs"] = rec_res.best_epoch

    drop = baseline["top1"] - recovered["top1"]
    report["top1_drop"] = round(float(drop), 4)
    if drop > 0.05:
        # 剪枝得不偿失：重新载入剪枝前的最优权重
        print(f"[pipeline] 剪枝后 top1 掉 {drop:.3f}，回退到剪枝前模型")
        ckpt = torch.load(str(_find_student_ckpt(work)), map_location="cpu", weights_only=False)
        from .model import build_model

        model = build_model(cfg.arch, n, pretrained=True)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        report["reverted"] = True
    else:
        report["reverted"] = False
    return model, report


def _find_student_ckpt(work: Path) -> Path:
    for cand in (work / "student" / "best.pt", work / "teacher" / "best.pt"):
        if cand.exists():
            return cand
    hits = sorted(work.rglob("best.pt"))
    if not hits:
        raise FileNotFoundError(f"{work} 下找不到任何 best.pt")
    return hits[0]


def stage_export(cfg: PipelineConfig, model, work: Path, class_ids: List[str]
                 ) -> Tuple[Path, Path, Dict[str, Any]]:
    fp32_path = work / FP32_NAME
    info = export_onnx.export_module_onnx(model, fp32_path, input_size=INPUT_SIZE)
    info.update({"arch": cfg.arch, "num_classes": len(class_ids), "class_ids": class_ids})

    portable_path = work / PORTABLE_NAME
    pstats: Dict[str, Any] = {}
    try:
        pstats = export_onnx.export_portable(fp32_path, portable_path)
    except Exception as exc:  # 纯 NumPy 后端是加分项，缺 onnx 包时不应中断构建
        print(f"[pipeline] 可移植图导出失败（不影响主路径）: {type(exc).__name__}: {exc}")
        portable_path = Path("")
    return fp32_path, portable_path, {"onnx": info, "portable": pstats}


def stage_quantize(cfg: PipelineConfig, model, fp32_path: Path, portable_path: Optional[Path],
                   train_ds, val_ds, class_ids: List[str], work: Path
                   ) -> Tuple[Path, Optional[Path], Optional[Path], Dict[str, Any]]:
    """8-bit 量化 + 体积/精度/延迟验收。

    返回 (fp32_path, portable_path, int8_path, report)。QAT 会用适配过 8-bit 网格的
    权重覆盖重导 FP32 与可移植图，所以这两个路径也要回传给打包阶段。
    """
    if not cfg.quantize:
        return fp32_path, portable_path, None, {"enabled": False}
    n_calib = min(cfg.calibration_samples, max(8, len(train_ds)))
    # raw_array() 走的是与推理完全同构的预处理且不施加增强，正是校准集要的东西
    calib = data.calibration_arrays(train_ds, n_calib, seed=cfg.seed)
    val_arrays = [val_ds.raw_array(i) for i in range(len(val_ds))]
    int8_path = work / INT8_NAME
    limits = cfg.resolve_limits()

    if cfg.qat:
        base = _base_train_cfg(cfg, epochs=cfg.qat_epochs, tag="qat", out_dir=work / "qat")
        base.num_classes = len(class_ids)
        rep = qat.qat_quantize_and_verify(
            model=model, fp32_path=fp32_path, int8_path=int8_path,
            calib_arrays=calib, val_arrays=val_arrays, val_labels=val_ds.labels,
            class_ids=class_ids, train_ds=train_ds, val_ds=val_ds, base_cfg=base,
            work=work / "qat", epochs=cfg.qat_epochs, batch_size=cfg.batch_size,
            lr=cfg.qat_lr, threads=cfg.benchmark_threads or 1,
            ptq_fallback=cfg.qat_ptq_fallback,
        )
        if portable_path and Path(portable_path).parent.exists():
            try:
                export_onnx.export_portable(fp32_path, portable_path)
            except Exception as exc:
                print(f"[pipeline] 可移植图重导失败（不影响主路径）: "
                      f"{type(exc).__name__}: {exc}")
    else:
        rep = quantize.quantize_and_verify(
            fp32_path, int8_path, calib, val_arrays, val_ds.labels, class_ids,
            threads=cfg.benchmark_threads or 1,
            max_model_mb=limits["model_mb"], max_latency_s=limits["latency_s"],
        )
    return fp32_path, portable_path, int8_path, rep.to_dict()


# --------------------------------------------------------------------------
# 打包
# --------------------------------------------------------------------------

def build_bundle(cfg: PipelineConfig, taxonomy: Taxonomy, fp32_path: Path,
                 portable_path: Optional[Path], int8_path: Optional[Path],
                 stages: Dict[str, Any], work: Path) -> ModelBundle:
    bundle_dir = cfg.root / "bundles" / f"{cfg.model_id}-v{cfg.version}"
    if cfg.clean and bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    q = stages.get("quantize", {}) or {}
    files: Dict[str, str] = {}
    if int8_path and Path(int8_path).exists():
        shutil.copyfile(int8_path, bundle_dir / INT8_NAME)
        files["int8"] = INT8_NAME
    shutil.copyfile(fp32_path, bundle_dir / FP32_NAME)
    files["fp32"] = FP32_NAME
    if portable_path and Path(portable_path).exists():
        shutil.copyfile(portable_path, bundle_dir / PORTABLE_NAME)
        files["portable"] = PORTABLE_NAME

    train_rep = stages.get("train", {}) or {}
    student = train_rep.get("student", {}) or {}
    metrics: Dict[str, Any] = {
        # 用实测值而不是训练日志里的值：QAT 之后 FP32 会按同一份权重重新导出
        "val_top1_fp32": (q.get("metrics_fp32") or {}).get("top1")
        or (student.get("metrics") or {}).get("top1"),
        "val_top3_fp32": (student.get("metrics") or {}).get("top3"),
        "val_macro_f1_fp32": (student.get("metrics") or {}).get("macro_f1"),
        "val_top1_int8": (q.get("metrics_int8") or {}).get("top1"),
        "val_top3_int8": (q.get("metrics_int8") or {}).get("top3"),
        "accuracy_drop_top1": q.get("accuracy_drop_top1"),
        "accuracy_drop_limit": ACCURACY_DROP_MAX,
        "accuracy_contract_ok": (q.get("accuracy_drop_top1") is not None
                                 and q["accuracy_drop_top1"] <= ACCURACY_DROP_MAX),
        "latency_int8_ms": q.get("latency_int8_ms"),
        "latency_fp32_ms": q.get("latency_fp32_ms"),
        "speedup": q.get("speedup"),
    }
    bench = stages.get("benchmark") or {}
    if bench:
        metrics["latency_p50_ms"] = (bench.get("latency_ms", {}).get("total_ms", {}) or {}).get("p50")
        metrics["latency_p95_ms"] = (bench.get("latency_ms", {}).get("total_ms", {}) or {}).get("p95")
        metrics["runtime_memory_delta_mb"] = (bench.get("memory_mb") or {}).get("runtime_delta_mb")
        metrics["budget_ok"] = bench.get("budget_ok")

    manifest = BundleManifest(
        model_id=cfg.model_id,
        version=cfg.version,
        arch=cfg.arch,
        num_classes=len(taxonomy),
        input_size=INPUT_SIZE,
        resize_size=RESIZE_SIZE,
        mean=list(IMAGE_MEAN),
        std=list(IMAGE_STD),
        files=files,
        quantization={
            "method": q.get("method", "none"),
            "weight_type": QUANT_WEIGHT_TYPE,
            "activation_type": "quint8",
            "per_channel": bool(q.get("per_channel", cfg.per_channel)),
            "calibration_samples": int(q.get("calibration_samples", 0) or 0),
            "compression_ratio": q.get("compression_ratio"),
        },
        budgets={
            "max_model_mb": cfg.resolve_limits()["model_mb"],
            "max_latency_s": cfg.resolve_limits()["latency_s"],
            "max_runtime_memory_mb": cfg.resolve_limits()["memory_mb"],
            "verified": bench.get("budgets") if bench else None,
        },
        metrics=metrics,
        backend_preference=["onnxruntime", "numpy"],
        region=cfg.region,
        notes=cfg.notes or "MobileNetV3-Small + 知识蒸馏 + 低秩剪枝 + INT8 静态量化",
        extra={
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pipeline": cfg.to_dict(),
            "dataset": (stages.get("dataset") or {}).get("stats"),
            "train": {k: v for k, v in student.items() if k != "history"},
            "teacher": train_rep.get("teacher"),
            "distillation": train_rep.get("distillation"),
            "prune": stages.get("prune"),
            "export": stages.get("export"),
            "quantize": q,
            "benchmark": bench,
        },
    )
    bundle = ModelBundle.create(bundle_dir, manifest, taxonomy=taxonomy,
                                advisory=load_advisory())
    print(f"[pipeline] bundle -> {bundle_dir}")
    return bundle


def attach_voicepack(cfg: PipelineConfig, bundle: ModelBundle) -> Dict[str, Any]:
    """把离线语音包塞进 bundle。失败不致命：设备上还能退回系统 TTS 或纯文字。"""
    info: Dict[str, Any] = {"enabled": cfg.build_voicepack, "languages": list(cfg.voice_languages)}
    if not cfg.build_voicepack:
        return info
    try:
        from ..tts.voicepack import build_voicepack
    except Exception as exc:  # pragma: no cover
        info["error"] = f"tts 模块不可用: {exc}"
        return info
    try:
        result = build_voicepack(list(cfg.voice_languages), bundle.voicepack_dir,
                                 taxonomy=bundle.taxonomy, advisory=bundle.advisory)
        info.update(result)
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[pipeline] 语音包构建失败（不影响识别功能）: {exc}")
    return info


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def run_pipeline(cfg: Optional[PipelineConfig] = None, verbose: bool = True) -> Dict[str, Any]:
    cfg = cfg or PipelineConfig()
    t_start = time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = cfg.root / "runs" / f"{cfg.model_id}-{stamp}"
    work.mkdir(parents=True, exist_ok=True)
    cfg.root.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] 工作目录 {work}")

    samples, class_ids = resolve_dataset(cfg)
    taxonomy = taxonomy_from_ids(class_ids)
    train_ds, val_ds, train_samples, val_samples, dstats = build_datasets(cfg, samples, class_ids)
    stages: Dict[str, Any] = {"dataset": {"class_ids": list(class_ids),
                                          "stats": dstats.to_dict(),
                                          "split": {"train": len(train_ds), "val": len(val_ds)}}}

    model, train_rep = stage_train(cfg, train_ds, val_ds, class_ids, work)
    stages["train"] = train_rep

    model, prune_rep = stage_prune(cfg, model, train_ds, val_ds, class_ids, work)
    stages["prune"] = prune_rep

    fp32_path, portable_path, export_rep = stage_export(cfg, model, work, class_ids)
    stages["export"] = export_rep

    fp32_path, portable_path, int8_path, quant_rep = stage_quantize(
        cfg, model, fp32_path, portable_path, train_ds, val_ds, class_ids, work)
    stages["quantize"] = quant_rep

    bundle = build_bundle(cfg, taxonomy, fp32_path, portable_path, int8_path, stages, work)
    stages["voicepack"] = attach_voicepack(cfg, bundle)

    # 验收：用真实验证图片跑端到端延迟与内存
    from ..eval.benchmark import benchmark_bundle, format_report, save_report

    probe_paths = [p for p, _ in val_samples[:8]] or [p for p, _ in train_samples[:8]]
    bench = benchmark_bundle(
        bundle.root, images=probe_paths, rounds=cfg.benchmark_rounds,
        threads=cfg.benchmark_threads, device_ram_mb=cfg.device_ram_mb,
        device_cost_cny=cfg.device_cost_cny,
    )
    bench["budgets"]["limits"] = cfg.resolve_limits()
    stages["benchmark"] = bench
    save_report(bench, bundle.root / BENCH_NAME)
    (bundle.root / "benchmark.txt").write_text(format_report(bench), encoding="utf-8")

    # 把验收结论回写进 manifest，设备上可直接读
    bundle.manifest.metrics.update({
        "latency_p50_ms": bench["latency_ms"]["total_ms"]["p50"],
        "latency_p95_ms": bench["latency_ms"]["total_ms"]["p95"],
        "runtime_memory_delta_mb": bench["memory_mb"]["runtime_delta_mb"],
        # 验收必须同时看设备预算与精度合同，只看体积/延迟/内存会漏掉量化掉点
        "accuracy_contract_ok": quant_ok(stages),
        "budget_ok": bool(bench["budget_ok"] and quant_ok(stages)),
    })
    bundle.manifest.budgets["verified"] = bench["budgets"]
    bundle.write_metadata()

    report = {
        "schema_version": "1.0",
        "model_id": cfg.model_id,
        "version": cfg.version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - t_start, 1),
        "work_dir": str(work),
        "bundle_dir": str(bundle.root),
        "config": cfg.to_dict(),
        "stages": stages,
        "budget_ok": bool(bench["budget_ok"] and quant_ok(stages)),
        "summary": summarize(stages, bundle, bench),
    }
    (bundle.root / REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (work / REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if cfg.pack_zip:
        zip_path = bundle.pack_zip()
        report["zip"] = str(zip_path)
        print(f"[pipeline] 打包 -> {zip_path}")

    if verbose:
        print()
        print(format_report(bench))
        print()
        for line in report["summary"]:
            print(f"[pipeline] {line}")
        print(f"[pipeline] 报告 -> {bundle.root / REPORT_NAME}")
    return report


def summarize(stages: Dict[str, Any], bundle: ModelBundle, bench: Dict[str, Any]) -> List[str]:
    """给构建者看的几行结论。"""
    q = stages.get("quantize", {}) or {}
    p = stages.get("prune", {}) or {}
    t = stages.get("train", {}) or {}
    lines: List[str] = []
    student = (t.get("student") or {}).get("metrics") or {}
    # 掉点是拿量化阶段自己那份 FP32 ONNX 当基准算的，汇总行必须用同一个基准。
    # 田间微调路径下 train 阶段记的是冻骨干 head-only 的成绩，而 FP32 ONNX 是
    # QAT 之后重导出的（BN 统计已漂移），两者不是一个数，混在一行会对不上账。
    fp32_top1 = (q.get("metrics_fp32") or {}).get("top1")
    if fp32_top1 is None:
        fp32_top1 = student.get("top1")
    lines.append(f"验证集 top1: FP32 {fp32_top1} -> INT8 "
                 f"{(q.get('metrics_int8') or {}).get('top1')} "
                 f"(掉 {q.get('accuracy_drop_top1')})")
    if student.get("top1") is not None and student.get("top1") != fp32_top1:
        lines.append(f"QAT 前训练权重 top1 {student['top1']}；上表 FP32 为 QAT 后重导出的 "
                     f"ONNX，BN 统计漂移使其降到 {fp32_top1}")
    if p.get("compression"):
        c = p["compression"]
        lines.append(f"低秩剪枝: 压缩 {c.get('layers_compressed')} 层，参数减少 "
                     f"{c.get('reduction', 0):.1%}")
    lines.append(f"模型体积: {bundle.model_size_mb()}MB / 上限 "
                 f"{bundle.manifest.budgets.get('max_model_mb')}MB")
    lat = bench["latency_ms"]["total_ms"]
    lines.append(f"端到端延迟 p50 {lat['p50']}ms / p95 {lat['p95']}ms，上限 3000ms")
    lines.append(f"运行内存增量 {bench['memory_mb']['runtime_delta_mb']}MB，上限 "
                 f"{bundle.manifest.budgets.get('max_runtime_memory_mb')}MB")
    lines.append(f"推理后端 {bench['backend']}")
    drop = q.get("accuracy_drop_top1")
    lines.append(f"量化精度合同: 掉点 {drop} / 上限 {ACCURACY_DROP_MAX} -> "
                 f"{'达标' if quant_ok({'quantize': q}) else '不达标'}")
    if bench["budget_ok"] and quant_ok({"quantize": q}):
        lines.append("预算验收: 全部通过（体积 / 延迟 / 内存 / 精度）")
    else:
        failed = list(bench["budgets"].get("failed") or [])
        if not quant_ok({"quantize": q}):
            failed.append("accuracy_drop")
        lines.append(f"预算验收: 有超标项 {failed}")
    return lines


def quant_ok(stages: Dict[str, Any]) -> bool:
    """量化精度是否守住文档 ≤3% 的合同。没跑量化时视为不适用，不算失败。"""
    q = stages.get("quantize") or {}
    if not q.get("enabled", True):
        return True
    drop = q.get("accuracy_drop_top1")
    if drop is None:
        return True
    return bool(drop <= ACCURACY_DROP_MAX)
