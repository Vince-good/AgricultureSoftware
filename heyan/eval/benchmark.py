"""设备基准与预算验收。

文档 3.2(4) 给了四条可以直接判定的硬指标：
    模型体积 ≤ 20MB、单图延迟 ≤ 3s、运行内存 ≤ 200MB、设备 RAM ≤ 2GB / 成本 ≤ 300 元。

这里把它们做成可执行的验收：任何一项不达标，`BudgetReport.ok` 就是 False，
构建流水线会直接把结论写进 bundle 的 manifest，设备上无需重测。

关于内存口径：桌面 Python 解释器自身就要吃掉上百 MB，拿整进程 RSS 去比 200MB
没有意义。因此这里以"加载模型 + 完成推理"带来的 RSS 增量作为边缘运行内存，
同时把绝对 RSS 一并记录下来供交叉核对。
"""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import (DEVICE_COST_MAX_CNY, DEVICE_RAM_MAX_MB, LATENCY_MAX_S,
                      MODEL_SIZE_MAX_MB, RUNTIME_MEMORY_MAX_MB)


# --------------------------------------------------------------------------
# 基础测量
# --------------------------------------------------------------------------

def rss_mb() -> float:
    """当前进程常驻内存（MB）。psutil 不可用时退回 Windows API / resource。"""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    try:
        import resource  # POSIX

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


def percentile_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "min": 0.0,
                "std": 0.0}
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "max": round(float(arr.max()), 3),
        "min": round(float(arr.min()), 3),
        "std": round(float(arr.std()), 3),
    }


def device_profile() -> Dict[str, Any]:
    """本机画像，用来对照"低端手机 RAM ≤ 2GB / 成本 ≤ 300 元"这条约束。"""
    import platform

    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_total_mb": None,
        "ram_available_mb": None,
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_total_mb"] = int(vm.total / (1024 * 1024))
        info["ram_available_mb"] = int(vm.available / (1024 * 1024))
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------
# 延迟
# --------------------------------------------------------------------------

def measure_latency(engine, images: Sequence[Any], rounds: int = 20,
                    warmup: int = 2) -> Dict[str, Any]:
    """分阶段测量端到端延迟。

    文档里的 3 秒是"从按下快门到出结果"的用户感受，所以必须把预处理、
    推理、严重程度估计和文案组装全部算进去，只测 `session.run` 会自欺欺人。
    """
    if not images:
        raise ValueError("延迟测量至少需要一张图片")
    stage_total: Dict[str, List[float]] = {
        "preprocess_ms": [], "infer_ms": [], "postprocess_ms": [], "total_ms": [],
    }
    n = len(images)

    def one(img) -> Dict[str, float]:
        t0 = time.perf_counter()
        arr = engine._to_array(img)
        x, cropped = engine._preprocess(arr)
        t1 = time.perf_counter()
        logits = np.asarray(engine.backend.run(x), dtype=np.float32).reshape(-1)
        t2 = time.perf_counter()
        from ..core.preprocess import softmax
        from ..core.severity import estimate_severity

        probs = softmax(logits)
        idx = int(np.argmax(probs))
        cls = engine.taxonomy.by_index(idx)
        estimate_severity(cls.stress, float(probs[idx]), cropped, class_id=cls.id)
        from ..advice import build_advice

        build_advice(cls.id, "mild", lang=engine.language, advisory=engine.advisory,
                     taxonomy=engine.taxonomy)
        t3 = time.perf_counter()
        return {
            "preprocess_ms": (t1 - t0) * 1000.0,
            "infer_ms": (t2 - t1) * 1000.0,
            "postprocess_ms": (t3 - t2) * 1000.0,
            "total_ms": (t3 - t0) * 1000.0,
        }

    for i in range(min(warmup, rounds)):
        one(images[i % n])
    for i in range(rounds):
        timings = one(images[i % n])
        for k, v in timings.items():
            stage_total[k].append(v)

    out: Dict[str, Any] = {k: percentile_stats(v) for k, v in stage_total.items()}
    out["rounds"] = rounds
    out["threads"] = getattr(engine.backend, "threads", None)
    out["backend"] = engine.backend.name
    return out


# --------------------------------------------------------------------------
# 内存
# --------------------------------------------------------------------------

def measure_memory(bundle_root: Path | str, backend: Optional[str] = None,
                   threads: int = 1, probe_images: Optional[Sequence[Any]] = None
                   ) -> Dict[str, Any]:
    """测量"加载模型 + 跑推理"造成的 RSS 增量。"""
    from ..core.engine import RecognitionEngine

    gc.collect()
    base = rss_mb()
    engine = RecognitionEngine(bundle_root, backend=backend, threads=threads)
    after_load = rss_mb()
    peak = after_load
    if probe_images:
        for img in probe_images:
            engine.recognize(img, with_severity=True)
            peak = max(peak, rss_mb())
    info = engine.info()
    engine.close()
    gc.collect()
    return {
        "rss_baseline_mb": round(base, 2),
        "rss_after_load_mb": round(after_load, 2),
        "rss_peak_mb": round(peak, 2),
        "load_delta_mb": round(after_load - base, 2),
        "runtime_delta_mb": round(peak - base, 2),
        "backend": info.get("backend", {}).get("backend"),
    }


# --------------------------------------------------------------------------
# 预算判定
# --------------------------------------------------------------------------

@dataclass
class BudgetCheck:
    name: str
    value: float
    limit: float
    unit: str
    ok: bool
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BudgetReport:
    checks: List[BudgetCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failed(self) -> List[BudgetCheck]:
        return [c for c in self.checks if not c.ok]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [c.to_dict() for c in self.checks],
            "failed": [c.name for c in self.failed],
        }


def check_budgets(*, model_size_mb: float, latency_p95_ms: float, runtime_memory_mb: float,
                  device_ram_mb: Optional[int] = None,
                  device_cost_cny: Optional[float] = None,
                  limits: Optional[Dict[str, float]] = None) -> BudgetReport:
    lim = {
        "model_mb": MODEL_SIZE_MAX_MB,
        "latency_s": LATENCY_MAX_S,
        "memory_mb": RUNTIME_MEMORY_MAX_MB,
        "device_ram_mb": DEVICE_RAM_MAX_MB,
        "device_cost_cny": DEVICE_COST_MAX_CNY,
    }
    lim.update(limits or {})
    checks = [
        BudgetCheck("model_size", round(float(model_size_mb), 3), lim["model_mb"], "MB",
                    model_size_mb <= lim["model_mb"],
                    "设备上实际部署的模型文件体积（INT8 优先，不含仅作对照的 FP32）"),
        BudgetCheck("latency_p95", round(float(latency_p95_ms) / 1000.0, 3), lim["latency_s"],
                    "s", latency_p95_ms <= lim["latency_s"] * 1000.0,
                    "端到端（预处理+推理+严重程度+文案）p95"),
        BudgetCheck("runtime_memory", round(float(runtime_memory_mb), 2), lim["memory_mb"], "MB",
                    runtime_memory_mb <= lim["memory_mb"],
                    "加载模型并完成推理的 RSS 增量"),
    ]
    if device_ram_mb is not None:
        checks.append(BudgetCheck(
            "device_ram", float(device_ram_mb), lim["device_ram_mb"], "MB",
            device_ram_mb <= lim["device_ram_mb"], "目标设备内存上限"))
    if device_cost_cny is not None:
        checks.append(BudgetCheck(
            "device_cost", float(device_cost_cny), lim["device_cost_cny"], "CNY",
            device_cost_cny <= lim["device_cost_cny"], "目标设备采购成本上限"))
    return BudgetReport(checks=checks)


# --------------------------------------------------------------------------
# 一键基准
# --------------------------------------------------------------------------

def benchmark_bundle(bundle_root: Path | str, images: Optional[Sequence[Any]] = None,
                     rounds: int = 20, threads: int = 1, backend: Optional[str] = None,
                     device_ram_mb: Optional[int] = None,
                     device_cost_cny: Optional[float] = None) -> Dict[str, Any]:
    """对已打包的 bundle 做完整验收，返回可直接写进 manifest 的字典。"""
    from ..core.bundle import ModelBundle
    from ..core.engine import RecognitionEngine

    bundle_root = Path(bundle_root)
    bundle = ModelBundle.load(bundle_root)
    images = list(images or [])
    if not images:
        images = [_synthetic_probe(bundle.manifest.input_size)]

    engine = RecognitionEngine(bundle_root, backend=backend, threads=threads)
    latency = measure_latency(engine, images, rounds=rounds)
    backend_info = engine.info()
    engine.close()

    memory = measure_memory(bundle_root, backend=backend, threads=threads,
                            probe_images=images[:1])
    model_size = bundle.model_size_mb()
    budgets = check_budgets(
        model_size_mb=model_size,
        latency_p95_ms=latency["total_ms"]["p95"],
        runtime_memory_mb=memory["runtime_delta_mb"],
        device_ram_mb=device_ram_mb,
        device_cost_cny=device_cost_cny,
    )
    return {
        "bundle": str(bundle_root),
        "model_id": bundle.manifest.model_id,
        "version": bundle.manifest.version,
        "device": device_profile(),
        "model_size": {
            "deployed_mb": model_size,
            "deployed_kind": bundle.deployable_kind(),
            "all_models_mb": bundle.all_models_mb(),
            "bundle_mb": bundle.size_mb(),
            "per_kind_mb": {k: round((bundle.root / n).stat().st_size / (1024 * 1024), 3)
                            for k, n in bundle.manifest.files.items()
                            if (bundle.root / n).exists()},
            "size_check": bundle.check_size_budget(),
        },
        "latency_ms": latency,
        "memory_mb": memory,
        "backend": backend_info.get("backend"),
        "budgets": budgets.to_dict(),
        "budget_ok": budgets.ok,
        "rounds": rounds,
        "threads": threads,
    }


def _synthetic_probe(size: int = 224) -> np.ndarray:
    """没有真实图片时的兜底探针：随机图足以测出延迟与内存量级。"""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (size * 2, size * 2, 3), dtype=np.uint8)


def format_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("=" * 68)
    lines.append(f"禾眼 HeYan 构建验收报告  {report.get('model_id')} v{report.get('version')}")
    lines.append("=" * 68)
    dev = report.get("device", {})
    lines.append(f"测量设备: {dev.get('machine')} / {dev.get('cpu_count')} 核 / "
                 f"RAM {dev.get('ram_total_mb')}MB")
    lines.append(f"推理后端: {(report.get('backend') or {}).get('backend')}  "
                 f"线程数={report.get('threads')}  轮次={report.get('rounds')}")
    lines.append("")
    lat = report.get("latency_ms", {}).get("total_ms", {})
    stages = report.get("latency_ms", {})
    lines.append("延迟 (ms)      p50      p95     mean      max")
    for key, label in (("preprocess_ms", "预处理"), ("infer_ms", "模型推理"),
                       ("postprocess_ms", "后处理"), ("total_ms", "端到端")):
        s = stages.get(key, {})
        lines.append(f"  {label:<10} {s.get('p50', 0):>8.1f} {s.get('p95', 0):>8.1f} "
                     f"{s.get('mean', 0):>8.1f} {s.get('max', 0):>8.1f}")
    lines.append("")
    mem = report.get("memory_mb", {})
    lines.append(f"内存: 基线 {mem.get('rss_baseline_mb')}MB -> 载入后 "
                 f"{mem.get('rss_after_load_mb')}MB -> 峰值 {mem.get('rss_peak_mb')}MB "
                 f"(增量 {mem.get('runtime_delta_mb')}MB)")
    size = report.get("model_size", {})
    lines.append(f"体积: 模型合计 {size.get('total_mb')}MB / bundle 合计 "
                 f"{size.get('bundle_mb')}MB")
    for kind, mb in (size.get("per_kind_mb") or {}).items():
        lines.append(f"        {kind:<10} {mb} MB")
    lines.append("")
    lines.append("预算判定:")
    for c in report.get("budgets", {}).get("checks", []):
        mark = "通过" if c["ok"] else "超标"
        lines.append(f"  [{mark}] {c['name']:<16} {c['value']:>10} {c['unit']:<4} "
                     f"(上限 {c['limit']} {c['unit']})  {c['note']}")
    lines.append("")
    lines.append(f"结论: {'全部指标达标' if report.get('budget_ok') else '存在超标项'}  "
                 f"(端到端 p95 {lat.get('p95', 0)}ms, 上限 3000ms)")
    lines.append("=" * 68)
    return "\n".join(lines)


def save_report(report: Dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
