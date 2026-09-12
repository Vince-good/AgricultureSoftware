"""8-bit 量化与预算校验（文档 3.2(2)(4)）。

主路径是静态量化 + QDQ 格式 + per-channel 权重量化，这是 ONNX Runtime 在 CPU 上
推荐、也是 MobileNet 类模型实测最快最稳的组合。校准集必须用真实田间分布的图片，
不能拿随机张量凑数，否则激活值范围估偏，精度会掉得莫名其妙。

量化完立刻做三件事：量体积、量精度、量延迟，任何一项超预算直接判定失败。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import (LATENCY_MAX_S, MODEL_SIZE_MAX_MB, QUANT_CALIBRATION_SAMPLES,
                      RUNTIME_MEMORY_MAX_MB)
from ..eval.metrics import summarize


@dataclass
class QuantizeReport:
    method: str = "static_qdq"
    fp32_path: str = ""
    int8_path: str = ""
    fp32_size_mb: float = 0.0
    int8_size_mb: float = 0.0
    compression_ratio: float = 0.0
    calibration_samples: int = 0
    per_channel: bool = True
    metrics_fp32: Dict[str, Any] = field(default_factory=dict)
    metrics_int8: Dict[str, Any] = field(default_factory=dict)
    accuracy_drop_top1: float = 0.0
    latency_fp32_ms: float = 0.0
    latency_int8_ms: float = 0.0
    speedup: float = 0.0
    budgets: Dict[str, Any] = field(default_factory=dict)
    ok: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class NumpyCalibrationReader:
    """把预先算好的 float32 NCHW 数组喂给 onnxruntime 的校准流程。"""

    def __init__(self, arrays: Sequence[np.ndarray], input_name: str = "input") -> None:
        self.arrays = [np.ascontiguousarray(a, dtype=np.float32) for a in arrays]
        self.input_name = input_name
        self._i = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._i >= len(self.arrays):
            return None
        arr = self.arrays[self._i]
        self._i += 1
        return {self.input_name: arr}

    def rewind(self) -> None:
        self._i = 0

    def __iter__(self):
        self.rewind()
        return self

    def __next__(self) -> Dict[str, np.ndarray]:
        item = self.get_next()
        if item is None:
            raise StopIteration
        return item


def _preprocess_model(src: Path, dst: Path) -> Path:
    """量化前先做形状推断与常量折叠，官方推荐步骤，能显著减少量化失败的算子。"""
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process

        quant_pre_process(str(src), str(dst), auto_merge=True, verbose=False)
        return dst
    except Exception as exc:  # pragma: no cover
        print(f"[quantize] 预处理失败，直接用原图量化: {exc}")
        return src


def quantize_int8(fp32_onnx: Path | str, out_onnx: Path | str,
                  calib_arrays: Sequence[np.ndarray], per_channel: bool = True,
                  input_name: str = "input", work_dir: Optional[Path] = None) -> Dict[str, Any]:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    fp32_onnx = Path(fp32_onnx)
    out_onnx = Path(out_onnx)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(work_dir) if work_dir else out_onnx.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    prepared = _preprocess_model(fp32_onnx, work_dir / f"{fp32_onnx.stem}.prep.onnx")
    reader = NumpyCalibrationReader(calib_arrays, input_name)

    attempts = [
        ("static_qdq", dict(quant_format=QuantFormat.QDQ, per_channel=per_channel)),
        ("static_qoperator", dict(quant_format=QuantFormat.QOperator, per_channel=per_channel)),
        ("static_qdq_per_tensor", dict(quant_format=QuantFormat.QDQ, per_channel=False)),
    ]
    last_exc: Optional[Exception] = None
    for method, extra in attempts:
        try:
            reader.rewind()
            quantize_static(
                str(prepared), str(out_onnx), reader,
                weight_type=QuantType.QUInt8,
                activation_type=QuantType.QUInt8,
                calibrate_method=CalibrationMethod.MinMax,
                extra_options={"ActivationSymmetric": False, "WeightSymmetric": False,
                               "EnableQdmPasses": True},
                **extra,
            )
            print(f"[quantize] 成功（{method}），{len(calib_arrays)} 张校准图 -> "
                  f"{out_onnx.name} ({out_onnx.stat().st_size/1024/1024:.2f} MB)")
            return {"method": method, "path": str(out_onnx),
                    "size_mb": round(out_onnx.stat().st_size / (1024 * 1024), 3),
                    "calibration_samples": len(calib_arrays), "per_channel": extra["per_channel"]}
        except Exception as exc:
            last_exc = exc
            print(f"[quantize] {method} 失败: {type(exc).__name__}: {exc}")

    # 全部失败则退回动态量化：只量化权重，卷积仍走 float，体积收益有限但一定能跑
    try:
        from onnxruntime.quantization import QuantType as QT, quantize_dynamic

        quantize_dynamic(str(prepared), str(out_onnx), weight_type=QT.QUInt8)
        print("[quantize] 退回动态量化")
        return {"method": "dynamic", "path": str(out_onnx),
                "size_mb": round(out_onnx.stat().st_size / (1024 * 1024), 3),
                "calibration_samples": len(calib_arrays), "per_channel": False,
                "warning": "静态量化失败，已退回动态量化"}
    except Exception as exc2:  # pragma: no cover
        raise RuntimeError(f"量化失败: {last_exc} / {exc2}") from exc2


def run_onnx(model_path: Path | str, arrays: Sequence[np.ndarray], threads: int = 1):
    """跑一遍 ONNX 模型，返回 (logits, 每次推理毫秒列表)。"""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(model_path), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    in_type = sess.get_inputs()[0].type
    logits, times = [], []
    for arr in arrays:
        x = np.ascontiguousarray(arr, dtype=np.float32)
        if "uint8" in in_type:  # QOperator 量化模型吃 uint8
            x = np.clip(np.rint(x), 0, 255).astype(np.uint8)
        t0 = time.perf_counter()
        out = sess.run(None, {name: x})[0]
        times.append((time.perf_counter() - t0) * 1000.0)
        logits.append(np.asarray(out, dtype=np.float32))
    return (np.concatenate(logits, axis=0) if logits else np.zeros((0, 1))), times


def latency_stats(times: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(times), dtype=np.float64)
    if arr.size == 0:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "max": 0.0, "min": 0.0}
    return {
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "mean": round(float(arr.mean()), 2),
        "max": round(float(arr.max()), 2),
        "min": round(float(arr.min()), 2),
    }


def quantize_and_verify(fp32_onnx: Path | str, int8_onnx: Path | str,
                        calib_arrays: Sequence[np.ndarray],
                        val_arrays: Sequence[np.ndarray], val_labels: Sequence[int],
                        class_ids: Sequence[str], threads: int = 1,
                        max_model_mb: float = MODEL_SIZE_MAX_MB,
                        max_latency_s: float = LATENCY_MAX_S) -> QuantizeReport:
    """量化 + 三重量化验收（体积/精度/延迟），返回结构化报告。"""
    fp32_onnx = Path(fp32_onnx); int8_onnx = Path(int8_onnx)
    report = QuantizeReport(
        fp32_path=str(fp32_onnx), int8_path=str(int8_onnx),
        fp32_size_mb=round(fp32_onnx.stat().st_size / (1024 * 1024), 3),
        calibration_samples=len(calib_arrays),
    )
    info = quantize_int8(fp32_onnx, int8_onnx, calib_arrays)
    report.method = info["method"]
    report.int8_size_mb = info["size_mb"]
    report.per_channel = bool(info.get("per_channel", True))
    report.compression_ratio = round(report.fp32_size_mb / max(report.int8_size_mb, 1e-6), 2)
    if info.get("warning"):
        report.notes.append(info["warning"])

    if val_arrays:
        logits32, t32 = run_onnx(fp32_onnx, val_arrays, threads=threads)
        logits8, t8 = run_onnx(int8_onnx, val_arrays, threads=threads)
        labels = np.asarray(val_labels, dtype=np.int64)
        report.metrics_fp32 = {k: v for k, v in summarize(logits32, labels, class_ids).items()
                               if k != "confusion_matrix"}
        report.metrics_int8 = {k: v for k, v in summarize(logits8, labels, class_ids).items()
                               if k != "confusion_matrix"}
        report.accuracy_drop_top1 = round(
            report.metrics_fp32["top1"] - report.metrics_int8["top1"], 4)
        report.latency_fp32_ms = latency_stats(t32)["p50"]
        report.latency_int8_ms = latency_stats(t8)["p50"]
        report.speedup = round(report.latency_fp32_ms / max(report.latency_int8_ms, 1e-6), 2)

    checks = {
        "model_size": {
            "value_mb": report.int8_size_mb, "limit_mb": max_model_mb,
            "ok": report.int8_size_mb <= max_model_mb,
        },
        "accuracy_drop": {
            "value": report.accuracy_drop_top1, "limit": 0.03,
            "ok": report.accuracy_drop_top1 <= 0.03,
        },
        "latency": {
            "value_ms": report.latency_int8_ms, "limit_ms": max_latency_s * 1000.0,
            "ok": report.latency_int8_ms <= max_latency_s * 1000.0,
        },
    }
    report.budgets = checks
    report.ok = all(c["ok"] for c in checks.values())
    for name, c in checks.items():
        if not c["ok"]:
            report.notes.append(f"预算超标: {name} -> {c}")

    print(f"[quantize] 体积 {report.fp32_size_mb}MB -> {report.int8_size_mb}MB "
          f"(x{report.compression_ratio}), top1 {report.metrics_fp32.get('top1','-')} -> "
          f"{report.metrics_int8.get('top1','-')} (掉 {report.accuracy_drop_top1}), "
          f"延迟 p50 {report.latency_fp32_ms}ms -> {report.latency_int8_ms}ms "
          f"(x{report.speedup}), 预算{'全部通过' if report.ok else '未通过'}")
    return report
