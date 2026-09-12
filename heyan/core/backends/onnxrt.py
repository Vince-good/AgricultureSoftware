"""ONNX Runtime 后端 —— 主力推理路径。

选 ONNX Runtime 的理由：同一份 INT8 模型能在 Windows / Linux ARM / Android
（onnxruntime-android AAR）上跑，避免为每个平台维护一套推理代码。
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from .base import Backend, InputSpec

_ORT_DTYPE = {
    "tensor(float)": ("float32", np.float32),
    "tensor(float16)": ("float16", np.float16),
    "tensor(double)": ("float64", np.float64),
    "tensor(uint8)": ("uint8", np.uint8),
    "tensor(int8)": ("int8", np.int8),
}


def ort_version() -> str:
    try:
        import onnxruntime as ort

        return ort.__version__
    except Exception:  # pragma: no cover
        return ""


class OnnxRuntimeBackend(Backend):
    name = "onnxruntime"
    requires = ("onnxruntime",)

    def _load(self) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        # 低端设备上不要放开所有核，否则会跟界面线程抢 CPU 导致掉帧
        n_threads = self.threads or max(1, min(4, os.cpu_count() or 1))
        opts.intra_op_num_threads = n_threads
        opts.inter_op_num_threads = 1
        opts.enable_mem_pattern = True

        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._threads_used = n_threads

        inp = self.session.get_inputs()[0]
        dtype_name, np_dtype = _ORT_DTYPE.get(inp.type, ("float32", np.float32))
        shape = tuple(int(d) if isinstance(d, int) and d > 0 else -1 for d in (inp.shape or ()))
        scale, zp = 1.0, 0
        meta = inp.type  # 量化参数在 ORT 的 python API 里不直接暴露，从 manifest 兜底
        q = (self.manifest.get("quantization") or {})
        if dtype_name in ("uint8", "int8") and q:
            scale = float(q.get("input_scale", 1.0) or 1.0)
            zp = int(q.get("input_zero_point", 0) or 0)
        self._np_dtype = np_dtype
        self._input_spec = InputSpec(name=inp.name, shape=shape, dtype=dtype_name,
                                     scale=scale, zero_point=zp)
        # 预热：第一次推理要做内存分配与图优化，不预热会把冷启动算进延迟指标里
        self._warmup()

    def _warmup(self, rounds: int = 1) -> None:
        size = self._input_spec.size if self._input_spec.size > 0 else 224
        if self._input_spec.wants_uint8:
            x = np.zeros((1, 3, size, size), dtype=self._np_dtype)
        else:
            x = np.zeros((1, 3, size, size), dtype=np.float32)
        for _ in range(rounds):
            self.session.run(None, {self._input_spec.name: x})

    def run(self, x: np.ndarray) -> np.ndarray:
        if x.dtype != self._np_dtype:
            if self._input_spec.wants_uint8:
                from ..preprocess import quantize_input

                x = quantize_input(x, self._input_spec.scale, self._input_spec.zero_point)
            else:
                x = x.astype(self._np_dtype)
        x = np.ascontiguousarray(x)
        out = self.session.run(None, {self._input_spec.name: x})[0]
        return np.asarray(out, dtype=np.float32)

    def info(self) -> Dict[str, Any]:
        base = super().info()
        base.update({
            "ort_version": ort_version(),
            "providers": self.session.get_providers(),
            "threads": self._threads_used,
            "outputs": [o.name for o in self.session.get_outputs()],
        })
        return base
