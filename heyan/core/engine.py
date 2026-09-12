"""识别引擎：整个识别核心的唯一入口。

界面层（Web/Kivy/命令行）只需要调用 `recognize()`，拿到 `RecognitionResult`
就能渲染，不需要知道后端是 onnxruntime 还是纯 numpy，也不知道量化方式。
这就是文档 3.4 说的"识别核心与交互界面解耦"。
"""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from ..advice import build_advice
from ..classes import ClassInfo, Taxonomy, load_taxonomy
from ..config import MIN_CONFIDENCE_FOR_VERDICT, TOP_K
from .backends.registry import create_backend
from .bundle import ModelBundle
from .preprocess import center_crop, normalize, preprocess, read_image, resize_short_side, softmax
from .result import Candidate, RecognitionResult, make_image_id
from .severity import estimate_severity

ImageInput = Union[str, Path, np.ndarray]


def device_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": __import__("os").cpu_count(),
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_total_mb"] = int(vm.total / (1024 * 1024))
        info["ram_available_mb"] = int(vm.available / (1024 * 1024))
    except Exception:
        pass
    return info


class RecognitionEngine:
    def __init__(self, bundle: Union[ModelBundle, str, Path], backend: Optional[str] = None,
                 language: str = "zh", threads: int = 0,
                 min_confidence: float = MIN_CONFIDENCE_FOR_VERDICT,
                 top_k: int = TOP_K, device: Optional[Dict[str, Any]] = None) -> None:
        if not isinstance(bundle, ModelBundle):
            # 允许直接传 bundle 目录/zip 路径：部署脚本和基准测量都这么用
            bundle = ModelBundle.load(bundle)
        self.bundle = bundle
        self.language = language
        self.min_confidence = float(min_confidence)
        self.top_k = int(top_k)
        self.device = device if device is not None else device_info()
        self.backend, self.model_path = create_backend(bundle, backend=backend, threads=threads)
        self.taxonomy: Taxonomy = bundle.taxonomy or load_taxonomy()
        self.advisory = bundle.advisory
        self.input_size = int(bundle.manifest.input_size)
        self.resize_size = int(bundle.manifest.resize_size)
        self.mean = tuple(bundle.manifest.mean)
        self.std = tuple(bundle.manifest.std)
        self._classes: List[ClassInfo] = list(self.taxonomy)
        if len(self._classes) != int(bundle.manifest.num_classes or len(self._classes)):
            raise ValueError(
                f"模型类别数 ({bundle.manifest.num_classes}) 与 labels.json ({len(self._classes)}) 不一致"
            )

    # ---------------- 构造 ----------------
    @classmethod
    def from_dir(cls, root: Path | str, **kwargs) -> "RecognitionEngine":
        return cls(ModelBundle.load(root), **kwargs)

    @classmethod
    def from_zip(cls, zip_path: Path | str, **kwargs) -> "RecognitionEngine":
        return cls(ModelBundle.load(Path(zip_path)), **kwargs)

    # ---------------- 推理 ----------------
    def _to_array(self, image: ImageInput) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        return read_image(image)

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cropped = center_crop(resize_short_side(img, self.resize_size), self.input_size)
        arr = normalize(cropped, self.mean, self.std)
        return arr[None, ...], cropped

    def infer(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        x, _ = self._preprocess(img)
        t0 = time.perf_counter()
        logits = self.backend.run(x)
        dt = (time.perf_counter() - t0) * 1000.0
        return np.asarray(logits, dtype=np.float32).reshape(-1), dt

    def recognize(self, image: ImageInput, image_id: Optional[str] = None,
                  language: Optional[str] = None, with_severity: bool = True) -> RecognitionResult:
        t_start = time.perf_counter()
        img = self._to_array(image)
        x, cropped = self._preprocess(img)

        t0 = time.perf_counter()
        logits = np.asarray(self.backend.run(x), dtype=np.float32).reshape(-1)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        probs = softmax(logits)
        order = np.argsort(-probs)[:max(self.top_k, 1)]
        candidates = [
            Candidate(
                class_id=self._classes[int(i)].id,
                index=int(i),
                name=self._classes[int(i)].name_zh,
                probability=float(probs[int(i)]),
                crop=self._classes[int(i)].crop,
                stress=self._classes[int(i)].stress,
                icon=self._classes[int(i)].icon,
            )
            for i in order
        ]

        top = candidates[0]
        low_confidence = top.probability < self.min_confidence
        if low_confidence:
            # 证据不足就不给病害结论，直接引导重拍
            fb = self.taxonomy.fallback
            if fb is not None and top.class_id != fb.id:
                candidates = [
                    Candidate(class_id=fb.id, index=fb.index, name=fb.name_zh,
                              probability=top.probability, crop=fb.crop, stress=fb.stress,
                              icon=fb.icon)
                ] + candidates
                top = candidates[0]

        lang = language or self.language
        severity_id, severity_score, evidence = ("none", 0.0, {"skipped": True})
        if with_severity:
            cls = self.taxonomy.by_id(top.class_id)
            severity_id, severity_score, evidence = estimate_severity(
                cls.stress, top.probability, cropped, class_id=cls.id
            )

        advice = build_advice(top.class_id, severity_id, lang=lang,
                              advisory=self.advisory, taxonomy=self.taxonomy)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return RecognitionResult(
            candidates=candidates[:max(self.top_k, 1)],
            advice=advice,
            severity_id=severity_id,
            severity_score=severity_score,
            confidence=top.probability,
            low_confidence=low_confidence,
            backend=self.backend.name,
            model_id=self.bundle.manifest.model_id,
            model_version=self.bundle.manifest.version,
            latency_ms=latency_ms,
            input_size=self.input_size,
            language=lang,
            image_id=image_id or make_image_id(),
            evidence=evidence,
            device=self.device,
        )

    def recognize_many(self, images: Iterable[ImageInput], language: Optional[str] = None,
                       with_severity: bool = False) -> List[RecognitionResult]:
        return [self.recognize(im, language=language, with_severity=with_severity) for im in images]

    # ---------------- 元信息 ----------------
    @property
    def class_ids(self) -> List[str]:
        return self.taxonomy.ids

    def info(self) -> Dict[str, Any]:
        return {
            "model_id": self.bundle.manifest.model_id,
            "version": self.bundle.manifest.version,
            "arch": self.bundle.manifest.arch,
            "num_classes": len(self._classes),
            "backend": self.backend.info(),
            "model_size_mb": self.bundle.model_size_mb(),
            "bundle_size_mb": self.bundle.size_mb(),
            "budgets": self.bundle.manifest.budgets,
            "metrics": self.bundle.manifest.metrics,
            "language": self.language,
            "min_confidence": self.min_confidence,
            "device": self.device,
        }

    def close(self) -> None:
        self.backend.close()

    def __enter__(self) -> "RecognitionEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<RecognitionEngine {self.bundle.manifest.model_id} "
                f"backend={self.backend.name} classes={len(self._classes)}>")
