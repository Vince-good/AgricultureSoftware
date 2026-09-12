"""识别结果的数据结构。界面层只读这些字段，不再自己解析模型输出。"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..advice import Advice


@dataclass(frozen=True)
class Candidate:
    class_id: str
    index: int
    name: str
    probability: float
    crop: str
    stress: str
    icon: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecognitionResult:
    candidates: List[Candidate]
    advice: Advice
    severity_id: str
    severity_score: float
    confidence: float
    low_confidence: bool
    backend: str
    model_id: str
    model_version: str
    latency_ms: float
    input_size: int
    language: str
    image_id: str
    created_at: float = field(default_factory=time.time)
    evidence: Dict[str, Any] = field(default_factory=dict)
    device: Dict[str, Any] = field(default_factory=dict)

    @property
    def top(self) -> Candidate:
        return self.candidates[0]

    @property
    def class_id(self) -> str:
        return self.top.class_id

    @property
    def needs_retake(self) -> bool:
        """证据不足时不给结论，只提示重拍 —— 田间误判的代价远高于多拍一张。"""
        return self.low_confidence or self.top.class_id == "unusable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "created_at": self.created_at,
            "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.created_at)),
            "class_id": self.class_id,
            "confidence": round(float(self.confidence), 4),
            "low_confidence": self.low_confidence,
            "needs_retake": self.needs_retake,
            "severity": {
                "id": self.severity_id,
                "score": round(float(self.severity_score), 4),
            },
            "candidates": [c.to_dict() for c in self.candidates],
            "advice": self.advice.to_dict(),
            "runtime": {
                "backend": self.backend,
                "model_id": self.model_id,
                "model_version": self.model_version,
                "latency_ms": round(float(self.latency_ms), 2),
                "input_size": self.input_size,
                "language": self.language,
            },
            "evidence": self.evidence,
            "device": self.device,
        }

    def summary_line(self) -> str:
        if self.needs_retake:
            return "未能识别，请靠近叶片重拍"
        return f"{self.advice.name}（{self.advice.severity_name}，置信度 {self.confidence:.0%}）"


def make_image_id(prefix: str = "img") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"


def optional(value: Any) -> Optional[Any]:
    return None if value in ("", None) else value
