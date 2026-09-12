"""离线 TTS 引擎抽象与合成结果结构。

所有引擎都必须满足同一条底线：**不联网**。云端 TTS（包括浏览器 Web Speech API）
在粤东西北的信号条件下不可靠，因此一律不纳入回退链。
"""

from __future__ import annotations

import wave
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import languages


@dataclass
class SynthResult:
    ok: bool
    text: str = ""
    language: str = ""
    engine: str = ""
    source: str = "none"        # voicepack | recording | engine | none
    wav_path: Optional[str] = None
    duration_s: float = 0.0
    sample_rate: int = 0
    error: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def exists(self) -> bool:
        return bool(self.wav_path) and Path(self.wav_path).exists()


def wav_info(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 22050
            return {
                "duration_s": round(frames / float(rate), 3),
                "sample_rate": rate,
                "channels": wf.getnchannels(),
                "sampwidth": wf.getsampwidth(),
                "bytes": int(path.stat().st_size),
            }
    except Exception as exc:
        return {"duration_s": 0.0, "sample_rate": 0, "channels": 0, "sampwidth": 0,
                "bytes": int(path.stat().st_size) if path.exists() else 0,
                "error": f"{type(exc).__name__}: {exc}"}


class TtsEngine(ABC):
    """合成引擎接口。实现方只负责"文字 -> wav 文件"。"""

    name = "base"
    offline = True

    @abstractmethod
    def available(self) -> bool:
        """当前机器上能不能用（依赖是否存在）。"""

    @abstractmethod
    def synthesize(self, text: str, language: str = "zh", out_path: Path | str = "",
                   rate: int = 0) -> SynthResult:
        """合成到 out_path；out_path 为空时由引擎自行决定临时路径。"""

    def voices(self, language: str = "zh") -> List[str]:
        return []

    def supports_language(self, language: str) -> bool:
        """能不能用**这门语言自己的语音**来合成。

        和 `available()` 的区别很关键：一台只装了普通话语音的机器上，
        `available()` 为真，但 `supports_language("yue")` 必须为假 ——
        否则我们会用普通话的嗓子去念粤语白话文，念出来是错的。
        这种情况下正确做法是跳过该语言，让运行期按回退链去播普通话。
        """
        return bool(self.voices(language))

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name, "available": self.available(), "offline": self.offline,
            "voices": self.voices(),
            "languages": [lang for lang in languages.SPOKEN_ORDER
                          if self.supports_language(lang)],
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<TtsEngine {self.name} available={self.available()}>"
