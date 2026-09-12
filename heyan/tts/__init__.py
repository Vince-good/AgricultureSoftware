"""离线语音子系统（文档 3.3(1)：离线 TTS，语音资源预置在安装包内）。

这里刻意**不做**顶层 eager import：`heyan.advice` 会 import `heyan.i18n`，
而 `heyan.i18n` 又 import `heyan.tts.languages`。如果本文件在导入时就拉进
`phrases`/`voicepack`（它们反过来 import `heyan.advice`），就会形成循环导入。
用 PEP 562 的模块级 `__getattr__` 做惰性转发，调用方写法不变，环也就断了。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

_SUBMODULES = ("languages", "base", "engines", "phrases", "voicepack", "synth")

__all__ = [
    "languages", "base", "engines", "phrases", "voicepack", "synth",
    "Synthesizer", "SpeechClip", "auto_engine", "build_voicepack", "VoicePack",
    "all_phrases", "TtsEngine", "SynthResult",
]


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    # 常用符号的快捷出口，省得调用方写三级路径
    _ALIASES = {
        "Synthesizer": ("synth", "Synthesizer"),
        "SpeechClip": ("synth", "SpeechClip"),
        "auto_engine": ("engines", "auto_engine"),
        "all_engines": ("engines", "all_engines"),
        "describe_engines": ("engines", "describe_engines"),
        "build_voicepack": ("voicepack", "build_voicepack"),
        "VoicePack": ("voicepack", "VoicePack"),
        "copy_into_bundle": ("voicepack", "copy_into_bundle"),
        "all_phrases": ("phrases", "all_phrases"),
        "Phrase": ("phrases", "Phrase"),
        "TtsEngine": ("base", "TtsEngine"),
        "SynthResult": ("base", "SynthResult"),
        "wav_info": ("base", "wav_info"),
        "LANGUAGES": ("languages", "LANGUAGES"),
        "fallback_chain": ("languages", "fallback_chain"),
    }
    if name in _ALIASES:
        import importlib

        mod_name, attr = _ALIASES[name]
        module = importlib.import_module(f"{__name__}.{mod_name}")
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(list(globals()) + __all__ + list(_SUBMODULES)))


if TYPE_CHECKING:  # pragma: no cover - 仅供静态检查
    from . import base, engines, languages, phrases, synth, voicepack
    from .base import SynthResult, TtsEngine
    from .engines import all_engines, auto_engine, describe_engines
    from .phrases import Phrase, all_phrases
    from .synth import SpeechClip, Synthesizer
    from .voicepack import VoicePack, build_voicepack, copy_into_bundle
