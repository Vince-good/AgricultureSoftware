"""运行期语音门面：把"一句话"变成"一个能播放的 wav"。

界面层只跟这一个类打交道，不需要知道语音到底来自哪里。解析顺序完全离线：

    1. 语音包里的真人方言录音      recordings/<lang>/<slug>.wav
    2. 语音包里预渲染的合成音        <lang>/<slug>.wav
    3. 按语言回退链重试 1~2（潮汕话 → 粤语 → 普通话）
    4. 本机 TTS 引擎现场合成（写进缓存目录，下次直接命中）
    5. 全部失败 -> ok=False，界面退回纯文字

第 4 步只是兜底。文档要求"所有语音资源预置在安装包内"，正常部署下
语音包已经覆盖全部封闭话术，运行期不会触发合成，也就没有那 1~2 秒的开销。

`degraded` 字段专门用来标记"你要的方言没有，我用了相近语言"这件事。
界面必须如实告诉用户，而不是假装念的是客家话 —— 这是适老化设计里的诚信问题。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import PATHS
from . import languages
from .base import SynthResult, TtsEngine, wav_info
from .voicepack import VoicePack, phrase_key


@dataclass
class SpeechClip:
    """一次语音解析的结果。界面拿 `path` 去播，拿 `degraded` 去决定要不要解释。"""

    ok: bool
    text: str = ""
    requested: str = ""            # 用户选择的语言
    language: str = ""             # 实际发声的语言
    source: str = "none"           # recording | voicepack | engine | none
    path: Optional[str] = None
    slug: str = ""
    duration_s: float = 0.0
    sample_rate: int = 0
    engine: str = ""
    degraded: bool = False         # 方言不可用，已回退到相近语言
    error: str = ""

    @property
    def exists(self) -> bool:
        return bool(self.path) and Path(self.path).exists()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["exists"] = self.exists
        if self.language and self.language != self.requested:
            d["language_name"] = languages.get(self.language).name_zh \
                if languages.is_supported(self.language) else self.language
        return d


class Synthesizer:
    """语音解析门面。线程安全（只读语音包 + 引擎调用本身是子进程）。"""

    def __init__(self, voicepack: Optional[VoicePack] = None,
                 bundle_root: Optional[Path] = None,
                 engine: Optional[TtsEngine] = None,
                 cache_dir: Optional[Path] = None,
                 rate: int = -1,
                 allow_runtime_synth: bool = True) -> None:
        self.bundle_root = Path(bundle_root) if bundle_root else None
        self.voicepack = voicepack if voicepack is not None else VoicePack.discover(
            self.bundle_root)
        self._engine = engine
        self.allow_runtime_synth = bool(allow_runtime_synth)
        self.rate = int(rate)
        self.cache_dir = Path(cache_dir) if cache_dir else PATHS.voicepacks / "_runtime_cache"

    # ---------------- 引擎 ----------------
    @property
    def engine(self) -> TtsEngine:
        if self._engine is None:
            from .engines import auto_engine

            self._engine = auto_engine()
        return self._engine

    # ---------------- 解析 ----------------
    def _from_pack(self, slug: str, text: str, lang: str) -> Optional[SpeechClip]:
        hit = None
        if slug:
            hit = self.voicepack.resolve_slug(slug, lang)
        if hit is None and text:
            hit = self.voicepack.resolve(text, lang)
        if hit is None:
            return None
        path, entry = hit
        chain = languages.fallback_chain(lang) if languages.is_supported(lang) else [lang]
        actual = entry.lang or lang
        return SpeechClip(
            ok=True, text=text or entry.text, requested=lang, language=actual,
            source="recording" if entry.source == "recording" else "voicepack",
            path=str(path), slug=entry.slug or slug,
            duration_s=entry.duration_s or wav_info(path).get("duration_s", 0.0),
            sample_rate=entry.sample_rate,
            engine=entry.engine,
            degraded=actual in chain and chain.index(actual) > 0,
        )

    def _effective_language(self, lang: str) -> Optional[str]:
        """本机引擎真正能发声的语言（按回退链找第一个）。"""
        engine = self.engine
        if not engine.available() or engine.name == "silent":
            return None
        chain = languages.fallback_chain(lang) if languages.is_supported(lang) else [lang]
        for code in chain:
            if engine.supports_language(code):
                return code
        return None

    def _runtime_synth(self, text: str, lang: str, slug: str = "") -> Optional[SpeechClip]:
        if not self.allow_runtime_synth or not text:
            return None
        effective = self._effective_language(lang)
        if effective is None:
            return None
        # 方言文本不能用相近语言的嗓子念，否则念出来是错的：
        # 回退时必须改用回退语言自己的文案。
        speak_text = text if effective == lang else None
        if speak_text is None:
            from ..i18n import STRINGS

            key = slug[len("ui_"):] if slug.startswith("ui_") else slug
            speak_text = (STRINGS.get(key, {}) or {}).get(effective)
            if not speak_text:
                return None
        name = f"{slug or 'phrase'}_{effective}_{phrase_key(speak_text)}.wav"
        out = self.cache_dir / effective / name
        if out.exists() and out.stat().st_size > 1024:
            info = wav_info(out)
            return SpeechClip(ok=True, text=speak_text, requested=lang, language=effective,
                              source="engine", path=str(out), slug=slug,
                              duration_s=info["duration_s"], sample_rate=info["sample_rate"],
                              engine=self.engine.name, degraded=effective != lang)
        out.parent.mkdir(parents=True, exist_ok=True)
        res: SynthResult = self.engine.synthesize(speak_text, effective, out, rate=self.rate)
        if not res.ok or not out.exists():
            return None
        return SpeechClip(ok=True, text=speak_text, requested=lang, language=effective,
                          source="engine", path=str(out), slug=slug,
                          duration_s=res.duration_s, sample_rate=res.sample_rate,
                          engine=self.engine.name, degraded=effective != lang)

    def clip(self, text: str = "", lang: str = "zh", slug: str = "") -> SpeechClip:
        """解析一段语音。text 与 slug 至少给一个。"""
        lang = lang if languages.is_supported(lang) else languages.DEFAULT_LANGUAGE
        hit = self._from_pack(slug, text, lang)
        if hit is not None:
            return hit
        hit = self._runtime_synth(text, lang, slug)
        if hit is not None:
            return hit
        return SpeechClip(ok=False, text=text, requested=lang, language="",
                          source="none", slug=slug,
                          error="没有预置语音，本机也没有可用的离线 TTS 引擎")

    def clip_for_slug(self, slug: str, lang: str = "zh") -> SpeechClip:
        from ..i18n import STRINGS

        text = ""
        key = slug[len("ui_"):] if slug.startswith("ui_") else slug
        if key in STRINGS:
            text = STRINGS[key].get(lang) or STRINGS[key].get("zh") or ""
        return self.clip(text, lang, slug=slug)

    def clip_for_result(self, result: Dict[str, Any], lang: str = "zh") -> SpeechClip:
        """识别结果 -> 播报语音。slug 用 `advice_<class_id>_<severity>` 约定。"""
        advice = (result or {}).get("advice") or {}
        class_id = advice.get("class_id") or (result or {}).get("class_id") or ""
        severity = ((advice.get("severity") or {}).get("id")
                    or (result or {}).get("severity", {}).get("id") or "none")
        if (result or {}).get("needs_retake"):
            # 证据不足时绝不念病害结论，只引导重拍
            return self.clip_for_slug("ui_low_confidence", lang)
        slug = f"advice_{class_id}_{severity}"
        text = advice.get("voice") or advice.get("name") or ""
        clip = self.clip(text, lang, slug=slug)
        if not clip.ok and severity != "none":
            # 该严重程度没有预渲染，退回只念类别名
            clip = self.clip(advice.get("name") or "", lang, slug=f"name_{class_id}")
        return clip

    # ---------------- 状态 ----------------
    def voiced_languages(self) -> List[str]:
        """当前部署下真正能发声的语言（语音包已覆盖 或 本机引擎支持）。"""
        packed = set(self.voicepack.languages())
        out: List[str] = []
        for code in languages.SPOKEN_ORDER:
            if code in packed or self._effective_language(code) == code:
                out.append(code)
        return out

    def status(self) -> Dict[str, Any]:
        engine = self.engine
        pack_stats = self.voicepack.stats()
        voiced = self.voiced_languages()
        return {
            "engine": engine.name,
            "engine_available": engine.available(),
            "offline": bool(engine.offline),
            "voicepack": pack_stats,
            "voiced_languages": voiced,
            "requested_languages": list(languages.SPOKEN_ORDER),
            "dialect_needs_recording": [c for c in languages.SPOKEN_ORDER
                                        if c not in voiced],
            "runtime_synth_allowed": self.allow_runtime_synth,
            "cache_dir": str(self.cache_dir),
            "languages": languages.describe(),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<Synthesizer engine={self.engine.name} roots={len(self.voicepack.roots)} "
                f"voiced={self.voiced_languages()}>")
