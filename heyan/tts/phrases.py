"""播报话术清单。

离线语音能不能做，取决于话术集合是不是封闭的。本系统的播报内容只有三类：
界面提示、识别结论（类别名 + 严重程度 + 处理建议）、系统状态提示，
全部可以提前枚举，因此可以在构建期一次性预渲染成 wav，运行期零合成开销。

每条话术都带一个语义 slug（如 `advice_rice_blast_severe`），
这样县农技站想换成真人方言录音时，只要按同名文件投放即可，不用改代码。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..advice import Advisory, _pick_voice, load_advisory
from ..classes import ClassInfo, Taxonomy, load_taxonomy
from . import languages

KIND_UI = "ui"
KIND_NAME = "name"
KIND_ADVICE = "advice"
KIND_SYSTEM = "system"


@dataclass(frozen=True)
class Phrase:
    slug: str
    text: str
    lang: str
    kind: str
    class_id: str = ""
    severity: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {"slug": self.slug, "text": self.text, "lang": self.lang, "kind": self.kind,
                "class_id": self.class_id, "severity": self.severity, "note": self.note}


# 系统状态提示不在 advisory.json 里，而是运行期才会出现的固定句子。
# 实体文案统一放在 `heyan.i18n`（键名以 `sys_` 开头），界面渲染与语音包共用一份，
# 这样屏上显示的和喇叭念出来的永远不会各说各话。
SYSTEM_KEY_PREFIX = "sys_"


def system_phrases_table(lang: str = "zh") -> Dict[str, str]:
    from ..i18n import STRINGS, t

    return {k: t(k, lang) for k in STRINGS if k.startswith(SYSTEM_KEY_PREFIX)}


# 兼容旧引用：{slug: {lang: text}}
def _build_system_phrases() -> Dict[str, Dict[str, str]]:
    from ..i18n import STRINGS

    return {k: dict(v) for k, v in STRINGS.items() if k.startswith(SYSTEM_KEY_PREFIX)}


SYSTEM_PHRASES: Dict[str, Dict[str, str]] = _build_system_phrases()

UI_SLUG_MAP = {
    "tap_to_shoot": "ui_tap_to_shoot",
    "analyzing": "ui_analyzing",
    "result_ready": "ui_result_ready",
    "retake": "ui_retake",
    "replay_voice": "ui_replay_voice",
    "nav_history": "ui_history",
    "low_confidence": "ui_low_confidence",
    "offline_ready": "ui_offline_ready",
}


def ui_phrases(lang: str = "zh") -> List[Phrase]:
    from ..advice import ui_phrases as _ui

    table = _ui(lang)
    out: List[Phrase] = []
    for key, text in table.items():
        if key.startswith(SYSTEM_KEY_PREFIX):
            continue  # 系统提示由 system_phrases 负责，避免同句两个 slug
        slug = UI_SLUG_MAP.get(key, f"ui_{key}")
        out.append(Phrase(slug=slug, text=text, lang=lang, kind=KIND_UI))
    return out


def system_phrases(lang: str = "zh") -> List[Phrase]:
    out: List[Phrase] = []
    for slug, text in system_phrases_table(lang).items():
        if not text:
            continue
        out.append(Phrase(slug=slug, text=text, lang=lang, kind=KIND_SYSTEM))
    return out


def class_phrases(lang: str = "zh", taxonomy: Optional[Taxonomy] = None,
                  advisory: Optional[Advisory] = None) -> List[Phrase]:
    """类别名 + 各严重程度下的播报稿。"""
    taxonomy = taxonomy or load_taxonomy()
    advisory = advisory or load_advisory()
    out: List[Phrase] = []
    for cls in taxonomy:
        name_text = _class_display_name(cls, lang)
        if name_text:
            out.append(Phrase(slug=f"name_{cls.id}", text=name_text, lang=lang,
                              kind=KIND_NAME, class_id=cls.id))
        entry = advisory.entries.get(cls.id, {})
        for sev_id in advisory.available_severities(cls.id):
            sev_entry = entry.get("by_severity", {}).get(sev_id, {})
            text = _pick_voice(sev_entry, cls, lang)
            if text:
                out.append(Phrase(slug=f"advice_{cls.id}_{sev_id}", text=text, lang=lang,
                                  kind=KIND_ADVICE, class_id=cls.id, severity=sev_id))
    return out


def _class_display_name(cls: ClassInfo, lang: str) -> str:
    if lang.startswith("en"):
        return cls.name_en
    if lang == "yue" and cls.voice_yue:
        return cls.voice_yue
    return cls.voice_zh or cls.name_zh


def all_phrases(lang: str = "zh", taxonomy: Optional[Taxonomy] = None,
                advisory: Optional[Advisory] = None) -> List[Phrase]:
    phrases: List[Phrase] = []
    seen = set()
    for p in (ui_phrases(lang) + class_phrases(lang, taxonomy, advisory)
              + system_phrases(lang)):
        if p.slug in seen:
            continue
        seen.add(p.slug)
        phrases.append(p)
    return phrases


def phrases_for_langs(langs: Sequence[str] = languages.SPOKEN_ORDER,
                      taxonomy: Optional[Taxonomy] = None,
                      advisory: Optional[Advisory] = None) -> Dict[str, List[Phrase]]:
    return {lang: all_phrases(lang, taxonomy, advisory) for lang in langs}


def phrase_count(langs: Sequence[str] = languages.SPOKEN_ORDER) -> Dict[str, int]:
    return {lang: len(all_phrases(lang)) for lang in langs}
