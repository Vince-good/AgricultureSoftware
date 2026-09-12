"""农艺建议知识库与多语言播报稿生成。

文档 3.3 要求播报内容为"病害名称、严重程度、处理建议"，且语音资源必须预置在安装包内。
这里把三样东西拼成一份 `Advice`，并给出固定话术列表（`voice_phrases`），
供 `tools/build_voicepack.py` 离线预渲染成 wav，运行时直接播放、零合成开销。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from .classes import ClassInfo, load_taxonomy
from .config import ADVISORY_PATH

# 中文书面语在粤语/客家话/潮汕话区通用，只有"说出来"的部分需要方言版本。
_SPOKEN_LANGS = ("zh", "yue", "hak", "teochew")
_SEVERITY_FALLBACK = ("severe", "moderate", "mild", "none")


@dataclass(frozen=True)
class SeverityLevel:
    id: str
    name_zh: str
    name_en: str
    order: int
    color: str


@dataclass
class Advice:
    class_id: str
    name: str
    summary: str
    severity_id: str
    severity_name: str
    severity_color: str
    actions: List[str]
    voice: str
    icon: str
    crop: str
    stress: str
    window_days: int = 0
    insurance_claimable: bool = False
    agro_input_category: str = ""
    agro_input_keywords: List[str] = field(default_factory=list)
    lang: str = "zh"
    is_fallback: bool = False

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "name": self.name,
            "summary": self.summary,
            "severity": {
                "id": self.severity_id,
                "name": self.severity_name,
                "color": self.severity_color,
            },
            "actions": list(self.actions),
            "voice": self.voice,
            "icon": self.icon,
            "crop": self.crop,
            "stress": self.stress,
            "window_days": self.window_days,
            "insurance_claimable": self.insurance_claimable,
            "agro_input": {
                "category": self.agro_input_category,
                "keywords": list(self.agro_input_keywords),
            },
            "lang": self.lang,
        }


@dataclass(frozen=True)
class Advisory:
    entries: Dict[str, dict]
    severities: Dict[str, SeverityLevel]
    schema_version: str = "1.0"

    def severity(self, sev_id: str) -> SeverityLevel:
        if sev_id not in self.severities:
            raise KeyError(f"未知严重程度: {sev_id}")
        return self.severities[sev_id]

    def available_severities(self, class_id: str) -> List[str]:
        entry = self.entries.get(class_id, {})
        return list(entry.get("by_severity", {}).keys())


@lru_cache(maxsize=4)
def load_advisory(path: Optional[Path] = None) -> Advisory:
    p = Path(path) if path else ADVISORY_PATH
    with open(p, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    severities = {
        s["id"]: SeverityLevel(
            id=s["id"], name_zh=s["name_zh"], name_en=s["name_en"],
            order=int(s.get("order", 0)), color=s.get("color", "#2E7D32"),
        )
        for s in raw.get("severity_levels", [])
    }
    return Advisory(entries=raw.get("entries", {}), severities=severities,
                    schema_version=str(raw.get("schema_version", "1.0")))


def _pick_voice(entry_sev: dict, cls: ClassInfo, lang: str) -> str:
    """按方言优先级挑播报稿：方言稿 → 粤/普通话稿 → 类别名。"""
    if lang in _SPOKEN_LANGS:
        for suffix in (lang, "yue" if lang in ("hak", "teochew") else "zh", "zh"):
            key = f"voice_{suffix}"
            if entry_sev.get(key):
                return str(entry_sev[key])
            if getattr(cls, key, ""):
                return str(getattr(cls, key))
        return cls.name_zh
    return cls.name_en


def _resolve_severity(advisory: Advisory, class_id: str, severity_id: Optional[str]) -> str:
    have = advisory.available_severities(class_id)
    if not have:
        return "none"
    if severity_id and severity_id in have:
        return severity_id
    for cand in _SEVERITY_FALLBACK:
        if cand in have:
            return cand
    return have[0]


def build_advice(class_id: str, severity_id: Optional[str] = None, lang: str = "zh",
                 advisory: Optional[Advisory] = None, taxonomy=None) -> Advice:
    """把类别 + 严重程度 + 语言拼成一份可直接显示/播报的建议。"""
    advisory = advisory or load_advisory()
    taxonomy = taxonomy or load_taxonomy()
    cls = taxonomy.by_id(class_id)
    entry = advisory.entries.get(class_id, {})
    sev_id = _resolve_severity(advisory, class_id, severity_id)
    sev = advisory.severity(sev_id) if sev_id in advisory.severities else SeverityLevel(
        sev_id, sev_id, sev_id, 0, "#2E7D32")
    sev_entry = entry.get("by_severity", {}).get(sev_id, {})

    if lang.startswith("en"):
        actions = list(sev_entry.get("actions_en") or sev_entry.get("actions_zh") or [])
        name = cls.name_en
        summary = str(entry.get("summary_zh", ""))
    else:
        actions = list(sev_entry.get("actions_zh") or [])
        name = cls.name_zh
        summary = str(entry.get("summary_zh", ""))

    agro = entry.get("agro_input", {}) or {}
    return Advice(
        class_id=cls.id,
        name=name,
        summary=summary,
        severity_id=sev.id,
        severity_name=sev.name_en if lang.startswith("en") else sev.name_zh,
        severity_color=sev.color,
        actions=actions,
        voice=_pick_voice(sev_entry, cls, lang),
        icon=cls.icon,
        crop=cls.crop,
        stress=cls.stress,
        window_days=int(entry.get("window_days", 0) or 0),
        insurance_claimable=bool(entry.get("insurance_claimable", False)),
        agro_input_category=str(agro.get("category_zh", "")),
        agro_input_keywords=list(agro.get("keywords", []) or []),
        lang=lang,
        is_fallback=cls.is_fallback,
    )


def voice_phrases(lang: str = "zh", advisory: Optional[Advisory] = None, taxonomy=None) -> List[str]:
    """枚举全部固定播报话术，用于离线语音包预渲染。

    离线 TTS 的关键不是"能不能合成"，而是"能不能在安装包里提前合成好"。
    本系统的结论话术是封闭集合，所以可以做到 100% 预置、运行时零合成。
    """
    advisory = advisory or load_advisory()
    taxonomy = taxonomy or load_taxonomy()
    phrases: List[str] = []
    seen = set()

    def add(text: str) -> None:
        text = (text or "").strip()
        if text and text not in seen:
            seen.add(text)
            phrases.append(text)

    for cls in taxonomy:
        add(_pick_voice({}, cls, lang))
        for sev_id in advisory.available_severities(cls.id):
            add(_pick_voice(advisory.entries[cls.id]["by_severity"][sev_id], cls, lang))
    return phrases


def ui_phrases(lang: str = "zh") -> Dict[str, str]:
    """界面固定文案，同样需要进语音包（提示音、失败提示等）。

    实体内容在 `heyan.i18n`，那里同时供界面渲染使用，保证"屏上写的"
    和"喇叭念的"永远是同一句话。这里保留函数只是为了不破坏既有调用方。
    """
    from .i18n import table

    return table(lang)
