"""播报语言与方言配置（文档 3.3(1)：普通话、粤语、客家话、潮汕话）。

一个必须说清楚的工程现实：离线 TTS 引擎里，普通话和粤语能找到现成语音
（Windows SAPI 的 Huihui / Tracy、espeak-ng 的 cmn / yue），而客家话与潮汕话
目前没有成熟的开源离线合成语音。硬凑一个"听起来像"的合成音，对老年用户
反而是负担。

所以这里的设计是：
  1. 语言与回退链在配置里显式声明，hak/teochew 先退到粤语、再退到普通话；
  2. 语音包目录支持直接投放真人录音（`recordings/<lang>/<key>.wav`），
     县农技站组织一次方言录音即可把该语言的合成音整体替换掉；
  3. 界面上的方言文案始终按所选语言显示，语音不可用时至少文字是对的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Language:
    code: str
    name_zh: str
    name_local: str
    name_en: str
    bcp47: str
    espeak_voice: str = ""
    piper_voices: Tuple[str, ...] = ()
    sapi_voices: Tuple[str, ...] = ()
    fallback: Optional[str] = None
    offline_tts_available: bool = True
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "code": self.code, "name_zh": self.name_zh, "name_local": self.name_local,
            "name_en": self.name_en, "bcp47": self.bcp47,
            "offline_tts_available": self.offline_tts_available,
            "fallback": self.fallback, "note": self.note,
        }


LANGUAGES: Dict[str, Language] = {
    "zh": Language(
        code="zh", name_zh="普通话", name_local="普通话", name_en="Mandarin",
        bcp47="zh-CN", espeak_voice="cmn",
        piper_voices=("zh_CN-huayan-medium",),
        sapi_voices=("Microsoft Huihui Desktop", "Microsoft Yaoyao Desktop",
                     "Microsoft Kangkang Desktop", "Huihui", "Yaoyao", "Kangkang",
                     "Microsoft Huihui", "Microsoft Yaoyao"),
        note="默认播报语言，离线语音齐全",
    ),
    "yue": Language(
        code="yue", name_zh="粤语", name_local="粵語", name_en="Cantonese",
        bcp47="zh-HK", espeak_voice="yue",
        sapi_voices=("Microsoft Tracy Desktop", "Microsoft Danny Desktop", "Tracy", "Danny",
                     "Microsoft Tracy", "Microsoft Danny"),
        fallback="zh",
        note="部分 Windows 需装粤语语音包；缺失时自动回退普通话",
    ),
    "hak": Language(
        code="hak", name_zh="客家话", name_local="客家話", name_en="Hakka",
        bcp47="zh-Hakka", espeak_voice="hak",
        fallback="yue", offline_tts_available=False,
        note="无成熟离线合成语音，默认回退粤语/普通话；支持投放真人录音替换",
    ),
    "teochew": Language(
        code="teochew", name_zh="潮汕话", name_local="潮州話", name_en="Teochew",
        bcp47="zh-min-nan", espeak_voice="nan",
        fallback="yue", offline_tts_available=False,
        note="无成熟离线合成语音，默认回退粤语/普通话；支持投放真人录音替换",
    ),
    "en": Language(
        code="en", name_zh="英语", name_local="English", name_en="English",
        bcp47="en-US", espeak_voice="en",
        piper_voices=("en_US-lessac-medium",),
        sapi_voices=("Microsoft David Desktop", "Microsoft Zira Desktop", "David", "Zira"),
        note="用于双语交互设计指南与对外演示",
    ),
}

DEFAULT_LANGUAGE = "zh"
SPOKEN_ORDER = ("zh", "yue", "hak", "teochew", "en")


def get(code: str) -> Language:
    lang = LANGUAGES.get(code)
    if lang is None:
        raise KeyError(f"不支持的语言: {code}，可选 {sorted(LANGUAGES)}")
    return lang


def is_supported(code: str) -> bool:
    return code in LANGUAGES


def fallback_chain(code: str) -> List[str]:
    """按回退链展开，例如 teochew -> ['teochew', 'yue', 'zh']。"""
    chain: List[str] = []
    seen = set()
    cur: Optional[str] = code
    while cur and cur not in seen:
        seen.add(cur)
        chain.append(cur)
        lang = LANGUAGES.get(cur)
        cur = lang.fallback if lang else None
    if "zh" not in chain and code != "en":
        chain.append("zh")
    return chain


def all_codes(spoken_only: bool = False) -> List[str]:
    if not spoken_only:
        return list(SPOKEN_ORDER)
    return [c for c in SPOKEN_ORDER if c in LANGUAGES]


def describe() -> List[Dict[str, object]]:
    return [LANGUAGES[c].to_dict() for c in SPOKEN_ORDER if c in LANGUAGES]


def display_names() -> Dict[str, str]:
    return {c: LANGUAGES[c].name_local for c in SPOKEN_ORDER if c in LANGUAGES}
