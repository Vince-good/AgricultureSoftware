"""预渲染离线语音包（文档 3.3(1)：所有语音资源预置在安装包内）。

目录结构：

    voicepack/
      index.json                     # slug/文本 -> 文件、时长、引擎
      zh/advice_rice_blast_severe.wav
      yue/...
      recordings/hak/ui_tap_to_shoot.wav   # 真人方言录音，优先级最高

运行期解析顺序（每一步都不联网）：
    1. recordings/<lang>/<slug>.wav     真人录音
    2. <lang>/<slug>.wav                预渲染合成音
    3. 上述两步按语言回退链重试（潮汕话→粤语→普通话）
    4. index 里按文本反查 slug，再走 1~3（应对没有 slug 的临时文本）

为什么要 recordings 这一层：客家话、潮汕话没有可用的开源离线合成语音，
但方言播报恰恰是文档里明确要求的。留一个"投放同名 wav 即替换"的口子，
让县农技站组织一次录音就能把合成音整体换成真人方言，不需要动代码。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import PATHS, VOICEPACK_ROOT
from . import languages
from .base import SynthResult, wav_info

INDEX_NAME = "index.json"
RECORDINGS_DIR = "recordings"
WAV_SUFFIX = ".wav"

_PUNCT = re.compile(r"[\s\u3000]+")
_TRAIL = "。.，,、；;！!？? "


def normalize_text(text: str) -> str:
    """归一化文本，让"稻瘟病。"和"稻瘟病"命中同一条语音。"""
    t = _PUNCT.sub("", str(text or ""))
    return t.strip(_TRAIL)


def phrase_key(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()[:16]


def slugify(text: str) -> str:
    """把任意文本压成文件名安全的 slug（用于没有预定义 slug 的话术）。"""
    base = re.sub(r"[^0-9A-Za-z_.-]+", "_", normalize_text(text)).strip("_")
    return (base[:40] or "phrase") + "_" + phrase_key(text)[:8]


@dataclass
class VoicePackEntry:
    slug: str
    text: str
    lang: str
    key: str
    file: str                 # 相对语音包根目录
    duration_s: float = 0.0
    sample_rate: int = 0
    engine: str = ""
    source: str = "synth"     # synth | recording
    kind: str = ""
    class_id: str = ""
    severity: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VoicePack:
    """只读的语音包解析器。可以同时挂多个根（bundle 内的 + 全局的）。"""

    def __init__(self, roots: Sequence[Path] = ()) -> None:
        self.roots: List[Path] = [Path(r) for r in roots if r and Path(r).exists()]
        self._index: Dict[str, Dict[str, VoicePackEntry]] = {}   # root -> lang -> ...
        self._by_text: Dict[str, Dict[str, str]] = {}            # root -> normtext(lang) -> slug
        self._scan()

    # ---------------- 发现 ----------------
    @classmethod
    def discover(cls, bundle_root: Optional[Path] = None,
                 extra_roots: Sequence[Path] = ()) -> "VoicePack":
        roots: List[Path] = []
        if bundle_root:
            cand = Path(bundle_root) / "voicepack"
            roots.append(cand if cand.exists() else Path(bundle_root))
        roots.extend(Path(r) for r in extra_roots)
        roots.append(PATHS.voicepacks)
        roots.append(VOICEPACK_ROOT)
        seen, uniq = set(), []
        for r in roots:
            key = str(Path(r).resolve()).lower()
            if key in seen or not Path(r).exists():
                continue
            seen.add(key)
            uniq.append(Path(r))
        return cls(uniq)

    def _scan(self) -> None:
        for root in self.roots:
            entries: Dict[str, VoicePackEntry] = {}
            index_path = root / INDEX_NAME
            if index_path.exists():
                try:
                    raw = json.loads(index_path.read_text(encoding="utf-8"))
                    for item in raw.get("entries", []):
                        rel = item.get("file", "")
                        if not rel or not (root / rel).exists():
                            continue
                        entries[f"{item.get('lang', 'zh')}::{item['slug']}"] = VoicePackEntry(
                            slug=item["slug"], text=item.get("text", ""),
                            lang=item.get("lang", "zh"), key=item.get("key", ""),
                            file=rel, duration_s=float(item.get("duration_s", 0.0) or 0.0),
                            sample_rate=int(item.get("sample_rate", 0) or 0),
                            engine=item.get("engine", ""), source=item.get("source", "synth"),
                            kind=item.get("kind", ""), class_id=item.get("class_id", ""),
                            severity=item.get("severity", ""),
                        )
                except Exception:
                    pass
            # 索引外的散落 wav 也收进来（现场录音常常是直接拷进目录的）
            for wav in sorted(root.rglob(f"*{WAV_SUFFIX}")):
                rel = wav.relative_to(root).as_posix()
                parts = rel.split("/")
                if parts[0] == RECORDINGS_DIR and len(parts) >= 3:
                    lang, slug = parts[1], Path(parts[-1]).stem
                    source = "recording"
                elif len(parts) >= 2:
                    lang, slug = parts[0], wav.stem
                    source = "synth"
                else:
                    continue
                tag = f"{lang}::{slug}"
                existing = entries.get(tag)
                if existing is None:
                    info = wav_info(wav)
                    entries[tag] = VoicePackEntry(
                        slug=slug, text=existing.text if existing else "", lang=lang,
                        key=phrase_key(existing.text) if existing and existing.text else "",
                        file=rel, duration_s=info["duration_s"],
                        sample_rate=info["sample_rate"], source=source)
                elif source == "recording" and existing.source != "recording":
                    # 真人录音覆盖合成音
                    info = wav_info(wav)
                    entries[tag] = VoicePackEntry(**{**existing.to_dict(), "file": rel,
                                                    "source": "recording",
                                                    "duration_s": info["duration_s"],
                                                    "sample_rate": info["sample_rate"]})
            self._index[str(root)] = entries
            by_text: Dict[str, str] = {}
            for tag, e in entries.items():
                if e.text:
                    by_text.setdefault(f"{e.lang}::{normalize_text(e.text)}", e.slug)
            self._by_text[str(root)] = by_text

    # ---------------- 查询 ----------------
    def _roots_ordered(self) -> List[Path]:
        return self.roots

    def _entry(self, root: Path, lang: str, slug: str) -> Optional[VoicePackEntry]:
        # 录音优先于合成音
        rec = self._index[str(root)].get(f"{lang}::{slug}")
        if rec is not None and rec.source == "recording":
            return rec
        for tag, e in self._index[str(root)].items():
            if tag.endswith(f"::{slug}") and e.lang == lang:
                return e
        return rec

    def resolve_slug(self, slug: str, lang: str = "zh") -> Optional[Tuple[Path, VoicePackEntry]]:
        chain = languages.fallback_chain(lang) if languages.is_supported(lang) else [lang, "zh"]
        for root in self._roots_ordered():
            for code in chain:
                e = self._entry(root, code, slug)
                if e is None:
                    continue
                path = root / e.file
                if path.exists():
                    return path, e
        return None

    def resolve(self, text: str, lang: str = "zh") -> Optional[Tuple[Path, VoicePackEntry]]:
        """按文本查语音。先直接当 slug 试，再走索引反查。"""
        norm = normalize_text(text)
        if not norm:
            return None
        direct = self.resolve_slug(text.strip(), lang)
        if direct:
            return direct
        chain = languages.fallback_chain(lang) if languages.is_supported(lang) else [lang, "zh"]
        for root in self._roots_ordered():
            table = self._by_text.get(str(root), {})
            for code in chain:
                slug = table.get(f"{code}::{norm}")
                if not slug:
                    continue
                hit = self._entry(root, code, slug)
                if hit is None:
                    continue
                path = root / hit.file
                if path.exists():
                    return path, hit
        return None

    def entries(self, lang: Optional[str] = None) -> List[VoicePackEntry]:
        out: List[VoicePackEntry] = []
        seen = set()
        for root in self.roots:
            for e in self._index[str(root)].values():
                if lang and e.lang != lang:
                    continue
                tag = (e.lang, e.slug)
                if tag in seen:
                    continue
                seen.add(tag)
                out.append(e)
        return sorted(out, key=lambda e: (e.lang, e.kind, e.slug))

    def languages(self) -> List[str]:
        return sorted({e.lang for e in self.entries()})

    def stats(self) -> Dict[str, Any]:
        entries = self.entries()
        per_lang: Dict[str, int] = {}
        recordings = 0
        total_s = 0.0
        for e in entries:
            per_lang[e.lang] = per_lang.get(e.lang, 0) + 1
            if e.source == "recording":
                recordings += 1
            total_s += e.duration_s
        return {
            "roots": [str(r) for r in self.roots],
            "entries": len(entries),
            "per_language": per_lang,
            "recordings": recordings,
            "total_duration_s": round(total_s, 2),
        }


# --------------------------------------------------------------------------
# 构建
# --------------------------------------------------------------------------

def build_voicepack(langs: Sequence[str], out_dir: Path | str, engine=None,
                    taxonomy=None, advisory=None, rate: int = 0, force: bool = False,
                    progress: Any = print) -> Dict[str, Any]:
    """把话术清单渲染成 wav，写出 index.json。"""
    from .phrases import all_phrases
    from .engines import auto_engine

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = engine or auto_engine()
    if not engine.available() or engine.name == "silent":
        return {"ok": False, "engine": engine.name, "entries": 0,
                "error": "没有可用的离线 TTS 引擎，语音包未生成（识别功能不受影响）"}

    index_path = out_dir / INDEX_NAME
    existing: List[Dict[str, Any]] = []
    if index_path.exists() and not force:
        try:
            existing = json.loads(index_path.read_text(encoding="utf-8")).get("entries", [])
        except Exception:
            existing = []
    have = {(e["lang"], e["slug"]) for e in existing
            if (out_dir / e.get("file", "")).exists()}

    entries: List[Dict[str, Any]] = list(existing)
    made = skipped = failed = 0
    failures: List[Dict[str, str]] = []
    unvoiced: List[str] = []
    rendered_langs: List[str] = []
    total_s = 0.0

    for lang in langs:
        if not languages.is_supported(lang):
            progress(f"[voicepack] 跳过未知语言 {lang}")
            continue
        if not engine.supports_language(lang):
            # 本机没有这门语言的离线语音。宁可缺，也不能用普通话的嗓子念粤语白话文 ——
            # 运行期 VoicePack 会按回退链去播已经正确的普通话版本。
            unvoiced.append(lang)
            chain = languages.fallback_chain(lang)[1:]
            progress(f"[voicepack] 跳过 {languages.get(lang).name_zh}：本机无该语言离线语音"
                     f"（运行期回退 {'->'.join(chain) or 'zh'}）")
            continue
        rendered_langs.append(lang)
        phrases = all_phrases(lang, taxonomy, advisory)
        lang_dir = out_dir / lang
        for p in phrases:
            if (lang, p.slug) in have:
                skipped += 1
                continue
            lang_dir.mkdir(parents=True, exist_ok=True)
            target = lang_dir / f"{p.slug}{WAV_SUFFIX}"
            res = engine.synthesize(p.text, lang, target, rate=rate)
            if not res.ok or not target.exists():
                failed += 1
                failures.append({"lang": lang, "slug": p.slug, "text": p.text,
                                 "error": res.error})
                if target.exists():
                    target.unlink()
                continue
            info = wav_info(target)
            total_s += info["duration_s"]
            entries.append({
                "slug": p.slug, "text": p.text, "lang": lang,
                "key": phrase_key(p.text), "file": f"{lang}/{p.slug}{WAV_SUFFIX}",
                "duration_s": info["duration_s"], "sample_rate": info["sample_rate"],
                "engine": engine.name, "source": "synth", "kind": p.kind,
                "class_id": p.class_id, "severity": p.severity,
            })
            have.add((lang, p.slug))
            made += 1
        progress(f"[voicepack] {languages.get(lang).name_zh}: 本语言累计 "
                 f"{sum(1 for e in entries if e['lang'] == lang)} 条")

    # 同 slug 后写的覆盖先写的，保持索引唯一
    dedup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for e in entries:
        dedup[(e["lang"], e["slug"])] = e
    final = sorted(dedup.values(), key=lambda e: (e["lang"], e.get("kind", ""), e["slug"]))
    payload = {
        "schema_version": "1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": engine.name,
        "sample_rate": getattr(engine, "sample_rate", 22050),
        "languages": rendered_langs,
        "requested_languages": list(langs),
        "unvoiced_languages": unvoiced,
        "entries": final,
        "stats": {
            "entries": len(final), "rendered": made, "skipped": skipped, "failed": failed,
            "total_duration_s": round(sum(float(e.get("duration_s", 0.0)) for e in final), 2),
            "size_mb": round(sum((out_dir / e["file"]).stat().st_size for e in final
                                 if (out_dir / e["file"]).exists()) / (1024 * 1024), 3),
        },
        "failures": failures[:20],
    }
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"ok": failed == 0, "engine": engine.name, "entries": len(final),
              "rendered": made, "skipped": skipped, "failed": failed,
              "out_dir": str(out_dir), "index": str(index_path),
              "languages": rendered_langs, "unvoiced_languages": unvoiced,
              "stats": payload["stats"], "failures": failures[:20]}
    progress(f"[voicepack] 完成：{made} 条新渲染，{skipped} 条已存在，{failed} 条失败 -> {out_dir}")
    return result


def copy_into_bundle(pack_dir: Path | str, bundle_dir: Path | str) -> Path:
    """把构建好的语音包拷进 bundle，使其自包含。"""
    src, dst = Path(pack_dir), Path(bundle_dir) / "voicepack"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst
