"""快速自检：导入链、i18n 覆盖率、话术清单、语音引擎探测。不产生音频。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import heyan
from heyan import i18n
from heyan.advice import build_advice, load_advisory, ui_phrases
from heyan.classes import load_taxonomy
from heyan.core import RecognitionEngine, ModelBundle
from heyan.tts import engines, languages, phrases, synth, voicepack

print("version:", heyan.__version__)
print()
print("== i18n coverage ==")
for lang, info in i18n.coverage().items():
    print(f"  {lang:<8} {info['keys']}/{info['total']} complete={info['complete']} "
          f"review={info['review_needed']}")
print()
print("== phrase counts per language ==")
for lang, n in phrases.phrase_count().items():
    print(f"  {lang:<8} {n}")
print()
tax = load_taxonomy()
adv = load_advisory()
print("taxonomy:", len(tax), "classes; crops:", sorted(tax.crops))
print("severities:", sorted(adv.severities))
print()
print("== sample advice (rice_blast / moderate) ==")
for lang in ("zh", "yue", "hak", "teochew", "en"):
    a = build_advice("rice_blast", "moderate", lang=lang, advisory=adv, taxonomy=tax)
    print(f"  [{lang}] {a.name} | {a.severity_name} | voice={a.voice[:46]}")
print()
print("== ui slug samples ==")
for p in phrases.ui_phrases("zh")[:6]:
    print(f"  {p.slug:<26} {p.text}")
for p in phrases.system_phrases("yue")[:4]:
    print(f"  {p.slug:<26} {p.text}")
print()
print("== engines ==")
for e in engines.all_engines():
    langs_ok = [l for l in languages.SPOKEN_ORDER if e.supports_language(l)]
    print(f"  {e.name:<8} available={e.available()} supports={langs_ok}")
print()
print("== synthesizer (no voicepack yet) ==")
s = synth.Synthesizer()
print("  voiced_languages:", s.voiced_languages())
c = s.clip_for_slug("ui_tap_to_shoot", "zh")
print("  ui_tap_to_shoot zh ->", c.ok, c.source, c.language, c.path, c.error)
c2 = s.clip("稻瘟病，中度", "yue", slug="advice_rice_blast_moderate")
print("  yue fallback ->", c2.ok, c2.source, c2.language, "degraded=", c2.degraded, c2.error)
