import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heyan.tts import engines

print("engines:", [(e.name, e.available()) for e in engines.all_engines()])
sapi = engines.SapiEngine()
for v in sapi.installed_voices():
    print("  voice:", v)
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/_smoke_zh.wav")
out.parent.mkdir(parents=True, exist_ok=True)
r = sapi.synthesize("稻瘟病，中度，建议尽快喷药", "zh", out)
print("zh ->", r.ok, r.wav_path, r.duration_s, r.error, r.extra)
r2 = sapi.synthesize("水稻有蟲害，要快啲處理", "yue", out.with_name("smoke_yue.wav"))
print("yue ->", r2.ok, r2.duration_s, r2.error, r2.extra)
