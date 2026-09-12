"""语音解析链测试：预置包优先、真人录音覆盖合成音、方言回退要诚实。

这里故意关掉运行期合成（allow_runtime_synth=False），
只验证"安装包自带语音"这条主路径，不依赖本机有没有 SAPI。
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from heyan.tts.synth import Synthesizer
from heyan.tts.voicepack import VoicePack, phrase_key, slugify


def _write_wav(path: Path, seconds: float = 0.2, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(b"".join(
            struct.pack("<h", int(6000 * math.sin(i / 9.0))) for i in range(frames)))


@pytest.fixture()
def synth(tmp_path):
    root = tmp_path / "pack"
    _write_wav(root / "zh" / "ui_hello.wav")
    _write_wav(root / "zh" / "ui_low_confidence.wav")
    _write_wav(root / "zh" / "advice_rice_blast_mild.wav")
    pack = VoicePack([root])
    return Synthesizer(voicepack=pack, allow_runtime_synth=False)


def test_slug_helpers_stable():
    assert phrase_key("你好 世界") == phrase_key("你好 世界")
    # 句尾标点归一化：同一句话加不加句号必须命中同一条语音
    assert phrase_key("稻瘟病。") == phrase_key("稻瘟病")
    assert phrase_key("你好世界") != phrase_key("再见世界")
    assert slugify("水稻·稻瘟病")


def test_pack_hit(synth):
    clip = synth.clip_for_slug("ui_hello", "zh")
    assert clip.ok and clip.exists
    assert clip.source == "voicepack"
    assert clip.language == "zh"
    assert clip.degraded is False
    assert clip.duration_s == pytest.approx(0.2, abs=0.05)


def test_dialect_fallback_is_flagged(synth):
    clip = synth.clip_for_slug("ui_hello", "teochew")
    assert clip.ok
    assert clip.language == "zh"
    assert clip.degraded is True, "回退必须标记，界面要如实告诉用户念的是普通话"


def test_recording_overrides_synth(tmp_path):
    root = tmp_path / "pack2"
    _write_wav(root / "zh" / "ui_hello.wav", seconds=0.2)
    _write_wav(root / "recordings" / "zh" / "ui_hello.wav", seconds=0.5)
    clip = Synthesizer(voicepack=VoicePack([root]),
                       allow_runtime_synth=False).clip_for_slug("ui_hello", "zh")
    assert clip.source == "recording"
    assert clip.duration_s == pytest.approx(0.5, abs=0.08)


def test_retake_never_speaks_diagnosis(synth):
    result = {"class_id": "rice_blast", "needs_retake": True, "confidence": 0.2,
              "severity": {"id": "mild", "score": 0.4},
              "advice": {"class_id": "rice_blast", "name": "稻瘟病",
                         "voice": "稻瘟病刚开始。", "severity": {"id": "mild"}}}
    clip = synth.clip_for_result(result, "zh")
    assert clip.ok
    assert clip.slug == "ui_low_confidence"


def test_missing_phrase_is_honest(synth):
    clip = synth.clip("包里没有的一句话", "zh", slug="advice_unknown_severe")
    assert clip.ok is False
    assert clip.error
    assert clip.path is None
