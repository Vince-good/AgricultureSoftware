"""具体 TTS 引擎实现。

优先级：Piper（神经语音，最自然）→ espeak-ng（轻量、跨平台、支持粤语）
        → Windows SAPI（系统自带，中文 Huihui 一定有）→ Silent（不发声）。

三者都是纯本地进程，没有任何网络调用。构建语音包时在构建机上跑一次，
边缘设备上只需要播放预渲染好的 wav，连引擎都不用装。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .base import SynthResult, TtsEngine, wav_info
from . import languages

SCRIPT_DIR = Path(__file__).resolve().parent
SAPI_SCRIPT = SCRIPT_DIR / "sapi_render.ps1"


def _which(*names: str) -> Optional[str]:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def _run(cmd: Sequence[str], timeout: int = 120) -> subprocess.CompletedProcess:
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace", creationflags=creationflags)


# --------------------------------------------------------------------------
# Windows SAPI
# --------------------------------------------------------------------------

class SapiEngine(TtsEngine):
    """Windows 系统自带语音。离线、零额外依赖，是本项目的保底方案。"""

    name = "sapi"
    offline = True

    def __init__(self, sample_rate: int = 22050, volume: int = 100) -> None:
        self.sample_rate = sample_rate
        self.volume = volume
        self._voices: Optional[List[Dict[str, str]]] = None
        self._powershell = _which("powershell", "pwsh")

    def available(self) -> bool:
        return os.name == "nt" and bool(self._powershell) and SAPI_SCRIPT.exists()

    def installed_voices(self) -> List[Dict[str, str]]:
        if self._voices is not None:
            return self._voices
        self._voices = []
        if not self.available():
            return self._voices
        try:
            proc = _run([self._powershell, "-NoProfile", "-NonInteractive",
                         "-ExecutionPolicy", "Bypass", "-File", str(SAPI_SCRIPT),
                         "-ListVoices"], timeout=60)
        except Exception:
            return self._voices
        for line in (proc.stdout or "").splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 3:
                self._voices.append({"name": parts[0], "culture": parts[1],
                                     "gender": parts[2],
                                     "enabled": (parts[3].lower() == "true") if len(parts) > 3
                                     else True})
        return self._voices

    def _pick_voice(self, language: str, strict: bool = False) -> Optional[str]:
        """挑一个已安装的 SAPI 语音。

        strict=True 时只接受"这门语言自己的语音"（粤语必须是 zh-HK），
        用于判断能否为该语言构建语音包；strict=False 时才允许按基础语种
        兜底（例如缺粤语语音时用普通话嗓），仅用于临时合成。
        """
        if not languages.is_supported(language):
            return None
        lang = languages.get(language)
        installed = {v["name"]: v for v in self.installed_voices()}
        for name in lang.sapi_voices:
            if name in installed:
                return name
        bcp = lang.bcp47
        for name, info in installed.items():
            if (info.get("culture") or "").lower() == bcp.lower():
                return name
        if strict:
            return None
        base = bcp.split("-")[0].lower()
        for name, info in installed.items():
            if (info.get("culture") or "").lower().startswith(base):
                return name
        return None

    def voices(self, language: str = "zh") -> List[str]:
        v = self._pick_voice(language, strict=True)
        return [v] if v else []

    def synthesize(self, text: str, language: str = "zh", out_path: Path | str = "",
                   rate: int = 0) -> SynthResult:
        if not self.available():
            return SynthResult(ok=False, text=text, language=language, engine=self.name,
                               error="SAPI 不可用（非 Windows 或缺 PowerShell）")
        text = (text or "").strip()
        if not text:
            return SynthResult(ok=False, text=text, language=language, engine=self.name,
                               error="空文本")
        out = Path(out_path) if out_path else Path(tempfile.gettempdir()) / \
            f"heyan_sapi_{abs(hash(text)) % 10**8}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(text)
            text_file = fh.name
        voice = self._pick_voice(language, strict=False) or ""
        culture = languages.get(language).bcp47 if languages.is_supported(language) else "zh-CN"
        try:
            proc = _run([self._powershell, "-NoProfile", "-NonInteractive",
                         "-ExecutionPolicy", "Bypass", "-File", str(SAPI_SCRIPT),
                         "-TextFile", text_file, "-OutFile", str(out),
                         "-Voice", voice, "-Culture", culture, "-Rate", str(int(rate)),
                         "-Volume", str(self.volume), "-SampleRate", str(self.sample_rate)],
                        timeout=180)
        except Exception as exc:
            return SynthResult(ok=False, text=text, language=language, engine=self.name,
                               error=f"{type(exc).__name__}: {exc}")
        finally:
            try:
                os.unlink(text_file)
            except OSError:
                pass

        if not out.exists() or out.stat().st_size < 1024:
            err = (proc.stderr or proc.stdout or "").strip().splitlines()
            return SynthResult(ok=False, text=text, language=language, engine=self.name,
                               error=err[-1] if err else "合成未产生音频",
                               extra={"voice": voice, "returncode": proc.returncode})
        info = wav_info(out)
        return SynthResult(ok=True, text=text, language=language, engine=self.name,
                           source="engine", wav_path=str(out),
                           duration_s=info["duration_s"], sample_rate=info["sample_rate"],
                           extra={"voice": voice})

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d["installed_voices"] = [v["name"] for v in self.installed_voices()]
        d["script"] = str(SAPI_SCRIPT)
        return d


# --------------------------------------------------------------------------
# espeak-ng
# --------------------------------------------------------------------------

class EspeakEngine(TtsEngine):
    """espeak-ng：Linux/树莓派/Termux 上最容易装到的离线引擎，支持 cmn 与 yue。"""

    name = "espeak"
    offline = True

    def __init__(self, sample_rate: int = 22050) -> None:
        self.sample_rate = sample_rate
        self.binary = _which("espeak-ng", "espeak")
        self._voice_cache: Dict[str, bool] = {}

    def available(self) -> bool:
        return bool(self.binary)

    def voices(self, language: str = "zh") -> List[str]:
        if not self.available():
            return []
        v = languages.get(language).espeak_voice if languages.is_supported(language) else ""
        return [v] if v and self._has_voice(v) else []

    def _has_voice(self, voice: str) -> bool:
        """espeak-ng 自带 cmn/yue，但 hak/nan 常常没有，必须实际查一次。"""
        if voice in self._voice_cache:
            return self._voice_cache[voice]
        ok = False
        try:
            proc = _run([self.binary, "--voices=" + voice], timeout=30)
            ok = proc.returncode == 0 and bool((proc.stdout or "").strip())
        except Exception:
            ok = False
        self._voice_cache[voice] = ok
        return ok

    def synthesize(self, text: str, language: str = "zh", out_path: Path | str = "",
                   rate: int = 0) -> SynthResult:
        if not self.available():
            return SynthResult(ok=False, text=text, language=language, engine=self.name,
                               error="未安装 espeak-ng")
        text = (text or "").strip()
        if not text:
            return SynthResult(ok=False, text=text, language=language, engine=self.name,
                               error="空文本")
        out = Path(out_path) if out_path else Path(tempfile.gettempdir()) / \
            f"heyan_espeak_{abs(hash(text)) % 10**8}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        voice = (languages.get(language).espeak_voice if languages.is_supported(language)
                 else "cmn") or "cmn"
        # espeak 的语速单位是词/分钟，rate 为相对偏移（-10..10 映射到 120..200 wpm）
        wpm = str(int(160 + max(-10, min(10, rate)) * 4))
        cmd = [self.binary, "-v", voice, "-s", wpm, "-w", str(out), text]
        try:
            proc = _run(cmd, timeout=120)
        except Exception as exc:
            return SynthResult(ok=False, text=text, language=language, engine=self.name,
                               error=f"{type(exc).__name__}: {exc}")
        if not out.exists() or out.stat().st_size < 1024:
            return SynthResult(ok=False, text=text, language=language, engine=self.name,
                               error=(proc.stderr or "合成失败").strip()[:200])
        info = wav_info(out)
        return SynthResult(ok=True, text=text, language=language, engine=self.name,
                           source="engine", wav_path=str(out),
                           duration_s=info["duration_s"], sample_rate=info["sample_rate"],
                           extra={"voice": voice})


# --------------------------------------------------------------------------
# Piper
# --------------------------------------------------------------------------

class PiperEngine(TtsEngine):
    """Piper 神经 TTS。音质最好，但需要单独下载语音模型，属于可选增强项。"""

    name = "piper"
    offline = True

    def __init__(self, voices_dir: Optional[Path] = None, sample_rate: int = 22050) -> None:
        self.binary = _which("piper")
        self.sample_rate = sample_rate
        self.voices_dir = Path(voices_dir) if voices_dir else \
            Path(os.environ.get("HEYAN_PIPER_VOICES", "")) if os.environ.get(
                "HEYAN_PIPER_VOICES") else None
        self._python_module = None
        try:
            import piper  # type: ignore

            self._python_module = piper
        except Exception:
            self._python_module = None

    def available(self) -> bool:
        return bool(self.binary) or self._python_module is not None

    def voices(self, language: str = "zh") -> List[str]:
        v = self._voice_file(language)
        return [v.name] if v else []

    def _voice_file(self, language: str) -> Optional[Path]:
        if not self.voices_dir or not self.voices_dir.exists():
            return None
        wanted = languages.get(language).piper_voices if languages.is_supported(language) else ()
        for name in wanted:
            for suffix in (".onnx", ""):
                cand = self.voices_dir / f"{name}{suffix}"
                if cand.exists():
                    return cand
        return None

    def synthesize(self, text: str, language: str = "zh", out_path: Path | str = "",
                   rate: int = 0) -> SynthResult:
        if not self.available():
            return SynthResult(ok=False, text=text, language=language, engine=self.name,
                               error="未安装 piper")
        voice = self._voice_file(language)
        if voice is None:
            return SynthResult(ok=False, text=text, language=language, engine=self.name,
                               error=f"缺少 {language} 的 piper 语音模型")
        text = (text or "").strip()
        out = Path(out_path) if out_path else Path(tempfile.gettempdir()) / \
            f"heyan_piper_{abs(hash(text)) % 10**8}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        length = 1.0 + max(-10, min(10, rate)) * 0.06
        cmd = [self.binary or sys.executable, "-m", str(voice), "-f", str(out),
               "--length_scale", f"{1.0 / length:.3f}"]
        if not self.binary:
            cmd = [sys.executable, "-m", "piper"] + cmd[1:]
        try:
            proc = subprocess.run(cmd, input=text, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=180)
        except Exception as exc:
            return SynthResult(ok=False, text=text, language=language, engine=self.name,
                               error=f"{type(exc).__name__}: {exc}")
        if not out.exists() or out.stat().st_size < 1024:
            return SynthResult(ok=False, text=text, language=language, engine=self.name,
                               error=(proc.stderr or "合成失败").strip()[:200])
        info = wav_info(out)
        return SynthResult(ok=True, text=text, language=language, engine=self.name,
                           source="engine", wav_path=str(out),
                           duration_s=info["duration_s"], sample_rate=info["sample_rate"],
                           extra={"voice": voice.name})


# --------------------------------------------------------------------------
# Silent
# --------------------------------------------------------------------------

class SilentEngine(TtsEngine):
    """什么都不合成。用于纯静音环境（例如批量构建时只要文字）与自动化测试。"""

    name = "silent"
    offline = True

    def available(self) -> bool:
        return True

    def synthesize(self, text: str, language: str = "zh", out_path: Path | str = "",
                   rate: int = 0) -> SynthResult:
        return SynthResult(ok=False, text=text, language=language, engine=self.name,
                           error="silent 引擎不产生音频")


# --------------------------------------------------------------------------
# 选择
# --------------------------------------------------------------------------

DEFAULT_ORDER = ("piper", "sapi", "espeak", "silent")


def all_engines(**kwargs) -> List[TtsEngine]:
    return [PiperEngine(), SapiEngine(), EspeakEngine(), SilentEngine()]


def describe_engines() -> List[Dict[str, Any]]:
    return [e.describe() for e in all_engines()]


def auto_engine(prefer: Optional[str] = None) -> TtsEngine:
    """按优先级挑第一个可用的引擎。"""
    engines = {e.name: e for e in all_engines()}
    order = ([prefer] if prefer else []) + [n for n in DEFAULT_ORDER if n != prefer]
    for name in order:
        eng = engines.get(name)
        if eng is not None and eng.available():
            return eng
    return SilentEngine()
