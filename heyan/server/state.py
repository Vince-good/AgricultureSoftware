"""服务进程内的共享状态：识别引擎、记录库、语音、界面设置。

三件事决定了这里的写法：

1. **惰性加载**。`create_app()` 不能因为模型包缺失就崩，界面得先起来，
   再告诉用户"还没有安装识别模型"（`sys_no_model`）。
2. **一把大锁串行化推理**。Flask 默认 threaded，多请求并发会把
   200MB 内存预算顶穿；低端设备上并发推理也只会互相拖慢。
3. **设置持久化**。语言/字号/语速/农户信息写进记录库的 meta 表，
   跟着数据库走，换浏览器或重装界面都不会丢。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import PATHS

# 界面可调项与默认值。字号/语速只影响前端渲染与 TTS 语速，不进模型。
DEFAULT_SETTINGS: Dict[str, Any] = {
    "language": "zh",
    "text_size": "large",       # standard | large | huge
    "voice_speed": "normal",    # slow | normal | fast
    "auto_voice": True,
    "save_records": True,
    "allow_runtime_synth": True,
}

DEFAULT_PROFILE: Dict[str, str] = {
    "farmer_id": "", "alias": "", "village": "", "town": "",
    "county": "", "region": "粤东西北", "plot_id": "",
}

_RATE_BY_SPEED = {"slow": -2, "normal": 0, "fast": 2}
_SETTINGS_KEY = "ui_settings"
_PROFILE_KEY = "farmer_profile"


class EngineUnavailable(RuntimeError):
    """模型包缺失或加载失败。路由层捕获后返回 503 + 人话，而不是 500 堆栈。"""


def resolve_bundle(explicit: Optional[Path | str] = None) -> Optional[Path]:
    """找模型包：显式指定优先，否则取 `artifacts/bundles` 下最新的那个。"""
    if explicit:
        p = Path(explicit).expanduser()
        return p if (p / "manifest.json").exists() else None
    root = PATHS.bundles
    if not root.exists():
        return None
    candidates = [c for c in root.iterdir() if (c / "manifest.json").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.stat().st_mtime)


class AppState:
    """一个服务进程一份。所有属性都是线程安全的惰性单例。"""

    def __init__(self, bundle: Optional[Path | str] = None, language: str = "zh",
                 backend: Optional[str] = None, threads: int = 1) -> None:
        self.bundle_path: Optional[Path] = resolve_bundle(bundle)
        self.backend = backend
        self.threads = int(threads)
        self.started_at = time.time()
        self._lock = threading.RLock()
        self._engine = None
        self._engine_error: str = ""
        self._engine_load_ms: float = 0.0
        self._store = None
        self._synth = None
        self._initial_language = language or "zh"
        self._inflight = 0

    # ---------------- 记录库 ----------------
    @property
    def store(self):
        with self._lock:
            if self._store is None:
                from ..data import RecordStore

                PATHS.ensure()
                self._store = RecordStore()
            return self._store

    # ---------------- 识别引擎 ----------------
    @property
    def engine_available(self) -> bool:
        return self._engine is not None or self.bundle_path is not None

    def engine(self, force_reload: bool = False):
        """取识别引擎。首次调用才真正加载模型（约 1~2 秒）。"""
        with self._lock:
            if self._engine is not None and not force_reload:
                return self._engine
            if self.bundle_path is None:
                self._engine_error = "no_bundle"
                raise EngineUnavailable("no_bundle")
            from ..core import RecognitionEngine

            t0 = time.perf_counter()
            try:
                engine = RecognitionEngine(self.bundle_path, backend=self.backend,
                                           language=self.settings.get("language", "zh"),
                                           threads=self.threads)
            except Exception as exc:  # noqa: BLE001 - 转成界面能念的一句话
                self._engine_error = f"{type(exc).__name__}: {exc}"
                raise EngineUnavailable(self._engine_error) from exc
            self._engine_load_ms = (time.perf_counter() - t0) * 1000.0
            self._engine = engine
            self._engine_error = ""
            return engine

    def warm(self) -> None:
        """后台预热：启动后立刻加载模型，避免用户第一张照片白等两秒。"""
        def _run() -> None:
            try:
                self.engine()
            except EngineUnavailable:
                pass

        threading.Thread(target=_run, name="heyan-warm", daemon=True).start()

    def engine_status(self) -> Dict[str, Any]:
        with self._lock:
            loaded = self._engine is not None
        out: Dict[str, Any] = {
            "available": self.bundle_path is not None,
            "loaded": loaded,
            "bundle_dir": str(self.bundle_path) if self.bundle_path else None,
            "load_ms": round(self._engine_load_ms, 1),
            "error": self._engine_error or None,
        }
        if loaded:
            info = self._engine.info()
            out.update({
                "model_id": info["model_id"], "version": info["version"],
                "arch": info["arch"], "num_classes": info["num_classes"],
                "backend": info["backend"], "model_size_mb": info["model_size_mb"],
                "bundle_size_mb": info["bundle_size_mb"], "budgets": info["budgets"],
                "metrics": info["metrics"], "min_confidence": info["min_confidence"],
                "device": info["device"],
            })
        return out

    # ---------------- 语音 ----------------
    @property
    def synth(self):
        with self._lock:
            allow = bool(self.settings.get("allow_runtime_synth", True))
            if self._synth is None or self._synth.allow_runtime_synth != allow:
                from ..tts.synth import Synthesizer

                self._synth = Synthesizer(
                    bundle_root=self.bundle_path,
                    rate=_RATE_BY_SPEED.get(str(self.settings.get("voice_speed", "normal")), 0),
                    allow_runtime_synth=allow,
                )
            return self._synth

    # ---------------- 设置与农户档案 ----------------
    def _read_meta_json(self, key: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
        raw = self.store.meta(key)
        merged = dict(defaults)
        if raw:
            try:
                merged.update({k: v for k, v in json.loads(raw).items() if k in defaults})
            except (ValueError, TypeError):
                pass
        return merged

    @property
    def settings(self) -> Dict[str, Any]:
        merged = self._read_meta_json(_SETTINGS_KEY, DEFAULT_SETTINGS)
        if not merged.get("language"):
            merged["language"] = self._initial_language
        return merged

    @property
    def profile(self) -> Dict[str, str]:
        return self._read_meta_json(_PROFILE_KEY, DEFAULT_PROFILE)

    def update_settings(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        current = self.settings
        for key, value in (patch or {}).items():
            if key not in DEFAULT_SETTINGS:
                continue
            default = DEFAULT_SETTINGS[key]
            if isinstance(default, bool):
                current[key] = bool(value)
            elif key == "language":
                from ..tts.languages import is_supported

                current[key] = str(value) if is_supported(str(value)) else current[key]
            else:
                current[key] = str(value)
        self.store.set_meta(_SETTINGS_KEY, json.dumps(current, ensure_ascii=False))
        if self._synth is not None:
            self._synth.rate = _RATE_BY_SPEED.get(str(current["voice_speed"]), 0)
            self._synth.allow_runtime_synth = bool(current["allow_runtime_synth"])
        return current

    def update_profile(self, patch: Dict[str, Any]) -> Dict[str, str]:
        current = self.profile
        for key, value in (patch or {}).items():
            if key in DEFAULT_PROFILE:
                current[key] = "" if value is None else str(value).strip()[:64]
        self.store.set_meta(_PROFILE_KEY, json.dumps(current, ensure_ascii=False))
        return current

    @property
    def language(self) -> str:
        return str(self.settings.get("language") or "zh")

    # ---------------- 统计 ----------------
    def record_count(self) -> int:
        try:
            return self.store.count()
        except Exception:  # noqa: BLE001 - 记录库不可写时界面仍要能用
            return 0

    def benchmark_summary(self) -> Dict[str, Any]:
        """从 bundle 里的 benchmark.json 取验收结论，供设置页如实展示。"""
        if not self.bundle_path:
            return {}
        path = self.bundle_path / "benchmark.json"
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        latency = raw.get("latency_ms") or {}
        total = latency.get("total_ms") or {}
        memory = raw.get("memory_mb") or {}
        size = raw.get("model_size") or {}
        budgets = raw.get("budgets") or {}
        return {
            "budget_ok": bool(raw.get("budget_ok")),
            "checks": budgets.get("checks") or [],
            "failed": budgets.get("failed") or [],
            "limits": budgets.get("limits") or {},
            "latency_p50_ms": total.get("p50"),
            "latency_p95_ms": total.get("p95"),
            "latency_mean_ms": total.get("mean"),
            "memory_delta_mb": memory.get("runtime_delta_mb"),
            "memory_peak_mb": memory.get("rss_peak_mb"),
            "model_size_mb": size.get("total_mb"),
            "per_kind_mb": size.get("per_kind_mb") or {},
            "device": raw.get("device") or {},
            "rounds": raw.get("rounds"),
            "threads": raw.get("threads"),
        }

    def classes(self) -> List[Dict[str, Any]]:
        """类别清单（含图标与作物），前端据此渲染图标与筛选项。"""
        from ..classes import load_taxonomy

        tax = load_taxonomy()
        return [{
            "id": c.id, "index": c.index, "crop": c.crop, "stress": c.stress,
            "name_zh": c.name_zh, "name_en": c.name_en, "icon": c.icon,
            "is_fallback": c.is_fallback,
        } for c in tax]

    def severities(self) -> List[Dict[str, Any]]:
        from ..advice import load_advisory

        adv = load_advisory()
        return [{"id": s.id, "name_zh": s.name_zh, "name_en": s.name_en,
                 "order": s.order, "color": s.color}
                for s in sorted(adv.severities.values(), key=lambda s: s.order)]

    def close(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.close()
                self._engine = None
            if self._store is not None:
                self._store.close()
                self._store = None
