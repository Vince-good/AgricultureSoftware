"""本地 Web 界面服务（文档 3.3 节：拍照 → 识别 → 语音播报）。

只依赖 flask + numpy + pillow：`heyan serve` 不需要装 torch，
边缘设备上跑界面和跑命令行识别用的是同一个 `heyan.core.RecognitionEngine`，
这就是文档 3.4(3) 要求的"识别核心与交互界面解耦"在服务端的体现。
"""

from __future__ import annotations

from .state import AppState, EngineUnavailable

__all__ = ["AppState", "EngineUnavailable", "create_app", "run_server"]


def __getattr__(name: str):  # pragma: no cover - 惰性导出，避免 import 就拉起 flask
    if name in ("create_app", "run_server"):
        from .app import create_app, run_server

        return {"create_app": create_app, "run_server": run_server}[name]
    raise AttributeError(name)
