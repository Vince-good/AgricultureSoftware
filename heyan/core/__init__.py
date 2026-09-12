"""识别核心。与界面完全解耦：只吃 numpy 数组，只吐 dataclass，不依赖 Flask/Kivy。"""

from .result import Candidate, RecognitionResult
from .engine import RecognitionEngine
from .bundle import BundleManifest, ModelBundle

__all__ = [
    "Candidate",
    "RecognitionResult",
    "RecognitionEngine",
    "BundleManifest",
    "ModelBundle",
]
