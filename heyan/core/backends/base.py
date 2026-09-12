"""推理后端抽象。识别核心只认这个接口，换硬件就是换实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class InputSpec:
    """模型输入约定。QDQ 量化模型吃 float，QOperator 量化模型吃 uint8。"""

    name: str
    shape: Tuple[int, ...]
    dtype: str = "float32"
    scale: float = 1.0
    zero_point: int = 0

    @property
    def wants_uint8(self) -> bool:
        return self.dtype in ("uint8", "quint8", "int8")

    @property
    def size(self) -> int:
        return int(self.shape[-1]) if self.shape else 224


class Backend(ABC):
    name = "base"
    requires: Tuple[str, ...] = ()

    def __init__(self, model_path: Path | str, manifest: Optional[Dict[str, Any]] = None,
                 threads: int = 0) -> None:
        self.model_path = Path(model_path)
        self.manifest = dict(manifest or {})
        self.threads = int(threads)
        self._input_spec: Optional[InputSpec] = None
        self._load()

    @abstractmethod
    def _load(self) -> None: ...

    @abstractmethod
    def run(self, x: np.ndarray) -> np.ndarray:
        """输入 (1,3,H,W)，输出 (1,C) 未归一化 logits。"""

    @property
    def input_spec(self) -> InputSpec:
        if self._input_spec is None:
            raise RuntimeError("后端尚未初始化输入规格")
        return self._input_spec

    def close(self) -> None:  # pragma: no cover - 默认无资源
        pass

    def info(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "model_path": str(self.model_path),
            "model_size_mb": round(self.model_path.stat().st_size / (1024 * 1024), 3)
            if self.model_path.exists() else None,
            "input": self._input_spec.__dict__ if self._input_spec else None,
        }

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass
class BackendProbe:
    name: str
    available: bool
    reason: str = ""
    version: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
