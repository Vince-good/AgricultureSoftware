"""TorchScript 后端 —— 只在开发机/训练机上用，用来对拍精度和排查量化误差。"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Backend, InputSpec


class TorchScriptBackend(Backend):
    name = "torch"
    requires = ("torch",)

    def _load(self) -> None:
        import torch

        self._torch = torch
        self.module = torch.jit.load(str(self.model_path), map_location="cpu")
        self.module.eval()
        size = int((self.manifest.get("preprocess") or {}).get("input_size", 224))
        self._input_spec = InputSpec(name="input", shape=(1, 3, size, size), dtype="float32")
        with torch.no_grad():
            self.module(torch.zeros(self._input_spec.shape))

    def run(self, x: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            out = self.module(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)))
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.detach().cpu().numpy().astype(np.float32)

    def info(self) -> Dict[str, Any]:
        base = super().info()
        base["torch_version"] = self._torch.__version__
        return base
