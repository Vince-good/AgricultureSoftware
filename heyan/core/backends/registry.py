"""后端选择。按设备能力自动降级：onnxruntime → 纯 numpy → torch(仅开发机)。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .base import Backend, BackendProbe

# 每种后端按优先级尝试的模型文件种类（对应 bundle.manifest.files 的 key）
MODEL_KINDS: Dict[str, Tuple[str, ...]] = {
    "onnxruntime": ("int8", "fp32", "onnx"),
    "numpy": ("portable", "fp32", "int8"),
    "torch": ("torchscript", "fp32"),
}

_IMPL = {
    "onnxruntime": ("heyan.core.backends.onnxrt", "OnnxRuntimeBackend"),
    "numpy": ("heyan.core.backends.numpy_graph", "NumpyGraphBackend"),
    "torch": ("heyan.core.backends.torch_backend", "TorchScriptBackend"),
}

_MODULE_FOR_PROBE = {"onnxruntime": "onnxruntime", "torch": "torch", "numpy": "numpy"}


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def probe(name: str) -> BackendProbe:
    mod = _MODULE_FOR_PROBE.get(name, name)
    if not _module_available(mod):
        return BackendProbe(name=name, available=False, reason=f"缺少依赖包: {mod}")
    try:
        m = importlib.import_module(mod)
        version = str(getattr(m, "__version__", ""))
    except Exception as exc:  # pragma: no cover
        return BackendProbe(name=name, available=False, reason=f"导入失败: {exc}")
    return BackendProbe(name=name, available=True, version=version)


def available_backends() -> List[BackendProbe]:
    return [probe(n) for n in ("onnxruntime", "numpy", "torch")]


def describe_backends() -> List[Dict[str, Any]]:
    return [p.__dict__ for p in available_backends()]


def _load_impl(name: str):
    module_name, cls_name = _IMPL[name]
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def resolve_model(bundle, backend: str) -> Optional[Path]:
    """在 bundle 里挑出该后端能吃的模型文件。"""
    kinds = MODEL_KINDS.get(backend, ("fp32",))
    for kind in kinds:
        p = bundle.model_path(kind)
        if p is not None:
            return p
    # 退化：按扩展名扫目录
    for ext in (".hgraph.npz", ".onnx", ".pt"):
        for cand in sorted(bundle.root.rglob(f"*{ext}")):
            return cand
    return None


def create_backend(bundle, backend: Optional[str] = None, threads: int = 0
                   ) -> Tuple[Backend, Path]:
    """创建后端实例。backend=None 时按 manifest 偏好顺序自动选第一个可用的。"""
    preference: Sequence[str] = (
        [backend] if backend else list(bundle.manifest.backend_preference or ["onnxruntime", "numpy"])
    )
    if backend and backend not in preference:
        preference = [backend] + list(preference)

    errors: List[str] = []
    for name in preference:
        if name not in _IMPL:
            errors.append(f"{name}: 未知后端")
            continue
        p = probe(name)
        if not p.available:
            errors.append(f"{name}: {p.reason}")
            continue
        model_path = resolve_model(bundle, name)
        if model_path is None:
            errors.append(f"{name}: bundle 里没有匹配的模型文件（需要 {MODEL_KINDS[name]}）")
            continue
        impl = _load_impl(name)
        return impl(model_path, bundle.manifest.to_dict(), threads=threads), model_path

    raise RuntimeError("没有可用的推理后端：\n  " + "\n  ".join(errors))


def pick_for_device(ram_mb: int = 2048) -> str:
    """给部署脚本用：按设备内存给出推荐后端。"""
    probes = {p.name: p.available for p in available_backends()}
    if probes.get("onnxruntime") and ram_mb >= 512:
        return "onnxruntime"
    if probes.get("numpy"):
        return "numpy"
    return "onnxruntime" if probes.get("onnxruntime") else "numpy"
