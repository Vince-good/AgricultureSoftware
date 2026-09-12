"""模型包（Model Bundle）。

一个 bundle 就是一个自包含目录，可以直接拷进 U 盘/SD 卡带到田里：

    bundles/heyan-mnv3s-int8-v1/
      manifest.json      # 结构、量化方式、实测指标、预算判定
      labels.json        # 类别体系快照（顺序即模型输出顺序）
      advisory.json      # 农艺建议快照
      model_int8.onnx    # 主推理模型（≤20MB）
      model_fp32.onnx    # 回退模型，供无 onnxruntime 的设备走纯 NumPy 解释器
      voicepack/         # 预渲染离线语音（zh/yue/...）

把 labels 和 advisory 一起快照进 bundle，是为了保证"模型 v3 配文案 v3"，
避免设备上更新了模型却还在念旧文案 —— 这类不一致在离线部署里最难排查。
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..advice import Advisory, load_advisory
from ..classes import Taxonomy, load_taxonomy
from ..config import (ADVISORY_PATH, LATENCY_MAX_S, MODEL_SIZE_MAX_MB, RUNTIME_MEMORY_MAX_MB,
                      TAXONOMY_PATH)

MANIFEST_NAME = "manifest.json"
LABELS_NAME = "labels.json"
ADVISORY_NAME = "advisory.json"
INT8_NAME = "model_int8.onnx"
FP32_NAME = "model_fp32.onnx"
VOICEPACK_DIR = "voicepack"


@dataclass
class BundleManifest:
    model_id: str
    version: str = "1.0.0"
    arch: str = "mobilenet_v3_small"
    num_classes: int = 0
    input_size: int = 224
    resize_size: int = 256
    mean: List[float] = None  # type: ignore[assignment]
    std: List[float] = None  # type: ignore[assignment]
    files: Dict[str, str] = None  # type: ignore[assignment]
    quantization: Dict[str, Any] = None  # type: ignore[assignment]
    budgets: Dict[str, float] = None  # type: ignore[assignment]
    metrics: Dict[str, Any] = None  # type: ignore[assignment]
    backend_preference: List[str] = None  # type: ignore[assignment]
    created_at: str = ""
    schema_version: str = "1.0"
    notes: str = ""
    region: str = ""
    extra: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        from ..config import IMAGE_MEAN, IMAGE_STD, INPUT_SIZE, RESIZE_SIZE

        self.mean = list(self.mean) if self.mean else list(IMAGE_MEAN)
        self.std = list(self.std) if self.std else list(IMAGE_STD)
        self.input_size = int(self.input_size or INPUT_SIZE)
        self.resize_size = int(self.resize_size or RESIZE_SIZE)
        self.files = dict(self.files or {})
        self.quantization = dict(self.quantization or {})
        self.budgets = dict(self.budgets or {
            "max_model_mb": MODEL_SIZE_MAX_MB,
            "max_latency_s": LATENCY_MAX_S,
            "max_runtime_memory_mb": RUNTIME_MEMORY_MAX_MB,
        })
        self.metrics = dict(self.metrics or {})
        self.backend_preference = list(self.backend_preference or ["onnxruntime", "numpy"])
        self.extra = dict(self.extra or {})
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "version": self.version,
            "created_at": self.created_at,
            "arch": self.arch,
            "region": self.region,
            "num_classes": int(self.num_classes),
            "preprocess": {
                "input_size": self.input_size,
                "resize_size": self.resize_size,
                "mean": self.mean,
                "std": self.std,
                "layout": "NCHW",
                "dtype": "float32",
            },
            "files": self.files,
            "quantization": self.quantization,
            "budgets": self.budgets,
            "metrics": self.metrics,
            "backend_preference": self.backend_preference,
            "notes": self.notes,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "BundleManifest":
        pre = raw.get("preprocess", {}) or {}
        known = {
            "model_id", "version", "arch", "num_classes", "input_size", "resize_size",
            "mean", "std", "files", "quantization", "budgets", "metrics",
            "backend_preference", "created_at", "schema_version", "notes", "region", "extra",
        }
        kwargs: Dict[str, Any] = {k: v for k, v in raw.items() if k in known}
        kwargs["input_size"] = pre.get("input_size", kwargs.get("input_size", 224))
        kwargs["resize_size"] = pre.get("resize_size", kwargs.get("resize_size", 256))
        kwargs["mean"] = pre.get("mean", kwargs.get("mean"))
        kwargs["std"] = pre.get("std", kwargs.get("std"))
        rest = {k: v for k, v in raw.items() if k not in known and k != "preprocess"}
        extra = dict(kwargs.get("extra") or {})
        extra.update(rest)
        kwargs["extra"] = extra
        return cls(**kwargs)


class ModelBundle:
    """bundle 目录的读写封装。"""

    def __init__(self, root: Path, manifest: BundleManifest,
                 taxonomy: Optional[Taxonomy] = None,
                 advisory: Optional[Advisory] = None) -> None:
        self.root = Path(root)
        self.manifest = manifest
        self._taxonomy = taxonomy
        self._advisory = advisory

    # ---------------- 载入 ----------------
    @classmethod
    def load(cls, root: Path | str) -> "ModelBundle":
        root = Path(root)
        if root.is_file() and root.suffix == ".zip":
            root = _unzip_bundle(root)
        mpath = root / MANIFEST_NAME
        if not mpath.exists():
            raise FileNotFoundError(f"不是有效的模型包（缺 {MANIFEST_NAME}）: {root}")
        with open(mpath, "r", encoding="utf-8") as fh:
            manifest = BundleManifest.from_dict(json.load(fh))

        taxonomy = None
        labels_path = root / LABELS_NAME
        if labels_path.exists():
            from ..classes import _parse  # 内部解析器，避免重复实现
            with open(labels_path, "r", encoding="utf-8") as fh:
                taxonomy = _parse(json.load(fh))

        advisory = None
        adv_path = root / ADVISORY_NAME
        if adv_path.exists():
            advisory = load_advisory(adv_path)
        return cls(root, manifest, taxonomy, advisory)

    @classmethod
    def create(cls, root: Path | str, manifest: BundleManifest,
               taxonomy: Optional[Taxonomy] = None,
               advisory: Optional[Advisory] = None) -> "ModelBundle":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        bundle = cls(root, manifest, taxonomy or load_taxonomy(), advisory or load_advisory())
        bundle.write_metadata()
        return bundle

    # ---------------- 元数据 ----------------
    def write_metadata(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / MANIFEST_NAME, "w", encoding="utf-8") as fh:
            json.dump(self.manifest.to_dict(), fh, ensure_ascii=False, indent=2)
        if self._taxonomy is not None:
            (self.root / LABELS_NAME).write_text(self._taxonomy.to_labels_json(), encoding="utf-8")
        src_advisory = Path(ADVISORY_PATH)
        dst_advisory = self.root / ADVISORY_NAME
        if src_advisory.exists() and not dst_advisory.exists():
            shutil.copyfile(src_advisory, dst_advisory)

    @property
    def taxonomy(self) -> Taxonomy:
        if self._taxonomy is None:
            self._taxonomy = load_taxonomy(TAXONOMY_PATH)
        return self._taxonomy

    @property
    def advisory(self) -> Advisory:
        if self._advisory is None:
            p = self.root / ADVISORY_NAME
            self._advisory = load_advisory(p if p.exists() else ADVISORY_PATH)
        return self._advisory

    # ---------------- 模型文件 ----------------
    def model_path(self, kind: str = "int8") -> Optional[Path]:
        name = self.manifest.files.get(kind)
        if not name:
            return None
        p = self.root / name
        return p if p.exists() else None

    def available_kinds(self) -> List[str]:
        return [k for k in self.manifest.files if (self.root / self.manifest.files[k]).exists()]

    def register_model(self, kind: str, src: Path | str, *, move: bool = False) -> Path:
        src = Path(src)
        dst_name = INT8_NAME if kind == "int8" else FP32_NAME if kind == "fp32" else f"model_{kind}.onnx"
        dst = self.root / dst_name
        self.root.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copyfile(src, dst)
        self.manifest.files[kind] = dst_name
        self.write_metadata()
        return dst

    @property
    def voicepack_dir(self) -> Path:
        return self.root / VOICEPACK_DIR

    def size_mb(self, kind: Optional[str] = None) -> float:
        if kind:
            p = self.model_path(kind)
            return round(p.stat().st_size / (1024 * 1024), 3) if p else 0.0
        total = sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())
        return round(total / (1024 * 1024), 3)

    def model_size_mb(self) -> float:
        """只算模型文件，用于 20MB 预算判定。"""
        return round(sum(
            (self.root / n).stat().st_size for n in self.manifest.files.values()
            if (self.root / n).exists()
        ) / (1024 * 1024), 3)

    def check_size_budget(self) -> Dict[str, Any]:
        size = self.model_size_mb()
        limit = float(self.manifest.budgets.get("max_model_mb", MODEL_SIZE_MAX_MB))
        return {"model_size_mb": size, "limit_mb": limit, "ok": size <= limit,
                "headroom_mb": round(limit - size, 3)}

    def pack_zip(self, dest: Optional[Path] = None) -> Path:
        dest = Path(dest) if dest else self.root.parent / f"{self.root.name}.zip"
        shutil.make_archive(str(dest.with_suffix("")), "zip", root_dir=str(self.root.parent),
                            base_dir=self.root.name)
        return dest

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ModelBundle {self.manifest.model_id} @ {self.root}>"


def _unzip_bundle(zip_path: Path) -> Path:
    import zipfile

    dest = zip_path.with_suffix("")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    inner = dest / dest.name
    return inner if (inner / MANIFEST_NAME).exists() else dest
