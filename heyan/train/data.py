"""数据集加载与小样本采样。

预处理直接复用 `heyan.core.preprocess`，保证训练与边缘推理逐比特一致 ——
训练时用 torchvision transforms、推理时用自己写的 numpy 版本，
是量化模型精度对不上的头号原因。
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..config import IMAGE_MEAN, IMAGE_STD, INPUT_SIZE, RESIZE_SIZE
from ..core.preprocess import center_crop, normalize, read_image, resize_short_side
from .augment import AugmentConfig, augment_array

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DatasetStats:
    total: int = 0
    per_class: Dict[str, int] = field(default_factory=dict)
    classes: List[str] = field(default_factory=list)
    min_per_class: int = 0
    max_per_class: int = 0
    few_shot: bool = False  # 每类 ≤ 20 张即为文献里说的小样本区间

    def to_dict(self) -> Dict[str, object]:
        return {
            "total": self.total,
            "num_classes": len(self.classes),
            "per_class": dict(self.per_class),
            "min_per_class": self.min_per_class,
            "max_per_class": self.max_per_class,
            "few_shot": self.few_shot,
        }


def discover_samples(root: Path | str, class_ids: Optional[Sequence[str]] = None
                     ) -> Tuple[List[Tuple[Path, int]], List[str]]:
    """ImageFolder 布局：<root>/<class_id>/<file>。类别顺序按名称排序，全程固定。"""
    root = Path(root)
    dirs = sorted([d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    ids = [d.name for d in dirs]
    if class_ids:
        wanted = list(class_ids)
        missing = [c for c in wanted if c not in ids]
        if missing:
            raise FileNotFoundError(f"数据集 {root} 缺少类别目录: {missing}")
        ids = wanted
        dirs = [root / c for c in ids]
    index = {cid: i for i, cid in enumerate(ids)}
    samples: List[Tuple[Path, int]] = []
    for d in dirs:
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                samples.append((f, index[d.name]))
    return samples, ids


@dataclass
class AliasedDiscovery:
    """按别名表扫描田间采集目录的结果。

    `dir_map` 与 `unknown_dirs` 一定要落进训练报告：小样本微调里"某个目录名
    没被认出来、整批照片被静默丢掉"是最难事后发现的事故。
    """

    samples: List[Tuple[Path, int]]
    class_ids: List[str]
    dir_map: Dict[str, str] = field(default_factory=dict)
    unknown_dirs: List[str] = field(default_factory=list)
    per_class: Dict[str, int] = field(default_factory=dict)
    empty_classes: List[str] = field(default_factory=list)
    root: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "root": self.root,
            "total": len(self.samples),
            "num_classes": len(self.class_ids),
            "dir_map": dict(self.dir_map),
            "unknown_dirs": list(self.unknown_dirs),
            "per_class": dict(self.per_class),
            "empty_classes": list(self.empty_classes),
        }


def discover_aliased_samples(root: Path | str, class_ids: Sequence[str],
                             aliases=None, strict: bool = True) -> AliasedDiscovery:
    """扫描 `<root>/<采集目录名>/<图片>`，用别名表把目录名翻成 class_id。

    与 `discover_samples` 的区别：后者要求目录名**就是** class_id（合成数据集
    的布局），而田间采集批次的目录名是团队自己起的（Maize_RustDisease 这种），
    必须过一遍 `heyan.classes.load_label_aliases`。

    标签索引一律按传入的 `class_ids` 定位，所以不同来源（真实照片 / 合成回放）
    扫出来的样本可以直接拼接，不会错位。
    """
    from ..classes import load_label_aliases

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"采集目录不存在: {root}")
    aliases = aliases if aliases is not None else load_label_aliases()
    ids = list(class_ids)
    index = {cid: i for i, cid in enumerate(ids)}

    dirs = sorted([d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    loose = [f.name for f in sorted(root.iterdir())
             if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    if loose:
        # 散落在根目录的图片没有类别归属，宁可不训也不能猜
        raise ValueError(f"{root} 根目录下有 {len(loose)} 张没有类别归属的图片"
                         f"（例如 {loose[:3]}），请放进对应的类别子目录")

    dir_map: Dict[str, str] = {}
    unknown: List[str] = []
    samples: List[Tuple[Path, int]] = []
    for d in dirs:
        cid = aliases.resolve(d.name)
        if cid is None:
            unknown.append(d.name)
            continue
        if cid not in index:
            # 目录能认出来，但不在本次训练的类别集合里（例如只微调玉米时扫到了水稻批次）
            unknown.append(d.name)
            continue
        dir_map[d.name] = cid
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                samples.append((f, index[cid]))

    if strict and unknown:
        raise ValueError(
            f"{root} 下有 {len(unknown)} 个目录无法映射到 class_id: {unknown}。"
            f"请在 heyan/assets/label_aliases.json 里补别名，"
            f"或传 strict=False 显式跳过（跳过会写进报告）。"
        )

    samples.sort(key=lambda t: (t[1], t[0].name))
    counter = Counter(lbl for _, lbl in samples)
    per_class = {cid: int(counter.get(i, 0)) for i, cid in enumerate(ids)}
    return AliasedDiscovery(
        samples=samples, class_ids=ids, dir_map=dir_map, unknown_dirs=unknown,
        per_class=per_class,
        empty_classes=[cid for cid, n in per_class.items() if n == 0],
        root=str(root),
    )


def read_label_csv(csv_path: Path | str, image_root: Path | str) -> Tuple[List[Tuple[Path, int]], List[str]]:
    """导入团队真实田间照片：labels.csv 两列 filename,class_id。

    调研只采到约 100 张照片，靠手工按文件夹分类很容易出错，用 csv 更好核对。
    """
    csv_path = Path(csv_path); image_root = Path(image_root)
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{csv_path} 是空的")
    cols = set(rows[0].keys())
    file_col = next((c for c in ("filename", "file", "path", "image") if c in cols), None)
    label_col = next((c for c in ("class_id", "label", "class") if c in cols), None)
    if not file_col or not label_col:
        raise ValueError(f"{csv_path} 需要包含 filename 与 class_id 两列，实际列: {sorted(cols)}")

    ids = sorted({str(r[label_col]).strip() for r in rows if str(r[label_col]).strip()})
    index = {c: i for i, c in enumerate(ids)}
    samples: List[Tuple[Path, int]] = []
    missing: List[str] = []
    for r in rows:
        name = str(r[file_col]).strip()
        cid = str(r[label_col]).strip()
        if not name or not cid:
            continue
        p = (image_root / name) if not Path(name).is_absolute() else Path(name)
        if not p.exists():
            missing.append(name)
            continue
        samples.append((p, index[cid]))
    if missing:
        raise FileNotFoundError(f"labels.csv 里 {len(missing)} 个文件找不到，例如: {missing[:5]}")
    return samples, ids


def stats(samples: Sequence[Tuple[Path, int]], class_ids: Sequence[str]) -> DatasetStats:
    counter = Counter(label for _, label in samples)
    per_class = {cid: int(counter.get(i, 0)) for i, cid in enumerate(class_ids)}
    counts = list(per_class.values())
    return DatasetStats(
        total=len(samples),
        per_class=per_class,
        classes=list(class_ids),
        min_per_class=min(counts) if counts else 0,
        max_per_class=max(counts) if counts else 0,
        few_shot=bool(counts) and max(counts) <= 20,
    )


class LeafDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[Path, int]], class_ids: Sequence[str],
                 input_size: int = INPUT_SIZE, resize_size: int = RESIZE_SIZE,
                 mean: Sequence[float] = IMAGE_MEAN, std: Sequence[float] = IMAGE_STD,
                 augment: Optional[AugmentConfig] = None, seed: Optional[int] = None,
                 cache: bool = False) -> None:
        self.samples = list(samples)
        self.class_ids = list(class_ids)
        self.input_size = input_size
        self.resize_size = resize_size
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.augment = augment
        self._rng = random.Random(seed)
        self._cache: Dict[Path, np.ndarray] = {} if cache else None  # type: ignore[assignment]

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def labels(self) -> List[int]:
        return [lbl for _, lbl in self.samples]

    def _load(self, path: Path) -> np.ndarray:
        if self._cache is not None and path in self._cache:
            return self._cache[path]
        arr = read_image(path)
        if self._cache is not None:
            self._cache[path] = arr
        return arr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        arr = self._load(path)
        if self.augment is not None and self.augment.enabled:
            arr = augment_array(arr, self.augment, random.Random(self._rng.randrange(2**31 - 1)))
        arr = center_crop(resize_short_side(arr, self.resize_size), self.input_size)
        chw = normalize(arr, self.mean, self.std)
        return torch.from_numpy(chw), int(label)

    def raw_array(self, idx: int) -> np.ndarray:
        """给量化校准用：输出与推理完全同构的 float32 NCHW。"""
        path, _ = self.samples[idx]
        arr = center_crop(resize_short_side(self._load(path), self.resize_size), self.input_size)
        return normalize(arr, self.mean, self.std)[None, ...]


class ClassBalancedSampler(Sampler[int]):
    """小样本数据各类数量常常差几倍，用重采样把每类的每轮迭代次数拉平。"""

    def __init__(self, labels: Sequence[int], seed: int = 0) -> None:
        self.labels = list(labels)
        self.seed = seed
        self.by_class: Dict[int, List[int]] = {}
        for i, lbl in enumerate(self.labels):
            self.by_class.setdefault(lbl, []).append(i)
        self.max_count = max(len(v) for v in self.by_class.values())
        self.num_classes = len(self.by_class)

    def __iter__(self) -> Iterable[int]:
        rng = random.Random(self.seed)
        out: List[int] = []
        for indices in self.by_class.values():
            out.extend(rng.choices(indices, k=self.max_count))
        rng.shuffle(out)
        return iter(out)

    def __len__(self) -> int:
        return self.max_count * self.num_classes


def split_samples(samples: Sequence[Tuple[Path, int]], val_ratio: float = 0.2,
                  seed: int = 42) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """按类别分层切分，保证每类都有验证样本（哪怕只有 3 张）。"""
    rng = random.Random(seed)
    by_class: Dict[int, List[Tuple[Path, int]]] = {}
    for item in samples:
        by_class.setdefault(item[1], []).append(item)
    train: List[Tuple[Path, int]] = []
    val: List[Tuple[Path, int]] = []
    for items in by_class.values():
        items = list(items)
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio))) if len(items) > 1 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def make_loaders(train_ds: LeafDataset, val_ds: LeafDataset, batch_size: int = 32,
                 balanced: bool = True, num_workers: int = 0, seed: int = 0):
    sampler = ClassBalancedSampler(train_ds.labels, seed=seed) if balanced else None
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=num_workers, drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False,
    )
    return train_loader, val_loader


def calibration_arrays(dataset: LeafDataset, n: int, seed: int = 0) -> List[np.ndarray]:
    """量化校准集。必须是真实分布的图，不能拿随机噪声糊弄。"""
    rng = random.Random(seed)
    idxs = list(range(len(dataset)))
    rng.shuffle(idxs)
    return [dataset.raw_array(i) for i in idxs[:min(n, len(idxs))]]
