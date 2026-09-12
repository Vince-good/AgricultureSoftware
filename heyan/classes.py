"""类别体系加载与查询。

类别定义全部放在 assets/taxonomy.json，训练、量化、推理、界面共用同一份，
避免"模型标签顺序"和"界面显示顺序"对不上这种边缘设备上的经典事故。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

from .config import TAXONOMY_PATH


@dataclass(frozen=True)
class ClassInfo:
    id: str
    index: int
    crop: str
    stress: str
    name_zh: str
    name_en: str
    icon: str
    voice_zh: str
    voice_yue: str = ""
    is_fallback: bool = False

    @property
    def is_stress(self) -> bool:
        return self.stress in ("disease", "nutrient", "water")


@dataclass(frozen=True)
class Taxonomy:
    classes: tuple[ClassInfo, ...]
    crops: Dict[str, dict]
    stress_types: Dict[str, dict]
    region: str = ""
    schema_version: str = "1.0"

    def __len__(self) -> int:
        return len(self.classes)

    def __iter__(self) -> Iterator[ClassInfo]:
        return iter(self.classes)

    @property
    def ids(self) -> List[str]:
        return [c.id for c in self.classes]

    @property
    def names_zh(self) -> List[str]:
        return [c.name_zh for c in self.classes]

    def by_id(self, class_id: str) -> ClassInfo:
        for c in self.classes:
            if c.id == class_id:
                return c
        raise KeyError(f"未知类别: {class_id}")

    def by_index(self, index: int) -> ClassInfo:
        return self.classes[index]

    def index_of(self, class_id: str) -> int:
        return self.by_id(class_id).index

    @property
    def fallback(self) -> Optional[ClassInfo]:
        """兜底类（unusable）。识别失败时用它，而不是硬给一个病害结论。"""
        for c in self.classes:
            if c.is_fallback:
                return c
        return None

    def crop_name(self, lang: str = "zh") -> Dict[str, str]:
        key = "name_zh" if lang.startswith("zh") or lang in ("yue", "hak", "teochew") else "name_en"
        return {cid: info.get(key, "") for cid, info in self.crops.items()}

    def to_labels_json(self) -> str:
        """导出为模型 bundle 内嵌的 labels.json（index -> id 顺序即训练顺序）。"""
        payload = {
            "schema_version": self.schema_version,
            "region": self.region,
            "crops": self.crops,
            "stress_types": self.stress_types,
            "classes": [c.__dict__ for c in self.classes],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


def _parse(raw: dict) -> Taxonomy:
    classes: List[ClassInfo] = []
    for i, item in enumerate(raw["classes"]):
        idx = int(item.get("index", i))
        if idx != i:
            raise ValueError(f"taxonomy.json 中类别 {item['id']} 的 index={idx} 与位置 {i} 不一致")
        classes.append(
            ClassInfo(
                id=item["id"],
                index=idx,
                crop=item["crop"],
                stress=item["stress"],
                name_zh=item["name_zh"],
                name_en=item["name_en"],
                icon=item.get("icon", "leaf_healthy"),
                voice_zh=item.get("voice_zh", item["name_zh"]),
                voice_yue=item.get("voice_yue", ""),
                is_fallback=bool(item.get("is_fallback", False)),
            )
        )
    return Taxonomy(
        classes=tuple(classes),
        crops=raw.get("crops", {}),
        stress_types=raw.get("stress_types", {}),
        region=raw.get("region", ""),
        schema_version=str(raw.get("schema_version", "1.0")),
    )


@lru_cache(maxsize=8)
def load_taxonomy(path: Optional[Path] = None) -> Taxonomy:
    p = Path(path) if path else TAXONOMY_PATH
    with open(p, "r", encoding="utf-8") as fh:
        return _parse(json.load(fh))


def taxonomy_from_ids(class_ids: Sequence[str]) -> Taxonomy:
    """训练目录里只有文件夹名时，用它按文件夹排序重建一个最小类别体系。"""
    base = load_taxonomy()
    known = {c.id: c for c in base.classes}
    classes = []
    for i, cid in enumerate(class_ids):
        if cid in known:
            src = known[cid]
            classes.append(
                ClassInfo(
                    id=src.id, index=i, crop=src.crop, stress=src.stress,
                    name_zh=src.name_zh, name_en=src.name_en, icon=src.icon,
                    voice_zh=src.voice_zh, voice_yue=src.voice_yue,
                    is_fallback=src.is_fallback,
                )
            )
        else:
            classes.append(
                ClassInfo(
                    id=cid, index=i, crop="none", stress="healthy",
                    name_zh=cid, name_en=cid, icon="leaf_healthy", voice_zh=cid,
                )
            )
    return Taxonomy(classes=tuple(classes), crops=base.crops, stress_types=base.stress_types,
                    region=base.region, schema_version=base.schema_version)
