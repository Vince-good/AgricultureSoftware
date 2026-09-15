"""类别体系加载与查询。

类别定义全部放在 assets/taxonomy.json，训练、量化、推理、界面共用同一份，
避免"模型标签顺序"和"界面显示顺序"对不上这种边缘设备上的经典事故。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from .config import LABEL_ALIASES_PATH, TAXONOMY_PATH


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


# --------------------------------------------------------------------------
# 采集目录名 / 标注名 -> class_id 的别名映射
# --------------------------------------------------------------------------
#
# 田间采集批次的目录命名不受控：团队按 Maize_RustDisease 建目录，而体系里的
# id 是 maize_rust；下一批可能又写成"玉米锈病"。这种对应关系如果只活在某个人的
# 脑子里，迟早会贴错标签，而小样本微调对贴错标签极其敏感（几十张图里错五张
# 就足以把一个新类带偏）。所以把映射固化成 assets/label_aliases.json，
# 可审阅、可追溯、改批次不用改代码。

_ALIAS_SEPARATORS = str.maketrans({c: " " for c in "_-./\\ \t"})


def normalize_label_name(name: str) -> str:
    """归一化目录名/标注名，让不同书写风格命中同一条别名。

    规则：转小写、去首尾空白、把 ``_ - . / \\`` 与空格统一抹平、去掉间隔号。
    因此 ``Maize_RustDisease``、``maize rust disease``、``MaizeRustDisease``
    归一化后都是 ``maizerustdisease``。
    """
    s = str(name).strip().lower()
    s = s.translate(_ALIAS_SEPARATORS)
    s = s.replace("\u00b7", "").replace("\u2022", "").replace("\u30fb", "")
    return "".join(s.split())


@dataclass(frozen=True)
class LabelAliases:
    """别名表。``lookup`` 是归一化别名 -> class_id，``by_class`` 供写报告用。"""

    lookup: Dict[str, str]
    by_class: Dict[str, List[str]]
    source: Optional[Path] = None

    def resolve(self, name: str, taxonomy: Optional[Taxonomy] = None) -> Optional[str]:
        """目录名/标注名 -> class_id；解析不出来返回 None，由调用方决定报错还是跳过。"""
        raw = str(name).strip()
        if not raw:
            return None
        key = normalize_label_name(raw)
        if key in self.lookup:
            return self.lookup[key]
        # 别名表可能落后于 taxonomy（新加了类但忘了登记别名），这里兜一层：
        # 目录名本身就是 class_id 时直接放行。
        tax = taxonomy if taxonomy is not None else load_taxonomy()
        for cid in tax.ids:
            if normalize_label_name(cid) == key:
                return cid
        return None

    def resolve_strict(self, name: str, taxonomy: Optional[Taxonomy] = None) -> str:
        cid = self.resolve(name, taxonomy)
        if cid is None:
            raise KeyError(
                f"无法把 {name!r} 映射到任何 class_id。"
                f"请在 {LABEL_ALIASES_PATH.name} 的 aliases 里补一条。"
            )
        return cid

    def unknown(self, names: Iterable[str],
                taxonomy: Optional[Taxonomy] = None) -> List[str]:
        """批量体检：返回解析不出来的名字，供 ingest 报告如实列出。"""
        return [n for n in names if self.resolve(n, taxonomy) is None]

    def aliases_for(self, class_id: str) -> List[str]:
        return list(self.by_class.get(class_id, []))


@lru_cache(maxsize=8)
def load_label_aliases(path: Optional[Path] = None) -> LabelAliases:
    """读取别名表。文件不存在时退化成"只认 class_id 本身"，不报错。"""
    p = Path(path) if path else LABEL_ALIASES_PATH
    lookup: Dict[str, str] = {}
    by_class: Dict[str, List[str]] = {}
    if p.exists():
        with open(p, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        aliases = raw.get("aliases") or {}
        if not isinstance(aliases, dict):
            raise ValueError(f"{p} 的 aliases 字段必须是 对象(class_id -> [别名])")
        for cid, names in aliases.items():
            if not isinstance(names, (list, tuple)):
                raise ValueError(f"{p} 中 {cid} 的别名必须是列表，收到 {type(names).__name__}")
            by_class[cid] = [str(n) for n in names]
            for n in [cid, *names]:
                key = normalize_label_name(n)
                if not key:
                    continue
                prev = lookup.get(key)
                if prev is not None and prev != cid:
                    raise ValueError(f"{p} 中别名 {n!r} 同时指向 {prev} 与 {cid}，必须唯一")
                lookup[key] = cid
    # class_id 自身永远可解析，哪怕别名表里没登记
    for cid in load_taxonomy().ids:
        lookup.setdefault(normalize_label_name(cid), cid)
        by_class.setdefault(cid, [])
    return LabelAliases(lookup=lookup, by_class=by_class,
                        source=p if p.exists() else None)
