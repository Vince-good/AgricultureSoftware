#!/usr/bin/env python3
"""导入团队田间调研照片（文档 3.2(3)：小样本微调的数据来源）。

调研只采到约 100 张照片，这种规模下"数据体检"比"数据量"更重要：
一张损坏的图、一个贴错标签的文件夹，就足以把小样本微调带偏。
所以本脚本在拷贝之前先做一遍完整体检：

  * 文件是否存在、能否解码、通道数是否为 3；
  * 分辨率分布（太小的图在 224 输入下信息量不足）；
  * 每类数量是否达到小样本下限（< min_per_class 会告警）；
  * 类别 id 是否在 taxonomy 内（拼写错误当场报出）；
  * 重复文件（同一张照片被贴了两个标签是最隐蔽的错误）。

支持两种输入布局：

  1. **labels.csv 模式**（`--labels-csv`）：两列 filename,class_id，标签已经
     人工核对过，适合混合批次或一图多标的情况；
  2. **目录名模式**（`--image-folder`）：`<root>/<采集目录名>/<图片>`，目录名
     经 `heyan/assets/label_aliases.json` 翻译成 class_id。团队实际采集就是
     这种布局（dateBase_Maize/Maize_RustDisease/...），不必先手工造 csv。
     认不出来的目录会**当场报错**而不是静默跳过 —— 小样本下少一个目录
     就是少一整类。

体检通过后，把图片按 ImageFolder 布局拷贝到 out_dir，并写出清洗后的
labels.csv 与 ingest_report.json，供 `heyan build --data-dir` 直接使用。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from heyan.classes import load_label_aliases, load_taxonomy  # noqa: E402
from heyan.core.preprocess import read_image  # noqa: E402

MIN_SIDE_WARN = 300  # 短边低于此值，中心裁剪 224 后细节损失大
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _csv_candidates(labels_csv: Path, image_root: Path, known: set
                    ) -> Tuple[List[Tuple[Path, str]], List[Dict[str, Any]], int]:
    """labels.csv -> (候选样本, 问题清单, 行数)。"""
    with open(labels_csv, "r", encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh)]
    if not rows:
        raise ValueError(f"{labels_csv} 是空的")
    cols = set(rows[0].keys())
    file_col = next((c for c in ("filename", "file", "path", "image") if c in cols), None)
    label_col = next((c for c in ("class_id", "label", "class") if c in cols), None)
    if not file_col or not label_col:
        raise ValueError(f"{labels_csv} 需要 filename 与 class_id 两列，实际: {sorted(cols)}")

    cands: List[Tuple[Path, str]] = []
    problems: List[Dict[str, Any]] = []
    for row in rows:
        name = str(row.get(file_col) or "").strip()
        cid = str(row.get(label_col) or "").strip()
        if not name or not cid:
            problems.append({"file": name, "issue": "空文件名或空标签"})
            continue
        if cid not in known:
            problems.append({"file": name, "issue": f"未知类别 {cid}（不在 taxonomy 内）"})
            continue
        src = image_root / name if not Path(name).is_absolute() else Path(name)
        if not src.exists():
            problems.append({"file": name, "issue": "文件不存在"})
            continue
        cands.append((src, cid))
    return cands, problems, len(rows)


def _folder_candidates(image_root: Path, known: set, aliases=None, strict: bool = True
                       ) -> Tuple[List[Tuple[Path, str]], List[Dict[str, Any]], Dict[str, Any]]:
    """`<root>/<采集目录名>/<图片>` -> 候选样本。目录名过别名表翻译成 class_id。"""
    aliases = aliases if aliases is not None else load_label_aliases()
    if not image_root.exists():
        raise FileNotFoundError(f"采集目录不存在: {image_root}")

    dirs = sorted([d for d in image_root.iterdir()
                   if d.is_dir() and not d.name.startswith(".")])
    if not dirs:
        raise ValueError(f"{image_root} 下没有任何类别子目录")

    cands: List[Tuple[Path, str]] = []
    problems: List[Dict[str, Any]] = []
    dir_map: Dict[str, str] = {}
    unknown: List[str] = []

    loose = [f.name for f in sorted(image_root.iterdir())
             if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    if loose:
        problems.append({"file": str(image_root), "issue":
                         f"根目录下有 {len(loose)} 张没有类别归属的图片（例如 {loose[:3]}）"})

    for d in dirs:
        cid = aliases.resolve(d.name)
        if cid is None:
            unknown.append(d.name)
            problems.append({"file": d.name, "issue":
                             f"目录名 {d.name!r} 无法映射到任何 class_id，"
                             f"请在 heyan/assets/label_aliases.json 补别名"})
            continue
        if cid not in known:
            unknown.append(d.name)
            problems.append({"file": d.name, "issue":
                             f"目录名 {d.name!r} 解析成 {cid}，但它不在 taxonomy 内"})
            continue
        dir_map[d.name] = cid
        files = [f for f in sorted(d.rglob("*"))
                 if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        if not files:
            problems.append({"file": d.name, "issue": f"目录 {d.name} 里没有图片"})
        cands.extend((f, cid) for f in files)

    if strict and unknown:
        raise ValueError(
            f"{image_root} 下有 {len(unknown)} 个目录认不出来: {unknown}。"
            f"补别名到 heyan/assets/label_aliases.json，或加 --allow-unknown 显式跳过。"
        )
    extra = {"dir_map": dir_map, "unknown_dirs": unknown,
             "aliases_source": str(aliases.source) if aliases.source else None}
    return cands, problems, extra


def _check_and_copy(cands: Sequence[Tuple[Path, str]], out_dir: Path, known: set,
                    problems: List[Dict[str, Any]], *, min_per_class: int,
                    copy: bool, rows: int, extra: Optional[Dict[str, Any]] = None
                    ) -> Dict[str, Any]:
    """体检 + 拷贝。csv 模式与目录名模式共用这一段，报告口径因此完全一致。"""
    accepted: List[Dict[str, Any]] = []
    seen_hash: Dict[str, str] = {}
    resolutions: List[Tuple[int, int]] = []
    per_class: Counter = Counter()

    for src, cid in cands:
        name = src.name
        digest = _sha256(src)
        if digest in seen_hash:
            problems.append({"file": name,
                             "issue": f"与 {seen_hash[digest]} 内容重复（疑似重复贴标）"})
            continue
        try:
            arr = read_image(src)
        except Exception as exc:  # noqa: BLE001 - 损坏文件要记录而不是中断
            problems.append({"file": name, "issue": f"无法解码: {type(exc).__name__}"})
            continue
        if arr.ndim != 3 or arr.shape[2] != 3:
            problems.append({"file": name, "issue": f"通道数异常 {arr.shape}"})
            continue
        h, w = arr.shape[:2]
        if min(h, w) < MIN_SIDE_WARN:
            problems.append({"file": name, "issue": f"分辨率偏低 {w}x{h}（仍保留，仅告警）",
                             "warn": True})
        seen_hash[digest] = name
        resolutions.append((w, h))
        per_class[cid] += 1
        accepted.append({"file": name, "class_id": cid, "src": str(src),
                         "width": int(w), "height": int(h), "sha256": digest})

    # 拷贝成 ImageFolder 布局
    copied = 0
    if copy and accepted:
        for item in accepted:
            cdir = out_dir / item["class_id"]
            cdir.mkdir(parents=True, exist_ok=True)
            dst = cdir / Path(item["file"]).name
            if dst.exists():
                stem = Path(item["file"]).stem
                dst = cdir / f"{stem}_{item['sha256'][:6]}{Path(item['file']).suffix}"
            shutil.copyfile(item["src"], dst)
            copied += 1
        with open(out_dir / "labels.csv", "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["filename", "class_id"])
            for item in accepted:
                writer.writerow([f"{item['class_id']}/{Path(item['file']).name}",
                                 item["class_id"]])

    hard = [p for p in problems if not p.get("warn")]
    warns = [p for p in problems if p.get("warn")]
    # 只统计本批出现过的类：混合训练时另一批数据会补齐其余类，
    # 把全 taxonomy 的空类都算成 missing 会淹掉真正需要关注的告警。
    touched = [cid for cid in known if per_class.get(cid, 0) > 0]
    short = [cid for cid in touched if per_class[cid] < min_per_class]
    res_arr = np.asarray(resolutions, dtype=np.float64) if resolutions else np.zeros((0, 2))

    report: Dict[str, Any] = {
        "out_dir": str(out_dir) if copy else None,
        "rows": rows,
        "accepted": len(accepted),
        "copied": copied,
        "rejected": len(hard),
        "warnings": len(warns),
        "per_class": {cid: int(per_class.get(cid, 0)) for cid in sorted(known)},
        "classes_present": touched,
        "classes_below_min": short,
        "min_per_class_required": min_per_class,
        "resolution": {
            "min_side": int(res_arr.min()) if len(res_arr) else None,
            "median_side": float(np.median(res_arr.min(axis=1))) if len(res_arr) else None,
            "max_side": int(res_arr.max()) if len(res_arr) else None,
        },
        "problems": problems,
        "few_shot": bool(per_class) and max(per_class.values()) <= 20,
        "verdict": ("可用于小样本微调" if not hard and accepted else
                    "存在必须修正的问题，见 problems"),
    }
    report.update(extra or {})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ingest_report.json").write_text(json.dumps(report, ensure_ascii=False,
                                                           indent=2), encoding="utf-8")
    return report


def ingest(labels_csv: Path | str, image_root: Path | str, out_dir: Path | str,
           min_per_class: int = 5, copy: bool = True) -> Dict[str, Any]:
    """labels.csv 模式（filename,class_id 两列）。"""
    labels_csv = Path(labels_csv)
    image_root = Path(image_root)
    out_dir = Path(out_dir)
    known = set(load_taxonomy().ids)
    cands, problems, rows = _csv_candidates(labels_csv, image_root, known)
    report = _check_and_copy(cands, out_dir, known, problems, min_per_class=min_per_class,
                             copy=copy, rows=rows)
    report["mode"] = "labels_csv"
    report["labels_csv"] = str(labels_csv)
    report["image_root"] = str(image_root)
    return report


def ingest_folders(image_root: Path | str, out_dir: Path | str, min_per_class: int = 5,
                   copy: bool = True, aliases=None, strict: bool = True) -> Dict[str, Any]:
    """目录名模式：`<root>/<采集目录名>/<图片>`，目录名经别名表翻成 class_id。

    团队实际采集就是这个布局（dateBase_Maize/Maize_RustDisease/...），
    不必先手工造一份 labels.csv。
    """
    image_root = Path(image_root)
    out_dir = Path(out_dir)
    known = set(load_taxonomy().ids)
    cands, problems, extra = _folder_candidates(image_root, known, aliases, strict)
    report = _check_and_copy(cands, out_dir, known, problems, min_per_class=min_per_class,
                             copy=copy, rows=len(cands), extra=extra)
    report["mode"] = "image_folder"
    report["image_root"] = str(image_root)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="导入并体检田间调研照片（labels.csv 或 目录名两种布局）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--labels-csv", help="filename,class_id 两列的清单")
    src.add_argument("--image-folder",
                     help="采集根目录，下含 <目录名>/<图片>，目录名走别名表映射")
    parser.add_argument("--image-root", help="labels.csv 模式下图片的根目录")
    parser.add_argument("--out", default="artifacts/data/field")
    parser.add_argument("--min-per-class", type=int, default=5)
    parser.add_argument("--no-copy", action="store_true", help="只体检不拷贝")
    parser.add_argument("--allow-unknown", action="store_true",
                        help="目录名模式下，认不出来的目录只告警不报错")
    parser.add_argument("--quiet", action="store_true", help="只打印结论行")
    args = parser.parse_args(argv)

    if args.labels_csv:
        root = Path(args.image_root) if args.image_root else Path(args.labels_csv).parent
        report = ingest(args.labels_csv, root, args.out,
                        min_per_class=args.min_per_class, copy=not args.no_copy)
    else:
        if args.image_root:
            parser.error("--image-folder 模式不需要 --image-root")
        report = ingest_folders(args.image_folder, args.out,
                                min_per_class=args.min_per_class, copy=not args.no_copy,
                                strict=not args.allow_unknown)

    if args.quiet:
        print(f"[ingest] {report['mode']} 接受 {report['accepted']} 张 / "
              f"{len(report['classes_present'])} 类 -> {report['out_dir']}")
        for k, v in (report.get("dir_map") or {}).items():
            print(f"[ingest]   {k} -> {v} ({report['per_class'].get(v, 0)} 张)")
        if report.get("unknown_dirs"):
            print(f"[ingest]   认不出来的目录: {report['unknown_dirs']}")
        print(f"[ingest] 结论: {report['verdict']}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["verdict"].startswith("可用") else 2


if __name__ == "__main__":
    raise SystemExit(main())
