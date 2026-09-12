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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from heyan.classes import load_taxonomy  # noqa: E402
from heyan.core.preprocess import read_image  # noqa: E402

MIN_SIDE_WARN = 300  # 短边低于此值，中心裁剪 224 后细节损失大


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest(labels_csv: Path | str, image_root: Path | str, out_dir: Path | str,
           min_per_class: int = 5, copy: bool = True) -> Dict[str, Any]:
    labels_csv = Path(labels_csv)
    image_root = Path(image_root)
    out_dir = Path(out_dir)
    taxonomy = load_taxonomy()
    known = set(taxonomy.ids)

    with open(labels_csv, "r", encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh)]
    if not rows:
        raise ValueError(f"{labels_csv} 是空的")
    cols = set(rows[0].keys())
    file_col = next((c for c in ("filename", "file", "path", "image") if c in cols), None)
    label_col = next((c for c in ("class_id", "label", "class") if c in cols), None)
    if not file_col or not label_col:
        raise ValueError(f"{labels_csv} 需要 filename 与 class_id 两列，实际: {sorted(cols)}")

    accepted: List[Dict[str, Any]] = []
    problems: List[Dict[str, Any]] = []
    seen_hash: Dict[str, str] = {}
    resolutions: List[Tuple[int, int]] = []
    per_class: Counter = Counter()

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
        digest = _sha256(src)
        if digest in seen_hash:
            problems.append({"file": name, "issue": f"与 {seen_hash[digest]} 内容重复（疑似重复贴标）"})
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
    short = [cid for cid in known if 0 < per_class.get(cid, 0) < min_per_class]
    missing = [cid for cid in known if per_class.get(cid, 0) == 0]
    res_arr = np.asarray(resolutions, dtype=np.float64) if resolutions else np.zeros((0, 2))

    report = {
        "labels_csv": str(labels_csv),
        "image_root": str(image_root),
        "out_dir": str(out_dir) if copy else None,
        "rows": len(rows),
        "accepted": len(accepted),
        "copied": copied,
        "rejected": len(hard),
        "warnings": len(warns),
        "per_class": {cid: int(per_class.get(cid, 0)) for cid in sorted(known)},
        "classes_below_min": short,
        "classes_missing": missing,
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
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ingest_report.json").write_text(json.dumps(report, ensure_ascii=False,
                                                           indent=2), encoding="utf-8")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="导入并体检田间调研照片")
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out", default="artifacts/data/field")
    parser.add_argument("--min-per-class", type=int, default=5)
    parser.add_argument("--no-copy", action="store_true", help="只体检不拷贝")
    args = parser.parse_args(argv)

    report = ingest(args.labels_csv, args.image_root, args.out,
                    min_per_class=args.min_per_class, copy=not args.no_copy)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["verdict"].startswith("可用") else 2


if __name__ == "__main__":
    raise SystemExit(main())
