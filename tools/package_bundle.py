#!/usr/bin/env python3
"""把模型包打成可离线分发的 zip，并附一份 sha256 校验单。

为什么需要这一步：
  模型包最终是拷到 SD 卡、用 U 盘传到农技站电脑上的，中间没有网络校验。
  拷贝中断或卡坏了，文件会静默少一截，到了田间才发现加载失败，代价太高。
  所以打包时顺手算一遍 sha256，接收方 `--verify` 一下就知道东西完不完整。

用法：
  python tools/package_bundle.py                  # 打包最新 bundle
  python tools/package_bundle.py --verify x.zip   # 校验收到的包
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heyan.core.bundle import ModelBundle  # noqa: E402
from heyan.data.schema import file_sha256  # noqa: E402
from heyan.server.state import resolve_bundle  # noqa: E402

CHUNK = 1 << 20


def _sha_file(path: Path) -> str:
    return file_sha256(path, CHUNK)


def package(bundle_dir: Path, out_dir: Path) -> int:
    bundle = ModelBundle.load(bundle_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{bundle_dir.name}.zip"

    zip_path = bundle.pack_zip(dest)
    digest = _sha_file(zip_path)
    checksum = out_dir / f"{zip_path.name}.sha256"
    checksum.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")

    size = bundle.check_size_budget()
    print(f"[package] bundle    -> {bundle_dir}")
    print(f"[package] zip       -> {zip_path}  ({zip_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"[package] sha256    -> {checksum}")
    print(f"[package] 模型体积  -> {size['model_size_mb']:.2f} MB / 上限 {size['limit_mb']:.0f} MB"
          f"  {'达标' if size['ok'] else '超标'}")
    if not size["ok"]:
        return 2
    return 0


def verify(zip_path: Path) -> int:
    if not zip_path.exists():
        print(f"[package] 找不到文件：{zip_path}")
        return 2
    checksum = zip_path.with_suffix(zip_path.suffix + ".sha256")
    if not checksum.exists():
        print(f"[package] 缺少校验单：{checksum}")
        return 2
    line = checksum.read_text(encoding="utf-8").strip().split()
    expected = line[0] if line else ""
    actual = _sha_file(zip_path)
    ok = actual == expected
    print(f"[package] {'校验通过' if ok else '校验失败：文件已损坏或被改动'}")
    print(f"[package] 期望 {expected}")
    print(f"[package] 实际 {actual}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", help="模型包目录（默认取 artifacts/bundles 下最新的一个）")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "dist"),
                    help="zip 与校验单的输出目录")
    ap.add_argument("--verify", metavar="ZIP", help="只校验一个已打包的 zip")
    args = ap.parse_args()

    if args.verify:
        return verify(Path(args.verify))

    bundle_dir = resolve_bundle(args.bundle)
    if bundle_dir is None:
        print("[package] 没找到模型包，先跑 python -m heyan.cli build")
        return 2
    return package(Path(bundle_dir), Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
