#!/usr/bin/env python3
"""生成禾眼的应用图标（PWA manifest 需要）。

为什么用脚本而不是塞一个二进制文件进仓库：
  图标要跟界面同一套配色（style.css 里的 --bar / --leaf-s），
  改配色时重跑一遍就行，不会出现"界面绿了图标还是旧绿"的漂移。

只依赖 Pillow，4 倍超采样后缩回目标尺寸，边缘是平滑的。
产出：icon-192.png / icon-512.png / maskable-512.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw  # noqa: E402

# 与 heyan/server/static/style.css 的 :root 变量保持一致
BAR = (18, 38, 26)             # --bar        深绿，图标底色
LEAF_LIGHT = (227, 240, 225)   # --leaf-s     浅绿，叶片主体
LEAF_ACCENT = (143, 199, 154)  # 中间调，叶脉与高光
SS = 4                         # 超采样倍数


def _cubic(p0, p1, p2, p3, steps: int = 64) -> List[Tuple[float, float]]:
    """三次贝塞尔采样。"""
    out = []
    for i in range(steps + 1):
        t = i / steps
        u = 1.0 - t
        x = (u ** 3 * p0[0] + 3 * u ** 2 * t * p1[0]
             + 3 * u * t ** 2 * p2[0] + t ** 3 * p3[0])
        y = (u ** 3 * p0[1] + 3 * u ** 2 * t * p1[1]
             + 3 * u * t ** 2 * p2[1] + t ** 3 * p3[1])
        out.append((x, y))
    return out


def _leaf_anchors(size: int, pad: float):
    """叶基与叶尖坐标（斜向右上）。

    `pad` 是内容到画布边的留白比例；maskable 图标要留更大的安全区，
    因为安卓会按圆形/圆角矩形裁切，裁到叶尖就难看了。
    """
    lo = size * pad
    hi = size * (1.0 - pad)
    span = hi - lo
    base = (lo + span * 0.16, hi - span * 0.10)
    tip = (hi - span * 0.10, lo + span * 0.14)
    return base, tip, span


def _leaf_outline(size: int, pad: float) -> List[Tuple[float, float]]:
    """一片叶子的闭合轮廓。"""
    base, tip, span = _leaf_anchors(size, pad)
    upper = _cubic(base,
                   (base[0] + span * 0.02, base[1] - span * 0.52),
                   (tip[0] - span * 0.50, tip[1] + span * 0.02),
                   tip)
    lower = _cubic(tip,
                   (tip[0] - span * 0.02, tip[1] + span * 0.54),
                   (base[0] + span * 0.52, base[1] - span * 0.02),
                   base)
    return upper + lower


def render(size: int, maskable: bool = False) -> Image.Image:
    """画一枚 `size`×`size` 的图标。"""
    big = size * SS
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if maskable:
        # 铺满整块，形状交给系统裁
        draw.rectangle([0, 0, big - 1, big - 1], fill=BAR + (255,))
        pad, radius = 0.27, 0.0
    else:
        radius = 0.22
        draw.rounded_rectangle([0, 0, big - 1, big - 1],
                               radius=int(big * radius), fill=BAR + (255,))
        pad = 0.19

    draw.polygon(_leaf_outline(big, pad), fill=LEAF_LIGHT + (255,))

    base, tip, span = _leaf_anchors(big, pad)
    width = max(2, int(big * 0.030))
    draw.line([base, tip], fill=BAR + (255,), width=width)

    # 三对侧脉：让"叶子"这个符号在 192px 甚至更小时也认得出来
    for frac, scale in ((0.34, 1.00), (0.53, 0.80), (0.72, 0.58)):
        px = base[0] + (tip[0] - base[0]) * frac
        py = base[1] + (tip[1] - base[1]) * frac
        reach = span * 0.21 * scale
        w = max(2, int(width * scale * 0.7))
        draw.line([(px, py), (px - reach * 0.66, py - reach * 0.88)],
                  fill=LEAF_ACCENT + (255,), width=w)
        draw.line([(px, py), (px + reach * 0.96, py + reach * 0.44)],
                  fill=LEAF_ACCENT + (255,), width=w)

    # 叶尖一点亮色，作为"识别到了"的视觉记忆点
    dot = max(3, int(big * 0.034))
    draw.ellipse([tip[0] - dot, tip[1] - dot, tip[0] + dot, tip[1] + dot],
                 fill=LEAF_ACCENT + (255,))

    out = img.resize((size, size), Image.LANCZOS)
    if not maskable:
        # 圆角外清干净，避免个别平台再叠一层白底
        mask = Image.new("L", (big, big), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [0, 0, big - 1, big - 1], radius=int(big * radius), fill=255)
        out.putalpha(mask.resize((size, size), Image.LANCZOS))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成禾眼 PWA 应用图标")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "heyan" / "server" / "static" / "icons",
                        help="输出目录（默认 heyan/server/static/icons）")
    parser.add_argument("--sizes", default="192,512",
                        help="要生成的普通图标边长，逗号分隔")
    parser.add_argument("--maskable-size", type=int, default=512,
                        help="maskable 图标边长，0 表示不生成")
    args = parser.parse_args(argv)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for raw in str(args.sizes).split(","):
        raw = raw.strip()
        if not raw:
            continue
        size = int(raw)
        path = out_dir / f"icon-{size}.png"
        render(size).save(path, "PNG", optimize=True)
        written.append(path)

    if args.maskable_size:
        path = out_dir / f"maskable-{args.maskable_size}.png"
        render(args.maskable_size, maskable=True).save(path, "PNG", optimize=True)
        written.append(path)

    for path in written:
        print(f"[icons] {path}  ({path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
