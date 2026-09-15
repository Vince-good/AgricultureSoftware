#!/usr/bin/env python3
"""生成"合成演示数据集"（ImageFolder 布局），用来跑通并验证整条流水线。

为什么需要它
------------
方案文档的技术路径是 预训练微调 → 知识蒸馏 → 通道剪枝 → ONNX 导出 → 8-bit 量化
→ 预算校验 → 打包 bundle。这条链子有 7 个环节，任何一环出错（预处理不一致、
量化掉精度、体积超标、labels 顺序错位）都会在**部署到田里之后**才暴露。
在没有真实田间照片的机器上，也需要能把这 7 环跑一遍、把体积/延迟/内存三个
硬预算验一遍的手段 —— 这就是本脚本的用途。

它**不是**用来得到可信精度数字的。合成图像的类别信号是人工注入的
（病斑形状、失绿程度、色调偏移），模型学到的是这些注入特征，
所以 demo 数据上的 top-1 只说明"流水线没坏"，不代表田间表现。
真实精度评估必须用团队田间调研照片，走 `tools/ingest_field_samples.py`。

生成内容
--------
全部类别目录（与 `heyan/assets/taxonomy.json` 一致，当前 17 类含玉米三态），每张图包含：
  田间背景（土壤/杂草，带低频明暗起伏）
+ 作物形态（水稻长披针叶 / 花生小叶 / 蔬菜阔叶 / 玉米宽大弓形叶，含叶脉）
+ 胁迫特征（梭形病斑、圆形叶斑、锈病疱点、大斑病长梭斑、失绿黄化、暗紫缺磷、
  萎蔫失水、霜霉斑块）
  + 全局扰动（曝光/色温/旋转/JPEG 压缩），制造类内方差

用法
----
    python tools/make_demo_dataset.py                 # 默认 30 张/类 → artifacts/data/demo_dataset
    python tools/make_demo_dataset.py --per-class 60 --seed 7
    python tools/make_demo_dataset.py --out D:/data/demo --labels-csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from heyan.classes import load_taxonomy  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "artifacts" / "data" / "demo_dataset"


# --------------------------------------------------------------------------
# 几何工具
# --------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _grid(size: int) -> Tuple[np.ndarray, np.ndarray]:
    """归一化坐标 [-1, 1]，缓存起来免得每张图都重新算。"""
    axis = (np.arange(size, dtype=np.float32) - (size - 1) / 2.0) / ((size - 1) / 2.0)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    return x, y


def _rotate(x: np.ndarray, y: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    ca, sa = math.cos(angle), math.sin(angle)
    return x * ca + y * sa, -x * sa + y * ca


def _soft(distance: np.ndarray, softness: float) -> np.ndarray:
    """把"到边界的距离"变成 0~1 的软掩膜，避免锯齿。"""
    return np.clip(distance / max(softness, 1e-4), 0.0, 1.0)


def _low_freq_noise(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    """低频噪声：小网格随机值放大后平滑，用来做叶面/土壤的明暗起伏。"""
    cells = max(2, int(cells))
    coarse = rng.random((cells, cells), dtype=np.float32)
    up = np.array(Image.fromarray(coarse).resize((size, size), Image.BILINEAR),
                  dtype=np.float32)
    return up


def _speckle(rng: np.random.Generator, size: int, count: int) -> np.ndarray:
    """细碎高频噪点，模拟传感器噪声与叶面绒毛。"""
    out = np.zeros((size, size), dtype=np.float32)
    if count <= 0:
        return out
    idx = rng.integers(0, size * size, size=count)
    out.reshape(-1)[idx] = rng.normal(0.0, 1.0, size=count).astype(np.float32)
    return out


# --------------------------------------------------------------------------
# 叶片形态
# --------------------------------------------------------------------------

def leaf_mask_rice(rng: np.random.Generator, size: int) -> Tuple[np.ndarray, Dict[str, float]]:
    """水稻：细长披针形叶片，略带弧度，铺满画面中部。"""
    x, y = _grid(size)
    angle = float(rng.uniform(-0.55, 0.55))
    xr, yr = _rotate(x, y, angle)
    a = float(rng.uniform(0.92, 1.05))       # 半长
    b = float(rng.uniform(0.13, 0.20))       # 半宽
    bow = float(rng.uniform(-0.22, 0.22))    # 弯曲度
    centre = bow * (xr ** 2)
    taper = np.sqrt(np.clip(1.0 - (xr / a) ** 2, 0.0, None))
    half_w = b * taper
    mask = _soft(half_w - np.abs(yr - centre), 0.035)
    mask *= _soft(a - np.abs(xr), 0.05)
    return mask, {"bow": bow, "half_w": b}


def leaf_mask_peanut(rng: np.random.Generator, size: int) -> Tuple[np.ndarray, Dict[str, float]]:
    """花生：复叶，3~4 片倒卵形小叶围绕叶轴排布。"""
    x, y = _grid(size)
    n = int(rng.integers(3, 5))
    base = float(rng.uniform(-math.pi, math.pi))
    mask = np.zeros((size, size), dtype=np.float32)
    radius = float(rng.uniform(0.42, 0.52))
    for i in range(n):
        theta = base + 2 * math.pi * i / n + float(rng.normal(0, 0.10))
        cx = radius * math.cos(theta)
        cy = radius * math.sin(theta)
        xr, yr = _rotate(x - cx, y - cy, theta + math.pi / 2)
        rx = float(rng.uniform(0.24, 0.30))
        ry = float(rng.uniform(0.30, 0.38))
        d = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)
        mask = np.maximum(mask, _soft(1.0 - d, 0.05))
    return mask, {"leaflets": float(n)}


def leaf_mask_vegetable(rng: np.random.Generator, size: int) -> Tuple[np.ndarray, Dict[str, float]]:
    """蔬菜：阔叶，边缘带锯齿。"""
    x, y = _grid(size)
    ang = np.arctan2(y, x)
    rad = np.hypot(x, y)
    phase = float(rng.uniform(0, 2 * math.pi))
    teeth = int(rng.integers(9, 16))
    amp = float(rng.uniform(0.04, 0.09))
    edge = float(rng.uniform(0.70, 0.86)) * (1.0 + amp * np.sin(teeth * ang + phase))
    mask = _soft(edge - rad, 0.05)
    return mask, {"teeth": float(teeth)}


def leaf_mask_maize(rng: np.random.Generator, size: int) -> Tuple[np.ndarray, Dict[str, float]]:
    """玉米：宽大披针形叶片，主脉明显，常呈下垂的弓形。

    与水稻同为平行脉，但叶片宽 1.5~2 倍、弧度更大 —— 田里拍玉米往往是
    一片叶子横贯整个画面，这个形态差异本身就是分类线索。
    """
    x, y = _grid(size)
    angle = float(rng.uniform(-0.62, 0.62))
    xr, yr = _rotate(x, y, angle)
    a = float(rng.uniform(1.00, 1.15))       # 半长：玉米叶通常铺满画面
    b = float(rng.uniform(0.22, 0.32))       # 半宽：明显宽于水稻
    bow = float(rng.uniform(-0.34, 0.34))    # 弯曲度：下垂的弓形
    centre = bow * (xr ** 2)
    taper = np.sqrt(np.clip(1.0 - (xr / a) ** 2, 0.0, None))
    # 叶基（靠茎的一侧）略窄、叶中部最宽，用 taper**0.7 把最宽处往中部推
    half_w = b * (taper ** 0.7)
    mask = _soft(half_w - np.abs(yr - centre), 0.035)
    mask *= _soft(a - np.abs(xr), 0.05)
    return mask, {"bow": bow, "half_w": b, "angle": angle}


MASK_BUILDERS = {"rice": leaf_mask_rice, "peanut": leaf_mask_peanut,
                 "vegetable": leaf_mask_vegetable, "maize": leaf_mask_maize}
# 类别没给出可用形态时（例如 unusable）随机挑一种作物轮廓，列表跟着 MASK_BUILDERS 走，
# 新增作物不必再来这里改一遍。
_SHAPE_CROPS = tuple(MASK_BUILDERS)


def veins(crop: str, rng: np.random.Generator, size: int, mask: np.ndarray) -> np.ndarray:
    """叶脉：返回 -1~1 的明暗调制，乘在叶色上。"""
    x, y = _grid(size)
    if crop in ("rice", "maize"):
        angle = float(rng.uniform(-0.55, 0.55))
        _, yr = _rotate(x, y, angle)
        # 玉米叶更宽，平行脉间距更大、主脉更突出
        freq = float(rng.uniform(20.0, 30.0)) if crop == "maize" else float(rng.uniform(34.0, 46.0))
        lateral = 0.10 * np.sin(freq * yr)
        if crop == "maize":
            mid = _soft(0.036 - np.abs(yr), 0.02)
            lateral = lateral + 0.12 * mid
        return lateral * mask
    # 主脉 + 侧脉
    mid = _soft(0.030 - np.abs(x), 0.02)
    laterals = np.zeros((size, size), dtype=np.float32)
    count = int(rng.integers(5, 8))
    for i in range(1, count + 1):
        yy = -1.0 + 2.0 * i / (count + 1)
        slope = float(rng.uniform(0.45, 0.75))
        laterals += _soft(0.022 - np.abs(y - yy - slope * np.abs(x)), 0.018)
    return (0.14 * mid + 0.10 * np.clip(laterals, 0, 1)) * mask


# --------------------------------------------------------------------------
# 颜色
# --------------------------------------------------------------------------

# (R, G, B) 基色，0~1
BASE_GREEN = {
    "rice": np.array([0.30, 0.55, 0.22], dtype=np.float32),
    "peanut": np.array([0.28, 0.50, 0.20], dtype=np.float32),
    "vegetable": np.array([0.22, 0.48, 0.19], dtype=np.float32),
    # 玉米叶色偏深、略带蓝调，与水稻的鲜绿区分开
    "maize": np.array([0.21, 0.46, 0.24], dtype=np.float32),
}
SOIL = np.array([0.30, 0.22, 0.15], dtype=np.float32)


def _mix(a: np.ndarray, b: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    t = np.asarray(t, dtype=np.float32)
    if t.ndim == 2:
        t = t[..., None]
    return a * (1.0 - t) + b * t


def background(rng: np.random.Generator, size: int) -> np.ndarray:
    """田间背景：土壤 + 零星杂草 + 阴影，明暗随位置起伏。"""
    shade = _low_freq_noise(rng, size, int(rng.integers(3, 6)))
    tone = float(rng.uniform(-0.10, 0.12))
    base = SOIL + tone
    img = base[None, None, :] * (0.72 + 0.52 * shade[..., None])
    weeds = _low_freq_noise(rng, size, int(rng.integers(6, 12)))
    weed_mask = _soft(weeds - float(rng.uniform(0.62, 0.78)), 0.08)
    weed_col = np.array([0.24, 0.36, 0.16], dtype=np.float32)
    img = _mix(img, weed_col[None, None, :] * (0.7 + 0.5 * shade[..., None]),
               0.6 * weed_mask)
    img += _speckle(rng, size, size * size // 900)[..., None] * 0.02
    return np.clip(img, 0.0, 1.0)


def leaf_texture(rng: np.random.Generator, size: int, crop: str) -> np.ndarray:
    """健康叶面的颜色纹理：基色 + 低频明暗 + 细微色斑。"""
    base = BASE_GREEN[crop].copy()
    shade = _low_freq_noise(rng, size, int(rng.integers(4, 8)))
    tint = float(rng.uniform(-0.05, 0.05))
    col = base + tint
    # 叶片内部有自然的深浅过渡
    img = col[None, None, :] * (0.80 + 0.40 * shade[..., None])
    mottle = _low_freq_noise(rng, size, int(rng.integers(14, 26)))
    img = img * (0.92 + 0.16 * mottle[..., None])
    img += _speckle(rng, size, size * size // 1600)[..., None] * 0.012
    return np.clip(img, 0.0, 1.0)


# --------------------------------------------------------------------------
# 胁迫特征
# --------------------------------------------------------------------------

YELLOW = np.array([0.82, 0.72, 0.20], dtype=np.float32)
BROWN = np.array([0.34, 0.20, 0.09], dtype=np.float32)
GREY = np.array([0.58, 0.57, 0.53], dtype=np.float32)
PURPLE = np.array([0.34, 0.16, 0.34], dtype=np.float32)
# 玉米锈病孢子堆的肉桂/铁锈色，比 BROWN 更红更亮
RUST = np.array([0.58, 0.27, 0.10], dtype=np.float32)
# 玉米大斑病的灰褐色病斑，比 BROWN 更灰、更浅
TAN = np.array([0.52, 0.47, 0.34], dtype=np.float32)


def _mask_axis(mask: np.ndarray, size: int) -> float:
    """从叶片掩膜估计主轴方向（弧度）。

    玉米大斑病的病斑是顺着叶轴长的长梭形，如果随机取向，合成图看起来就是
    "撒了一把瓜子"，与真实照片的形态学差异太大，模型学到的会是错误的特征。
    这里对掩膜做二阶矩（等价于 PCA 主方向），不需要把叶片的旋转角从
    mask builder 一路传进来，任何作物都能用。
    """
    x, y = _grid(size)
    w = mask.astype(np.float64)
    total = float(w.sum())
    if total < 16.0:
        return 0.0
    cx = float((x * w).sum() / total)
    cy = float((y * w).sum() / total)
    dx, dy = x - cx, y - cy
    sxx = float((dx * dx * w).sum())
    syy = float((dy * dy * w).sum())
    sxy = float((dx * dy * w).sum())
    return 0.5 * math.atan2(2.0 * sxy, sxx - syy)


def _blotches(rng: np.random.Generator, size: int, count: int,
              rx: Tuple[float, float], ry: Tuple[float, float],
              ratio_lock: bool = False) -> List[Tuple[np.ndarray, float]]:
    """随机撒若干椭圆斑块，返回 (掩膜, 角度) 列表。"""
    x, y = _grid(size)
    out = []
    for _ in range(int(count)):
        cx = float(rng.uniform(-0.72, 0.72))
        cy = float(rng.uniform(-0.72, 0.72))
        a = float(rng.uniform(*rx))
        b = a * float(rng.uniform(0.28, 0.42)) if ratio_lock else float(rng.uniform(*ry))
        ang = float(rng.uniform(0, math.pi))
        xr, yr = _rotate(x - cx, y - cy, ang)
        d = np.sqrt((xr / max(a, 1e-3)) ** 2 + (yr / max(b, 1e-3)) ** 2)
        out.append((_soft(1.0 - d, 0.10), ang))
    return out


def apply_blast(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                size: int, severity: float) -> np.ndarray:
    """稻瘟病（叶瘟）：梭形病斑，深褐边缘 + 灰白中心。"""
    x, y = _grid(size)
    count = int(round(rng.integers(3, 8) * (0.5 + severity)))
    for _ in range(max(count, 1)):
        cx = float(rng.uniform(-0.6, 0.6))
        cy = float(rng.uniform(-0.6, 0.6))
        a = float(rng.uniform(0.14, 0.26)) * (0.7 + 0.6 * severity)
        b = a * float(rng.uniform(0.24, 0.36))
        ang = float(rng.uniform(0, math.pi))
        xr, yr = _rotate(x - cx, y - cy, ang)
        d = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
        outer = _soft(1.0 - d, 0.14) * mask
        inner = _soft(1.0 - d / 0.52, 0.18) * mask
        img = _mix(img, BROWN[None, None, :], 0.85 * outer)
        img = _mix(img, GREY[None, None, :], 0.80 * inner)
    return img


def apply_round_spots(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                      size: int, severity: float, halo: bool = True) -> np.ndarray:
    """圆形/椭圆形叶斑（花生叶斑病、胡麻叶斑病），带或不带黄晕。"""
    count = int(round(rng.integers(8, 22) * (0.5 + severity)))
    for m, _ in _blotches(rng, size, count, (0.035, 0.085), (0.030, 0.075)):
        m = m * mask
        if halo:
            halo_m = _soft(m - 0.35, 0.55) * mask
            img = _mix(img, YELLOW[None, None, :], 0.55 * halo_m)
        img = _mix(img, BROWN[None, None, :], 0.88 * m)
    return img


def apply_downy_mildew(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                       size: int, severity: float) -> np.ndarray:
    """霜霉病：受叶脉限制的角状黄斑 + 灰紫色霉层。"""
    count = int(round(rng.integers(4, 10) * (0.5 + severity)))
    for m, _ in _blotches(rng, size, count, (0.16, 0.34), (0.13, 0.28)):
        m = m * mask
        img = _mix(img, YELLOW[None, None, :], 0.72 * m)
    fuzz = _low_freq_noise(rng, size, int(rng.integers(18, 30)))
    fuzz_m = _soft(fuzz - float(rng.uniform(0.55, 0.70)), 0.10) * mask
    grey_purple = 0.5 * (GREY + PURPLE)
    img = _mix(img, grey_purple[None, None, :], (0.35 + 0.35 * severity) * fuzz_m)
    return img


def apply_rust(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
               size: int, severity: float) -> np.ndarray:
    """玉米锈病：针尖到芝麻大的锈红色孢子堆密布叶面，常沿叶脉成行，几乎不带黄晕。

    和花生/水稻的圆叶斑区别在尺度与密度：锈病的疱点非常小、数量上百、成行排列，
    而圆叶斑是十几个大而边界清楚的斑。所以这里先按叶片主轴算出垂直方向的条带，
    用它调制疱点密度（模拟"一串串铁锈"），再叠一层老熟疱点的深色，制造色阶层次。
    """
    axis = _mask_axis(mask, size)
    x, y = _grid(size)
    perp = -x * math.sin(axis) + y * math.cos(axis)
    band = 0.45 + 0.55 * (0.5 + 0.5 * np.sin(
        perp * float(rng.uniform(26.0, 44.0)) + float(rng.uniform(0, math.pi))))
    pust = np.zeros((size, size), dtype=np.float32)
    count = int(round(rng.integers(45, 95) * (0.45 + severity)))
    for m, _ in _blotches(rng, size, count, (0.008, 0.022), (0.008, 0.022)):
        pust = np.maximum(pust, m)
    pust = np.clip(pust * band * mask, 0.0, 1.0)
    img = _mix(img, RUST[None, None, :], (0.70 + 0.25 * severity) * pust[..., None])
    aged = _soft(pust - float(rng.uniform(0.55, 0.75)), 0.30)
    img = _mix(img, BROWN[None, None, :], 0.55 * aged[..., None])
    return img


def apply_northern_leaf_blight(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                               size: int, severity: float) -> np.ndarray:
    """玉米大斑病：长梭形灰褐色病斑，长轴顺着叶片伸展，边缘深褐、中部灰白。

    病斑长宽比 6~10:1，比稻瘟病的梭形斑更长更窄；长轴在叶片主轴附近摆动而不是
    完全平行，重病叶整体失绿发黄。主轴取自 mask 的 PCA，所以叶片旋转多少都跟得上。
    """
    axis = _mask_axis(mask, size)
    x, y = _grid(size)
    count = int(round(rng.integers(3, 8) * (0.5 + severity)))
    for _ in range(max(count, 1)):
        cx = float(rng.uniform(-0.62, 0.62))
        cy = float(rng.uniform(-0.62, 0.62))
        half_len = float(rng.uniform(0.10, 0.26)) * (0.7 + 0.6 * severity)
        half_wid = half_len * float(rng.uniform(0.10, 0.18))
        ang = axis + float(rng.uniform(-0.35, 0.35))
        xr, yr = _rotate(x - cx, y - cy, ang)
        d = np.sqrt((xr / max(half_len, 1e-3)) ** 2 + (yr / max(half_wid, 1e-3)) ** 2)
        lesion = _soft(1.0 - d, 0.12) * mask
        core = _soft(1.0 - d / 0.62, 0.16) * mask
        img = _mix(img, BROWN[None, None, :], 0.70 * lesion[..., None])
        img = _mix(img, TAN[None, None, :], 0.80 * core[..., None])
    if severity > 0.6:
        yellowing = 0.30 * (severity - 0.6) / 0.4
        img = _mix(img, YELLOW[None, None, :], yellowing * mask[..., None])
    return img


def apply_chlorosis(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                    size: int, severity: float, vein_green: bool = True) -> np.ndarray:
    """缺氮：整体失绿黄化，叶脉附近保留绿色（脉间失绿）。"""
    yellowing = np.clip(0.30 + 0.62 * severity + rng.normal(0, 0.06), 0.0, 1.0)
    img = _mix(img, YELLOW[None, None, :], yellowing * mask[..., None])
    if vein_green:
        keep = veins("vegetable", rng, size, mask)
        keep = np.clip(keep, 0, 1)
        base = BASE_GREEN["vegetable"] * 1.05
        img = _mix(img, base[None, None, :], 0.55 * keep[..., None] * mask[..., None])
    # 老叶先端先黄，加一点纵向梯度
    x, y = _grid(size)
    grad = np.clip((y + 1.0) / 2.0, 0, 1)
    img = _mix(img, YELLOW[None, None, :], 0.18 * severity * grad[..., None] * mask[..., None])
    return img


def apply_purple(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                 size: int, severity: float) -> np.ndarray:
    """缺磷：叶色暗绿转紫，叶背与叶缘最先显症。"""
    dark = np.array([0.13, 0.26, 0.12], dtype=np.float32)
    img = _mix(img, dark[None, None, :], (0.35 + 0.3 * severity) * mask[..., None])
    x, y = _grid(size)
    edge = np.clip(np.hypot(x, y), 0, 1)
    purple_m = _soft(edge - 0.30, 0.45) * mask
    img = _mix(img, PURPLE[None, None, :], (0.30 + 0.5 * severity) * purple_m[..., None])
    return img


def apply_water_stress(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                       size: int, severity: float) -> np.ndarray:
    """水分胁迫：整体褪绿发灰、失去光泽，叶缘卷曲干枯。"""
    dull = np.array([0.52, 0.52, 0.34], dtype=np.float32)
    img = _mix(img, dull[None, None, :], (0.35 + 0.35 * severity) * mask[..., None])
    x, y = _grid(size)
    edge = np.clip(np.hypot(x, y), 0, 1)
    rim = _soft(edge - float(rng.uniform(0.42, 0.58)), 0.30) * mask
    dry = np.array([0.62, 0.50, 0.28], dtype=np.float32)
    img = _mix(img, dry[None, None, :], (0.40 + 0.45 * severity) * rim[..., None])
    # 失去蜡质光泽 = 高频对比下降，靠轻微模糊表现
    return img


def apply_unusable(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                   size: int, severity: float) -> np.ndarray:
    """无法识别：没有叶片 / 严重失焦 / 欠曝 / 手指遮挡。"""
    mode = int(rng.integers(0, 4))
    x, y = _grid(size)
    if mode == 0:
        # 只有背景，压根没拍到叶子
        img = img * (1.0 - mask[..., None])
    elif mode == 1:
        # 严重失焦
        radius = int(rng.integers(9, 18))
        img = np.array(Image.fromarray((img * 255).astype(np.uint8))
                       .filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0
    elif mode == 2:
        # 欠曝，几乎全黑
        img = img * float(rng.uniform(0.10, 0.25))
    else:
        # 手指/杂物遮挡大半画面
        cx, cy = float(rng.uniform(-0.3, 0.3)), float(rng.uniform(-0.3, 0.3))
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        cover = _soft(float(rng.uniform(0.55, 0.85)) - d, 0.12)
        skin = np.array([0.62, 0.45, 0.34], dtype=np.float32)
        img = _mix(img, skin[None, None, :], 0.95 * cover[..., None])
    return img


# class_id -> (渲染函数, 是否额外模糊)
STRESS_APPLIERS = {
    "rice_blast": apply_blast,
    "rice_brown_spot": apply_round_spots,
    "peanut_leaf_spot": apply_round_spots,
    "vegetable_downy_mildew": apply_downy_mildew,
    "maize_rust": apply_rust,
    "maize_northern_leaf_blight": apply_northern_leaf_blight,
    "rice_n_deficiency": apply_chlorosis,
    "peanut_n_deficiency": apply_chlorosis,
    "vegetable_p_deficiency": apply_purple,
    "rice_water_stress": apply_water_stress,
    "peanut_water_stress": apply_water_stress,
    "vegetable_water_stress": apply_water_stress,
    "unusable": apply_unusable,
}


# --------------------------------------------------------------------------
# 单张图渲染
# --------------------------------------------------------------------------

def render_image(rng: np.random.Generator, class_id: str, crop: str, size: int) -> np.ndarray:
    """渲染一张 size×size 的 RGB uint8 图像。"""
    img = background(rng, size)
    if class_id == "unusable" and rng.random() < 0.35:
        # 有一部分"无法识别"样本是压根没有叶片的空镜
        crop_for_shape = str(rng.choice(_SHAPE_CROPS))
        mask, _ = MASK_BUILDERS[crop_for_shape](rng, size)
        return _finalize(apply_unusable(img, np.zeros_like(mask), rng, size, 1.0),
                         rng, size, blur=0)

    shape_crop = crop if crop in MASK_BUILDERS else str(rng.choice(_SHAPE_CROPS))
    mask, _ = MASK_BUILDERS[shape_crop](rng, size)

    if class_id != "unusable":
        texture = leaf_texture(rng, size, shape_crop)
        vein_mod = 1.0 + veins(shape_crop, rng, size, mask)
        texture = np.clip(texture * vein_mod[..., None], 0.0, 1.0)
        img = _mix(img, texture, mask[..., None])
        # 叶片在背景上留一点投影，增强立体感
        shadow = _soft(mask - 0.5, 0.5)
        img = img * (1.0 - 0.16 * np.roll(shadow, 6, axis=0)[..., None])

    applier = STRESS_APPLIERS.get(class_id)
    if applier is not None:
        severity = float(rng.uniform(0.15, 1.0))
        img = applier(img, mask, rng, size, severity)

    blur = 0
    if class_id == "unusable":
        blur = 0  # apply_unusable 内部已处理
    elif "water_stress" in class_id:
        blur = int(rng.integers(1, 3))
    return _finalize(img, rng, size, blur=blur)


def _finalize(img: np.ndarray, rng: np.random.Generator, size: int, blur: int = 0) -> np.ndarray:
    """全局扰动：曝光/色温/对比度/旋转/噪声/压缩。制造类内方差的关键一步。"""
    img = np.clip(img, 0.0, 1.0)
    # 曝光与对比度
    gain = float(rng.uniform(0.78, 1.24))
    bias = float(rng.uniform(-0.07, 0.07))
    img = np.clip((img - 0.5) * float(rng.uniform(0.86, 1.16)) + 0.5, 0.0, 1.0)
    img = np.clip(img * gain + bias, 0.0, 1.0)
    # 色温偏移（早晚光、阴天）
    warm = np.array([float(rng.uniform(0.94, 1.10)),
                     float(rng.uniform(0.96, 1.05)),
                     float(rng.uniform(0.90, 1.08))], dtype=np.float32)
    img = np.clip(img * warm[None, None, :], 0.0, 1.0)
    # 轻微旋转（手持拍摄不会正对）
    if rng.random() < 0.7:
        deg = float(rng.uniform(-14, 14))
        arr8 = (img * 255).astype(np.uint8)
        arr8 = np.array(Image.fromarray(arr8).rotate(deg, resample=Image.BILINEAR,
                                                      fillcolor=(60, 45, 32)))
        img = arr8.astype(np.float32) / 255.0
    if blur > 0:
        img = np.array(Image.fromarray((img * 255).astype(np.uint8))
                       .filter(ImageFilter.GaussianBlur(blur)), dtype=np.float32) / 255.0
    img += _speckle(rng, size, size * size // 700)[..., None] * float(rng.uniform(0.004, 0.016))
    return np.clip(img, 0.0, 1.0)


# --------------------------------------------------------------------------
# 数据集构建
# --------------------------------------------------------------------------

@dataclass
class BuildReport:
    out_dir: str
    per_class: Dict[str, int]
    total: int
    size: int
    seed: int
    elapsed_s: float
    labels_csv: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "out_dir": self.out_dir, "total": self.total, "size": self.size,
            "seed": self.seed, "elapsed_s": round(self.elapsed_s, 2),
            "per_class": self.per_class, "labels_csv": self.labels_csv,
            "note": "合成演示数据：仅用于验证流水线与体积/延迟/内存预算，不代表田间精度",
        }


def build_dataset(out_dir: Path | str, per_class: int = 30, seed: int = 42,
                  size: int = 320, write_labels_csv: bool = False,
                  progress: bool = True) -> BuildReport:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy()
    t0 = time.perf_counter()
    rows: List[Tuple[str, str]] = []
    counts: Dict[str, int] = {}

    total_expected = per_class * len(list(taxonomy))
    done = 0
    for cls in taxonomy:
        cdir = out_dir / cls.id
        cdir.mkdir(parents=True, exist_ok=True)
        # 每类用独立子种子，重跑同一 seed 得到完全相同的数据集
        cls_seed = (seed * 1000003 + abs(hash(cls.id)) % 100000) % (2 ** 31 - 1)
        rng = np.random.default_rng(cls_seed)
        counts[cls.id] = 0
        for i in range(per_class):
            arr = render_image(rng, cls.id, cls.crop, size)
            name = f"{cls.id}_{i:04d}.jpg"
            quality = int(rng.integers(78, 95))   # 压缩质量也参与扰动
            Image.fromarray((arr * 255).astype(np.uint8)).save(cdir / name, quality=quality)
            rows.append((f"{cls.id}/{name}", cls.id))
            counts[cls.id] += 1
            done += 1
            if progress and (done % 50 == 0 or done == total_expected):
                print(f"[demo-data] {done}/{total_expected} 张", flush=True)

    csv_path = None
    if write_labels_csv:
        csv_path = out_dir / "labels.csv"
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["filename", "class_id"])
            writer.writerows(rows)

    import json

    report = BuildReport(out_dir=str(out_dir), per_class=counts, total=len(rows),
                         size=size, seed=seed, elapsed_s=time.perf_counter() - t0,
                         labels_csv=str(csv_path) if csv_path else None)
    with open(out_dir / "dataset_info.json", "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, ensure_ascii=False, indent=2)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="生成合成演示数据集（ImageFolder 布局），用于跑通流水线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录")
    parser.add_argument("--per-class", type=int, default=30, help="每类图片数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（同种子结果完全可复现）")
    parser.add_argument("--size", type=int, default=320, help="图像边长（正方形）")
    parser.add_argument("--labels-csv", action="store_true",
                        help="额外写出 labels.csv（filename,class_id 两列）")
    parser.add_argument("--quiet", action="store_true", help="不打印进度")
    args = parser.parse_args(argv)

    report = build_dataset(args.out, per_class=args.per_class, seed=args.seed,
                           size=args.size, write_labels_csv=args.labels_csv,
                           progress=not args.quiet)
    print(f"\n[demo-data] 完成：{report.total} 张 / {len(report.per_class)} 类 → {report.out_dir}")
    print(f"[demo-data] 耗时 {report.elapsed_s:.1f}s，尺寸 {report.size}×{report.size}")
    if report.labels_csv:
        print(f"[demo-data] 标签清单：{report.labels_csv}")
    print("[demo-data] 提醒：合成数据只能验证流水线与资源预算，田间精度请用真实照片评估。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
