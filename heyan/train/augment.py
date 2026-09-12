"""田间场景数据增强。

只用 numpy + Pillow 实现，不依赖 torch，因此也能被演示数据生成器和
离线批量复核脚本复用。

增强项针对调研记录的真实拍摄条件设计：不均匀光照、复杂背景（杂草/土壤/邻作物）、
随意拍摄角度、早期胁迫表观细微。公开数据集那种"干净单叶黑背景"图像直接用，
下田就会掉点 —— 这是小样本农业模型最常见的翻车原因。
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

Array = np.ndarray  # uint8 HWC RGB


@dataclass
class AugmentConfig:
    rotate_deg: float = 25.0
    scale_range: Tuple[float, float] = (0.75, 1.25)
    shift_range: float = 0.12
    flip_h: bool = True
    flip_v: bool = True
    brightness: Tuple[float, float] = (0.65, 1.45)
    contrast: Tuple[float, float] = (0.7, 1.4)
    saturation: Tuple[float, float] = (0.6, 1.5)
    hue_shift: float = 12.0
    uneven_light_p: float = 0.55
    shadow_patch_p: float = 0.35
    noise_sigma: float = 7.0
    blur_p: float = 0.25
    motion_blur_p: float = 0.18
    jpeg_p: float = 0.45
    jpeg_quality: Tuple[int, int] = (45, 90)
    occlusion_p: float = 0.25
    seed: Optional[int] = None
    enabled: bool = True


def _rng(cfg: AugmentConfig) -> random.Random:
    return random.Random(cfg.seed) if cfg.seed is not None else random.Random()


def random_rotate_scale_shift(img: Image.Image, cfg: AugmentConfig, rng: random.Random
                              ) -> Image.Image:
    angle = rng.uniform(-cfg.rotate_deg, cfg.rotate_deg)
    scale = rng.uniform(*cfg.scale_range)
    w, h = img.size
    tx = rng.uniform(-cfg.shift_range, cfg.shift_range) * w
    ty = rng.uniform(-cfg.shift_range, cfg.shift_range) * h
    # 平移用仿射矩阵，避免出现黑角
    matrix = (scale, 0, tx + (w - scale * w) / 2, 0, scale, ty + (h - scale * h) / 2)
    return img.transform(img.size, Image.AFFINE, matrix, resample=Image.BILINEAR,
                         fillcolor=(0, 0, 0)) if angle == 0 else _rotate_affine(img, angle, matrix)


def _rotate_affine(img: Image.Image, angle: float, matrix: Tuple[float, ...]) -> Image.Image:
    import math

    rad = math.radians(angle)
    cos, sin = math.cos(rad), math.sin(rad)
    base = np.array(matrix, dtype=np.float64).reshape(2, 3)
    rot = np.array([[cos, -sin, 0.0], [sin, cos, 0.0]])
    w, h = img.size
    center = np.array([w / 2.0, h / 2.0])
    # 先绕中心旋转，再套用缩放平移
    m = rot @ np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]])[:2]
    m = np.hstack([m, (center - m @ np.array([0.0, 0.0])).reshape(2, 1)])
    combined = base[:, :2] @ m[:, :2]
    offset = base[:, :2] @ m[:, 2] + base[:, 2]
    final = np.hstack([combined, offset.reshape(2, 1)]).flatten()
    return img.transform(img.size, Image.AFFINE, tuple(float(v) for v in final),
                         resample=Image.BILINEAR)


def random_flip(img: Image.Image, cfg: AugmentConfig, rng: random.Random) -> Image.Image:
    if cfg.flip_h and rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if cfg.flip_v and rng.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def color_jitter(arr: Array, cfg: AugmentConfig, rng: random.Random) -> Array:
    im = Image.fromarray(arr)
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(*cfg.brightness))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(*cfg.contrast))
    im = ImageEnhance.Color(im).enhance(rng.uniform(*cfg.saturation))
    out = np.asarray(im, dtype=np.float32)
    if cfg.hue_shift > 0:
        shift = rng.uniform(-cfg.hue_shift, cfg.hue_shift)
        out[..., 0] = np.clip(out[..., 0] + shift * 1.4, 0, 255)
        out[..., 2] = np.clip(out[..., 2] - shift * 0.8, 0, 255)
    return np.clip(out, 0, 255).astype(np.uint8)


def uneven_lighting(arr: Array, rng: random.Random, strength: float = 0.55) -> Array:
    """模拟侧光/云影造成的整幅亮度渐变，田间照片几乎每张都有。"""
    h, w = arr.shape[:2]
    yy = np.linspace(-1.0, 1.0, h)[:, None]
    xx = np.linspace(-1.0, 1.0, w)[None, :]
    angle = rng.uniform(0, 2 * np.pi)
    grad = np.cos(angle) * xx + np.sin(angle) * yy
    grad = (grad - grad.min()) / (grad.ptp() + 1e-6)
    gain = (1.0 - strength) + grad * (2.0 * strength)
    out = arr.astype(np.float32) * gain[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def shadow_patch(arr: Array, rng: random.Random) -> Array:
    """随机暗斑，模拟人手、植株自身或邻作物的阴影。"""
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(rng.randint(1, 2)):
        ch, cw = int(h * rng.uniform(0.15, 0.4)), int(w * rng.uniform(0.15, 0.4))
        y0 = rng.randint(0, max(0, h - ch)); x0 = rng.randint(0, max(0, w - cw))
        mask[y0:y0 + ch, x0:x0 + cw] = 1.0
    if mask.max() == 0:
        return arr
    mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(
        radius=max(h, w) // 24 + 1)), dtype=np.float32) / 255.0
    depth = rng.uniform(0.25, 0.55)
    out = arr.astype(np.float32) * (1.0 - mask[..., None] * depth)
    return np.clip(out, 0, 255).astype(np.uint8)


def occlude(arr: Array, rng: random.Random) -> Array:
    """随机遮挡，逼模型不要只盯着一小块区域做判断。"""
    h, w = arr.shape[:2]
    out = arr.copy()
    for _ in range(rng.randint(1, 3)):
        ch = int(h * rng.uniform(0.05, 0.18)); cw = int(w * rng.uniform(0.05, 0.18))
        y0 = rng.randint(0, max(0, h - ch)); x0 = rng.randint(0, max(0, w - cw))
        out[y0:y0 + ch, x0:x0 + cw] = rng.randint(0, 255)
    return out


def gaussian_noise(arr: Array, rng: random.Random, sigma: float) -> Array:
    if sigma <= 0:
        return arr
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, sigma, arr.shape)
    return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def motion_blur(arr: Array, rng: random.Random) -> Array:
    size = rng.choice([3, 5])
    kernel = np.zeros((size, size), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[:, size // 2] = 1.0
    else:
        kernel[size // 2, :] = 1.0
        if rng.random() < 0.3:
            kernel = np.diag(np.ones(size, dtype=np.float32))
    kernel /= kernel.sum()
    from PIL import ImageFilter as F

    im = Image.fromarray(arr).filter(F.Kernel((size, size), tuple(kernel.flatten()), scale=1.0,
                                             offset=size // 2))
    return np.asarray(im, dtype=np.uint8)


def jpeg_artifacts(arr: Array, rng: random.Random, quality: Tuple[int, int]) -> Array:
    q = rng.randint(*quality)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def augment_array(arr: Array, cfg: Optional[AugmentConfig] = None,
                  rng: Optional[random.Random] = None) -> Array:
    """对 uint8 HWC RGB 数组做一整套田间风格增强。"""
    cfg = cfg or AugmentConfig()
    rng = rng or _rng(cfg)
    if not cfg.enabled:
        return arr

    im = Image.fromarray(arr)
    im = random_rotate_scale_shift(im, cfg, rng)
    im = random_flip(im, cfg, rng)
    out = np.asarray(im, dtype=np.uint8)

    out = color_jitter(out, cfg, rng)
    if rng.random() < cfg.uneven_light_p:
        out = uneven_lighting(out, rng)
    if rng.random() < cfg.shadow_patch_p:
        out = shadow_patch(out, rng)
    if rng.random() < cfg.occlusion_p:
        out = occlude(out, rng)
    if cfg.noise_sigma > 0:
        out = gaussian_noise(out, rng, cfg.noise_sigma * rng.uniform(0.3, 1.0))
    if rng.random() < cfg.blur_p:
        out = np.asarray(Image.fromarray(out).filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.2))),
                         dtype=np.uint8)
    if rng.random() < cfg.motion_blur_p:
        out = motion_blur(out, rng)
    if rng.random() < cfg.jpeg_p:
        out = jpeg_artifacts(out, rng, cfg.jpeg_quality)
    return out


def center_crop_resize(arr: Array, size: int) -> Array:
    im = Image.fromarray(arr)
    w, h = im.size
    short = min(w, h)
    if short != size:
        scale = size / float(short)
        im = im.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.BILINEAR)
    w, h = im.size
    box = ((w - size) // 2, (h - size) // 2, (w - size) // 2 + size, (h - size) // 2 + size)
    return np.asarray(im.crop(box), dtype=np.uint8)


class Augmenter:
    """有状态封装，方便 DataLoader 里多线程使用。"""

    def __init__(self, cfg: Optional[AugmentConfig] = None, seed: Optional[int] = None) -> None:
        self.cfg = cfg or AugmentConfig()
        self._rng = random.Random(seed)

    def __call__(self, arr: Array) -> Array:
        return augment_array(arr, self.cfg, random.Random(self._rng.randrange(2**31 - 1)))

    def many(self, arr: Array, n: int) -> Sequence[Array]:
        return [self(arr) for _ in range(n)]
