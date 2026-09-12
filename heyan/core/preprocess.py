"""图像预处理。

只依赖 numpy + Pillow，这两样在 300 元级开发板和 Termux 上都装得上。
MobileNetV3 官方预处理是"短边缩放到 256 → 中心裁剪 224 → ImageNet 均值方差归一化"，
这里保持完全一致，否则量化模型的精度会莫名其妙掉几个点。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ..config import IMAGE_MEAN, IMAGE_STD, INPUT_SIZE, RESIZE_SIZE

PathLike = Union[str, Path]
ImageArray = np.ndarray  # uint8, HWC, RGB


def read_image(path: PathLike, max_side: Optional[int] = 1280) -> ImageArray:
    """读取图片为 uint8 HWC RGB。

    手机直出的照片动辄 4000×3000，先在读取阶段降采样到 max_side，
    这是低端设备内存预算（≤200MB）里最容易被忽略的一处泄漏。
    """
    from PIL import Image

    with Image.open(path) as im:
        return _pil_to_array(im, max_side)


def read_image_bytes(data: bytes, max_side: Optional[int] = 1280) -> ImageArray:
    """从内存字节流读图（HTTP 上传走这条路），语义与 `read_image` 完全一致。

    单独开一个入口而不是先落盘再读，是为了不在低端设备的存储上留临时文件；
    两个函数共用 `_pil_to_array`，避免"网页识别和命令行识别结果不一样"这类漂移。
    """
    import io

    from PIL import Image

    with Image.open(io.BytesIO(bytes(data))) as im:
        return _pil_to_array(im, max_side)


def _pil_to_array(im, max_side: Optional[int]) -> ImageArray:
    from PIL import ImageOps

    im = ImageOps.exif_transpose(im)  # 手机竖拍照片的方向藏在 EXIF 里
    im = im.convert("RGB")
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def _pil_resize(img: ImageArray, size: Tuple[int, int]) -> ImageArray:
    from PIL import Image

    im = Image.fromarray(img)
    im = im.resize(size, Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def resize_short_side(img: ImageArray, size: int = RESIZE_SIZE) -> ImageArray:
    h, w = img.shape[:2]
    if min(h, w) == size:
        return img
    scale = size / float(min(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return _pil_resize(img, (new_w, new_h))


def center_crop(img: ImageArray, size: int = INPUT_SIZE) -> ImageArray:
    h, w = img.shape[:2]
    if h < size or w < size:  # 极端小图先放大，保证输出尺寸恒定
        img = _pil_resize(img, (max(w, size), max(h, size)))
        h, w = img.shape[:2]
    top = (h - size) // 2
    left = (w - size) // 2
    return img[top:top + size, left:left + size, :]


def normalize(img: ImageArray, mean: Sequence[float] = IMAGE_MEAN,
              std: Sequence[float] = IMAGE_STD) -> np.ndarray:
    """uint8 HWC → float32 CHW，值域按 ImageNet 统计量归一化。"""
    arr = img.astype(np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def preprocess(img: ImageArray, size: int = INPUT_SIZE, resize: int = RESIZE_SIZE,
               mean: Sequence[float] = IMAGE_MEAN, std: Sequence[float] = IMAGE_STD,
               add_batch: bool = True) -> np.ndarray:
    out = center_crop(resize_short_side(img, resize), size)
    arr = normalize(out, mean, std)
    return arr[None, ...] if add_batch else arr


def preprocess_path(path: PathLike, **kwargs) -> np.ndarray:
    return preprocess(read_image(path), **kwargs)


def quantize_input(arr: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """float32 → uint8，配合 QDQ 量化模型的输入量化参数。"""
    q = np.rint(arr / scale) + zero_point
    return np.clip(q, 0, 255).astype(np.uint8)


def dequantize_input(arr: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (arr.astype(np.float32) - float(zero_point)) * scale


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def letterbox(img: ImageArray, size: int = INPUT_SIZE, fill: Tuple[int, int, int] = (114, 114, 114)
              ) -> Tuple[ImageArray, float, Tuple[int, int]]:
    """等比缩放后补边到正方形。检测类模型或需要保留全幅画面的场景用它。"""
    h, w = img.shape[:2]
    scale = size / float(max(h, w))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = _pil_resize(img, (new_w, new_h))
    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, scale, (left, top)
