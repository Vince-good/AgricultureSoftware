"""严重程度估计。

分类网络给的是"是不是这个病"，给不出"有多重"。而农户真正要问的是后者 ——
文档 3.3 要求播报内容包含严重程度。

这里的做法是：在预处理后的图像上算一个可解释的"胁迫像素占比"作为面积代理，
再与分类置信度融合，落到轻度/中度/重度三档。不引入第二个模型，
因为它必须在 300 元级设备的 3 秒预算内和分类推理一起跑完。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

# 三档阈值：胁迫面积占比达到多少算中度/重度
MILD_MAX = 0.12
MODERATE_MAX = 0.30

# 置信度对严重程度的贡献权重。面积代理受光照影响大，所以不让它单独说了算。
AREA_WEIGHT = 0.7
CONF_WEIGHT = 0.3


def rgb_to_hsv(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """uint8 RGB → (H 角度 0-360, S 0-1, V 0-1)。纯 numpy，避免引入 opencv。"""
    a = arr.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    mask = delta > 1e-6

    rm = np.where(mask & (cmax == r), ((g - b) / np.where(mask, delta, 1)) % 6.0, 0.0)
    gm = np.where(mask & (cmax == g), ((b - r) / np.where(mask, delta, 1)) + 2.0, 0.0)
    bm = np.where(mask & (cmax == b), ((r - g) / np.where(mask, delta, 1)) + 4.0, 0.0)
    h = np.where(mask & (cmax == r), rm, h)
    h = np.where(mask & (cmax == g), gm, h)
    h = np.where(mask & (cmax == b), bm, h)
    h = (h * 60.0) % 360.0

    s = np.where(cmax > 1e-6, delta / np.where(cmax > 1e-6, cmax, 1.0), 0.0)
    return h, s, cmax


def _downsample(img: np.ndarray, target: int = 128) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= target:
        return img
    step = max(1, int(round(max(h, w) / float(target))))
    return img[::step, ::step, :]


def plant_ratio(hsv: Tuple[np.ndarray, np.ndarray, np.ndarray],
                loose: bool = False) -> np.ndarray:
    """植株像素掩膜：绿色 + 黄化/褪色的叶片都算，排除土壤和阴影。"""
    h, s, v = hsv
    green = (h >= 55) & (h <= 175) & (s >= 0.12) & (v >= 0.10)
    if not loose:
        return green
    # 黄化叶（缺氮）与紫红叶（缺磷）已经不是绿色，需要放宽色相范围
    yellow = (h >= 35) & (h < 55) & (s >= 0.18) & (v >= 0.15)
    purple = (((h < 350) | (h >= 330)) & (s >= 0.18) & (v >= 0.10))
    return green | yellow | purple


def stress_pixel_ratio(stress: str, img: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """按胁迫类型计算异常像素占植株面积的比例。"""
    small = _downsample(img)
    hsv = rgb_to_hsv(small)
    h, s, v = hsv
    loose = stress in ("nutrient", "water")
    plant = plant_ratio(hsv, loose=loose)
    plant_n = int(plant.sum())
    total = int(plant.size)
    evidence: Dict[str, Any] = {
        "plant_coverage": round(plant_n / max(total, 1), 4),
        "mean_value": round(float(v.mean()), 4),
        "mean_saturation": round(float(s.mean()), 4),
        "stress": stress,
    }
    if plant_n < int(0.03 * total):
        # 画面里几乎没有植株，面积代理没有意义
        evidence["reliable"] = False
        return 0.0, evidence

    evidence["reliable"] = True

    if stress == "disease":
        # 坏死斑：偏褐/偏黄褐（色相 < 55）或很暗的死组织，且仍在植株轮廓内
        necrotic = plant & (((h < 55) & (s >= 0.15)) | ((v < 0.20) & (s >= 0.10)))
        ratio = float(necrotic.sum()) / plant_n
        evidence["necrotic_ratio"] = round(ratio, 4)
    elif stress == "nutrient":
        # 缺氮＝失绿黄化；缺磷＝暗紫。两者都表现为"偏离健康绿"
        yellowed = plant & (h >= 35) & (h < 70) & (s >= 0.18)
        purpled = plant & (((h < 25) | (h >= 330)) & (s >= 0.20))
        pale = plant & (s < 0.22) & (h >= 55) & (h <= 110)
        ratio = float((yellowed | purpled | pale).sum()) / plant_n
        evidence["yellowed_ratio"] = round(float(yellowed.sum()) / plant_n, 4)
        evidence["purpled_ratio"] = round(float(purpled.sum()) / plant_n, 4)
    elif stress == "water":
        # 失水叶片反光变弱、颜色发暗发灰：低饱和 + 低明度，且整体方差升高
        dull = plant & (s < 0.28) & (v < 0.45)
        ratio = float(dull.sum()) / plant_n
        evidence["dull_ratio"] = round(ratio, 4)
        evidence["value_std"] = round(float(v[plant].std()) if plant_n else 0.0, 4)
    else:
        ratio = 0.0

    return float(np.clip(ratio, 0.0, 1.0)), evidence


def estimate_severity(stress: str, confidence: float, img: np.ndarray | None = None,
                      class_id: str = "") -> Tuple[str, float, Dict[str, Any]]:
    """返回 (severity_id, score, evidence)。健康/无效类恒为 none。"""
    if stress not in ("disease", "nutrient", "water"):
        return "none", 0.0, {"stress": stress, "reason": "non-stress class"}

    if img is None:
        score = float(np.clip((confidence - 0.5) * 2.0, 0.0, 1.0))
        return _bucket(score), score, {"stress": stress, "image": None}

    area, evidence = stress_pixel_ratio(stress, img)
    # 面积归一化：35% 以上异常面积按满格算
    area_norm = float(np.clip(area / 0.35, 0.0, 1.0))
    conf_norm = float(np.clip((confidence - 0.4) / 0.6, 0.0, 1.0))
    if evidence.get("reliable"):
        score = AREA_WEIGHT * area_norm + CONF_WEIGHT * conf_norm
    else:
        # 面积不可信时退回只用置信度，宁可保守
        score = conf_norm * 0.6
        evidence["fallback_to_confidence"] = True
    score = float(np.clip(score, 0.0, 1.0))
    evidence["area_ratio"] = round(area, 4)
    evidence["area_normalized"] = round(area_norm, 4)
    evidence["confidence"] = round(float(confidence), 4)
    evidence["score"] = round(score, 4)
    evidence["class_id"] = class_id
    return _bucket(score), score, evidence


def _bucket(score: float) -> str:
    if score < MILD_MAX / MODERATE_MAX:
        return "mild"
    if score < 0.72:
        return "moderate"
    return "severe"


def severity_from_ratio(area_ratio: float) -> str:
    """只按面积占比分档，供离线批量复核历史照片时使用。"""
    if area_ratio < MILD_MAX:
        return "mild"
    if area_ratio < MODERATE_MAX:
        return "moderate"
    return "severe"
