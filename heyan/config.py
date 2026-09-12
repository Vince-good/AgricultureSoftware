"""全局工程约束与默认配置。

这里的数值不是随手写的默认值，而是方案文档里的硬性验收指标，
`heyan.eval.benchmark` 会用同一组常量判定构建产物是否达标。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# 文档来源的硬约束（研究问题 2 / 3.2 节）
# --------------------------------------------------------------------------

# 量化后的模型文件体积上限（3.2(2)(4)：采用量化技术将模型体积控制在 20MB 以内）
MODEL_SIZE_MAX_MB = 20.0

# 单张图片端到端识别延迟上限（研究问题 2：离线、低延迟 ≤ 3 秒）
LATENCY_MAX_S = 3.0

# 推理时进程内存占用上限（3.2(4)：内存占用控制在 200MB 以下）
RUNTIME_MEMORY_MAX_MB = 200.0

# 目标设备画像：低端智能手机 RAM ≤ 2GB，或成本 ≤ 300 元的嵌入式设备
DEVICE_RAM_MAX_MB = 2048
DEVICE_COST_MAX_CNY = 300

# 移动端可接受的最低可用精度（低于此值只允许提示"请重拍"，不给出结论）
MIN_CONFIDENCE_FOR_VERDICT = 0.45

# 量化后 top1 精度损失上限（3.2(2)(4)：精度损失不超过 3%）。
# 实测纯 PTQ 在本项目的 MobileNetV3-Small 上会掉 15~47 个百分点，
# 只有把伪量化放进训练回路（QAT）才能兑现这条合同，见 heyan/train/qat.py。
ACCURACY_DROP_MAX = 0.03

# --------------------------------------------------------------------------
# 模型与预处理
# --------------------------------------------------------------------------

# 3.2(1)(4)：MobileNetV3 在"精度—速度—体积"三维权衡下最为均衡
DEFAULT_ARCH = "mobilenet_v3_small"
DEFAULT_TEACHER_ARCH = "mobilenet_v3_large"

INPUT_SIZE = 224
RESIZE_SIZE = 256  # 短边缩放到 256 再中心裁剪 224，MobileNet 官方预处理
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

# 8-bit 量化：权重 per-channel 对称 int8，激活 per-tensor 非对称 uint8。
# 主路径是量化感知训练（QAT），PTQ 仅作为没有训练条件时的降级方案。
QUANT_WEIGHT_TYPE = "qint8"
QUANT_ACTIVATION_TYPE = "quint8"
QUANT_CALIBRATION_SAMPLES = 128
QUANT_QAT_EPOCHS = 6
QUANT_QAT_LR = 3e-5

TOP_K = 3

# --------------------------------------------------------------------------
# 交互（3.3 节：拍照 → 识别 → 语音播报 三步极简流程）
# --------------------------------------------------------------------------

# 界面最小字号与触控目标尺寸，面向 60 岁以上、低识字率用户
UI_MIN_FONT_PX = 20
UI_TOUCH_TARGET_PX = 72

# 语音播报时长上限：一句话讲完，超出则截断到备用短话术
VOICE_MAX_SECONDS = 12.0

# 默认播报语言与回退链：普通话 / 粤语 / 客家话 / 潮汕话
DEFAULT_LANGUAGE = "zh"
SUPPORTED_LANGUAGES = ("zh", "yue", "hak", "teochew", "en")

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_ROOT / "assets"
TAXONOMY_PATH = ASSETS_DIR / "taxonomy.json"
ADVISORY_PATH = ASSETS_DIR / "advisory.json"
VOICEPACK_ROOT = ASSETS_DIR / "voicepacks"


def _default_root() -> Path:
    env = os.environ.get("HEYAN_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd() / "artifacts"


@dataclass(frozen=True)
class Paths:
    """运行期目录约定。可通过环境变量 HEYAN_HOME 整体重定向（便于 SD 卡/U 盘部署）。"""

    root: Path = field(default_factory=_default_root)

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def bundles(self) -> Path:
        return self.root / "bundles"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def records(self) -> Path:
        return self.root / "records"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def outbox(self) -> Path:
        """待外发的对接文件（保险/补贴/农资）。离线优先：先生成，后有网或走 U 盘时再发。"""
        return self.root / "outbox"

    @property
    def voicepacks(self) -> Path:
        return self.root / "voicepacks"

    @property
    def db(self) -> Path:
        return self.records / "heyan.db"

    def ensure(self) -> "Paths":
        for p in (self.models, self.bundles, self.data, self.records, self.exports,
                  self.outbox, self.voicepacks):
            p.mkdir(parents=True, exist_ok=True)
        return self


PATHS = Paths()
