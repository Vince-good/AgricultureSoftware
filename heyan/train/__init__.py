"""训练与压缩流水线。

对应文档 3.2 节的技术路径：预训练模型微调（小样本）→ 知识蒸馏 → 通道剪枝
→ ONNX 导出 → 8-bit 量化 → 体积/延迟/内存预算校验 → 打包 bundle。

这一层只在构建机上运行，边缘设备不需要 torch。
"""

from __future__ import annotations

__all__ = ["augment", "data", "model", "finetune", "distill", "prune", "export_onnx",
           "quantize", "pipeline"]
