"""Validate the QAT export chain: torch fake-quant -> ONNX QDQ -> onnxruntime."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collections
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

WORK = Path("artifacts/_qat_probe")
WORK.mkdir(parents=True, exist_ok=True)


class ActFQ(nn.Module):
    """Per-tensor asymmetric uint8 fake-quant, exported as QuantizeLinear+DequantizeLinear."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor([0.02], dtype=torch.float32))
        self.register_buffer("zp", torch.tensor([128], dtype=torch.uint8))

    def forward(self, x):
        return torch.fake_quantize_per_tensor_affine(
            x, float(self.scale.item()), int(self.zp.item()), 0, 255)


class ConvFQ(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.in_fq = ActFQ()
        self.out_fq = ActFQ()
        self.register_buffer("w_scale", torch.full((cout,), 0.01, dtype=torch.float32))
        self.register_buffer("w_zp", torch.zeros(cout, dtype=torch.int32))

    def forward(self, x):
        w = torch.fake_quantize_per_channel_affine(
            self.conv.weight, self.w_scale, self.w_zp, 0, -128, 127)
        return self.out_fq(F_conv(self.in_fq(x), w))


def F_conv(x, w):
    return torch.nn.functional.conv2d(x, w, None, padding=1)


net = nn.Sequential(ConvFQ(3, 4), nn.ReLU(), ConvFQ(4, 14), nn.AdaptiveAvgPool2d(1), nn.Flatten())
net.eval()
dummy = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    ref = net(dummy)

out = WORK / "tiny_qat.onnx"
torch.onnx.export(net, (dummy,), str(out), input_names=["input"], output_names=["logits"],
                  opset_version=17, do_constant_folding=False, dynamo=False)

m = onnx.load(str(out))
print("ops:", collections.Counter(n.op_type for n in m.graph.node))

sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
got = sess.run(None, {"input": dummy.numpy().astype(np.float32)})[0]
print("torch:", np.round(ref.numpy().ravel()[:5], 5))
print("ort  :", np.round(got.ravel()[:5], 5))
print("max abs diff:", float(np.abs(ref.numpy() - got).max()))

