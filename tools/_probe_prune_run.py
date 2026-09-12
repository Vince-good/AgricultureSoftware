"""Run the real low_rank_compress on the trained student and check forward parity."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from heyan.train.model import build_model
from heyan.train import prune

m = build_model("mobilenet_v3_small", num_classes=14, pretrained=False)
b = torch.load("artifacts/runs/heyan-mnv3s-int8-20260913-054400/student/best.pt",
               map_location="cpu", weights_only=False)
m.load_state_dict(b["state_dict"], strict=True)
m.eval()

x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    ref = m(x)
print("baseline logits[0][:5]", ref[0][:5].tolist())

rep = prune.low_rank_compress(m, energy=0.85, min_channels=24, min_gain=0.15,
                              min_linear=256, verbose=True)
m.eval()
with torch.no_grad():
    out = m(x)
print("compressed logits[0][:5]", out[0][:5].tolist())
print("reduction", rep.reduction, "layers", rep.layers_compressed)
print("pred match:", (ref.argmax(1) == out.argmax(1)).tolist())
print("max abs logit diff:", float((ref - out).abs().max()))
