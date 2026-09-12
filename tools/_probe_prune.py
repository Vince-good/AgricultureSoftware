"""Probe 1x1 conv shapes / SVD gain profile on MobileNetV3-Small (throwaway)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch, torchvision
from heyan.train.prune import _find_pointwise_convs, _rank_for_energy
from heyan.train.model import build_model

CKPT = Path("artifacts/runs/heyan-mnv3s-int8-20260913-054400/student/best.pt")
m = build_model("mobilenet_v3_small", num_classes=14, pretrained=False)
if CKPT.exists():
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = blob.get("state_dict", blob.get("model", blob)) if isinstance(blob, dict) else blob
    m.load_state_dict(sd, strict=False)
    print("loaded trained student weights")
else:
    print("no checkpoint, random init")
for min_ch in (24, 48, 96, 128):
    t = _find_pointwise_convs(m, min_ch)
    print(f"min_channels={min_ch}: {len(t)} candidates")

t = _find_pointwise_convs(m, 24)
print("\nlayer shapes + rank/gain by energy")
for energy in (0.95, 0.9, 0.85, 0.8):
    rows = []
    for parent, name, conv in t:
        w = conv.weight.detach().numpy().reshape(conv.out_channels, conv.in_channels)
        s = np.linalg.svd(w, compute_uv=False)
        r = _rank_for_energy(s, energy)
        before = conv.in_channels * conv.out_channels
        after = conv.in_channels * r + r * conv.out_channels
        gain = (before - after) / max(before, 1)
        rows.append((name, conv.out_channels, conv.in_channels, r, gain))
    applied = [x for x in rows if x[4] >= 0.15 and x[3] < min(x[1], x[2])]
    print(f"\nenergy={energy}: {len(applied)}/{len(rows)} pass min_gain=0.15")
    for x in rows[:14]:
        print(f"   {x[0]:<20} {x[1]}x{x[2]} rank={x[3]} gain={x[4]:.3f}")
