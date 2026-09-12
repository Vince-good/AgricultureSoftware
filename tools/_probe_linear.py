"""Probe Linear-layer SVD gain on the trained student (throwaway)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
import torch.nn as nn
from heyan.train.model import build_model
from heyan.train.prune import _rank_for_energy

m = build_model("mobilenet_v3_small", num_classes=14, pretrained=False)
b = torch.load("artifacts/runs/heyan-mnv3s-int8-20260913-054400/student/best.pt",
               map_location="cpu", weights_only=False)
missing, unexpected = m.load_state_dict(b["state_dict"], strict=True)
print("loaded strict OK")
total = sum(p.numel() for p in m.parameters())
print(f"total params {total:,}")
print("\nLinear layers:")
for name, mod in m.named_modules():
    if isinstance(mod, nn.Linear):
        print(f"  {name}: {mod.out_features}x{mod.in_features} = {mod.weight.numel():,}")

lins = [(n, mo) for n, mo in m.named_modules()
        if isinstance(mo, nn.Linear) and min(mo.in_features, mo.out_features) >= 256]
for energy in (0.95, 0.9):
    print(f"\nenergy={energy}")
    for name, mod in lins:
        w = mod.weight.detach().numpy()
        s = np.linalg.svd(w, compute_uv=False)
        r = _rank_for_energy(s, energy)
        before = mod.in_features * mod.out_features
        after = mod.in_features * r + r * mod.out_features
        print(f"  {name}: {mod.out_features}x{mod.in_features} rank={r} "
              f"gain={(before-after)/before:.3f} params {before:,}->{after:,}")

print("\nSE-block convs (to exclude):")
for name, mod in m.named_modules():
    if type(mod).__name__ == "SqueezeExcitation":
        for cn, cc in mod.named_children():
            if isinstance(cc, nn.Conv2d):
                print(f"  {name}.{cn}: {cc.out_channels}x{cc.in_channels} = {cc.weight.numel():,}")
