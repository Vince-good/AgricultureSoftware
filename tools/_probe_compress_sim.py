"""Simulate low-rank compression yield (no mutation, no training)."""
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
m.load_state_dict(b["state_dict"], strict=True)
m.eval()
total = sum(p.numel() for p in m.parameters())


def conv_targets(min_channels, skip_se):
    out = []
    for parent in m.modules():
        if skip_se and type(parent).__name__ == "SqueezeExcitation":
            continue
        for name, child in parent.named_children():
            if isinstance(child, nn.Conv2d) and child.kernel_size == (1, 1) \
                    and child.groups == 1 and child.dilation == (1, 1):
                if min(child.in_channels, child.out_channels) >= min_channels:
                    out.append((f"{type(parent).__name__}.{name}", child.out_channels,
                                child.in_channels, child.weight.detach().numpy()
                                .reshape(child.out_channels, child.in_channels)))
    return out


def lin_targets(min_features):
    out = []
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and min(mod.in_features, mod.out_features) >= min_features:
            out.append((name, mod.out_features, mod.in_features, mod.weight.detach().numpy()))
    return out


for energy in (0.95, 0.9, 0.85, 0.8):
    for min_gain in (0.15, 0.25):
        saved = 0
        applied = 0
        detail = []
        cands = [(n, o, i, w, "conv") for n, o, i, w in conv_targets(24, True)]
        cands += [(n, o, i, w, "lin") for n, o, i, w in lin_targets(256)]
        for name, oc, ic, w, kind in cands:
            s = np.linalg.svd(w, compute_uv=False)
            r = _rank_for_energy(s, energy)
            before = oc * ic
            after = ic * r + r * oc
            gain = (before - after) / max(before, 1)
            if gain >= min_gain and r < min(oc, ic):
                saved += before - after
                applied += 1
                detail.append((name, kind, oc, ic, r, round(gain, 3)))
        print(f"energy={energy} min_gain={min_gain}: applied={applied} "
              f"params {total:,} -> {total-saved:,} (-{saved/total:.1%})")
        if energy == 0.9 and min_gain == 0.15:
            for d in detail:
                print("     ", d)
