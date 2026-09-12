"""预下载 torchvision 预训练权重。

文档 3.2(3) 的技术路径依赖 ImageNet 预训练权重做微调，而边缘构建机
（往往是县农技站的一台普通笔记本）不一定随时有网。这里把权重一次性
拉进本地缓存，之后的构建流程即可完全离线执行。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heyan.train.model import ARCHS  # noqa: E402


def main(argv: list[str]) -> int:
    import torch
    import torchvision.models as tvm

    from heyan.train.model import _resolve_weights

    names = argv or ["mobilenet_v3_small", "mobilenet_v3_large"]
    home = Path(torch.hub.get_dir())
    print(f"[weights] torch hub 缓存目录: {home}")
    failed = []
    for arch in names:
        spec = ARCHS[arch]
        try:
            weights = _resolve_weights(tvm, spec["weights"])
            getattr(tvm, spec["factory"])(weights=weights)
            print(f"[weights] OK  {arch}")
        except Exception as exc:
            failed.append((arch, exc))
            print(f"[weights] FAIL {arch}: {type(exc).__name__}: {exc}")
    for f in sorted(home.rglob("*")):
        if f.is_file():
            print(f"  {f.stat().st_size / 1024 / 1024:6.2f} MB  {f.name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
