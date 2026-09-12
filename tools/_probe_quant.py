"""Verify INT8 quantization works under the restricted-tempdir shim."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from heyan.train.quantize import quantize_int8, _mkdtemp_usable

print("mkdtemp usable natively?", _mkdtemp_usable())
src = Path("artifacts/runs/heyan-mnv3s-int8-20260913-054400/model_fp32.onnx")
out = Path("artifacts/_probe_quant/model_int8.onnx")
out.parent.mkdir(parents=True, exist_ok=True)
calib = [np.random.rand(1, 3, 224, 224).astype(np.float32) for _ in range(8)]
t0 = time.time()
res = quantize_int8(src, out, calib, per_channel=True, input_name="input",
                    work_dir=out.parent)
print("result:", res)
print(f"elapsed {time.time()-t0:.1f}s")
