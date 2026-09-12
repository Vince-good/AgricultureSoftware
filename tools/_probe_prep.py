"""Isolate whether quant_pre_process itself breaks the FP32 graph."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from heyan.train import data as ddata
from heyan.train.quantize import restricted_fs_tempdir, run_onnx, _preprocess_model
from heyan.eval.metrics import summarize

BUNDLE = Path("artifacts/bundles/heyan-mnv3s-int8-v1.0.0")
FP32 = BUNDLE / "model_fp32.onnx"
WORK = Path("artifacts/_qsweep"); WORK.mkdir(parents=True, exist_ok=True)

samples, class_ids = ddata.discover_samples(Path("artifacts/data/demo_dataset"))
train_s, val_s = ddata.split_samples(samples, 0.2, 42)
train_ds = ddata.LeafDataset(train_s, list(class_ids), augment=None, cache=True)
val_ds = ddata.LeafDataset(val_s, list(class_ids), augment=None, cache=True)
calib = ddata.calibration_arrays(train_ds, 128, seed=42)
val_arrays = [val_ds.raw_array(i) for i in range(len(val_ds))]
labels = np.asarray(val_ds.labels, dtype=np.int64)

orig, _ = run_onnx(FP32, val_arrays, threads=1)
print(f"FP32 original      top1={summarize(orig, labels, class_ids)['top1']:.4f}")

prep = _preprocess_model(FP32, WORK / "prep.onnx")
print("prep path:", prep)
p, _ = run_onnx(prep, val_arrays, threads=1)
print(f"FP32 preprocessed  top1={summarize(p, labels, class_ids)['top1']:.4f}")
print("max |orig-prep| logit diff:", float(np.abs(orig - p).max()))


class Reader:
    def __init__(self, arrays, name="input"):
        self.a = [np.ascontiguousarray(x, dtype=np.float32) for x in arrays]
        self.n = name; self.i = 0
    def get_next(self):
        if self.i >= len(self.a): return None
        x = self.a[self.i]; self.i += 1; return {self.n: x}
    def rewind(self): self.i = 0


# Quantize straight from the ORIGINAL graph, skipping quant_pre_process entirely
out = WORK / "noprep_int8.onnx"
r = Reader(calib); r.rewind()
with restricted_fs_tempdir():
    quantize_static(str(FP32), str(out), r, quant_format=QuantFormat.QDQ,
                    per_channel=True, weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8,
                    calibrate_method=CalibrationMethod.MinMax,
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
q, _ = run_onnx(out, val_arrays, threads=1)
print(f"INT8 no-preprocess top1={summarize(q, labels, class_ids)['top1']:.4f} "
      f"size={out.stat().st_size/1024/1024:.2f}MB")
