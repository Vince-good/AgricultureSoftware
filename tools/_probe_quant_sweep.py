"""Sweep INT8 quantization configs against the accuracy-drop contract."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from heyan.train import data as ddata
from heyan.train.quantize import restricted_fs_tempdir, run_onnx, _preprocess_model
from heyan.eval.metrics import summarize
from heyan.classes import load_taxonomy

BUNDLE = Path("artifacts/bundles/heyan-mnv3s-int8-v1.0.0")
FP32 = BUNDLE / "model_fp32.onnx"
WORK = Path("artifacts/_qsweep"); WORK.mkdir(parents=True, exist_ok=True)

tax = load_taxonomy()
class_ids = tax.ids

samples = ddata.discover_samples(Path("artifacts/data/demo_dataset"), class_ids)
train_s, val_s = ddata.split_samples(samples, 0.2, 42)
train_ds = ddata.LeafDataset(train_s, list(class_ids), augment=None, cache=True)
val_ds = ddata.LeafDataset(val_s, list(class_ids), augment=None, cache=True)
calib = ddata.calibration_arrays(train_ds, 128, seed=42)
val_arrays = [val_ds.raw_array(i) for i in range(len(val_ds))]
labels = np.asarray(val_ds.labels, dtype=np.int64)
print(f"calib={len(calib)} val={len(val_arrays)}")


class Reader:
    def __init__(self, arrays, name="input"):
        self.a = [np.ascontiguousarray(x, dtype=np.float32) for x in arrays]
        self.n = name; self.i = 0
    def get_next(self):
        if self.i >= len(self.a): return None
        x = self.a[self.i]; self.i += 1; return {self.n: x}
    def rewind(self): self.i = 0


logits32, _ = run_onnx(FP32, val_arrays, threads=1)
base = summarize(logits32, labels, class_ids)
print(f"FP32 top1={base['top1']:.4f}\n")

prep = _preprocess_model(FP32, WORK / "prep.onnx")
g = onnx.load(str(prep))
convs = [n.name for n in g.graph.node if n.op_type == "Conv"]
gemms = [n.name for n in g.graph.node if n.op_type in ("Gemm", "MatMul")]
print(f"nodes: {len(convs)} Conv, {len(gemms)} Gemm/MatMul")
print("first conv:", convs[:1], "last gemm:", gemms[-1:])

CONFIGS = {
    "A_current_u8_minmax": dict(weight_type=QuantType.QUInt8, activation_type=QuantType.QUInt8,
                                calibrate_method=CalibrationMethod.MinMax, per_channel=True,
                                extra_options={"ActivationSymmetric": False, "WeightSymmetric": False}),
    "B_s8w_minmax": dict(weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                         calibrate_method=CalibrationMethod.MinMax, per_channel=True,
                         extra_options={"ActivationSymmetric": False, "WeightSymmetric": True}),
    "C_s8w_percentile": dict(weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                             calibrate_method=CalibrationMethod.Percentile, per_channel=True,
                             extra_options={"ActivationSymmetric": False, "WeightSymmetric": True,
                                            "percentile": 99.999}),
    "D_s8w_entrophy": dict(weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                           calibrate_method=CalibrationMethod.Entropy, per_channel=True,
                           extra_options={"ActivationSymmetric": False, "WeightSymmetric": True}),
    "E_s8w_sym_both": dict(weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
                           calibrate_method=CalibrationMethod.MinMax, per_channel=True,
                           extra_options={"ActivationSymmetric": True, "WeightSymmetric": True}),
    "F_conv_only": dict(weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                        calibrate_method=CalibrationMethod.MinMax, per_channel=True,
                        op_types_to_quantize=["Conv"],
                        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True}),
}
EXCL = {
    "G_C_plus_exclude": ("C_s8w_percentile", [convs[0]] + gemms[-1:]),
    "H_E_plus_exclude": ("E_s8w_sym_both", [convs[0]] + gemms[-1:]),
}

rows = []
for name, kw in CONFIGS.items():
    out = WORK / f"{name}.onnx"
    r = Reader(calib); r.rewind()
    t0 = time.time()
    try:
        with restricted_fs_tempdir():
            quantize_static(str(prep), str(out), r, quant_format=QuantFormat.QDQ, **kw)
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__}: {e}"); continue
    lg, _ = run_onnx(out, val_arrays, threads=1)
    m = summarize(lg, labels, class_ids)
    sz = out.stat().st_size / 1024 / 1024
    drop = base["top1"] - m["top1"]
    rows.append((name, m["top1"], drop, sz, time.time() - t0))
    print(f"{name:<22} top1={m['top1']:.4f} drop={drop:+.4f} size={sz:.2f}MB ({time.time()-t0:.1f}s)")

for name, (basecfg, excl) in EXCL.items():
    kw = dict(CONFIGS[basecfg]); kw["nodes_to_exclude"] = excl
    out = WORK / f"{name}.onnx"
    r = Reader(calib); r.rewind()
    t0 = time.time()
    try:
        with restricted_fs_tempdir():
            quantize_static(str(prep), str(out), r, quant_format=QuantFormat.QDQ, **kw)
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__}: {e}"); continue
    lg, _ = run_onnx(out, val_arrays, threads=1)
    m = summarize(lg, labels, class_ids)
    sz = out.stat().st_size / 1024 / 1024
    drop = base["top1"] - m["top1"]
    rows.append((name, m["top1"], drop, sz, time.time() - t0))
    print(f"{name:<22} top1={m['top1']:.4f} drop={drop:+.4f} size={sz:.2f}MB ({time.time()-t0:.1f}s)")

print("\n=== ranked by drop ===")
for n, t, d, s, _ in sorted(rows, key=lambda x: x[2]):
    print(f"  {n:<22} top1={t:.4f} drop={d:+.4f} {'PASS' if d<=0.03 else 'FAIL'} size={s:.2f}MB")
