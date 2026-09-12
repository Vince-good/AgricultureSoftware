"""Locate the INT8 accuracy collapse: pruned low-rank graph vs. pre-prune graph,
and HardSwish/HardSigmoid exclusion."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)

from heyan.eval.metrics import summarize
from heyan.train import data as ddata
from heyan.train.quantize import restricted_fs_tempdir, run_onnx

WORK = Path("artifacts/_qsweep")
WORK.mkdir(parents=True, exist_ok=True)

samples, class_ids = ddata.discover_samples(Path("artifacts/data/demo_dataset"))
train_s, val_s = ddata.split_samples(samples, 0.2, 42)
train_ds = ddata.LeafDataset(train_s, list(class_ids), augment=None, cache=True)
val_ds = ddata.LeafDataset(val_s, list(class_ids), augment=None, cache=True)
calib = ddata.calibration_arrays(train_ds, 128, seed=42)
val_arrays = [val_ds.raw_array(i) for i in range(len(val_ds))]
labels = np.asarray(val_ds.labels, dtype=np.int64)


class Reader:
    def __init__(self, arrays, name="input"):
        self.a = [np.ascontiguousarray(x, dtype=np.float32) for x in arrays]
        self.n = name
        self.i = 0

    def get_next(self):
        if self.i >= len(self.a):
            return None
        x = self.a[self.i]
        self.i += 1
        return {self.n: x}

    def rewind(self):
        self.i = 0


def measure(model_path, tag):
    logits, _ = run_onnx(model_path, val_arrays, threads=1)
    top1 = summarize(logits, labels, class_ids)["top1"]
    size = Path(model_path).stat().st_size / 1024 / 1024
    print(f"{tag:<46} top1={top1:.4f}  {size:.2f}MB", flush=True)
    return top1


def quant(src, dst, exclude=None, ops=None, method=CalibrationMethod.Percentile,
          wtype=QuantType.QInt8, reduce_range=False):
    r = Reader(calib)
    r.rewind()
    kw = {}
    if exclude:
        kw["nodes_to_exclude"] = list(exclude)
    if ops:
        kw["op_types_to_quantize"] = list(ops)
    with restricted_fs_tempdir():
        quantize_static(str(src), str(dst), r, quant_format=QuantFormat.QDQ,
                        per_channel=True, weight_type=wtype,
                        activation_type=QuantType.QUInt8,
                        calibrate_method=method, reduce_range=reduce_range,
                        extra_options={"ActivationSymmetric": False,
                                       "WeightSymmetric": True}, **kw)
    return dst


PREPRUNE = Path("artifacts/runs/heyan-mnv3s-int8-20260913-054400/model_fp32.onnx")
PRUNED = Path("artifacts/bundles/heyan-mnv3s-int8-v1.0.0/model_fp32.onnx")

print("=== pre-prune FP32 graph ===", flush=True)
measure(PREPRUNE, "FP32 pre-prune")
quant(PREPRUNE, WORK / "pre_int8.onnx")
measure(WORK / "pre_int8.onnx", "INT8 pre-prune (percentile)")

print("\n=== pruned low-rank FP32 graph ===", flush=True)
measure(PRUNED, "FP32 pruned")

g = onnx.load(str(PRUNED))
hard = [n.name for n in g.graph.node if n.op_type in ("HardSwish", "HardSigmoid")]
se = sorted({n.name for n in g.graph.node if ".block.1." in n.name or "avgpool" in n.name})
clf = [n.name for n in g.graph.node if "classifier" in n.name]
first = g.graph.node[0].name
last_conv = [n.name for n in g.graph.node if n.op_type == "Conv"][-1]
print(f"hard={len(hard)} se={len(se)} clf={clf} first={first} lastconv={last_conv}", flush=True)

quant(PRUNED, WORK / "pr_excl_hard.onnx", exclude=hard)
measure(WORK / "pr_excl_hard.onnx", "INT8 pruned, exclude HardSwish/HardSigmoid")

quant(PRUNED, WORK / "pr_excl_hard_se.onnx", exclude=sorted(set(hard) | set(se)))
measure(WORK / "pr_excl_hard_se.onnx", "INT8 pruned, exclude hard+SE")

quant(PRUNED, WORK / "pr_excl_hard_se_clf.onnx",
      exclude=sorted(set(hard) | set(se) | set(clf) | {first, last_conv}))
measure(WORK / "pr_excl_hard_se_clf.onnx", "INT8 pruned, exclude hard+SE+clf+ends")

quant(PRUNED, WORK / "pr_reduce_range.onnx", reduce_range=True)
measure(WORK / "pr_reduce_range.onnx", "INT8 pruned, reduce_range(7bit)")
