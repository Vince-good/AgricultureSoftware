"""End-to-end QAT smoke test on the existing student checkpoint."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from heyan.eval.metrics import summarize
from heyan.train import data as ddata
from heyan.train import qat
from heyan.train.export_onnx import load_checkpoint
from heyan.train.quantize import run_onnx

CKPT = Path("artifacts/runs/heyan-mnv3s-int8-20260913-055751/student/best.pt")
WORK = Path("artifacts/_qat_probe")
WORK.mkdir(parents=True, exist_ok=True)

samples, class_ids = ddata.discover_samples(Path("artifacts/data/demo_dataset"))
train_s, val_s = ddata.split_samples(samples, 0.2, 42)
train_ds = ddata.LeafDataset(train_s, list(class_ids), augment=None, cache=True)
val_ds = ddata.LeafDataset(val_s, list(class_ids), augment=None, cache=True)
calib = ddata.calibration_arrays(train_ds, 128, seed=42)
val_arrays = [val_ds.raw_array(i) for i in range(len(val_ds))]
labels = np.asarray(val_ds.labels, dtype=np.int64)

from heyan.train.finetune import TrainConfig

model, ckpt, cids = load_checkpoint(CKPT, "mobilenet_v3_small")
print("class ids:", len(cids), flush=True)

with torch.no_grad():
    model.eval()
    tl = np.concatenate([model(torch.from_numpy(np.ascontiguousarray(a))).numpy()
                         for a in val_arrays], axis=0)
print("torch FP32 top1 =", round(summarize(tl, labels, class_ids)["top1"], 4), flush=True)

res = qat.run_qat(model, train_ds, val_ds, class_ids, TrainConfig(),
                  calib, WORK / "run", epochs=2, batch_size=16, lr=3e-5, verbose=True)
qmodel = res["model"]

info = qat.export_qat_onnx(qmodel, WORK / "qat_int8.onnx")
print("ops:", info["ops"], flush=True)

with torch.no_grad():
    qat.set_mode(qmodel, observer=False, fake_quant=True)
    ql = np.concatenate([qmodel(torch.from_numpy(np.ascontiguousarray(a))).numpy()
                         for a in val_arrays], axis=0)
print("torch QAT  top1 =", round(summarize(ql, labels, class_ids)["top1"], 4), flush=True)

l8, t8 = run_onnx(WORK / "qat_int8.onnx", val_arrays, threads=1)
print("ORT   INT8 top1 =", round(summarize(l8, labels, class_ids)["top1"], 4),
      " p50ms=", round(float(np.median(t8)), 2), flush=True)
print("max |torch-ort| logit diff =", float(np.abs(ql - l8).max()), flush=True)
