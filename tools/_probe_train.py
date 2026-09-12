import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from heyan.train import data as hdata, augment, model as hmodel, finetune
from heyan.train.pipeline import PipelineConfig, resolve_dataset, build_datasets

cfg = PipelineConfig(data_dir="artifacts/data/demo_dataset", epochs=1, teacher_epochs=1)
samples, class_ids = resolve_dataset(cfg)
train_ds, val_ds, tr, va, stats = build_datasets(cfg, samples, class_ids)
x, y = train_ds[0]
print("sample tensor", tuple(x.shape), x.dtype, "label", y)
xb, yb = next(iter(torch.utils.data.DataLoader(train_ds, batch_size=4)))
print("batch", tuple(xb.shape), tuple(yb.shape))
m = hmodel.build_model("mobilenet_v3_large", len(class_ids), pretrained=True)
m.eval()
with torch.no_grad():
    out = m(xb)
print("teacher fwd ok", tuple(out.shape))
try:
    res = finetune.train(train_ds, val_ds, class_ids,
                         finetune.TrainConfig(arch="mobilenet_v3_large", num_classes=len(class_ids),
                                              epochs=1, batch_size=4, out_dir="artifacts/_probe_train"),
                         device=torch.device("cpu"))
    print("train ok", res.metrics.get("top1"))
except Exception:
    traceback.print_exc()
