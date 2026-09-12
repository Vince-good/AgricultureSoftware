"""分类指标。纯 numpy，训练和边缘端复核都能用。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1)
    if logits.size == 0:
        return 0.0
    k = min(k, logits.shape[1])
    topk = np.argsort(-logits, axis=1)[:, :k]
    hit = (topk == labels[:, None]).any(axis=1)
    return float(hit.mean())


def confusion_matrix(preds: Sequence[int], labels: Sequence[int], n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for p, t in zip(preds, labels):
        if 0 <= int(t) < n and 0 <= int(p) < n:
            cm[int(t), int(p)] += 1
    return cm


def per_class_report(cm: np.ndarray, class_ids: Sequence[str]) -> Dict[str, Dict[str, float]]:
    report: Dict[str, Dict[str, float]] = {}
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted_as = cm.sum(axis=0).astype(np.float64)
    for i, cid in enumerate(class_ids):
        precision = tp[i] / predicted_as[i] if predicted_as[i] > 0 else 0.0
        recall = tp[i] / support[i] if support[i] > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        report[cid] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(support[i]),
        }
    return report


def summarize(logits: np.ndarray, labels: np.ndarray, class_ids: Sequence[str],
              ks: Sequence[int] = (1, 3)) -> Dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1)
    preds = np.argmax(logits, axis=1)
    cm = confusion_matrix(preds, labels, len(class_ids))
    out: Dict[str, Any] = {
        "n": int(len(labels)),
        "num_classes": len(class_ids),
        "macro_accuracy": round(float((preds == labels).mean()), 4) if len(labels) else 0.0,
        "per_class": per_class_report(cm, class_ids),
    }
    for k in ks:
        out[f"top{k}"] = round(topk_accuracy(logits, labels, k), 4)
    precs = [v["precision"] for v in out["per_class"].values() if v["support"] > 0]
    recs = [v["recall"] for v in out["per_class"].values() if v["support"] > 0]
    f1s = [v["f1"] for v in out["per_class"].values() if v["support"] > 0]
    out["macro_precision"] = round(float(np.mean(precs)), 4) if precs else 0.0
    out["macro_recall"] = round(float(np.mean(recs)), 4) if recs else 0.0
    out["macro_f1"] = round(float(np.mean(f1s)), 4) if f1s else 0.0
    out["confusion_matrix"] = cm.tolist()
    return out


def worst_classes(report: Dict[str, Dict[str, float]], n: int = 3) -> List[str]:
    ranked = sorted(report.items(), key=lambda kv: kv[1]["f1"])
    return [cid for cid, _ in ranked[:n]]


def accuracy_drop(before: Dict[str, Any], after: Dict[str, Any], key: str = "top1") -> float:
    """量化/剪枝后的精度损失，文档要求控制在 3% 以内。"""
    return round(float(before.get(key, 0.0)) - float(after.get(key, 0.0)), 4)
