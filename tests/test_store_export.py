"""记录库与导出链路测试：哈希链、同意门、匿名化。

这三件事是文档 4.3 节（数据主权）的合同条款：
  - 记录成链，删改必须被完整性校验如实发现；
  - 保险/补贴/农资三方取数必须先过农户同意；
  - 统计用途默认匿名。
"""

from __future__ import annotations

import io
import json
import time

import pytest

from heyan.data import Exporter, RecordStore


def _jpeg_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (60, 120, 60)).save(buf, "JPEG", quality=80)
    return buf.getvalue()


def _result(class_id: str, confidence: float, severity: str = "mild") -> dict:
    now = time.time()
    return {
        "class_id": class_id,
        "confidence": confidence,
        "needs_retake": False,
        "created_at": now,
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
        "image_id": f"img-{class_id}",
        "severity": {"id": severity, "score": 0.5, "name": severity,
                     "color": "#4C8C2B"},
        "candidates": [{"class_id": class_id, "probability": confidence}],
        "runtime": {"model_id": "heyan-mnv3s-int8", "model_version": "1.0.0",
                    "arch": "mobilenet_v3_small", "backend": "test"},
        "advice": {"class_id": class_id, "name": class_id,
                   "voice": "test voice", "severity": {"id": severity}},
    }


@pytest.fixture()
def store(tmp_path):
    db = RecordStore(db_path=tmp_path / "t.db")
    yield db
    db.close()


def _add(store, class_id, **kwargs):
    record_id = f"rec-test-{class_id}"
    rel, digest, w, h = store.save_image(_jpeg_bytes(), record_id)
    return store.add_result(_result(class_id, 0.8), record_id=record_id,
                            image_ref=rel, image_sha256=digest,
                            observation={"image_width": w, "image_height": h},
                            **kwargs)


def _package(result) -> dict:
    path = [f for f in result.files if f.endswith(".json")][0]
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_chain_and_filters(store):
    # 交换格式约定：村址在农户档案下，地块在田块下（data/schema.py）
    _add(store, "rice_blast", farmer={"alias": "老李", "id": "F001",
                                      "village": "测试村"},
         field={"plot_id": "P1"})
    _add(store, "rice_healthy", farmer={"alias": "老李", "id": "F001",
                                        "village": "测试村"},
         field={"plot_id": "P1"})
    _add(store, "peanut_leaf_spot", farmer={"alias": "阿妹", "id": "F002",
                                            "village": "邻村"},
         field={"plot_id": "P2"})

    assert store.count() == 3
    assert store.count(class_id="rice_blast") == 1
    assert store.count(village="测试村") == 2
    assert store.count(farmer_id="F002") == 1
    rows = store.list(limit=10, order="asc", full=False)
    assert [r["record_id"] for r in rows][0] == "rec-test-rice_blast"

    report = store.verify_integrity()
    assert report["ok"] is True
    assert report["checked"] == 3
    assert report["problems"] == []


def test_delete_breaks_chain_honestly(store):
    _add(store, "rice_blast")
    _add(store, "rice_healthy")
    store.delete("rec-test-rice_blast")
    report = store.verify_integrity()
    assert report["ok"] is False, "删除必须被如实报告，不能假装链还完整"
    assert report["problems"]


def test_followup_and_consent(store):
    _add(store, "rice_blast")
    rec = store.update_followup("rec-test-rice_blast", action_taken="喷药",
                                effect="better")
    assert rec["followup"]["effect"] == "better"
    rec = store.update_consent("rec-test-rice_blast", actor="老李",
                               share_insurance=True)
    assert rec["consent"]["share_insurance"] is True
    assert rec["consent"].get("share_subsidy") is not True


def test_export_consent_gate_and_anonymize(store, tmp_path):
    _add(store, "rice_blast", farmer={"alias": "老李", "id": "F001"})
    exporter = Exporter(store, out_root=tmp_path / "exports")

    # 保险用途：没同意就必须一条都不给，并且如实说明被挡了几条
    result = exporter.export(fmt="json", purpose="insurance", channel="local")
    assert result.ok is True
    assert result.excluded_by_consent == 1
    package = _package(result)
    assert package["record_count"] == 0

    store.update_consent("rec-test-rice_blast", share_insurance=True)
    result = exporter.export(fmt="both", purpose="insurance", channel="local")
    assert result.ok is True
    package = _package(result)
    assert package["record_count"] == 1

    # 统计用途：默认匿名，导出件里不允许出现农户别名
    stats = exporter.export(fmt="json", purpose="statistics", channel="local")
    assert "老李" not in json.dumps(_package(stats), ensure_ascii=False)
    assert stats.anonymized is True


def test_image_downsampled_on_save(store):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4000, 3000), (30, 90, 40)).save(buf, "JPEG", quality=90)
    rel, digest, w, h = store.save_image(buf.getvalue(), "rec-big")
    assert max(w, h) <= 1280
    path = store.image_path(rel)
    assert path.exists() and path.stat().st_size < len(buf.getvalue())
