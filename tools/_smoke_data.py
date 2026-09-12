"""数据层端到端冒烟：入库 -> schema 校验 -> 哈希链 -> 导出 -> 三条对接。

跑法：python tools/_smoke_data.py。所有产物写进临时目录，不碰 artifacts/。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heyan.advice import build_advice
from heyan.classes import load_taxonomy
from heyan.data import (AgriSupplyAdapter, Exporter, InsuranceAdapter, Outbox,
                        RecordStore, SubsidyAdapter, build_record, registry, validate,
                        validate_export, verify_chain)
from heyan.data.integration import _sha256
from heyan.data.schema import verify_record

CASES = [
    ("rice_blast", "severe", 0.91, "rice"),
    ("rice_n_deficiency", "mild", 0.78, "rice"),
    ("rice_healthy", "none", 0.95, "rice"),
    ("peanut_leaf_spot", "moderate", 0.66, "peanut"),
    ("vegetable_p_deficiency", "mild", 0.52, "vegetable"),
    ("unusable", "none", 0.21, "none"),
]

VILLAGES = [("石坑村", "河口镇", "紫金县"), ("下輋村", "黄坑镇", "仁化县"),
            ("大塘村", "石潭镇", "英德市")]

SEV_SCORE = {"none": 0.0, "mild": 0.3, "moderate": 0.6, "severe": 0.9}


def fake_result(class_id: str, severity: str, confidence: float, crop: str,
                idx: int) -> dict:
    tax = load_taxonomy()
    info = tax.by_id(class_id)
    advice = build_advice(class_id, severity, "zh")
    low = confidence < 0.45
    when = time.time() - (len(CASES) * 3 - idx) * 3600
    return {
        "image_id": f"img-smoke-{idx:03d}",
        "created_at": when,
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(when)),
        "class_id": class_id,
        "confidence": confidence,
        "low_confidence": low,
        "needs_retake": low or class_id == "unusable",
        "severity": {"id": severity, "score": SEV_SCORE[severity]},
        "candidates": [{"class_id": class_id, "index": info.index, "name": info.name_zh,
                        "probability": confidence, "crop": crop, "stress": info.stress,
                        "icon": info.icon}],
        "advice": advice.to_dict(),
        "runtime": {"backend": "smoke", "model_id": "heyan-smoke",
                    "model_version": "0.0.1", "latency_ms": 210.5, "input_size": 224,
                    "language": "zh", "arch": "mobilenet_v3_small",
                    "quantization": "int8"},
        "evidence": {"leaf_ratio": 0.42, "lesion_pixels": 1200},
        "device": {"model": "smoke-device", "ram_mb": 2048},
    }


def main() -> int:
    # 临时目录放在工作区内：沙箱只允许写工作区，系统 TEMP 会被拒绝
    base = (Path(sys.argv[1]) if len(sys.argv) > 1 else
            Path(__file__).resolve().parents[1] / "artifacts" / "_smoke")
    base.mkdir(parents=True, exist_ok=True)
    # 不用 tempfile.mkdtemp：它建的 0700 目录在本机沙箱下连 listdir 都被拒
    tmp = base / f"data-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    state = {"ok": True}

    def check(label: str, good: bool, detail: str = "") -> None:
        state["ok"] = state["ok"] and bool(good)
        print(f"[{'PASS' if good else 'FAIL'}] {label}"
              + (f" -- {detail}" if detail else ""))

    store = RecordStore(db_path=tmp / "records" / "heyan.db")
    store.set_meta("device_id", "smoke-device-01")
    store.set_meta("app_version", "0.1.0")

    n = 0
    for i, (cid, sev, conf, crop) in enumerate(CASES):
        for f in range(3):
            village, town, county = VILLAGES[(i + f) % len(VILLAGES)]
            field_info = {
                "plot_id": f"plot-{i}-{f}", "crop": crop, "area_mu": 1.5 + f,
                "growth_stage": "分蘖期", "irrigation": "rainfed",
            }
            if f != 2:  # 第三张故意没有定位，用于验证风险标记
                field_info["location"] = {"lat": 23.4 + i * 0.05,
                                          "lon": 113.2 + f * 0.05,
                                          "accuracy_m": 12.0, "source": "gps"}
            store.add_result(
                fake_result(cid, sev, conf, crop, n),
                farmer={"id": f"hh-{(i + f) % 4 + 1}", "alias": f"农户{(i + f) % 4 + 1}",
                        "phone_masked": "138****5678", "age_band": "60-69",
                        "is_senior": True, "village": village, "town": town,
                        "county": county, "region": "粤东西北"},
                field=field_info,
                observation={"capture_mode": "camera"},
                consent={"share_insurance": f != 2, "share_subsidy": True,
                         "share_supplier": f == 0, "share_research": True},
                image_ref=f"images/rec-{n}.jpg", image_sha256=f"{n:064x}",
                device_id="smoke-device-01", app_version="0.1.0",
            )
            n += 1
    check("入库条数", len(store) == n == 18, f"n={n}")

    records = store.all_ordered()
    bad = [(r["record_id"], e[:3]) for r in records for e in [validate(r)[1]] if e]
    check("全部记录符合 RECORD_SCHEMA", not bad, json.dumps(bad[:2], ensure_ascii=False))

    chain = verify_chain(records)
    check("哈希链完整", chain["ok"], json.dumps(chain["problems"][:2], ensure_ascii=False))

    tampered = json.loads(json.dumps(records, ensure_ascii=False))
    tampered[3]["diagnosis"]["confidence"] = 0.999
    check("篡改诊断可被发现", not verify_chain(tampered)["ok"])

    stats = store.stats()
    check("统计聚合", stats["total"] == 18 and len(stats["by_class"]) >= 5,
          f"by_class={len(stats['by_class'])} consented={stats['consented']}")

    rid = records[0]["record_id"]
    old_hash = records[0]["integrity"]["record_sha256"]
    store.update_followup(rid, action_taken="喷施三环唑", product_used="三环唑 75%",
                          effect="better", reviewed_by="农技站-李")
    updated = store.get(rid)
    check("批注回写不改动证据核心摘要",
          updated["integrity"]["record_sha256"] == old_hash)
    check("批注回写后哈希链仍连续", verify_chain(store.all_ordered())["ok"])
    check("批注回写留痕",
          updated["integrity"]["annotation_log"][-1]["kind"] == "followup")
    check("批注摘要自洽", verify_record(updated)[0])

    sneaky = json.loads(json.dumps(updated, ensure_ascii=False))
    sneaky["consent"]["share_insurance"] = not sneaky["consent"]["share_insurance"]
    check("偷改授权可被发现", not verify_record(sneaky)[0])

    exporter = Exporter(store, out_root=tmp / "exports")
    for fmt in ("json", "csv", "jsonl"):
        for purpose in ("statistics", "insurance", "subsidy"):
            res = exporter.export(fmt=fmt, purpose=purpose, channel="local")
            primary = [p for p in res.files
                       if not p.endswith(("MANIFEST.json", ".schema.json"))]
            good = res.ok and bool(primary)
            if fmt == "json" and good:
                vok, verr = validate_export(primary[0])
                good = good and vok
                res.errors.extend(verr[:3])
            check(f"导出 {fmt}/{purpose}", good,
                  f"records={res.record_count} excluded={res.excluded_by_consent} "
                  f"scope={res.chain_scope} errors={res.errors[:2]}")

    stat_res = exporter.export(fmt="json", purpose="statistics")
    check("脱敏包声明为派生链", stat_res.chain_scope == "derived",
          f"scope={stat_res.chain_scope}")
    stat_pkg = json.loads(Path([p for p in stat_res.files
                                if p.endswith(".json") and "schema" not in p][0]
                               ).read_text(encoding="utf-8"))
    check("统计导出已脱敏",
          all(not (r.get("farmer") or {}).get("alias")
              and not (r.get("field") or {}).get("location")
              for r in stat_pkg["records"]))
    check("派生记录保留原始摘要可回溯",
          all(r["integrity"].get("source_record_sha256")
              and r["integrity"].get("derived") == "anonymized"
              for r in stat_pkg["records"]))

    ins_res = exporter.export(fmt="json", purpose="insurance")
    check("授权筛选包声明为子集", ins_res.chain_scope == "subset",
          f"scope={ins_res.chain_scope} excluded={ins_res.excluded_by_consent}")

    ins_pkg = json.loads(Path([p for p in ins_res.files
                               if p.endswith(".json") and "schema" not in p][0]
                              ).read_text(encoding="utf-8"))
    check("保险导出仅含授权记录",
          all(r["consent"]["share_insurance"] for r in ins_pkg["records"]))

    outbox = Outbox(tmp / "outbox")
    adapters = [InsuranceAdapter(outbox=outbox), SubsidyAdapter(outbox=outbox),
                AgriSupplyAdapter(outbox=outbox)]
    for adapter in adapters:
        adapter.device_id = "smoke-device-01"
        adapter.app_version = "0.1.0"
        res = adapter.dispatch(store)
        check(f"对接 {adapter.name} 生成", res.ok, f"errors={res.errors[:2]}")
        if not res.ok:
            continue
        doc = json.loads(Path(res.files[0]).read_text(encoding="utf-8"))
        vok, verr = validate(doc, adapter.schema)
        check(f"对接 {adapter.name} 符合 schema", vok,
              json.dumps(verr[:3], ensure_ascii=False))
        print(f"       summary={json.dumps(res.summary, ensure_ascii=False)[:200]}")

    empty = adapters[0].dispatch(store, filters={"village": "不存在村"})
    check("无记录时不生成文件", not empty.ok and bool(empty.errors))

    supply_entries = [e for e in outbox.entries("pending") if e.adapter == "agri_supply"]
    blob = json.dumps(json.loads(Path(supply_entries[0].path).read_text(encoding="utf-8")),
                      ensure_ascii=False)
    check("农资报表已聚合脱敏",
          "农户1" not in blob and "138****5678" not in blob and '"lat"' not in blob)

    store.update_consent(rid, share_supplier=True, actor="农户本人")
    reco = adapters[2].recommend(store.get(rid))
    check("本机农资推荐可用", reco["local_only"] and reco["agro_input_category"] != "",
          f"category={reco['agro_input_category']} urgency={reco['urgency']}")

    pending = outbox.entries("pending")
    check("outbox pending 计数", len(pending) == 3, f"{len(pending)}")
    moved = outbox.ack(pending[0].id, "已交保险系统")
    check("outbox 状态迁移", moved is not None and moved.state == "sent")
    doc = json.loads(Path(moved.path).read_text(encoding="utf-8"))
    check("迁移未改写 payload",
          doc["integrity"]["payload_sha256"] == _sha256(doc["payload"]))
    check("sent 目录可见", len(outbox.entries("sent")) == 1)
    flush = outbox.flush(tmp / "usb", channel=pending[1].channel)
    check("outbox flush 到 U 盘目录", flush["ok"] and flush["count"] >= 1,
          str(flush["errors"]))

    registry.configure(device_id="smoke-device-01", app_version="0.1.0")
    schemas = registry.write_schemas(tmp / "schemas")
    check("registry 导出三份 schema", len(schemas) == 3, str([p.name for p in schemas]))
    check("registry.describe", len(registry.describe()) == 3)
    check("registry.preview 不落盘", registry.preview("insurance", store, limit=3)["ok"])

    store.close()
    print(f"\n临时目录：{tmp}")
    print("结果：" + ("全部通过" if state["ok"] else "存在失败项"))
    return 0 if state["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
