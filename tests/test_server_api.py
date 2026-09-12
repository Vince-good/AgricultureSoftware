"""Web 服务接口契约测试。

用 Flask test client 跑，不占端口、不起线程池；
断言的都是界面前端真正依赖的字段，改坏任何一个前端就会白屏或念错话。
"""

from __future__ import annotations

import io
import json


def test_static_shell(client):
    for path, marker in (("/", "禾眼"), ("/sw.js", "PRECACHE"),
                         ("/manifest.webmanifest", "start_url")):
        resp = client.get(path)
        assert resp.status_code == 200, path
        assert marker in resp.get_data(as_text=True), path
    resp = client.get("/icons/icon-192.png")
    assert resp.status_code == 200
    assert resp.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_bootstrap_contract(client):
    body = client.get("/api/bootstrap").get_json()
    assert body["app"]["flow"] == ["capture", "recognize", "listen"]
    assert body["app"]["ui"]["min_font_px"] >= 20
    assert body["app"]["ui"]["touch_target_px"] >= 72
    assert body["model"]["available"] is True
    assert body["benchmark"]["budget_ok"] is True
    assert len(body["classes"]) == 14
    assert {c["id"] for c in body["classes"]} >= {"rice_blast", "unusable"}
    # 五种语言的话术表必须齐键，界面不允许出现裸 key
    for key in ("flip_camera", "sys_no_model", "record_detail"):
        assert key in body["i18n"], key


def test_recognize_and_voice(client, demo_image):
    data = {"file": (io.BytesIO(demo_image.read_bytes()), "leaf.jpg")}
    resp = client.post("/api/recognize?lang=zh", data=data,
                       content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    result = body["result"]
    assert body["ok"] is True
    assert result["class_id"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["severity"]["id"] in ("none", "mild", "moderate", "severe")
    assert result["record_id"]
    # 语音包已预置：必须命中 voicepack，而不是运行期合成
    voice = result["voice"]
    assert voice["ok"] is True
    assert voice["source"] == "voicepack"
    assert voice["url"].startswith("/api/voice?slug=")

    wav = client.get(voice["url"])
    assert wav.status_code == 200
    assert wav.headers["X-HeYan-Source"] == "voicepack"
    assert wav.data[:4] == b"RIFF"

    # 粤语本机无语音包：如实回退普通话并标记 degraded，而不是假装会念粤语
    yue = client.get(voice["url"].replace("lang=zh", "lang=yue"))
    assert yue.status_code == 200
    assert yue.headers["X-HeYan-Degraded"] == "1"
    assert yue.headers["X-HeYan-Language"] == "zh"


def test_recognize_rejects_garbage(client):
    data = {"file": (io.BytesIO(b"not an image at all"), "x.jpg")}
    resp = client.post("/api/recognize", data=data,
                       content_type="multipart/form-data")
    assert resp.status_code == 400
    assert resp.get_json()["code"] == "decode_failed"


def test_records_crud_and_image(client, demo_image):
    data = {"file": (io.BytesIO(demo_image.read_bytes()), "leaf.jpg")}
    record_id = client.post(
        "/api/recognize", data=data,
        content_type="multipart/form-data").get_json()["result"]["record_id"]

    listing = client.get("/api/records?limit=5").get_json()
    assert listing["total"] >= 1
    assert listing["items"][0]["record_id"] == record_id

    detail = client.get(f"/api/records/{record_id}").get_json()["record"]
    assert detail["diagnosis"]["class_id"]
    assert detail["observation"]["image_ref"]

    img = client.get(f"/api/records/{record_id}/image")
    assert img.status_code == 200 and img.data[:3] == b"\xff\xd8\xff"

    follow = client.post(f"/api/records/{record_id}/followup",
                         json={"action_taken": "喷了三环唑", "effect": "better"})
    assert follow.get_json()["record"]["followup"]["effect"] == "better"

    consent = client.post(f"/api/records/{record_id}/consent",
                          json={"share_insurance": True})
    assert consent.get_json()["consent"]["share_insurance"] is True

    deleted = client.delete(f"/api/records/{record_id}")
    assert deleted.status_code == 200
    assert client.get(f"/api/records/{record_id}").status_code == 404


def test_export_and_integrity(client, demo_image):
    data = {"file": (io.BytesIO(demo_image.read_bytes()), "leaf.jpg")}
    client.post("/api/recognize", data=data, content_type="multipart/form-data")

    resp = client.post("/api/export",
                       json={"fmt": "both", "purpose": "insurance",
                             "channel": "local", "anonymize": True})
    body = resp.get_json()
    assert resp.status_code == 200 and body["ok"] is True
    assert body["download_url"]
    dl = client.get(body["download_url"])
    assert dl.status_code == 200 and dl.data

    # 前面的用例删过记录，链在这里应当如实报"断"，而不是假装完整
    integrity = client.get("/api/integrity").get_json()
    assert {"ok", "checked", "problems", "head_sha256"} <= set(integrity)
    assert integrity["ok"] is False
    assert any("chain broken" in p["problem"] for p in integrity["problems"])


def test_export_traversal_guard(client):
    resp = client.get("/api/export/file?name=..%2F..%2Frecords%2Fheyan.db")
    assert resp.status_code == 404


def test_settings_roundtrip(client):
    resp = client.post("/api/settings",
                       json={"text_size": "huge", "voice_speed": "slow",
                             "profile": {"village": "测试村", "alias": "老李"}})
    body = resp.get_json()
    assert body["settings"]["text_size"] == "huge"
    assert body["profile"]["village"] == "测试村"
    # 非法语言不允许写进设置
    body = client.post("/api/settings", json={"language": "xx"}).get_json()
    assert body["settings"]["language"] != "xx"


def test_adapters_and_schemas(client):
    adapters = client.get("/api/adapters").get_json()
    names = {a["name"] for a in adapters["adapters"]}
    assert {"insurance", "subsidy", "agri_supply"} <= names
    schemas = client.get("/api/schemas").get_json()
    assert schemas["record"]["id"].endswith("record-1.0.json")
    assert schemas["export"]["field_count"] > 0
    assert "record_id" in schemas["record"]["required"]


def test_unknown_api_is_json_404(client):
    resp = client.get("/api/nope")
    assert resp.status_code == 404
    assert resp.get_json()["code"] == "not_found"


def test_spa_fallback_for_non_api(client):
    resp = client.get("/some/deep/route")
    assert resp.status_code == 200
    assert "禾眼" in resp.get_data(as_text=True)
