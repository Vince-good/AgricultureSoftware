"""Flask 应用：把识别核心、记录库、语音、导出对接包成本地 Web 服务。

设计约束（都来自文档）：
  - **全离线**：不引用任何 CDN、字体服务或统计脚本，断网可用（3.1 节）；
  - **三步流程**：界面只有拍照 → 识别 → 播报，其余功能收进"记录/设置"两页（3.3 节）；
  - **界面与核心解耦**：这里只做 HTTP 编解码，识别逻辑一行不写（3.4(3)）；
  - **如实告知**：方言语音缺失、模型未安装、置信度不足，都返回明确字段，
    前端必须显示出来，不能假装成功。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from flask import Flask, jsonify, request, send_file, send_from_directory

from ..config import PATHS, UI_MIN_FONT_PX, UI_TOUCH_TARGET_PX
from ..i18n import DIALECT_REVIEW_NEEDED, coverage, table
from .state import AppState, EngineUnavailable

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_MB = 24
_IMAGE_OK = ("image/jpeg", "image/png", "image/webp", "image/bmp")


class ApiError(Exception):
    """带 HTTP 状态码的业务错误。前端拿到的永远是一句能念给用户听的话。"""

    def __init__(self, message: str, status: int = 400, code: str = "bad_request",
                 **extra: Any) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.extra = extra


# --------------------------------------------------------------------------
# 请求解析
# --------------------------------------------------------------------------

def _options() -> Dict[str, Any]:
    """把 query / form / json 三种传参方式合并成一份，前端怎么方便怎么来。"""
    out: Dict[str, Any] = dict(request.args.items())
    if request.form:
        out.update(request.form.to_dict())
    if request.is_json:
        body = request.get_json(silent=True) or {}
        if isinstance(body, dict):
            out.update(body)
    return out


def _read_upload() -> Tuple[bytes, str]:
    """取出上传图片字节。支持 multipart 的 file 字段和裸 body 两种上传。"""
    if request.files:
        upload = request.files.get("file") or next(iter(request.files.values()), None)
        if upload is None:
            raise ApiError("没有收到图片", 400, "no_file")
        data = upload.read()
        name = upload.filename or "photo.jpg"
        mime = upload.mimetype or ""
    else:
        data = request.get_data(cache=False) or b""
        name = str(request.args.get("filename") or "photo.jpg")
        mime = request.content_type or ""
    if not data:
        raise ApiError("图片是空的，请重拍", 400, "empty_file")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ApiError(f"图片超过 {MAX_UPLOAD_MB}MB，请重拍或压缩", 413, "too_large")
    if mime and mime.split(";")[0].strip() not in _IMAGE_OK:
        raise ApiError("只支持 JPEG/PNG/WebP/BMP 图片", 415, "bad_type")
    return data, name


def _as_bool(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on", "是")


def _lang(state: AppState, opts: Dict[str, Any]) -> str:
    from ..tts.languages import is_supported

    code = str(opts.get("lang") or state.language or "zh")
    return code if is_supported(code) else "zh"


def _voice_payload(state: AppState, result: Dict[str, Any], lang: str) -> Dict[str, Any]:
    """识别结果 → 播报语音。同时返回元信息，让前端能如实说明降级情况。"""
    clip = state.synth.clip_for_result(result, lang)
    payload: Dict[str, Any] = {
        "ok": bool(clip.ok and clip.exists),
        "text": clip.text,
        "slug": clip.slug,
        "source": clip.source,
        "language": clip.language,
        "requested": clip.requested,
        "degraded": bool(clip.degraded),
        "duration_s": round(float(clip.duration_s or 0.0), 2),
        "error": clip.error or None,
    }
    if payload["ok"]:
        payload["url"] = f"/api/voice?slug={clip.slug}&lang={lang}"
    return payload


def _record_brief(row: Dict[str, Any]) -> Dict[str, Any]:
    """记录列表项：只给列表渲染需要的字段，避免把整份 payload 塞进首屏。"""
    return {
        "record_id": row.get("record_id"),
        "created_at": row.get("created_at"),
        "created_at_unix": row.get("created_at_unix"),
        "class_id": row.get("class_id"),
        "class_name": row.get("class_name"),
        "crop": row.get("crop"),
        "stress": row.get("stress"),
        "severity": row.get("severity"),
        "severity_score": row.get("severity_score"),
        "confidence": row.get("confidence"),
        "needs_retake": bool(row.get("needs_retake")),
        "village": row.get("village"),
        "plot_id": row.get("plot_id"),
        "image_ref": row.get("image_ref"),
        "has_image": bool(row.get("image_ref")),
    }


def _safe_export_path(name: str) -> Path:
    """导出文件下载：必须落在导出目录内，杜绝 ../ 穿越。

    导出件按 `exports/<export_id>/...` 分目录存放，所以这里接受
    一层或多层相对路径，但解析后必须仍在导出根目录之内。
    """
    root = PATHS.exports.resolve()
    rel = Path(str(name).replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise ApiError("找不到这个导出文件", 404, "not_found")
    target = (root / rel).resolve()
    if root not in target.parents or not target.is_file():
        raise ApiError("找不到这个导出文件", 404, "not_found")
    return target


# --------------------------------------------------------------------------
# 应用工厂
# --------------------------------------------------------------------------

def create_app(bundle: Optional[Path | str] = None, language: str = "zh",
               host: str = "127.0.0.1", port: int = 8765,
               backend: Optional[str] = None, threads: int = 1,
               warm: bool = True) -> Flask:
    """创建 Flask 应用。`bundle` 为空时自动取 artifacts/bundles 下最新的模型包。"""
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    app.config["JSON_AS_ASCII"] = False
    app.config["HEYAN_HOST"] = host
    app.config["HEYAN_PORT"] = int(port)

    state = AppState(bundle=bundle, language=language, backend=backend, threads=threads)
    app.extensions["heyan"] = state

    # ---------------- 静态资源 ----------------
    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/sw.js")
    def service_worker():
        # 必须挂在根路径，作用域才能覆盖整个站点
        resp = send_from_directory(STATIC_DIR, "sw.js")
        resp.headers["Content-Type"] = "application/javascript; charset=utf-8"
        resp.headers["Service-Worker-Allowed"] = "/"
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.get("/manifest.webmanifest")
    def manifest():
        resp = send_from_directory(STATIC_DIR, "manifest.webmanifest")
        resp.headers["Content-Type"] = "application/manifest+json; charset=utf-8"
        return resp

    @app.get("/icons/<path:name>")
    def icons(name: str):
        return send_from_directory(STATIC_DIR / "icons", name)

    # ---------------- 首屏 ----------------
    @app.get("/api/bootstrap")
    def api_bootstrap():
        opts = _options()
        lang = _lang(state, opts)
        from ..tts import languages

        try:
            voice = state.synth.status()
        except Exception as exc:  # noqa: BLE001 - 语音不可用不该拖垮整个界面
            voice = {"engine": "unavailable", "engine_available": False,
                     "voiced_languages": [], "error": f"{type(exc).__name__}: {exc}"}
        payload: Dict[str, Any] = {
            "app": {
                "name": "禾眼", "name_en": "HeYan", "version": __version(),
                "region": "粤东西北",
                "ui": {"min_font_px": UI_MIN_FONT_PX,
                       "touch_target_px": UI_TOUCH_TARGET_PX},
                "flow": ["capture", "recognize", "listen"],
            },
            "language": lang,
            "i18n": table(lang),
            "languages": languages.describe(),
            "dialect_review_needed": list(DIALECT_REVIEW_NEEDED),
            "i18n_coverage": coverage(),
            "model": state.engine_status(),
            "benchmark": state.benchmark_summary(),
            "classes": state.classes(),
            "severities": state.severities(),
            "voice": voice,
            "settings": state.settings,
            "profile": state.profile,
            "records_total": state.record_count(),
            "paths": {"records_db": str(PATHS.db), "exports": str(PATHS.exports),
                      "outbox": str(PATHS.outbox)},
            "server_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return jsonify(payload)

    @app.get("/api/health")
    def api_health():
        return jsonify({
            "ok": True, "engine": state.engine_status()["available"],
            "bundle": str(state.bundle_path) if state.bundle_path else None,
            "records": state.record_count(), "uptime_s": round(time.time() - state.started_at, 1),
        })

    # ---------------- 识别 ----------------
    @app.post("/api/recognize")
    def api_recognize():
        from ..core.preprocess import read_image_bytes
        from ..data.schema import make_record_id

        opts = _options()
        lang = _lang(state, opts)
        data, _filename = _read_upload()

        try:
            img = read_image_bytes(data)
        except Exception as exc:  # noqa: BLE001 - 损坏的图片要给"请重拍"，不是 500
            raise ApiError("图片读不出来，请重拍一次", 400, "decode_failed",
                           detail=f"{type(exc).__name__}: {exc}") from exc
        if img.size == 0 or min(img.shape[:2]) < 8:
            raise ApiError("图片太小，请靠近叶片重拍", 400, "too_small")

        try:
            result = state.recognize(img, language=lang)
        except EngineUnavailable as exc:
            raise ApiError("还没有安装识别模型，请先在电脑上构建模型包", 503,
                           "no_model", detail=str(exc)) from exc
        payload = result.to_dict()
        payload["language"] = lang

        record_id: Optional[str] = None
        if _as_bool(opts.get("save"), state.settings.get("save_records", True)):
            profile = state.profile
            record_id = make_record_id(payload["created_at"], payload["image_id"])
            rel, digest, width, height = state.store.save_image(data, record_id)
            # 交换格式里村址属于农户档案、地块属于田块（见 data/schema.py），
            # 写错位置会让记录库的 village/farmer_id 列永远是空的。
            farmer = {
                "id": profile.get("farmer_id") or None,
                "alias": profile.get("alias") or None,
                "village": profile.get("village") or None,
                "town": profile.get("town") or None,
                "county": profile.get("county") or None,
                "region": profile.get("region") or None,
            }
            field = {"plot_id": profile.get("plot_id") or None}
            farmer = {k: v for k, v in farmer.items() if v}
            field = {k: v for k, v in field.items() if v}
            record = state.store.add_result(
                payload, record_id=record_id, image_ref=rel, image_sha256=digest,
                farmer=farmer or None, field=field or None,
                observation={"image_width": width, "image_height": height,
                             "source": "web_capture"},
            )
            payload["record_id"] = record.get("record_id")
            payload["image_url"] = f"/api/records/{record_id}/image"
        else:
            payload["record_id"] = None

        payload["voice"] = _voice_payload(state, payload, lang)
        return jsonify({"ok": True, "result": payload})

    # ---------------- 语音 ----------------
    @app.get("/api/voice")
    def api_voice():
        opts = _options()
        lang = _lang(state, opts)
        slug = str(opts.get("slug") or "")
        text = str(opts.get("text") or "")
        record_id = str(opts.get("record_id") or "")

        if record_id:
            record = state.store.get(record_id)
            if record is None:
                raise ApiError("找不到这条记录", 404, "not_found")
            clip = state.synth.clip_for_result(_record_as_result(record), lang)
        elif slug:
            clip = state.synth.clip_for_slug(slug, lang)
        elif text:
            clip = state.synth.clip(text, lang)
        else:
            raise ApiError("要播什么？请给 slug / text / record_id", 400, "missing_target")

        if not clip.ok or not clip.exists:
            return jsonify({"ok": False, "reason": clip.error or "no_voice",
                            "text": clip.text, "slug": clip.slug, "language": lang}), 404
        resp = send_file(str(clip.path), mimetype="audio/wav", conditional=True)
        resp.headers["X-HeYan-Source"] = clip.source
        resp.headers["X-HeYan-Language"] = clip.language
        resp.headers["X-HeYan-Degraded"] = "1" if clip.degraded else "0"
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp

    @app.get("/api/voice/status")
    def api_voice_status():
        return jsonify({"ok": True, **state.synth.status()})

    # ---------------- 记录 ----------------
    @app.get("/api/records")
    def api_records():
        opts = _options()
        limit = max(1, min(int(opts.get("limit") or 40), 200))
        offset = max(0, int(opts.get("offset") or 0))
        order = "asc" if str(opts.get("order", "desc")).lower() == "asc" else "desc"
        filters: Dict[str, Any] = {}
        for key in ("class_id", "crop", "stress", "severity", "village", "farmer_id"):
            if opts.get(key):
                filters[key] = str(opts[key])
        if opts.get("q"):
            filters["query"] = str(opts["q"])
        if opts.get("needs_retake") in ("0", "1"):
            filters["needs_retake"] = opts["needs_retake"] == "1"
        if opts.get("since"):
            filters["since"] = float(opts["since"])
        if opts.get("until"):
            filters["until"] = float(opts["until"])
        rows = state.store.list(limit=limit, offset=offset, order=order, full=False, **filters)
        return jsonify({
            "ok": True, "items": [_record_brief(r) for r in rows],
            "total": state.store.count(**filters), "limit": limit, "offset": offset,
        })

    @app.get("/api/records/<record_id>")
    def api_record(record_id: str):
        record = state.store.get(record_id)
        if record is None:
            raise ApiError("找不到这条记录", 404, "not_found")
        lang = _lang(state, _options())
        out = dict(record)
        out["image_url"] = f"/api/records/{record_id}/image" if record.get(
            "observation", {}).get("image_ref") else None
        out["voice"] = _voice_payload(state, _record_as_result(record), lang)
        return jsonify({"ok": True, "record": out})

    @app.get("/api/records/<record_id>/image")
    def api_record_image(record_id: str):
        record = state.store.get(record_id)
        if record is None:
            raise ApiError("找不到这条记录", 404, "not_found")
        ref = (record.get("observation") or {}).get("image_ref")
        path = state.store.image_path(ref) if ref else None
        if path is None:
            raise ApiError("这条记录没有留照片", 404, "no_image")
        return send_file(str(path), mimetype="image/jpeg", max_age=3600)

    @app.delete("/api/records/<record_id>")
    def api_record_delete(record_id: str):
        ok = state.store.delete(record_id)
        if not ok:
            raise ApiError("找不到这条记录", 404, "not_found")
        return jsonify({"ok": True, "deleted": record_id,
                        "note": "删除会打断哈希链，完整性校验会如实报告"})

    @app.post("/api/records/<record_id>/followup")
    def api_record_followup(record_id: str):
        opts = _options()
        effect = opts.get("effect")
        record = state.store.update_followup(
            record_id,
            action_taken=str(opts.get("action_taken") or ""),
            product_used=str(opts.get("product_used") or ""),
            effect=str(effect) if effect not in (None, "") else None,
            reviewed_by=str(opts.get("reviewed_by") or ""),
            notes=str(opts.get("notes") or ""),
        )
        if record is None:
            raise ApiError("找不到这条记录", 404, "not_found")
        return jsonify({"ok": True, "record": record})

    @app.post("/api/records/<record_id>/consent")
    def api_record_consent(record_id: str):
        opts = _options()
        flags = {k: _as_bool(v, False) for k, v in opts.items()
                 if k.startswith("share_") and k in ("share_insurance", "share_subsidy",
                                                     "share_supplier", "share_research")}
        record = state.store.update_consent(record_id, actor=str(opts.get("actor") or ""),
                                           **flags)
        if record is None:
            raise ApiError("找不到这条记录", 404, "not_found")
        return jsonify({"ok": True, "consent": record.get("consent")})

    @app.get("/api/stats")
    def api_stats():
        return jsonify({"ok": True, "stats": state.store.stats()})

    @app.get("/api/integrity")
    def api_integrity():
        return jsonify({"ok": True, **state.store.verify_integrity()})

    # ---------------- 设置 ----------------
    @app.post("/api/settings")
    def api_settings():
        opts = _options()
        settings = state.update_settings({k: v for k, v in opts.items()
                                          if k not in ("profile",)})
        profile = state.profile
        if isinstance(opts.get("profile"), dict):
            profile = state.update_profile(opts["profile"])
        return jsonify({"ok": True, "settings": settings, "profile": profile,
                        "i18n": table(str(settings["language"]))})

    # ---------------- 导出与对接 ----------------
    @app.get("/api/devices")
    def api_devices():
        from ..data import detect_bluetooth, find_usb_targets

        return jsonify({"ok": True, "usb": find_usb_targets(),
                        "bluetooth": detect_bluetooth()})

    @app.post("/api/export")
    def api_export():
        from ..data import Exporter

        opts = _options()
        fmt = str(opts.get("fmt") or "json")
        if fmt not in ("json", "csv", "both"):
            raise ApiError("导出格式只能是 json / csv / both", 400, "bad_format")
        purpose = str(opts.get("purpose") or "statistics")
        channel = str(opts.get("channel") or "local")
        if channel not in ("local", "usb", "bluetooth", "download"):
            raise ApiError("导出通道只能是 local / usb / bluetooth / download",
                           400, "bad_channel")
        filters = opts.get("filters") if isinstance(opts.get("filters"), dict) else None
        exporter = Exporter(state.store)
        result = exporter.export(
            fmt=fmt, purpose=purpose, channel=channel, filters=filters,
            anonymize=opts.get("anonymize"), usb_target=opts.get("usb_target"),
        )
        payload = result.to_dict()
        payload["ok"] = bool(result.ok)
        # 无论走哪个通道，本机都留了一份；给个下载链接，方便农技站直接取走。
        # 用相对导出根的路径，保住 <export_id> 这一层目录。
        primary = _primary_export_file(result.files, fmt)
        if primary:
            from urllib.parse import quote

            rel = Path(primary).resolve().relative_to(PATHS.exports.resolve())
            payload["download_url"] = (f"/api/export/file?name={quote(rel.as_posix())}")
        else:
            payload["download_url"] = None
        return jsonify(payload), (200 if result.ok else 500)

    @app.get("/api/export/file")
    def api_export_file():
        path = _safe_export_path(str(request.args.get("name") or ""))
        return send_file(str(path), as_attachment=True, download_name=path.name)

    @app.get("/api/adapters")
    def api_adapters():
        from ..data import registry

        return jsonify({"ok": True, "adapters": registry.describe(),
                        "status": registry.status()})

    @app.post("/api/adapters/dispatch")
    def api_adapters_dispatch():
        from ..data import registry

        opts = _options()
        raw = opts.get("names")
        names = [n.strip() for n in raw.split(",")] if isinstance(raw, str) and raw else None
        results = registry.dispatch_all(state.store, names)
        return jsonify({"ok": True, "results": [r.to_dict() for r in results]})

    @app.get("/api/schemas")
    def api_schemas():
        from ..data import registry
        from ..data.schema import EXPORT_SCHEMA, RECORD_SCHEMA, schema_summary

        return jsonify({
            "ok": True,
            "record": schema_summary(RECORD_SCHEMA),
            "export": schema_summary(EXPORT_SCHEMA),
            "adapters": registry.describe(),
        })

    # ---------------- 错误处理 ----------------
    @app.errorhandler(ApiError)
    def _on_api_error(exc: ApiError):
        return jsonify({"ok": False, "code": exc.code, "message": exc.message,
                        **exc.extra}), exc.status

    @app.errorhandler(413)
    def _on_too_large(_exc):
        return jsonify({"ok": False, "code": "too_large",
                        "message": f"图片超过 {MAX_UPLOAD_MB}MB，请重拍或压缩"}), 413

    @app.errorhandler(EngineUnavailable)
    def _on_engine(exc: EngineUnavailable):
        return jsonify({"ok": False, "code": "no_model",
                        "message": "还没有安装识别模型，请先在电脑上构建模型包",
                        "detail": str(exc)}), 503

    @app.errorhandler(404)
    def _on_404(_exc):
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "code": "not_found", "message": "没有这个接口"}), 404
        return send_from_directory(STATIC_DIR, "index.html")

    @app.errorhandler(Exception)
    def _on_error(exc: Exception):
        app.logger.exception("unhandled error")
        return jsonify({"ok": False, "code": "internal_error",
                        "message": "识别失败，请再试一次",
                        "detail": f"{type(exc).__name__}: {exc}"}), 500

    if warm:
        state.warm()
    return app


# --------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------

def _record_as_result(record: Dict[str, Any]) -> Dict[str, Any]:
    """库里的交换格式记录 → 播报所需的精简结果结构。"""
    diag = record.get("diagnosis") or {}
    advice = record.get("advice") or {}
    severity = diag.get("severity") or {}
    return {
        "class_id": diag.get("class_id", ""),
        "needs_retake": bool(diag.get("needs_retake")),
        "confidence": diag.get("confidence", 0.0),
        "severity": {"id": severity.get("id", "none"), "score": severity.get("score", 0.0)},
        "advice": {
            "class_id": diag.get("class_id", ""),
            "name": diag.get("class_name", ""),
            "voice": advice.get("voice_text", ""),
            "severity": {"id": severity.get("id", "none")},
        },
    }


def _primary_export_file(files: Any, fmt: str) -> Optional[str]:
    files = [Path(f) for f in (files or [])]
    if not files:
        return None
    for suffix in (f".{fmt}", ".json", ".csv", ".jsonl"):
        for path in files:
            if path.name.endswith(suffix):
                return str(path)
    return str(files[0])


def __version() -> str:
    try:
        from .. import __version__ as v

        return str(v)
    except Exception:  # noqa: BLE001
        return "1.0.0"


def run_server(app: Flask, host: str = "127.0.0.1", port: int = 8765,
               debug: bool = False) -> None:
    """启动服务。默认只监听回环：田间设备上不该对外开端口。"""
    state: AppState = app.extensions.get("heyan")
    engine = state.engine_status()
    port = _bind_port(app, host, port)
    print(f"[heyan] 界面地址 http://{host}:{port}")
    if engine["available"]:
        print(f"[heyan] 模型包 {engine['bundle_dir']}")
    else:
        print("[heyan] 警告：没有找到模型包，界面会提示先运行 `heyan build`")
    app.run(host=host, port=int(port), debug=bool(debug), threaded=True, use_reloader=False)


def _bind_port(app: Flask, host: str, port: int, tries: int = 10) -> int:
    """端口被占就顺延。

    田间电脑上常驻软件（输入法、网盘、打印机服务）经常悄悄占端口，
    直接崩掉只会让农技员以为软件坏了；顺延并打印真实地址更诚实。
    Windows 上被占用可能报 10048，也可能报 10013（权限式拒绝），都算占用。
    """
    import errno
    import socket

    for offset in range(tries):
        candidate = int(port) + offset
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, candidate))
        except OSError as exc:
            probe.close()
            if exc.errno in (errno.EACCES, errno.EADDRINUSE, 10013, 10048) or \
                    getattr(exc, "winerror", 0) in (10013, 10048):
                if offset == 0:
                    print(f"[heyan] 端口 {candidate} 被占用，尝试顺延…")
                continue
            raise
        probe.close()
        if candidate != int(port):
            print(f"[heyan] 改用端口 {candidate}")
        app.config["HEYAN_PORT"] = candidate
        return candidate
    raise OSError(f"端口 {port}~{port + tries - 1} 都被占用，请用 --port 指定其它端口")
