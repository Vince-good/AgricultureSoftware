"""公网暴露这条路的契约测试：口令闸门、限流、响应头、隧道客户端接线。

这里不去加载模型（那由 conftest 的 app 夹具负责，只在最后一个测试里用一下），
其余测试都拿一个只注册了三个假路由的迷你 Flask app 来验闸门本身的行为 ——
闸门跟识别逻辑无关，跑得快才好反复跑。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from flask import Flask, jsonify

from heyan.server import guard, tunnel

TOKEN = "test-pass-123"
HTML = {"Accept": "text/html"}


def _mini_app(token=None, public=False, limits=None) -> Flask:
    app = Flask(__name__)
    guard.install(app, token=token, public=public, limits=limits)

    @app.get("/")
    def index():
        return "UI-OK"

    @app.get("/sw.js")
    def sw():
        return "SW-OK"

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True})

    @app.post("/api/recognize")
    def recognize():
        return jsonify({"ok": True})

    @app.get("/api/records")
    def records():
        return jsonify({"ok": True, "records": []})

    return app


# ------------------------------------------------------------------ 闸门
def test_loopback_stays_open_and_unhardened():
    """不设口令时行为必须和以前一模一样：老部署方式不能被这次改动弄坏。"""
    client = _mini_app().test_client()
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert "X-Frame-Options" not in resp.headers


def test_api_denied_without_token():
    client = _mini_app(token=TOKEN).test_client()
    resp = client.get("/api/health")
    assert resp.status_code == 401
    body = resp.get_json()
    assert body["ok"] is False
    assert body["code"] == "auth_required"
    assert body["message"], "错误信息要能直接念给用户听"


def test_api_accepts_header_token():
    client = _mini_app(token=TOKEN).test_client()
    resp = client.get("/api/health", headers={guard.TOKEN_HEADER: TOKEN})
    assert resp.status_code == 200


def test_query_token_sets_cookie_and_cleans_url():
    client = _mini_app(token=TOKEN).test_client()
    resp = client.get(f"/?{guard.TOKEN_QUERY}={TOKEN}&lang=yue", headers=HTML)
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/?lang=yue"), "口令要从地址里摘掉，其余参数保留"
    assert guard.TOKEN_COOKIE in resp.headers.get("Set-Cookie", "")
    cookie = resp.headers["Set-Cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie
    # 带上刚拿到的 cookie 再访问就该放行
    client.set_cookie(guard.TOKEN_COOKIE, TOKEN)
    assert client.get("/api/health").status_code == 200


def test_wrong_token_gets_gate_page_in_browser_but_json_for_api():
    app = _mini_app(token=TOKEN)
    client = app.test_client()
    page = client.get("/", headers=HTML)
    assert page.status_code == 401
    assert "访问口令" in page.get_data(as_text=True)
    assert page.headers["Content-Type"].startswith("text/html")
    api = client.get("/api/health")
    assert api.status_code == 401
    assert api.get_json()["code"] == "auth_required"


def test_static_asset_never_gets_html_gate():
    """把闸门页塞给 sw.js 只会换来一句看不懂的 MIME 报错，得给干净的 401。"""
    client = _mini_app(token=TOKEN).test_client()
    resp = client.get("/sw.js")
    assert resp.status_code == 401
    assert not resp.headers["Content-Type"].startswith("text/html")


def test_auth_form_round_trip():
    client = _mini_app(token=TOKEN).test_client()
    bad = client.post("/__auth", data={"token": "wrong"})
    assert bad.status_code == 401
    assert "口令不对" in bad.get_data(as_text=True)
    good = client.post("/__auth", data={"token": TOKEN})
    assert good.status_code == 302
    assert good.headers["Location"].endswith("/")
    assert guard.TOKEN_COOKIE in good.headers.get("Set-Cookie", "")


def test_generate_token_is_url_safe_and_unique():
    tokens = {guard.generate_token() for _ in range(20)}
    assert len(tokens) == 20
    for tok in tokens:
        assert len(tok) >= 12
        assert all(c.isalnum() or c in "-_" for c in tok), "要能在手机上顺手打出来"


# ------------------------------------------------------------------ 限流
def test_rate_limit_only_in_public_mode():
    limits = {"default": (3, 60.0), "recognize": (2, 60.0)}
    open_client = _mini_app(limits=limits).test_client()
    for _ in range(6):
        assert open_client.get("/api/health").status_code == 200

    client = _mini_app(public=True, limits=limits).test_client()
    for _ in range(3):
        assert client.get("/api/health").status_code == 200
    blocked = client.get("/api/health")
    assert blocked.status_code == 429
    assert blocked.get_json()["code"] == "rate_limited"
    assert blocked.get_json()["retry_after_s"] == 60
    assert blocked.headers["Retry-After"] == "60"


def test_rate_limit_buckets_are_independent():
    """识别的额度单独算：刷接口的人不该顺手把拍照识别也堵死。"""
    limits = {"default": (10, 60.0), "recognize": (1, 60.0)}
    client = _mini_app(public=True, limits=limits).test_client()
    assert client.post("/api/recognize").status_code == 200
    assert client.post("/api/recognize").status_code == 429
    assert client.get("/api/health").status_code == 200


def test_rate_limiter_window_slides():
    import time

    limiter = guard.RateLimiter({"default": (2, 0.08)})
    assert limiter.allow("a")
    assert limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b"), "不同来源不共用配额"
    time.sleep(0.1)  # 窗口滑过去之后配额要自己回来，不需要重启服务
    assert limiter.allow("a")


# ------------------------------------------------------------------ 加固头
def test_public_mode_hardens_responses():
    client = _mini_app(token=TOKEN, public=True).test_client()
    headers = {guard.TOKEN_HEADER: TOKEN}
    resp = client.get("/api/health", headers=headers)
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Referrer-Policy"] == "no-referrer"
    assert "noindex" in resp.headers["X-Robots-Tag"]


def test_records_are_never_cached_in_public_mode():
    client = _mini_app(token=TOKEN, public=True).test_client()
    resp = client.get("/api/records", headers={guard.TOKEN_HEADER: TOKEN})
    assert resp.headers["Cache-Control"] == "no-store"


# ------------------------------------------------------------------ 播报
def test_share_url_and_banner():
    assert guard.share_url("https://x.ngrok-free.app/", TOKEN) == \
        f"https://x.ngrok-free.app/?{guard.TOKEN_QUERY}={TOKEN}"
    assert guard.share_url("https://x.ngrok-free.app", None) == "https://x.ngrok-free.app"

    quiet = guard.banner_lines(TOKEN, public=False)
    assert quiet and "公网" not in quiet[0]

    lines = "\n".join(guard.banner_lines(TOKEN, public=True,
                                        share_urls=["https://x.ngrok-free.app"]))
    assert "警告：公网模式" in lines
    assert TOKEN in lines
    assert f"https://x.ngrok-free.app/?{guard.TOKEN_QUERY}={TOKEN}" in lines
    assert "Ctrl+C" in lines

    open_lines = "\n".join(guard.banner_lines(None, public=True))
    assert "对全网敞开" in open_lines, "没口令时必须把话说重"


# ------------------------------------------------------------------ 隧道
NGROK_LOG = '''t=2026-09-15T10:11:12+0800 lvl=info msg="starting web service" obj=web addr=127.0.0.1:4040
t=2026-09-15T10:11:13+0800 lvl=info msg="started tunnel" obj=tunnels name="command_line (8080)" addr=http://localhost:8080 url=https://a1b2-123-45.ngrok-free.app
t=2026-09-15T10:11:13+0800 lvl=info msg="docs and examples" url=https://ngrok.com/docs
'''

CPOLAR_LOG = '''2026/09/15 10:11:12 Init cpolar client with authtoken...
2026/09/15 10:11:13 Tunnel established at https://abcd1234.r6.cn
2026/09/15 10:11:13 See dashboard at https://dashboard.cpolar.com/get-started
'''


def test_parse_public_urls_ngrok():
    urls = tunnel.parse_public_urls(NGROK_LOG)
    assert urls == ["https://a1b2-123-45.ngrok-free.app"]


def test_parse_public_urls_cpolar_and_noise():
    urls = tunnel.parse_public_urls(CPOLAR_LOG)
    assert urls == ["https://abcd1234.r6.cn"], "官网/控制台链接不能当成隧道地址"


def test_parse_public_urls_dedupes_and_handles_empty():
    assert tunnel.parse_public_urls("") == []
    assert tunnel.parse_public_urls(None) == []
    twice = CPOLAR_LOG + CPOLAR_LOG
    assert tunnel.parse_public_urls(twice) == ["https://abcd1234.r6.cn"]


def test_build_command_matches_each_client_syntax():
    ngrok = tunnel.build_command("ngrok", "ngrok.exe", 8080)
    assert ngrok[:3] == ["ngrok.exe", "http", "8080"]
    assert "--log=stdout" in ngrok

    cpolar = tunnel.build_command("cpolar", "cpolar.exe", 8080,
                                   subdomain="heyan", region="cn")
    assert cpolar[:3] == ["cpolar.exe", "http", "8080"]
    assert "-log=stdout" in cpolar
    assert "-subdomain=heyan" in cpolar and "-region=cn" in cpolar

    fancy = tunnel.build_command("ngrok", "ngrok.exe", 9000, subdomain="heyan-abc",
                                 region="ap")
    assert "--domain" in fancy and "heyan-abc" in fancy and "--region" in fancy

    with pytest.raises(ValueError, match="不支持"):
        tunnel.build_command("frp", "frpc.exe", 8080)


def test_find_client_prefers_path_then_local_bin(monkeypatch, tmp_path):
    monkeypatch.setattr(tunnel.shutil, "which", lambda name: None)
    monkeypatch.setattr(tunnel, "PATHS", type("P", (), {"root": tmp_path})())
    assert tunnel.find_client() is None
    assert tunnel.find_client("ngrok") is None

    (tmp_path / "bin").mkdir()
    exe = tmp_path / "bin" / "cpolar.exe"
    exe.write_text("stub", encoding="utf-8")
    assert tunnel.find_client() == ("cpolar", str(exe))
    assert tunnel.find_client("ngrok") is None, "指定了 ngrok 就不能拿 cpolar 顶"

    monkeypatch.setattr(tunnel.shutil, "which",
                        lambda name: "C:/bin/ngrok.exe" if name == "ngrok" else None)
    assert tunnel.find_client() == ("ngrok", "C:/bin/ngrok.exe")


def test_server_command_forces_tunnel_mode():
    cmd = tunnel._server_command(8080, TOKEN, None, "zh")  # noqa: SLF001
    assert "--tunnel" in cmd, "隧道模式下服务必须只监听回环并强制鉴权"
    assert "--access-token" in cmd and TOKEN in cmd
    assert "127.0.0.1" not in cmd, "回环由 --tunnel 保证，别再写死一遍"
    with_bundle = tunnel._server_command(8080, TOKEN, "artifacts/bundles/x", "zh")
    assert "--bundle" in with_bundle  # noqa: SLF001


def test_fetch_ngrok_tunnels_reads_local_api():
    payload = json.dumps({"tunnels": [
        {"public_url": "http://a1b2.ngrok-free.app"},
        {"public_url": "https://a1b2.ngrok-free.app"},
        {"public_url": ""},
    ]}).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib 命名
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):  # 静音，别把测试输出弄脏
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        port = httpd.server_address[1]
        urls = tunnel.fetch_ngrok_tunnels(api_base=f"http://127.0.0.1:{port}", timeout=3)
        assert urls == ["https://a1b2.ngrok-free.app"], "只认 https，且要去重去空"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_fetch_ngrok_tunnels_swallows_absent_api():
    """没起 ngrok 时 4040 端口是空的，这里必须安静返回空列表而不是抛异常。"""
    assert tunnel.fetch_ngrok_tunnels(api_base="http://127.0.0.1:1", timeout=0.5) == []


# ------------------------------------------------------------------ 真服务接线
def test_health_reports_guard_status(client):
    """真实 app 上也要能看到闸门状态，运维排查时不用去猜。"""
    body = client.get("/api/health").get_json()
    assert body["ok"] is True
    assert body["guard"] == {"auth": False, "public": False, "rate_limited": False}
