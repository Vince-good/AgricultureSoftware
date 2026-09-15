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


def test_public_mode_trusts_proxy_scheme_so_redirects_stay_https():
    """隧道在服务商那边终止 TLS，不认 X-Forwarded-Proto 就会把人从 https 甩回 http。"""
    client = _mini_app(token=TOKEN, public=True).test_client()
    resp = client.get(f"/?{guard.TOKEN_QUERY}={TOKEN}",
                      headers={**HTML, "X-Forwarded-Proto": "https",
                               "X-Forwarded-For": "203.0.113.7"})
    assert resp.status_code == 302
    assert resp.headers["Location"].startswith("https://"), "跳转必须留在 https"
    assert guard.TOKEN_QUERY not in resp.headers["Location"], "口令仍要从地址里摘掉"
    assert "secure" in resp.headers["Set-Cookie"].lower(), "https 下 cookie 要带 Secure"


def test_non_public_mode_ignores_spoofed_proxy_scheme():
    """回环/局域网直连时不该信代理头，否则一个头就能篡改跳转和 cookie 属性。"""
    client = _mini_app(token=TOKEN).test_client()
    resp = client.get(f"/?{guard.TOKEN_QUERY}={TOKEN}",
                      headers={**HTML, "X-Forwarded-Proto": "https"})
    assert resp.status_code == 302
    assert resp.headers["Location"].startswith("http://")
    assert "secure" not in resp.headers["Set-Cookie"].lower()


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


CLOUDFLARED_LOG = '''2026-09-15T10:11:12Z INF Thank you for trying Cloudflare Tunnel. Report issues at https://github.com/cloudflare/cloudflared/issues/new
2026-09-15T10:11:12Z INF See https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/
2026-09-15T10:11:13Z INF Your quick Tunnel has been created! Visit it at (it may take some time to be reachable, it is normal): https://crop-leaf-demo.trycloudflare.com
2026-09-15T10:11:13Z INF Registered tunnel connection connIndex=0
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


def test_parse_public_urls_cloudflared_needs_an_exact_match():
    urls = tunnel.parse_public_urls(CLOUDFLARED_LOG, "cloudflared")
    assert urls == ["https://crop-leaf-demo.trycloudflare.com"]

    # 不点名客户端时黑名单滤不掉 GitHub/文档链接 —— 这正是白名单存在的理由
    loose = tunnel.parse_public_urls(CLOUDFLARED_LOG)
    assert any("github.com" in u or "developers.cloudflare.com" in u for u in loose)


def test_cloudflared_api_host_is_never_handed_out_as_a_share_url():
    """实测踩到的坑：隧道没建起来时，日志里唯一的 https 地址是 api.trycloudflare.com。

    分享链接是带着访问口令的，认错域名等于把口令送给别人，所以这里必须认不出来，
    让 run() 走失败分支打印日志，而不是高高兴兴发一个错链接。
    """
    failed = ('2026-09-15T13:57:05Z INF Requesting new quick Tunnel on trycloudflare.com...\n'
              'failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel": '
              'context deadline exceeded (Client.Timeout exceeded while awaiting headers)\n')
    assert tunnel.parse_public_urls(failed, "cloudflared") == []
    assert tunnel.parse_public_urls(failed) == [], "不点名客户端时黑名单也得拦住 API 端点"
    assert tunnel.failure_hint(failed) == "failed to request quick tunnel"


def test_failure_hint_spots_ngrok_missing_authtoken():
    assert tunnel.failure_hint("ERR_NGROK_105: Your account is not configured") == "err_ngrok_"
    assert tunnel.failure_hint("INF Registered tunnel connection connIndex=0") is None
    assert tunnel.failure_hint(None) is None


def test_clients_order_prefers_the_zero_signup_one():
    assert tunnel.CLIENTS[0] == "cloudflared", "免注册的那家应该先被自动探测到"
    assert set(tunnel.CLIENTS) == {"cloudflared", "ngrok", "cpolar"}
    assert tunnel.INSTALL_HINTS["cloudflared"], "每家都得给安装指引"


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


def test_build_command_cloudflared_quick_tunnel():
    cmd = tunnel.build_command("cloudflared", "cloudflared.exe", 8091)
    assert cmd[1] == "tunnel"
    assert "--url" in cmd and "http://127.0.0.1:8091" in cmd, "必须指回环，别把服务敞到网卡上"
    assert "--no-autoupdate" in cmd, "跑着的隧道不该被自动更新打断"

    with_region = tunnel.build_command("cloudflared", "cloudflared.exe", 8091, region="ap")
    assert "--region" in with_region and "ap" in with_region

    with pytest.raises(ValueError, match="随机域名"):
        tunnel.build_command("cloudflared", "cloudflared.exe", 8091, subdomain="heyan")


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
