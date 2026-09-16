"""对外暴露时的访问闸门：口令鉴权 + 限流 + 响应头收紧。

为什么必须有这一层：
  这套服务原本是"田间设备上只监听回环"的单机软件，接口一个都没设防 ——
  `/api/records` 直接返回带姓名、村、地块、农户编号的记录，`DELETE /api/records/<id>`
  能删证据，`POST /api/recognize` 能塞 24MB 图片，`/api/bootstrap` 还会把本机
  绝对路径交出去。一旦用内网穿透挂到公网，等于把这些无差别交给全网。
  所以 `--tunnel` 模式下口令是强制的，不给就自动生成一个并打印出来。

刻意不做的事：
  - 不做多用户/角色/审计。现场是"一台主机 + 几个信得过的人"，做账号体系
    只会让人把口令写在便利贴上，安全上更差；
  - 不上 CSP。界面大量使用内联脚本与 style，硬套 CSP 要么形同虚设
    （unsafe-inline）要么把功能弄坏，得不偿失；
  - 不做 HTTPS 终结。隧道客户端（ngrok/cpolar）已经给了公网 HTTPS，
    本机这一跳留在回环上明文反而更简单、也没有额外证书要管。
"""

from __future__ import annotations

import hmac
import secrets
import threading
import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode

from flask import Response, g, jsonify, make_response, redirect, request

TOKEN_COOKIE = "heyan_token"
TOKEN_HEADER = "X-HeYan-Token"
TOKEN_QUERY = "token"
# 口令有效期：一个田间工作日足够长，又不至于让链接被转发一周后还能用
TOKEN_TTL_S = 12 * 3600

# 每分钟配额：识别是 CPU 密集的（一次 7ms 推理 + 图片解码），公网上一旦被人
# 拿脚本刷，主机就没法给现场用了，所以单独给一档更紧的额度。
DEFAULT_LIMITS: Dict[str, Tuple[int, float]] = {
    "recognize": (12, 60.0),
    "default": (240, 60.0),
}


def generate_token(nbytes: int = 12) -> str:
    """生成一个手机上也好输入的口令（16 个 URL 安全字符，约 72 bit 熵）。"""
    return secrets.token_urlsafe(nbytes)


class RateLimiter:
    """滑动窗口限流，按"来源 + 类别"计数。

    不存任何个人信息，只存时间戳；窗口过期的记录在每次访问时顺手清掉，
    长期运行不会涨内存。
    """

    def __init__(self, limits: Optional[Dict[str, Tuple[int, float]]] = None) -> None:
        self._limits = dict(limits or DEFAULT_LIMITS)
        self._hits: Dict[str, deque] = {}
        self._lock = threading.Lock()

    @property
    def limits(self) -> Dict[str, Tuple[int, float]]:
        """当前生效的额度。启动横幅要如实报数，不能自己另算一份。"""
        return dict(self._limits)

    def allow(self, source: str, category: str = "default") -> bool:
        quota, window = self._limits.get(category) or self._limits["default"]
        now = time.monotonic()
        key = f"{category}:{source}"
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and now - bucket[0] > window:
                bucket.popleft()
            if len(bucket) >= quota:
                return False
            bucket.append(now)
            return True

    def retry_after(self, category: str = "default") -> int:
        _, window = self._limits.get(category) or self._limits["default"]
        return int(window)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


def client_source() -> str:
    """取"谁在请求"。

    隧道模式下服务只监听回环，唯一的对端就是 cloudflared/ngrok/cpolar 进程，
    所以 X-Forwarded-For 的第一跳是可信的真实客户端 IP；直连时退化成
    remote_addr。令牌也参与区分，避免同一出口 IP 下多人互相挤配额。
    """
    forwarded = request.headers.get("X-Forwarded-For", "")
    ip = (forwarded.split(",")[0].strip() if forwarded else "") or \
        (request.remote_addr or "unknown")
    token = request.cookies.get(TOKEN_COOKIE, "")
    return f"{ip}|{token[-6:]}" if token else ip


def request_category() -> str:
    return "recognize" if request.path == "/api/recognize" else "default"


GATE_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>禾眼 HeYan · 需要访问口令</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { margin: 0; min-height: 100vh; display: flex; align-items: center;
         justify-content: center; padding: 24px; background: #f4f6f4;
         font: 16px/1.6 system-ui, -apple-system, "Microsoft YaHei", sans-serif;
         color: #1d2a24; }
  .card { width: 100%; max-width: 420px; background: #fff; border: 1px solid #d8e0da;
          border-radius: 8px; padding: 24px; }
  h1 { margin: 0 0 4px; font-size: 20px; letter-spacing: 0; }
  p { margin: 0 0 16px; color: #52615a; font-size: 14px; }
  label { display: block; font-size: 13px; color: #52615a; margin-bottom: 6px; }
  input { width: 100%; padding: 12px; font-size: 16px; border: 1px solid #c3cec7;
          border-radius: 6px; background: #fbfcfb; }
  input:focus { outline: 2px solid #2f7d55; outline-offset: 1px; }
  button { margin-top: 16px; width: 100%; padding: 12px; font-size: 16px;
           border: 0; border-radius: 6px; background: #2f7d55; color: #fff; }
  button:active { background: #266645; }
  .err { margin: 12px 0 0; padding: 10px 12px; border-radius: 6px; font-size: 14px;
         background: #fdecec; color: #93261f; border: 1px solid #f3c9c6; }
  .hint { margin-top: 16px; font-size: 12px; color: #7b877f; }
</style>
</head>
<body>
  <form class="card" method="post" action="/__auth">
    <h1>禾眼 HeYan</h1>
    <p>这台主机通过内网穿透对外开放，需要口令才能进入。</p>
    <label for="token">访问口令</label>
    <input id="token" name="token" type="password" autocomplete="current-password"
           autofocus required>
    <button type="submit">进入</button>
    __ERROR__
    <p class="hint">口令由主机持有人给出，12 小时内有效。链接请勿转发到公开群。</p>
  </form>
</body>
</html>
"""

_GATE_ERROR = ('<p class="err">口令不对，请向主机持有人确认后重试。</p>')


def _gate_page(with_error: bool = False) -> Response:
    html = GATE_HTML.replace("__ERROR__", _GATE_ERROR if with_error else "")
    resp = make_response(html, 401)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _api_denied(message: str, code: str, status: int,
                **extra) -> Tuple[Response, int]:
    return jsonify({"ok": False, "code": code, "message": message, **extra}), status


def _wants_html() -> bool:
    """判断这次请求是不是"人在用浏览器打开页面"。

    静态资源（sw.js、图标、清单）和 /api/* 一律不给 HTML 闸门页：
    把一坨 HTML 塞给 Service Worker 只会得到一个看不懂的 MIME 报错。
    """
    if request.path.startswith("/api/"):
        return False
    if request.method != "GET":
        return False
    last = request.path.rstrip("/").rsplit("/", 1)[-1]
    if "." in last:  # 带扩展名的静态文件
        return False
    accept = request.headers.get("Accept", "")
    return "text/html" in accept or accept in ("", "*/*")


def _strip_token_from_url() -> str:
    """把 ?token=… 从地址里摘掉，其余查询参数原样保留。"""
    args = request.args.to_dict(flat=False)
    args.pop(TOKEN_QUERY, None)
    query = urlencode(args, doseq=True)
    return f"{request.base_url}?{query}" if query else request.base_url


def install(app, token: Optional[str] = None, public: bool = False,
            limits: Optional[Dict[str, Tuple[int, float]]] = None) -> Dict[str, object]:
    """把闸门挂到 app 上。`token` 为空表示不鉴权（纯回环时的原有行为）。"""
    cfg: Dict[str, object] = {
        "token": token or None,
        "public": bool(public),
        "limiter": RateLimiter(limits),
    }
    app.config["HEYAN_GUARD"] = cfg
    app.config["HEYAN_PUBLIC"] = bool(public)

    if bool(public) and not app.config.get("HEYAN_PROXY_FIXED"):
        # 隧道在服务商那边终止 TLS，Flask 自己看到的 scheme 永远是 http。
        # 不认 X-Forwarded-Proto，?token= 的 302 就会把人从 https 甩回 http，
        # cookie 也拿不到 Secure 标志。只信 for/proto，不信 host：
        # 跳转目标的主机名继续由真实 Host 头决定，不给伪造域名的机会。
        from werkzeug.middleware.proxy_fix import ProxyFix

        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
        app.config["HEYAN_PROXY_FIXED"] = True

    @app.before_request
    def _guard():  # noqa: ANN202 - Flask 钩子
        cfg = app.config["HEYAN_GUARD"]
        limiter: RateLimiter = cfg["limiter"]
        expected: Optional[str] = cfg["token"]

        # 限流放在鉴权之前：否则猜口令本身就是免费的，还能顺手把主机压垮
        if cfg["public"]:
            category = request_category()
            if not limiter.allow(client_source(), category):
                retry = limiter.retry_after(category)
                if request.path.startswith("/api/"):
                    body, status = _api_denied("请求太频繁，请稍后再试", "rate_limited",
                                               429, retry_after_s=retry)
                    body.headers["Retry-After"] = str(retry)
                    return body, status
                resp = make_response("Too Many Requests", 429)
                resp.headers["Retry-After"] = str(retry)
                return resp

        if not expected:
            return None

        supplied = (request.headers.get(TOKEN_HEADER)
                    or request.args.get(TOKEN_QUERY)
                    or request.cookies.get(TOKEN_COOKIE))
        # 故意不读 request.form：那会把 24MB 的上传图片提前整个解析一遍
        if supplied and hmac.compare_digest(str(supplied), expected):
            # 链接里带的口令换成 HttpOnly cookie：地址栏和历史记录里就不留痕了
            if request.cookies.get(TOKEN_COOKIE) != expected:
                g.heyan_set_cookie = True
            if request.args.get(TOKEN_QUERY):
                return redirect(_strip_token_from_url(), code=302)
            return None

        if request.path == "/__auth" and request.method == "POST":
            return None  # 交给下面的处理函数渲染"口令错误"
        if _wants_html():
            return _gate_page()
        return _api_denied("需要访问口令，请用主机给你的完整链接打开", "auth_required", 401)

    @app.post("/__auth")
    def _auth_form():  # noqa: ANN202
        cfg = app.config["HEYAN_GUARD"]
        expected: Optional[str] = cfg["token"]
        supplied = str(request.form.get("token", ""))
        if not expected or not hmac.compare_digest(supplied, expected):
            return _gate_page(with_error=True)
        g.heyan_set_cookie = True
        return redirect("/", code=302)

    @app.after_request
    def _harden(resp):  # noqa: ANN001, ANN202
        cfg = app.config["HEYAN_GUARD"]
        if getattr(g, "heyan_set_cookie", False) and cfg["token"]:
            resp.set_cookie(TOKEN_COOKIE, cfg["token"], max_age=TOKEN_TTL_S,
                            httponly=True, samesite="Lax", path="/",
                            secure=request.is_secure)
        if cfg["public"]:
            resp.headers.setdefault("X-Content-Type-Options", "nosniff")
            # 界面上有删记录、改同意状态这类按钮，被 iframe 套走点击劫持代价太高
            resp.headers.setdefault("X-Frame-Options", "DENY")
            resp.headers.setdefault("Referrer-Policy", "no-referrer")
            resp.headers.setdefault("X-Robots-Tag", "noindex, nofollow, noarchive")
            if request.path.startswith("/api/records") or request.path == "/api/export":
                # 农户记录可能落在别人的电脑/公用平板上，不许中间缓存
                resp.headers["Cache-Control"] = "no-store"
        return resp

    return cfg


def status(app) -> Dict[str, object]:
    """给 /api/health 用：说清当前是不是敞开的，别让人误以为安全。"""
    cfg = app.config.get("HEYAN_GUARD") or {}
    return {
        "auth": bool(cfg.get("token")),
        "public": bool(cfg.get("public")),
        "rate_limited": bool(cfg.get("public")),
    }


def share_url(base: str, token: Optional[str]) -> str:
    """拼出可以直接发给别人的完整链接。

    裸域名（没有路径）要补一个 `/`：`https://x.app?token=a` 虽然也能打开，
    但发到微信里被截断/转义的概率比带斜杠的高，不如一次给对。
    """
    base = str(base).strip()
    if not token:
        return base
    if "/" not in base.split("://", 1)[-1]:
        base += "/"
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}{TOKEN_QUERY}={token}"


def banner_lines(token: Optional[str], public: bool,
                 share_urls: Optional[Iterable[str]] = None) -> List[str]:
    """启动时要如实交代的话：谁能进来、进来能干什么、怎么关。

    `share_urls` 为空时只给出口令与拼链接的方法 —— `serve --tunnel` 自己
    并不知道隧道会分到哪个公网域名，那条完整链接由 `heyan tunnel` 打印。
    """
    lines: List[str] = []
    if not public:
        if token:
            lines.append(f"[heyan] 已开启口令鉴权（{TOKEN_TTL_S // 3600} 小时有效）")
        return lines
    lines.append("[heyan] 警告：公网模式 —— 任何拿到链接的人都能查看和修改本机全部识别记录")
    if token:
        lines.append(f"[heyan] 访问口令 {token}（{TOKEN_TTL_S // 3600} 小时内有效）")
    else:
        lines.append("[heyan] 警告：没有设置口令，接口对全网敞开！请加 --access-token")
    limits = DEFAULT_LIMITS
    lines.append(f"[heyan] 限流已开：识别 {limits['recognize'][0]} 次/分钟，"
                 f"其余 {limits['default'][0]} 次/分钟（按来源计）")
    urls = list(share_urls or [])
    for url in urls:
        lines.append(f"[heyan] 分享链接 {share_url(url, token)}")
    if not urls and token:
        lines.append(f"[heyan] 拿到公网域名后，分享链接就是 https://<域名>/?{TOKEN_QUERY}={token}")
    lines.append("[heyan] 用完就关：Ctrl+C 停止服务，隧道进程会一并退出")
    return lines
