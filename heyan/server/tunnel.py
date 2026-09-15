"""内网穿透：把回环上的界面交给 ngrok / cpolar，换一个公网 HTTPS 地址。

为什么走隧道而不是路由器端口映射：
  田间主机多半挂在 4G 路由或家宽 NAT 后面，压根没有公网 IP，端口映射无从下手。
  隧道是从内网主动往外连的长连接，不用动路由器、不用找运营商要公网地址。

安全前提（不满足就别开）：
  服务只监听 127.0.0.1，由隧道客户端从本机连进来；同时必须开启口令鉴权
  （heyan.server.guard）。少了这两条，等于把带姓名/村/地块的农户记录
  和“删除记录”的接口一起挂到全网。`heyan tunnel` 会强制这么做，
  没给口令就自动生成一个。

附带的好处：
  隧道给的是公网 HTTPS，对方浏览器把页面当“安全上下文”，页内实时取景和
  Service Worker 都能用 —— 这点反而比局域网 http 直连体验更好。

免费档的现实（如实告知，别当生产入口）：
  - 域名每次重启都变（要固定域名得付费档或 cpolar 的二级域名）；
  - 有带宽与并发连接上限，几十人同时刷会卡；
  - ngrok 免费档会给浏览器请求插一张警告页，前端已带
    `ngrok-skip-browser-warning` 头绕过，但对方首次打开仍可能看到一次；
  - 隧道服务商能看到全部过路流量（HTTPS 只在浏览器↔服务商之间加密，
    服务商到本机这一跳是它自己解开的），涉密数据不要走公网。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import PATHS
from . import guard

ROOT = Path(__file__).resolve().parents[2]
CLIENTS: Tuple[str, ...] = ("ngrok", "cpolar")

_URL_RE = re.compile(r"https://[A-Za-z0-9._-]+(?::\d+)?")
# 客户端帮助文本里的官网/控制台链接，不是隧道地址，必须滤掉
_NOISE_HOSTS = {
    "ngrok.com", "dashboard.ngrok.com", "api.ngrok.com", "login.ngrok.com",
    "cpolar.com", "www.cpolar.com", "dashboard.cpolar.com", "cpolar.cn",
}

INSTALL_HINTS: Dict[str, List[str]] = {
    "ngrok": [
        "  1) 到 https://dashboard.ngrok.com/signup 注册（免费档即可），拿 authtoken",
        "  2) 下载 https://ngrok.com/download 并把 ngrok.exe 放进 PATH",
        "  3) 执行一次：ngrok config add-authtoken <你的TOKEN>",
    ],
    "cpolar": [
        "  1) 到 https://www.cpolar.com/signup 注册（国内直连，免费档即可）",
        "  2) 下载客户端并把 cpolar.exe 放进 PATH",
        "  3) 执行一次：cpolar authtoken <你的TOKEN>",
    ],
}


def find_client(preferred: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """返回 (客户端名, 可执行文件路径)；`preferred` 指定了就只找它。

    先看 PATH，再看 artifacts/bin —— 后者已被 .gitignore 忽略，
    可以把下载下来的 ngrok.exe/cpolar.exe 放在那儿，不用去改系统环境变量。
    """
    local_bin = PATHS.root / "bin"
    names = (preferred,) if preferred else CLIENTS
    for name in names:
        if name not in CLIENTS:
            continue
        exe = shutil.which(name) or shutil.which(f"{name}.exe")
        if exe:
            return name, exe
        for cand in (local_bin / f"{name}.exe", local_bin / name):
            if cand.is_file():
                return name, str(cand)
    return None


def build_command(client: str, exe: str, port: int,
                  subdomain: Optional[str] = None,
                  region: Optional[str] = None) -> List[str]:
    """拼隧道命令。ngrok 用双横线参数，cpolar 用单横线，别混。"""
    port = str(int(port))
    if client == "ngrok":
        cmd = [exe, "http", port, "--log=stdout", "--log-level=info"]
        if subdomain:
            cmd += ["--domain", subdomain]
        if region:
            cmd += ["--region", region]
        return cmd
    if client == "cpolar":
        cmd = [exe, "http", port, "-log=stdout", "-log-level=info"]
        if subdomain:
            cmd += [f"-subdomain={subdomain}"]
        if region:
            cmd += [f"-region={region}"]
        return cmd
    raise ValueError(f"不支持的隧道客户端：{client}（可选 {', '.join(CLIENTS)}）")


def parse_public_urls(text: Optional[str]) -> List[str]:
    """从隧道客户端输出里抠出公网地址，按出现顺序去重。

    覆盖两家真实输出：
      ngrok   : `msg="started tunnel" ... url=https://abcd-1234.ngrok-free.app`
      cpolar : `Tunnel established at https://abcd1234.r6.cn`
    """
    out: List[str] = []
    for match in _URL_RE.finditer(text or ""):
        url = match.group(0).rstrip(".,;:)\"'")
        host = url.split("://", 1)[1].split("/", 1)[0].split(":", 1)[0].lower()
        if host in _NOISE_HOSTS or host in ("localhost", "127.0.0.1"):
            continue
        if "." not in host:
            continue
        if url not in out:
            out.append(url)
    return out


def fetch_ngrok_tunnels(api_base: str = "http://127.0.0.1:4040",
                        timeout: float = 2.0) -> List[str]:
    """读 ngrok 的本地 API。这是最权威的来源，日志级别不够时也能拿到地址。

    cpolar 的本地面板（127.0.0.1:9200）要账号密码，这里不猜，统一走日志解析。
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{api_base}/api/tunnels", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return []
    urls: List[str] = []
    for tun in data.get("tunnels") or []:
        url = (tun.get("public_url") or "").strip()
        if url.startswith("https://") and url not in urls:
            urls.append(url)
    return urls


def wait_for_server(port: int, token: Optional[str] = None, timeout: float = 90.0,
                    proc: Optional[subprocess.Popen] = None) -> Optional[Dict]:
    """轮询 /api/health 直到服务真能干活。服务进程已经死了就立刻放弃。"""
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout
    last: Optional[Dict] = None
    url = f"http://127.0.0.1:{int(port)}/api/health"
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return None
        req = urllib.request.Request(url)
        if token:
            req.add_header(guard.TOKEN_HEADER, token)
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                last = json.loads(resp.read().decode("utf-8"))
                if last.get("ok"):
                    return last
        except urllib.error.HTTPError as exc:
            # 4xx 说明服务已经起来了（只是这次请求没被接受），不必再等
            if exc.code < 500:
                return {"ok": True, "http": exc.code}
            last = {"ok": False, "reason": f"HTTP {exc.code}"}
        except Exception as exc:  # noqa: BLE001 - 启动期什么错都可能，等下一次轮询
            last = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
        time.sleep(0.5)
    return last


def _server_command(port: int, token: str, bundle: Optional[str],
                    lang: str) -> List[str]:
    return [sys.executable, "-X", "utf8", "-m", "heyan.cli", "serve", "--tunnel",
            "--port", str(int(port)), "--access-token", token, "--lang", lang] + \
        (["--bundle", bundle] if bundle else [])


def _terminate(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _tail(path: Path, lines: int = 25) -> str:
    text = _read(path).strip().splitlines()
    return "\n".join(f"    {ln}" for ln in text[-lines:])


def _wait_for_public_url(client: str, proc: subprocess.Popen, log_path: Path,
                         timeout: float = 120.0) -> List[str]:
    """边等隧道建立边找公网地址：ngrok 优先问本地 API，两家都兜底解析日志。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return parse_public_urls(_read(log_path))
        if client == "ngrok":
            urls = fetch_ngrok_tunnels()
            if urls:
                return urls
        urls = parse_public_urls(_read(log_path))
        if urls:
            return urls
        time.sleep(1.0)
    return parse_public_urls(_read(log_path))


def run(port: int = 8080, client: Optional[str] = None, token: Optional[str] = None,
        subdomain: Optional[str] = None, region: Optional[str] = None,
        bundle: Optional[str] = None, lang: str = "zh",
        log_dir: Optional[Path] = None, timeout: float = 120.0) -> int:
    """起服务 + 起隧道，打印能直接发出去的链接；Ctrl+C 时把两个进程都收干净。"""
    token = token or os.environ.get("HEYAN_ACCESS_TOKEN") or guard.generate_token()
    logs = Path(log_dir or (PATHS.root / "logs"))
    logs.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    server_log = logs / f"tunnel-server-{stamp}.log"
    tunnel_log = logs / f"tunnel-{client or 'auto'}-{stamp}.log"

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    found = find_client(client)
    if not found:
        names = (client,) if client else CLIENTS
        print(f"[tunnel] 没找到可用的隧道客户端（{'/'.join(names)}）。任选一个装好再跑：")
        for name in names:
            print(f"[tunnel] {name}:")
            for line in INSTALL_HINTS.get(name, []):
                print(line)
        return 3
    name, exe = found

    server_proc = tunnel_proc = None
    try:
        with open(server_log, "wb") as slog:
            server_proc = subprocess.Popen(_server_command(port, token, bundle, lang),
                                           cwd=str(ROOT), env=env,
                                           stdout=slog, stderr=subprocess.STDOUT)
        print(f"[tunnel] 服务启动中（只监听 127.0.0.1:{port}，口令鉴权已开）…")
        if not wait_for_server(port, token, timeout=timeout, proc=server_proc):
            print(f"[tunnel] 服务没起来，日志：{server_log}")
            print(_tail(server_log))
            return 4

        cmd = build_command(name, exe, port, subdomain=subdomain, region=region)
        print(f"[tunnel] 启动隧道：{' '.join(cmd)}")
        with open(tunnel_log, "wb") as tlog:
            tunnel_proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                                           stdout=tlog, stderr=subprocess.STDOUT)

        public_urls = _wait_for_public_url(name, tunnel_proc, tunnel_log, timeout=timeout)
        if not public_urls:
            print(f"[tunnel] 没拿到公网地址，隧道日志：{tunnel_log}")
            print(_tail(tunnel_log))
            print("[tunnel] 最常见原因是 authtoken 没配："
                  "ngrok config add-authtoken <TOKEN> / cpolar authtoken <TOKEN>")
            return 5

        print()
        for line in guard.banner_lines(token, public=True, share_urls=public_urls):
            print(line)
        print(f"[tunnel] 本机地址 http://127.0.0.1:{port}/?{guard.TOKEN_QUERY}={token}")
        print(f"[tunnel] 隧道日志 {tunnel_log}")
        print(f"[tunnel] 服务日志 {server_log}")
        print("[tunnel] 保持这个窗口开着；按 Ctrl+C 同时关掉服务和隧道。")
        tunnel_proc.wait()
        return 0
    except KeyboardInterrupt:
        print("\n[tunnel] 收到中断，正在关闭…")
        return 0
    finally:
        _terminate(tunnel_proc)
        _terminate(server_proc)
