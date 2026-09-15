"""局域网访问这条路的契约测试。

要守住的三件事：
  1. 绑 0.0.0.0 时打印出来的必须是别人真能打开的地址，不是 0.0.0.0 本身；
  2. 证书和私钥要么成对给出、要么明确报错，绝不静默退回 HTTP
     （静默退回的唯一现场症状是"相机突然不能用"，排查成本极高）；
  3. --lan / --ssl-* 真的传到了 run_server，而不是停在 argparse 里。

这些测试不加载模型：run_server 用一个假的 app 顶替，只验证它对外说了什么、
往 app.run 里塞了什么。
"""

from __future__ import annotations

import ipaddress
import socket

import pytest

from heyan import cli
from heyan.server import app as server_app


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


class _StubEngine:
    def engine_status(self):
        return {"available": True, "bundle_dir": "artifacts/bundles/heyan-mnv3s-int8-v1.1.0"}


class _StubApp:
    """只实现 run_server 会用到的那一点点 Flask 接口。"""

    def __init__(self):
        self.extensions = {"heyan": _StubEngine()}
        self.config = {}
        self.run_kwargs = None

    def run(self, **kwargs):
        self.run_kwargs = kwargs


# ------------------------------------------------------------------ IP 探测
def test_discover_lan_ipv4_returns_routable_addresses():
    ips = server_app.discover_lan_ipv4()
    assert isinstance(ips, list)
    for ip in ips:
        addr = ipaddress.ip_address(ip)  # 不是合法 IP 就直接炸
        assert addr.version == 4
        assert not addr.is_loopback
        assert not addr.is_link_local
    assert len(ips) == len(set(ips)), "重复地址会让用户看到两行一样的链接"


def test_describe_endpoints_loopback_only():
    assert server_app.describe_endpoints("127.0.0.1", 8765) == ["http://127.0.0.1:8765"]


def test_describe_endpoints_lan_never_prints_wildcard(monkeypatch):
    monkeypatch.setattr(server_app, "discover_lan_ipv4", lambda: ["192.168.1.20", "10.0.0.5"])
    urls = server_app.describe_endpoints("0.0.0.0", 8080, "https")
    assert "https://127.0.0.1:8080" in urls
    assert "https://192.168.1.20:8080" in urls
    assert "https://10.0.0.5:8080" in urls
    assert not any("0.0.0.0" in u for u in urls), "0.0.0.0 不是能点开的地址"


def test_describe_endpoints_lan_without_ip_says_so(monkeypatch):
    monkeypatch.setattr(server_app, "discover_lan_ipv4", lambda: [])
    urls = server_app.describe_endpoints("::", 8080)
    assert urls[0] == "http://127.0.0.1:8080"
    assert any("ipconfig" in u for u in urls), "探测不到 IP 时要告诉用户下一步做什么"


# ------------------------------------------------------------------ HTTPS
def test_build_ssl_context_none_when_absent():
    assert server_app.build_ssl_context(None, None) is None


@pytest.mark.parametrize("cert,key", [("c.pem", None), (None, "k.pem")])
def test_build_ssl_context_rejects_half_a_pair(cert, key):
    with pytest.raises(ValueError, match="同时给出"):
        server_app.build_ssl_context(cert, key)


def test_build_ssl_context_rejects_missing_file(tmp_path):
    cert = tmp_path / "heyan.crt"
    cert.write_text("not really a cert", encoding="utf-8")
    with pytest.raises(ValueError, match="不存在"):
        server_app.build_ssl_context(str(cert), str(tmp_path / "nope.key"))


def test_build_ssl_context_returns_pair(tmp_path):
    cert = tmp_path / "heyan.crt"
    key = tmp_path / "heyan.key"
    cert.write_text("c", encoding="utf-8")
    key.write_text("k", encoding="utf-8")
    assert server_app.build_ssl_context(str(cert), str(key)) == (str(cert), str(key))


# ------------------------------------------------------------------ 启动播报
def test_run_server_announces_lan_urls_and_firewall(capsys, monkeypatch):
    monkeypatch.setattr(server_app, "discover_lan_ipv4", lambda: ["192.168.1.20"])
    stub = _StubApp()
    port = _free_port()
    server_app.run_server(stub, host="0.0.0.0", port=port)
    out = capsys.readouterr().out
    assert f"http://192.168.1.20:{port}" in out
    assert "netsh advfirewall" in out, "对外监听时必须把放行端口的命令交代清楚"
    assert "HTTP 方式下浏览器禁用" in out, "要说清纯 HTTP 少了什么，别让人以为坏了"
    assert stub.run_kwargs["ssl_context"] is None
    assert stub.run_kwargs["threaded"] is True


def test_run_server_https_announces_https_and_passes_context(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(server_app, "discover_lan_ipv4", lambda: ["192.168.1.20"])
    cert = tmp_path / "heyan.crt"
    key = tmp_path / "heyan.key"
    cert.write_text("c", encoding="utf-8")
    key.write_text("k", encoding="utf-8")
    stub = _StubApp()
    port = _free_port()
    server_app.run_server(stub, host="0.0.0.0", port=port,
                          ssl_cert=str(cert), ssl_key=str(key))
    out = capsys.readouterr().out
    assert f"https://192.168.1.20:{port}" in out
    assert stub.run_kwargs["ssl_context"] == (str(cert), str(key))


def test_run_server_loopback_stays_quiet_about_lan(capsys):
    stub = _StubApp()
    port = _free_port()
    server_app.run_server(stub, host="127.0.0.1", port=port)
    out = capsys.readouterr().out
    assert f"http://127.0.0.1:{port}" in out
    assert "netsh" not in out, "只监听回环时不该劝人开防火墙"


# ------------------------------------------------------------------ CLI 接线
def test_cli_serve_flags_are_wired():
    args = server_app and cli.build_parser().parse_args(
        ["serve", "--lan", "--port", "8080", "--ssl-cert", "c.pem", "--ssl-key", "k.pem"])
    assert args.lan is True
    assert args.ssl_cert == "c.pem"
    assert args.ssl_key == "k.pem"
    assert args.host == "127.0.0.1", "--lan 不该改写 --host 的默认值，只在运行期覆盖"


def test_cmd_serve_lan_binds_all_interfaces(monkeypatch):
    calls = {}

    def fake_create_app(**kwargs):
        calls["create"] = kwargs
        return _StubApp()

    def fake_run_server(app, **kwargs):
        calls["run"] = kwargs

    monkeypatch.setattr(server_app, "create_app", fake_create_app)
    monkeypatch.setattr(server_app, "run_server", fake_run_server)
    args = cli.build_parser().parse_args(["serve", "--lan", "--port", "8123"])
    assert cli.cmd_serve(args) == 0
    assert calls["create"]["host"] == "0.0.0.0"
    assert calls["run"]["host"] == "0.0.0.0"
    assert calls["run"]["port"] == 8123


def test_cmd_serve_reports_bad_ssl_pair(monkeypatch, capsys):
    monkeypatch.setattr(server_app, "create_app", lambda **kw: _StubApp())

    def boom(app, **kwargs):
        raise ValueError("--ssl-cert 和 --ssl-key 必须同时给出")

    monkeypatch.setattr(server_app, "run_server", boom)
    args = cli.build_parser().parse_args(["serve", "--ssl-cert", "only-cert.pem"])
    assert cli.cmd_serve(args) == 2
    assert "同时给出" in capsys.readouterr().out
