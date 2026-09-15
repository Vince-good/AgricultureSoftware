#!/usr/bin/env python3
"""给局域网访问生成自签名 HTTPS 证书（零 pip 依赖，优先借用 Git 自带的 openssl）。

为什么需要它：
  别人用 http://<你的局域网IP>:8080 打开界面时，浏览器判定为“不安全上下文”，
  会直接禁掉 getUserMedia（页内实时取景）和 Service Worker（离线缓存/装到桌面）。
  识别、语音播报、记录、导出照常能用，快门会自动退化成系统相机/相册选图，
  所以纯 HTTP 也能干活 —— 只是体验差一档。要拿回完整体验就得给这个端口配 HTTPS。
  田间现场没域名也没公网 CA，只能自签名：对方浏览器第一次会拦一道警告，
  点“高级 → 继续前往”即可；想彻底不弹，把生成的 .crt 装进对方设备的受信任根证书。

为什么不直接 pip install cryptography：
  项目硬约束是全离线，现场机器常常连不上 PyPI。这里先试 cryptography，
  没有就找 openssl —— Windows 上装过 Git 就自带（形如
  E:\\git_soft\\Git\\usr\\bin\\openssl.exe），于是一条依赖都不用装。

用法：
    python tools/make_dev_cert.py
    python tools/make_dev_cert.py --out artifacts/certs --days 825 --ip 192.168.1.20
    python -m heyan.cli serve --lan --port 8080 \
        --ssl-cert artifacts/certs/heyan-lan.crt --ssl-key artifacts/certs/heyan-lan.key

安全提示：私钥默认落在 artifacts/（已被 .gitignore 忽略），脚本还会在证书目录里
再写一个内容为 `*` 的 .gitignore 兜底，防止有人把 --out 指到仓库别处再把私钥提交上去。
这套证书只用于局域网内测，别拿到公网当正经证书用。
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heyan.server.app import discover_lan_ipv4  # noqa: E402

# Chrome 从 2019 年起拒收有效期超过 825 天的 TLS 证书，超了连“继续前往”都不给
MAX_DAYS = 825


def find_openssl() -> Path | None:
    """找 openssl：先看 PATH，再顺着 git 的安装位置摸 Git for Windows 自带的那份。"""
    found = shutil.which("openssl")
    if found:
        return Path(found)

    # git.exe 一般在 <Git>/cmd/ 或 <Git>/bin/，openssl 在 <Git>/usr/bin/ 与 <Git>/mingw64/bin/
    candidates: list[Path] = []
    git = shutil.which("git")
    if git:
        base = Path(git).resolve().parent.parent
        candidates += [base / "usr" / "bin" / "openssl.exe",
                       base / "mingw64" / "bin" / "openssl.exe"]
    for env in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(env)
        if root:
            candidates += [Path(root) / "Git" / "usr" / "bin" / "openssl.exe",
                           Path(root) / "OpenSSL-Win64" / "bin" / "openssl.exe"]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def collect_ips(extra: list[str] | None = None) -> list[str]:
    """本机局域网 IP + 用户显式指定的 IP（另一块网卡、或计划中的固定地址）。"""
    ips: list[str] = []
    for ip in list(discover_lan_ipv4()) + list(extra or []):
        ip = (ip or "").strip()
        if ip and ip not in ips:
            ips.append(ip)
    return ips


def build_san(ips: list[str]) -> str:
    """拼 subjectAltName。现代浏览器只认 SAN、不看 CN，缺这段证书直接无效。"""
    parts = ["DNS:localhost", "IP:127.0.0.1"]
    for ip in ips:
        parts.append(f"IP:{ip}")
    return ",".join(parts)


def gen_with_openssl(openssl: Path, crt: Path, key: Path, days: int, san: str,
                     cn: str) -> None:
    cmd = [
        str(openssl), "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256",
        "-keyout", str(key), "-out", str(crt), "-days", str(days),
        "-subj", f"/CN={cn}/O=HeYan",
        # CA:TRUE + keyCertSign：让这份自签名证书同时充当自己的签发者，
        # 装进受信任根之后就能彻底消掉浏览器警告（纯 leaf 证书做不到这点）。
        "-addext", "basicConstraints=critical,CA:TRUE",
        "-addext", "keyUsage=critical,digitalSignature,keyEncipherment,keyCertSign",
        "-addext", "extendedKeyUsage=serverAuth",
        "-addext", f"subjectAltName={san}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"[cert] openssl 失败（{proc.returncode}）：\n{proc.stderr.strip()}")


def gen_with_cryptography(crt: Path, key: Path, days: int, ips: list[str],
                          cn: str) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn),
                         x509.NameAttribute(NameOID.ORGANIZATION_NAME, "HeYan")])
    now = datetime.now(timezone.utc)
    san = x509.SubjectAlternativeName(
        [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
        + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]
    )
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False,
                                     key_encipherment=True, data_encipherment=False,
                                     key_agreement=False, key_cert_sign=True,
                                     crl_sign=False, encipher_only=False,
                                     decipher_only=False), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
                       critical=False)
        .add_extension(san, critical=False)
        .sign(private_key, hashes.SHA256())
    )
    key.write_bytes(private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def harden(path: Path) -> None:
    """收紧私钥权限。Windows 上 chmod 能力有限，尽力而为，不假装成功。"""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成局域网自签名 HTTPS 证书")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "certs"),
                    help="输出目录（默认 artifacts/certs）")
    ap.add_argument("--name", default="heyan-lan", help="文件名前缀")
    ap.add_argument("--cn", default="HeYan LAN", help="证书 CN")
    ap.add_argument("--days", type=int, default=MAX_DAYS,
                    help=f"有效期天数，上限 {MAX_DAYS}（浏览器硬限制）")
    ap.add_argument("--ip", action="append", default=[],
                    help="额外写进 SAN 的 IP，可重复；默认自动探测本机局域网 IP")
    ap.add_argument("--force", action="store_true", help="已存在则覆盖")
    args = ap.parse_args(argv)

    if not 1 <= args.days <= MAX_DAYS:
        print(f"[cert] --days 必须在 1~{MAX_DAYS} 之间（更长的有效期浏览器会拒收）")
        return 2

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    crt = out / f"{args.name}.crt"
    key = out / f"{args.name}.key"
    if (crt.exists() or key.exists()) and not args.force:
        print(f"[cert] 已存在：{crt}")
        print("[cert] 要重新生成请加 --force（换证书后，已信任旧证书的设备会重新报警告）")
        return 1

    ips = collect_ips(args.ip)
    try:
        import cryptography  # noqa: F401

        engine = "cryptography"
    except ImportError:
        engine = "openssl"

    if engine == "cryptography":
        gen_with_cryptography(crt, key, args.days, ips, args.cn)
    else:
        openssl = find_openssl()
        if openssl is None:
            print("[cert] 既没有 cryptography 也没找到 openssl。二选一：")
            print("         pip install cryptography")
            print("         或装 Git for Windows / OpenSSL 后重试")
            return 3
        print(f"[cert] 使用 {openssl}")
        gen_with_openssl(openssl, crt, key, args.days, build_san(ips), args.cn)

    harden(key)
    (out / ".gitignore").write_text("*\n", encoding="utf-8")

    print(f"[cert] 证书 {crt}")
    print(f"[cert] 私钥 {key}")
    print(f"[cert] SAN 覆盖 IP：{', '.join(ips) if ips else '（只有 localhost）'}")
    print("[cert] 启动 HTTPS 服务：")
    print(f'         python -m heyan.cli serve --lan --port 8080 '
          f'--ssl-cert "{crt}" --ssl-key "{key}"')
    print("[cert] 对方首次打开会有证书警告，点“高级 → 继续前往”即可；")
    print(f"         想免警告就把 {crt.name} 拷给对方，装进“受信任的根证书颁发机构”"
          "（手机：设置 → 安全 → 加密与凭据 → 安装证书 → CA 证书）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
