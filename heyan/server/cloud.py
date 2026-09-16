"""云端 / 容器启动入口。

本地 `heyan serve` 是田间口径：默认只监听回环、端口被占就顺延、用 Flask 自带的
开发服务器。这三条搬到云上每一条都会出事，所以单独给一个入口：

  * 监听 0.0.0.0 —— 容器外面的负载均衡要能连进来，绑回环等于谁都连不上；
  * 端口取 `$PORT` —— HF Spaces / Render / Railway / Fly 都是平台分配端口，
    写死 8080 会直接部署失败；没有 `$PORT` 时用 7860（HF Spaces 的约定端口）；
  * 端口不可用就退出，绝不自作主张顺延 —— 平台只健康检查它分配的那一个端口，
    悄悄换端口等于"日志显示起来了、链接永远打不开"，比直接崩掉更难查；
  * 用 waitress 起服务（没装就退回 Flask 开发服务器，并把这个降级说明白）；
  * 一律按公网暴露处理：限流、安全响应头、认 X-Forwarded-For / X-Forwarded-Proto
    （TLS 在平台那侧终止，不认这两个头的话 cookie 拿不到 Secure，跳转也会甩回 http）。

口令是**可选**的：不设 `HEYAN_ACCESS_TOKEN` 就是"评委点开链接直接用"；
设了就多一道口令页，链接用 `?token=...` 分享，首次访问后换成 HttpOnly cookie。

环境变量（都可选）
------------------
    PORT / HEYAN_PORT      监听端口，默认 7860
    HOST / HEYAN_HOST      监听地址，默认 0.0.0.0
    HEYAN_ACCESS_TOKEN     访问口令；不设则拿到链接的人都能用
    HEYAN_BUNDLE           指定模型包目录；不设则取 HEYAN_HOME/bundles 下最新的一个
    HEYAN_LANG             界面语言，默认 zh
    HEYAN_THREADS          waitress 的 HTTP 工作线程数，默认 8
    HEYAN_INFER_THREADS    单次推理用几个 CPU 线程，默认 1（留给并发）
    HEYAN_RATE_RECOGNIZE   每个来源每分钟识别次数上限，默认 30
    HEYAN_RATE_DEFAULT     每个来源每分钟其他请求上限，默认 600
    HEYAN_HOME             运行期根目录（记录库、导出件都写在这里，必须可写）
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7860  # HF Spaces Docker SDK 的约定端口
DEFAULT_THREADS = 8
DEFAULT_INFER_THREADS = 1
DEFAULT_LANG = "zh"

# 云端默认比本机宽一档：比赛现场评委很可能都挤在会场同一个出口 IP 后面，
# 按本机口径（12 次/分钟）限流会把一整排人一起挡在门外。
CLOUD_LIMITS: Dict[str, Tuple[int, float]] = {
    "recognize": (30, 60.0),
    "default": (600, 60.0),
}


def _env(*names: str) -> Optional[str]:
    """按顺序取第一个非空环境变量。空串等同于没设。"""
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"[cloud] {name} 不是整数：{raw!r}") from None
    if value < minimum:
        raise SystemExit(f"[cloud] {name} 不能小于 {minimum}：{value}")
    return value


def resolve_host(explicit: Optional[str] = None) -> str:
    """监听地址。云端必须是 0.0.0.0，否则容器外面进不来。"""
    return explicit or _env("HEYAN_HOST", "HOST") or DEFAULT_HOST


def resolve_port(explicit: Optional[int] = None) -> int:
    """监听端口：显式参数 > HEYAN_PORT > 平台注入的 PORT > 7860。"""
    if explicit:
        return int(explicit)
    raw = _env("HEYAN_PORT", "PORT")
    if raw is None:
        return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError:
        raise SystemExit(f"[cloud] 端口环境变量不是整数：{raw!r}") from None
    if not 1 <= port <= 65535:
        raise SystemExit(f"[cloud] 端口超出 1~65535：{port}")
    return port


def resolve_limits() -> Dict[str, Tuple[int, float]]:
    """限流额度。默认用云端口径，可用环境变量按现场人数再放宽。"""
    limits = dict(CLOUD_LIMITS)
    for key, name in (("recognize", "HEYAN_RATE_RECOGNIZE"),
                      ("default", "HEYAN_RATE_DEFAULT")):
        raw = _env(name)
        if raw is None:
            continue
        try:
            per_minute = int(raw)
        except ValueError:
            raise SystemExit(f"[cloud] {name} 不是整数：{raw!r}") from None
        if per_minute <= 0:
            raise SystemExit(f"[cloud] {name} 必须是正整数：{per_minute}")
        limits[key] = (per_minute, 60.0)
    return limits


def build_app(bundle: Optional[str] = None, language: Optional[str] = None,
              threads: Optional[int] = None, infer_threads: Optional[int] = None):
    """按云端口径建应用：公网加固 + 可选口令 + 宽松限流 + 启动即预热模型。

    预热（warm）很重要：平台的健康检查不会等第一次推理，冷启动那一下
    往往就是评委点开链接的第一下。
    """
    from .app import create_app

    return create_app(
        bundle=bundle or _env("HEYAN_BUNDLE"),
        language=language or _env("HEYAN_LANG") or DEFAULT_LANG,
        host=resolve_host(),
        port=resolve_port(),
        threads=infer_threads if infer_threads is not None
        else _int_env("HEYAN_INFER_THREADS", DEFAULT_INFER_THREADS),
        warm=True,
        access_token=_env("HEYAN_ACCESS_TOKEN"),
        public=True,
        limits=resolve_limits(),
    )


def _banner(app: Any, host: str, port: int) -> None:
    from ..config import PATHS

    state = app.extensions.get("heyan")
    cfg = app.config.get("HEYAN_GUARD") or {}
    engine = state.engine_status() if state is not None else {}
    print(f"[cloud] 监听 {host}:{port}（容器外通过平台分配的公网地址访问）")
    print(f"[cloud] 运行期根目录 HEYAN_HOME = {PATHS.root}")
    if engine.get("available"):
        print(f"[cloud] 模型包 {engine.get('bundle_dir')}")
        # engine_status()["backend"] 是后端的自述字典（含 model_path、providers…），
        # 日志里只要一行结论：用的哪个后端、模型多大。
        backend = engine.get("backend") or {}
        if isinstance(backend, dict) and backend:
            size = backend.get("model_size_mb")
            suffix = f"，模型 {size}MB" if size else ""
            print(f"[cloud] 推理后端 {backend.get('backend')}{suffix}")
        elif engine.get("loaded"):
            print(f"[cloud] 推理后端 {backend}")
        else:
            print(f"[cloud] 模型包已就位，引擎将在首次请求时加载："
                  f"{engine.get('error') or '预热未完成'}")
    else:
        print("[cloud] 警告：没找到模型包，界面会提示先构建。"
              "确认 artifacts/bundles 已经打进镜像，或用 HEYAN_BUNDLE 指过去")
    # 不复用 guard.banner_lines：那套文案是内网穿透口径（"Ctrl+C 停止"、
    # "请加 --access-token"），在容器里是错的。这里按云端实情自己说。
    print("[cloud] 警告：公网模式 —— 任何拿到链接的人都能查看和修改全部识别记录")
    if cfg.get("token"):
        print("[cloud] 口令鉴权：已开启（HEYAN_ACCESS_TOKEN）")
        print(f"[cloud] 分享链接形如 https://<平台域名>/?token={cfg['token']}")
    else:
        print("[cloud] 口令鉴权：未开启 —— 拿到链接的人都能用，"
              "适合评委演示；要收紧就设 HEYAN_ACCESS_TOKEN")
    limiter = cfg.get("limiter")
    limits = limiter.limits if limiter is not None else {}
    # 缺 recognize 档时 RateLimiter 会退回 default 档，横幅照实说
    default_quota = limits.get("default") or (0, 0.0)
    recognize_quota = limits.get("recognize") or default_quota
    print(f"[cloud] 限流已开：识别 {recognize_quota[0]} 次/分钟，"
          f"其余 {default_quota[0]} 次/分钟（按来源计）")
    print("[cloud] 健康检查 GET /api/health")


def serve(app: Any = None, host: Optional[str] = None, port: Optional[int] = None,
          threads: Optional[int] = None) -> None:
    """阻塞式起服务。端口不可用直接抛，不顺延（理由见模块 docstring）。"""
    host = host or resolve_host()
    port = port or resolve_port()
    threads = threads if threads is not None else _int_env("HEYAN_THREADS", DEFAULT_THREADS)
    if app is None:
        app = build_app(threads=threads)
    _banner(app, host, port)
    try:
        from waitress import serve as _waitress_serve
    except ImportError:
        print("[cloud] 警告：没装 waitress，退回 Flask 开发服务器——能跑，但并发和"
              "稳定性都差一档。装一下：pip install -r requirements-cloud.txt")
        app.run(host=host, port=port, threaded=True, use_reloader=False)
        return
    print(f"[cloud] waitress 起服务，{threads} 个工作线程")
    _waitress_serve(app, host=host, port=port, threads=threads)


def main(argv: Optional[list] = None) -> int:
    host, port = resolve_host(), resolve_port()
    try:
        serve(host=host, port=port)
    except OSError as exc:
        # 端口被占 / 无权限：云端的正确反应是失败退出，让平台重启或报错，
        # 而不是换个端口假装成功。
        print(f"[cloud] 起不来：{host}:{port} -> {exc}")
        print("[cloud] 云端端口由平台指定，不做顺延。确认 $PORT 没被别的进程占用。")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
