"""交接守卫：仓库里不许出现写死的绝对路径。

这个项目要能"整个目录拷到任何一台机器、任何盘符下直接跑"，运行期产物还能用
`HEYAN_HOME` 整体重定向到 SD 卡 / U 盘。路径因此一律由「项目根 + 环境变量」推导
（见 `heyan/config.py` 的 `PACKAGE_ROOT` 与 `_default_root`）。

这条约束靠人肉 review 守不住：新写一个脚本就可能顺手钉一个盘符进去，连注释里的
示例路径都会误导接手的人。所以把它变成测试——扫仓库内所有文本文件，出现盘符路径
或写死的 POSIX 系统目录就直接红。
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SELF_REL = Path(__file__).resolve().relative_to(REPO).as_posix()

# 只扫文本。图片、模型、语音包是二进制，读成字符串除了噪声什么也得不到。
TEXT_SUFFIXES = {
    ".py", ".pyi", ".ps1", ".bat", ".cmd", ".sh",
    ".md", ".txt", ".rst",
    ".json", ".toml", ".cfg", ".ini", ".yaml", ".yml",
    ".js", ".mjs", ".html", ".css", ".webmanifest", ".svg",
}
# 无后缀但确实是文本的文件
TEXT_NAMES = {".gitignore", ".gitattributes", ".dockerignore", "Dockerfile"}

# 构建产物、数据集与缓存：体积大且不是手写的，不在守卫范围内
SKIP_DIRS = {
    ".git", ".hg", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "node_modules", "build", "dist",
    "artifacts", "data", "runs", "dateBase_Maize",
}
SKIP_DIR_PREFIXES = ("pytest-cache-files-",)

# 盘符路径：E:\foo、D:/foo。前面不能紧挨字母/数字/斜杠，
# 否则 http:// 里的 "p:/" 会被误判成盘符。
DRIVE_PATH = re.compile(r"(?<![A-Za-z0-9_/])[A-Za-z]:[\\/]")

# 写死的 POSIX 系统目录：同样会让换机器部署跑不起来。
POSIX_ROOT = re.compile(r"/(?:home|root|Users|mnt|media|opt|var|usr|etc|srv)/")

# 白名单：单测里故意造的假安装位置，用来验证"PATH 上找到了就用它"，
# 不是真实路径，也不参与部署。
DRIVE_PATH_ALLOWLIST = {
    "tests/test_public_access.py",
}


def _walk(root: Path):
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except OSError:  # 权限异常的目录直接跳过，别让守卫自己崩掉
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name in SKIP_DIRS or entry.name.startswith(SKIP_DIR_PREFIXES):
                    continue
                stack.append(entry)
            elif entry.suffix.lower() in TEXT_SUFFIXES or entry.name in TEXT_NAMES:
                yield entry


def _scanned_files() -> list[Path]:
    return [p for p in _walk(REPO) if p.is_file()]


def _lines(path: Path) -> list[str]:
    return path.read_bytes().decode("utf-8", errors="replace").splitlines()


def test_scanner_covers_the_repo():
    """守卫自己别空转：一个文件都没扫到，等于什么都没检查。"""
    rels = {p.relative_to(REPO).as_posix() for p in _scanned_files()}
    assert len(rels) >= 40, f"只扫到 {len(rels)} 个文本文件，扫描口径可能失灵"
    for expected in ("heyan/config.py", "heyan/server/app.py",
                     "heyan/tts/engines.py", "docs/operations.md", "README.md"):
        assert expected in rels, f"{expected} 没被扫到"
    for skipped in ("artifacts", "dateBase_Maize"):
        assert not any(r.startswith(skipped + "/") for r in rels), f"{skipped} 不该进扫描范围"


def test_no_hardcoded_drive_paths():
    """任何文本文件里都不许出现写死的盘符路径，注释和 docstring 也算。"""
    offenders: list[str] = []
    for path in _scanned_files():
        rel = path.relative_to(REPO).as_posix()
        if rel == SELF_REL or rel in DRIVE_PATH_ALLOWLIST:
            continue
        for lineno, line in enumerate(_lines(path), start=1):
            if DRIVE_PATH.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()[:120]}")
    assert not offenders, (
        "发现写死的盘符路径。请改成基于项目根目录推导"
        "（Path(__file__).parent、./artifacts/，或 HEYAN_HOME 重定向）：\n  "
        + "\n  ".join(offenders)
    )


def test_no_posix_system_roots():
    """同样不许钉死 Linux 系统目录；shebang 与注释里的目录说明放行。"""
    offenders: list[str] = []
    for path in _scanned_files():
        rel = path.relative_to(REPO).as_posix()
        if rel == SELF_REL:
            continue
        for lineno, line in enumerate(_lines(path), start=1):
            stripped = line.lstrip()
            # shebang（#!/usr/bin/env python3）是跨平台惯例，注释不影响运行
            if stripped.startswith("#") or stripped.startswith("//"):
                continue
            if POSIX_ROOT.search(line):
                offenders.append(f"{rel}:{lineno}: {stripped[:120]}")
    assert not offenders, "发现写死的 POSIX 系统路径：\n  " + "\n  ".join(offenders)


def test_tools_derive_root_from_file():
    """tools/ 下的脚本一律用 __file__ 推导项目根，不依赖从哪个目录启动。"""
    scripts = sorted((REPO / "tools").glob("*.py"))
    assert scripts, "tools/ 下没有脚本，扫描口径变了"
    missing = [p.name for p in scripts if "__file__" not in p.read_text(encoding="utf-8")]
    assert not missing, f"这些脚本没用 __file__ 定位项目根：{missing}"


def test_tts_script_is_file_relative():
    """跨进程调用的 sapi_render.ps1 按包位置定位，与启动目录、盘符都无关。"""
    from heyan.tts import engines

    assert engines.SCRIPT_DIR == REPO / "heyan" / "tts"
    assert engines.SAPI_SCRIPT == REPO / "heyan" / "tts" / "sapi_render.ps1"
    assert engines.SAPI_SCRIPT.is_file()


def _run_python(code: str, extra_env: dict | None = None) -> list[str]:
    """在干净子进程里跑一句代码。

    conftest 在导入期就把 HEYAN_HOME 设成了测试沙箱，进程内改不干净；
    这里显式清掉再按需注入，才能测到"默认行为"和"重定向行为"两种情况。
    """
    env = dict(os.environ)
    env.pop("HEYAN_HOME", None)
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONIOENCODING"] = "utf-8"
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code],
        capture_output=True, text=True, encoding="utf-8",
        env=env, cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()


def test_default_root_follows_cwd():
    """没设 HEYAN_HOME 时，运行期根目录跟着当前工作目录走，不钉在开发机上。"""
    out = _run_python("from heyan import config; print(config.PATHS.root)")
    assert Path(out[0]) == (Path.cwd() / "artifacts").resolve()


def test_root_follows_heyan_home(tmp_path):
    """HEYAN_HOME 指到哪儿，产物就落在哪儿——U 盘 / SD 卡部署全靠这个开关。"""
    target = tmp_path / "usb" / "heyan"
    out = _run_python(
        "from heyan import config; print(config.PATHS.root); print(config.PATHS.bundles)",
        {"HEYAN_HOME": str(target)},
    )
    assert Path(out[0]) == target.resolve()
    assert Path(out[1]) == target.resolve() / "bundles"


def test_bundled_assets_are_package_relative(tmp_path):
    """随包分发的资源（taxonomy / advisory / 语音包）按包位置定位，不跟 HEYAN_HOME 跑。"""
    out = _run_python(
        "from heyan import config; print(config.TAXONOMY_PATH); print(config.VOICEPACK_ROOT)",
        {"HEYAN_HOME": str(tmp_path / "elsewhere")},
    )
    assets = REPO / "heyan" / "assets"
    assert Path(out[0]) == assets / "taxonomy.json"
    assert Path(out[1]) == assets / "voicepacks"
    assert Path(out[0]).is_file()


def test_find_openssl_never_falls_back_to_a_pinned_drive(tmp_path):
    """证书脚本找 openssl 只看 PATH 与 ProgramFiles；都拿不到时返回 None，
    而不是回落到某个写死的安装盘。"""
    script = REPO / "tools" / "make_dev_cert.py"
    spec = importlib.util.spec_from_file_location("_make_dev_cert_probe", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    saved_path = os.environ.get("PATH")
    saved_pf = {k: os.environ.get(k) for k in ("ProgramFiles", "ProgramFiles(x86)")}
    try:
        os.environ["PATH"] = str(tmp_path)  # 空目录：PATH 上什么都没有
        for key in saved_pf:
            os.environ.pop(key, None)
        found = module.find_openssl()
        assert found is None, f"PATH 与 ProgramFiles 都清空后仍找到 {found}，说明有兜底的写死路径"
    finally:
        if saved_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = saved_path
        for key, value in saved_pf.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
