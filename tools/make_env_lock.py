#!/usr/bin/env python3
"""把当前 Python 环境钉成一份可复现的 requirements.lock.txt。

为什么需要它：
  `requirements.txt` 写的是 `torch>=2.1` 这种下限。换一台机器重新 `pip install`，
  装到的是那天最新的版本 —— 界面照样能开，但"一模一样"就没了：numpy 大版本动一下，
  INT8 推理的数值可能差最后一位；torch 从 `+cpu` 变成 CUDA 版，装包体积差 2GB，
  离线机器上还可能根本装不下来。

  这个脚本读 `requirements.txt` / `requirements-edge.txt` 声明的顶层依赖，顺着已安装
  的包元数据把传递依赖一起解出来，写成 `==` 精确锁定的清单。

  锁文件里会出现 `torch==2.14.0+cpu` 这种带本地版本号的条目，PyPI 上没有，所以脚本
  会在文件头自动补一行 `--extra-index-url https://download.pytorch.org/whl/cpu`。
  少了这行，新机器上的 pip 要么去找 CUDA 版，要么直接报 "No matching distribution"。

  本机装了但项目用不上的包（别的项目留下的 python-docx、MyQR 之类）不会进锁文件：
  这里算的是从声明顶层包往下的依赖闭包，不是 `pip freeze` 的全量快照。

用法：
    python tools/make_env_lock.py                  # 生成/刷新 requirements.lock.txt
    python tools/make_env_lock.py --check          # 在新机器上核对，不一致退出码 1
    python tools/make_env_lock.py --out lock.txt   # 换个输出路径
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from packaging.requirements import InvalidRequirement, Requirement

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DECLARED = ("requirements.txt", "requirements-edge.txt")
DEFAULT_OUT = ROOT / "requirements.lock.txt"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"

def canonical(name: str) -> str:
    """PEP 503 归一化：大小写、下划线、点、连杠线视为同一个名字。"""
    return re.sub(r"[-_.]+", "-", name).lower()


def declared_top_level(root: Path = ROOT) -> List[str]:
    """从声明文件里挑出顶层包名，忽略注释、`-r` 引用、pip 选项和本机用不上的标记。"""
    names: List[str] = []
    for rel in DECLARED:
        path = root / rel
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            req = _parse(line)
            if req is None or not _marker_applies(req):
                continue
            if canonical(req.name) not in {canonical(n) for n in names}:
                names.append(req.name)
    return names


def _parse(text: str) -> Optional[Requirement]:
    try:
        return Requirement(text)
    except InvalidRequirement:
        return None


def _marker_applies(req: Requirement) -> bool:
    """环境标记在当前机器上成立吗。

    torch 的元数据里挂着一串 `platform_system == "Linux"` 的 CUDA 包，还有
    `python_version < "3.11"` 才要的 tomli、`extra == "..."` 的可选扩展。
    这些本机都没装也不该装，算进闭包会让锁文件锁不上（脚本会报"锁不全"）。
    """
    return req.marker is None or bool(req.marker.evaluate())


def installed_dists() -> Dict[str, metadata.Distribution]:
    out: Dict[str, metadata.Distribution] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            out.setdefault(canonical(name), dist)
    return out


def dependency_closure(seeds: List[str],
                       dists: Dict[str, metadata.Distribution]) -> Tuple[List[str], List[str]]:
    """广度优先解出依赖闭包。返回 (闭包内的规范名, 声明了却没装的包名)。"""
    seen: Set[str] = set()
    missing: Set[str] = set()
    queue = [canonical(s) for s in seeds]
    while queue:
        key = queue.pop(0)
        if key in seen or key in missing:
            continue
        dist = dists.get(key)
        if dist is None:
            missing.add(key)
            continue
        seen.add(key)
        for raw in dist.requires or ():
            req = _parse(raw.strip())
            if req is None or not _marker_applies(req):
                continue
            queue.append(canonical(req.name))
    return sorted(seen), sorted(missing)


def pinned_rows(names: List[str],
                dists: Dict[str, metadata.Distribution]) -> List[Tuple[str, str, str]]:
    """(规范名, 展示名, 版本) 三元组，按规范名排序，保证多次生成结果一致。"""
    rows: List[Tuple[str, str, str]] = []
    for key in names:
        dist = dists[key]
        rows.append((key, dist.metadata["Name"] or key, dist.version))
    return rows


def render(rows: List[Tuple[str, str, str]], python_version: str, platform: str) -> str:
    needs_cpu_index = any("+" in ver for _, _, ver in rows)
    lines = [
        "# HeYan 环境锁：由 tools/make_env_lock.py 生成，别手改（改完重跑脚本即可）。",
        f"# 参考机器：Python {python_version} / {platform}",
        "# 新机器上装：python -m pip install -r requirements.lock.txt",
        "# 装完核对：  python tools/make_env_lock.py --check",
    ]
    if needs_cpu_index:
        lines += [
            "# 带 +cpu 的本地版本号 PyPI 上没有，必须加下面这个索引才装得到：",
            f"--extra-index-url {CPU_INDEX}",
        ]
    lines.append("")
    lines += [f"{display}=={ver}" for _, display, ver in rows]
    return "\n".join(lines) + "\n"


def parse_lock(path: Path) -> Tuple[Dict[str, str], Optional[str], Optional[str]]:
    """读锁文件，返回 (规范名 -> 版本, 参考 Python 版本, 参考平台)。"""
    pins: Dict[str, str] = {}
    python_version: Optional[str] = None
    platform_tag: Optional[str] = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if python_version is None and line.startswith("# 参考机器：Python "):
            tail = line.split("Python ", 1)[1]
            python_version = tail.split(" / ", 1)[0].strip()
            platform_tag = tail.split(" / ", 1)[1].strip() if " / " in tail else None
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if "==" not in line:
            continue
        name, _, ver = line.partition("==")
        pins[canonical(name.strip())] = ver.strip()
    return pins, python_version, platform_tag


def python_here() -> str:
    info = sys.version_info
    return f"{info.major}.{info.minor}.{info.micro}"


def check(path: Path) -> int:
    """在新机器上核对环境。一致返回 0，有出入返回 1。"""
    if not path.is_file():
        print(f"[env-lock] 找不到 {path.name}，先在原机器上跑一次 tools/make_env_lock.py")
        return 1
    pins, ref_python, platform_tag = parse_lock(path)
    dists = installed_dists()
    here = python_here()

    bad: List[str] = []
    if ref_python and ref_python != here:
        bad.append(f"Python 版本：锁文件 {ref_python}，本机 {here}")
    if platform_tag and platform_tag != sys.platform:
        # 依赖闭包是按标记算的，跨平台锁不住：Linux 那边还需要 CUDA/triton 一串包
        bad.append(f"平台：锁文件 {platform_tag}，本机 {sys.platform}"
                   "（这份锁只对同平台有效，换平台请在那台机器上重新生成）")
    for key in sorted(pins):
        dist = dists.get(key)
        if dist is None:
            bad.append(f"{key}：锁文件要 {pins[key]}，本机没装")
        elif dist.version != pins[key]:
            bad.append(f"{key}：锁文件要 {pins[key]}，本机是 {dist.version}")

    if bad:
        print(f"[env-lock] 与 {path.name} 不一致（{len(bad)} 处）：")
        for line in bad:
            print(f"[env-lock]   - {line}")
        print(f"[env-lock] 对齐办法：python -m pip install -r {path.name}")
        return 1
    print(f"[env-lock] 环境一致：Python {here}，{len(pins)} 个包全部对得上")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="生成/核对可复现的环境锁文件")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"锁文件路径（默认 {DEFAULT_OUT.name}）")
    parser.add_argument("--check", action="store_true",
                        help="只核对不写文件，环境不一致退出码 1")
    args = parser.parse_args(argv)

    if args.check:
        return check(args.out)

    seeds = declared_top_level()
    if not seeds:
        print("[env-lock] 没从 requirements*.txt 里读到任何依赖，拒绝生成空锁文件")
        return 2
    dists = installed_dists()
    names, missing = dependency_closure(seeds, dists)
    if missing:
        print(f"[env-lock] 这些声明过的包本机没装，锁不全：{', '.join(missing)}")
        print("[env-lock] 先 pip install -r requirements.txt 再重跑本脚本")
        return 2

    rows = pinned_rows(names, dists)
    text = render(rows, python_here(), sys.platform)
    args.out.write_text(text, encoding="utf-8")
    print(f"[env-lock] 已写 {args.out.name}（{len(rows)} 个包，Python {python_here()}）")
    local = [f"{display}=={ver}" for _, display, ver in rows if "+" in ver]
    if local:
        print(f"[env-lock] 含本地版本号：{', '.join(local)}（已自动补 CPU 索引行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
