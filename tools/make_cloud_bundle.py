#!/usr/bin/env python3
"""生成能直接推上云的部署目录（HF Spaces / 任意 Docker 主机通用）。

为什么要单独打包，而不是把整个仓库推上去：
  仓库根的 .gitignore 把 artifacts/ 整体挡在版本库外——模型和语音包太大，
  GitHub 推不上去。可云端镜像又必须带着模型包，否则容器起来只会显示
  "请先构建模型包"。两边口径天生冲突，所以给云端单独备一份目录：
  代码 + 一个模型包，再加上 HF Spaces 认的 README 头、git-lfs 规则，
  以及一份"不忽略模型"的 .gitignore。

  顺带把本机的东西挡在外面：artifacts/records/heyan.db 是真实识别记录，
  打进镜像等于把自己的数据交给每一个点开链接的人；artifacts/data 里的演示
  数据集同理，几十 MB 的无用负载。这个脚本按白名单拷贝，不做全量复制。

用法：
  python tools/make_cloud_bundle.py                     # -> artifacts/cloud
  python tools/make_cloud_bundle.py --slim              # 去掉 fp32/npz，省 11.9MB
  python tools/make_cloud_bundle.py --voice-langs zh    # 只带普通话，再省 17MB
  python tools/make_cloud_bundle.py --zip               # 顺手压成 zip，scp / U 盘搬运
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heyan.core.bundle import ModelBundle  # noqa: E402
from heyan.data.schema import file_sha256  # noqa: E402
from heyan.server.state import resolve_bundle  # noqa: E402
from heyan.tts.voicepack import VoicePack  # noqa: E402

# 输出目录里放这个标记文件，用来区分"上次是我生成的"和"用户自己的东西"。
# 清理前先看标记：没有标记就拒绝覆盖，绝不顺手 rmtree 掉别人的目录。
MARKER = ".heyan-cloud-staging"

# 随镜像一起上云的代码。tests / docs 不进镜像（.dockerignore 挡着），
# 但进 Space 仓库——队友在网页上就能翻到文档，也方便云端自己跑一遍测试。
CODE_DIRS = ("heyan", "tools", "tests", "docs")
CODE_FILES = ("Dockerfile", ".dockerignore", "requirements-cloud.txt",
              "requirements.txt", "pyproject.toml")

SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                  ".ipynb_checkpoints"}

# 少任何一个，云端启动就会炸：manifest 决定能不能 load，labels / advisory 决定
# 类别名和防治建议，int8 是唯一的推理模型，benchmark.json 设置页要读。
REQUIRED_BUNDLE_FILES = ("manifest.json", "labels.json", "advisory.json",
                         "model_int8.onnx", "benchmark.json")

# --slim 省掉的大件。fp32 是没有 onnxruntime 时的回退、npz 是纯 NumPy 解释器
# 用的图，云端镜像一定装了 onnxruntime，这两个都用不上（合计 11.9MB）。
# build_report.json 只是训练时写下的构建记录，运行期没人读。
SLIM_SKIP = ("model_fp32.onnx", "model_portable.hgraph.npz", "build_report.json")

# HF Spaces 的 Docker SDK 约定端口，必须和 Dockerfile 的 EXPOSE、
# heyan/server/cloud.py 的 DEFAULT_PORT 三者一致（测试会盯着这条）。
APP_PORT = 7860

LFS_PATTERNS = ("*.onnx", "*.wav", "*.npz", "*.pt", "*.pth", "*.zip",
                "*.jpg", "*.jpeg", "*.png", "*.db")


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def _mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def _dir_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1024 / 1024


def _ignore(_src: str, names: list) -> set:
    """copytree 的过滤器：缓存目录和字节码一律不带。"""
    return {n for n in names if n in SKIP_DIR_NAMES or n.endswith((".pyc", ".pyo"))}


def _clean_out(out: Path, force: bool) -> None:
    """清空输出目录。只清自己生成过的，别的目录要求 --force。"""
    if not out.exists():
        out.mkdir(parents=True, exist_ok=True)
        return
    if not any(out.iterdir()):
        return
    if not (out / MARKER).exists() and not force:
        raise SystemExit(
            f"[cloud-bundle] {out} 已存在且不是本脚本生成的目录。\n"
            f"[cloud-bundle] 换个 --out，或者确认可以清空后加 --force。")
    shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# 拷贝
# --------------------------------------------------------------------------
def copy_code(out: Path) -> int:
    """把代码拷进部署目录。返回文件数。"""
    copied = 0
    for name in CODE_DIRS:
        src = ROOT / name
        if not src.is_dir():
            print(f"[cloud-bundle] 跳过不存在的目录 {name}/")
            continue
        shutil.copytree(src, out / name, ignore=_ignore, dirs_exist_ok=True)
        copied += sum(1 for _ in (out / name).rglob("*") if _.is_file())
    for name in CODE_FILES:
        src = ROOT / name
        if not src.is_file():
            print(f"[cloud-bundle] 跳过不存在的文件 {name}")
            continue
        shutil.copy2(src, out / name)
        copied += 1
    return copied


def copy_bundle(src: Path, out: Path, slim: bool, langs: list) -> Path:
    """按白名单拷一个模型包。返回部署目录里的包路径。"""
    dest = out / "artifacts" / "bundles" / src.name
    dest.mkdir(parents=True, exist_ok=True)

    skip = set(SLIM_SKIP) if slim else set()
    missing = []
    for item in sorted(src.iterdir()):
        if item.name == "voicepack":
            continue  # 语音包单独处理，要按语言裁
        if item.is_dir():
            # 模型包里不该有别的子目录；真出现了也不猜，照实报出来
            print(f"[cloud-bundle] 警告：模型包里有意外目录 {item.name}/，已跳过")
            continue
        if item.name in skip:
            print(f"[cloud-bundle] --slim：不带 {item.name}（{_mb(item):.2f} MB）")
            continue
        shutil.copy2(item, dest / item.name)
    for name in REQUIRED_BUNDLE_FILES:
        if not (dest / name).exists():
            missing.append(name)
    if missing:
        raise SystemExit(f"[cloud-bundle] 模型包缺关键文件：{'、'.join(missing)}\n"
                         f"[cloud-bundle] 源目录：{src}")

    copy_voicepack(src / "voicepack", dest / "voicepack", langs)
    return dest


def copy_voicepack(src: Path, dest: Path, langs: list) -> None:
    """拷语音包，只留选中的语言，并把 index.json 一起裁掉。

    裁 index.json 不是必须的——voicepack 的扫描器会跳过文件不存在的条目，
    resolve_slug 还有 zh 回退链，删掉 en 目录也不会 404。但索引里留着几百条
    指向空气的记录，界面统计和日志都会说谎，所以一并过滤干净。
    """
    if not src.is_dir():
        print("[cloud-bundle] 模型包里没有 voicepack/，界面会退回在线合成或静音")
        dest.mkdir(parents=True, exist_ok=True)
        return
    dest.mkdir(parents=True, exist_ok=True)
    wanted = set(langs)

    for item in sorted(src.iterdir()):
        if item.is_file():
            continue  # index.json 最后单独生成
        if item.name in wanted:
            shutil.copytree(item, dest / item.name, ignore=_ignore, dirs_exist_ok=True)
        elif item.name == "recordings":
            # 真人录音也按语言分目录，同样只留选中的
            for sub in sorted(item.iterdir()):
                if sub.is_dir() and sub.name in wanted:
                    shutil.copytree(sub, dest / item.name / sub.name,
                                    ignore=_ignore, dirs_exist_ok=True)
        else:
            print(f"[cloud-bundle] 语音包：不带 {item.name}/（{_dir_mb(item):.1f} MB）")

    index_src = src / "index.json"
    if not index_src.is_file():
        return
    raw = json.loads(index_src.read_text(encoding="utf-8"))
    before = len(raw.get("entries", []))
    raw["entries"] = [e for e in raw.get("entries", []) if e.get("lang") in wanted]
    for key in ("languages", "requested_languages", "unvoiced_languages"):
        if isinstance(raw.get(key), list):
            raw[key] = [c for c in raw[key] if c in wanted or key == "unvoiced_languages"]
    stats = raw.get("stats")
    if isinstance(stats, dict):
        stats["entries"] = len(raw["entries"])
        stats["trimmed_from"] = before
        stats["trimmed_languages"] = sorted(wanted)
    (dest / "index.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cloud-bundle] 语音索引：{before} 条 -> {len(raw['entries'])} 条"
          f"（保留 {'、'.join(sorted(wanted))}）")


# --------------------------------------------------------------------------
# 生成云端要的几个配置文件
# --------------------------------------------------------------------------
def write_readme(out: Path, title: str, bundle_name: str, langs: list) -> None:
    """HF Spaces 靠仓库根 README.md 顶部的 YAML 头认这是个 Docker Space。

    没有这段头，HF 会当成静态页面或者 Gradio，构建直接失败。
    app_port 必须等于容器真正监听的端口。
    """
    body = f"""---
title: {title}
emoji: 🌽
colorFrom: green
colorTo: yellow
sdk: docker
app_port: {APP_PORT}
pinned: false
license: mit
short_description: 农作物病虫害离线识别（MobileNetV3-Small INT8）
---

# {title}

面向小农户的农作物病虫害识别与防治建议。识别在容器内完成，不依赖任何外部服务。

- 模型包：`{bundle_name}`（MobileNetV3-Small，INT8 量化，17 类）
- 离线语音：{'、'.join(langs)}
- 启动命令：`python -m heyan.server.cloud`（监听 0.0.0.0，端口取 `$PORT`，默认 {APP_PORT}）
- 健康检查：`GET /api/health`

完整文档在仓库的 `docs/` 目录：部署见 `docs/deploy-cloud.md`，日常运维见
`docs/operations.md`。

## 环境变量（都可选）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `PORT` / `HEYAN_PORT` | {APP_PORT} | 监听端口 |
| `HEYAN_ACCESS_TOKEN` | 空 | 设了就要口令才能进；不设则拿到链接的人都能用 |
| `HEYAN_THREADS` | 8 | waitress 的 HTTP 工作线程数 |
| `HEYAN_RATE_RECOGNIZE` | 30 | 每个来源每分钟识别次数上限 |
| `HEYAN_RATE_DEFAULT` | 600 | 每个来源每分钟其他请求上限 |
| `HEYAN_HOME` | 容器内 artifacts | 记录库和导出件的落盘位置，必须可写 |

## 已知取舍

免费档容器闲置约 48 分钟会休眠，评委第一次点开链接需要等十几秒冷启动。
容器文件系统不持久，重启后识别记录（sqlite）会清空——演示够用，别当数据库。
"""
    (out / "README.md").write_text(body, encoding="utf-8")


def write_gitattributes(out: Path) -> None:
    """git-lfs 规则。

    GitHub / HF 都拒绝单文件超过 100MB 的普通提交，语音包和 onnx 是二进制，
    不走 lfs 的话 push 会以 "file exceeds" 失败——这正是之前推不上去的原因。
    """
    lines = ["# 云端 Space 仓库：模型与语音包必须走 git-lfs，否则 push 会被拒。"]
    lines += [f"{p} filter=lfs diff=lfs merge=lfs -text" for p in LFS_PATTERNS]
    (out / ".gitattributes").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gitignore(out: Path) -> None:
    """部署目录自己的 .gitignore。

    故意不忽略 artifacts/ 和 *.onnx / *.wav：仓库根那份是为了把大文件挡在
    GitHub 外，而 Space 仓库恰恰必须带上它们。直接复制根 .gitignore 的话，
    `git add .` 会静默漏掉整个模型包，构建出来的容器是空的。

    但运行期产物要忽略：PATHS.ensure() 在启动时会建出 records / exports /
    outbox 等目录，识别记录就落在 artifacts/records/heyan.db。在部署目录里
    本地试跑过一次，再 `git add .` 就会把这份记录推上公网。
    """
    body = """# 云端 Space 仓库：只忽略缓存和本地垃圾。
# 注意这里**不能**忽略 artifacts/ 或 *.onnx / *.wav —— 模型包和语音包
# 必须提交上去（走 git-lfs，见 .gitattributes），否则容器里没有模型。
__pycache__/
*.py[cod]
.pytest_cache/
pytest-cache-files-*/
*.egg-info/
.venv/
venv/
.DS_Store
Thumbs.db

# 运行期产物：容器里由服务自己创建，不该进版本库。
# 如果你在部署目录里本地试跑过，这一节能挡住识别记录被误推上公网。
artifacts/records/
artifacts/exports/
artifacts/outbox/
artifacts/models/
artifacts/data/
artifacts/voicepacks/
*.db
*.sqlite3
"""
    (out / ".gitignore").write_text(body, encoding="utf-8")


# --------------------------------------------------------------------------
# 自检：拷完就在本地把包读一遍，别等云端构建失败了才发现
# --------------------------------------------------------------------------
def verify(bundle_dest: Path) -> list:
    problems = []
    try:
        bundle = ModelBundle.load(bundle_dest)
    except Exception as exc:  # noqa: BLE001 —— 自检要把任何异常都变成一条人话
        return [f"模型包加载失败：{type(exc).__name__}: {exc}"]

    kinds = bundle.available_kinds()
    if "int8" not in kinds:
        problems.append(f"没有可用的 int8 模型（available_kinds={kinds}）")
    n_classes = bundle.manifest.num_classes
    try:
        n_labels = len(bundle.taxonomy.classes)
    except Exception as exc:  # noqa: BLE001
        n_labels = -1
        problems.append(f"labels.json 读不出来：{type(exc).__name__}: {exc}")
    if n_labels >= 0 and n_labels != n_classes:
        problems.append(f"labels.json 有 {n_labels} 类，manifest 声明 {n_classes} 类")

    voice_dir = bundle_dest / "voicepack"
    if voice_dir.is_dir():
        pack = VoicePack([voice_dir])
        found = pack.languages()
        if not found:
            problems.append("voicepack 里一条可用语音都没有")
        else:
            sample = pack.entries(found[0])[:1]
            for entry in sample:
                if pack.resolve_slug(entry.slug, found[0]) is None:
                    problems.append(f"语音条目解析不到文件：{entry.slug}")
            print(f"[cloud-bundle] 语音自检：语言 {'、'.join(found)}，"
                  f"共 {len(pack.entries())} 条")
    return problems


def make_zip(out: Path) -> Path:
    dest = out.parent / f"{out.name}.zip"
    if dest.exists():
        dest.unlink()
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out.parent).as_posix())
    return dest


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", help="模型包目录（默认取 artifacts/bundles 下最新的一个）")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "cloud"),
                    help="部署目录输出位置（默认 artifacts/cloud，已被 .gitignore 忽略）")
    ap.add_argument("--slim", action="store_true",
                    help="不带 fp32 / npz / build_report，省约 12MB（云端用不到）")
    ap.add_argument("--voice-langs", default="",
                    help="只带这些语言的语音包，逗号分隔（如 zh）。默认全带")
    ap.add_argument("--space-title", default="Heyan Crop Doctor",
                    help="HF Spaces 上显示的名字")
    ap.add_argument("--zip", action="store_true", help="顺手压成一个 zip")
    ap.add_argument("--force", action="store_true",
                    help="允许清空一个不是本脚本生成的 --out 目录")
    args = ap.parse_args()

    src_bundle = resolve_bundle(args.bundle)
    if src_bundle is None:
        print("[cloud-bundle] 没找到模型包，先跑 python -m heyan.cli build")
        return 2
    src_bundle = Path(src_bundle)

    langs = [c.strip() for c in args.voice_langs.split(",") if c.strip()]
    if not langs:
        pack_dir = src_bundle / "voicepack"
        langs = sorted(d.name for d in pack_dir.iterdir()
                       if d.is_dir() and d.name != "recordings") if pack_dir.is_dir() else []
        if not langs:
            langs = ["zh"]

    out = Path(args.out).expanduser()
    if not out.is_absolute():
        out = ROOT / out
    _clean_out(out, args.force)

    print(f"[cloud-bundle] 源模型包 -> {src_bundle}")
    print(f"[cloud-bundle] 输出目录 -> {out}")
    n_code = copy_code(out)
    print(f"[cloud-bundle] 代码 {n_code} 个文件")

    bundle_dest = copy_bundle(src_bundle, out, args.slim, langs)
    (out / MARKER).write_text(
        f"generated by tools/make_cloud_bundle.py\nsource bundle: {src_bundle.name}\n",
        encoding="utf-8")

    write_readme(out, args.space_title, src_bundle.name, langs)
    write_gitattributes(out)
    write_gitignore(out)

    problems = verify(bundle_dest)
    if problems:
        print("[cloud-bundle] 自检没通过：")
        for p in problems:
            print(f"[cloud-bundle]   - {p}")
        return 1

    int8 = bundle_dest / "model_int8.onnx"
    print(f"[cloud-bundle] 自检通过：int8 模型 sha256 {file_sha256(int8)[:16]}… "
          f"({_mb(int8):.2f} MB)")
    print(f"[cloud-bundle] 体积：代码 {_dir_mb(out / 'heyan') + _dir_mb(out / 'tools'):.1f} MB，"
          f"模型包 {_dir_mb(bundle_dest):.1f} MB，合计 {_dir_mb(out):.1f} MB")

    if args.zip:
        zip_path = make_zip(out)
        print(f"[cloud-bundle] zip -> {zip_path}（{_mb(zip_path):.1f} MB）"
              f"  sha256 {file_sha256(zip_path)[:16]}…")

    n_files = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"[cloud-bundle] 完成：{n_files} 个文件")
    print("")
    print("[cloud-bundle] 下一步（HF Spaces，详细步骤见 docs/deploy-cloud.md）：")
    print(f"  cd {out}")
    print("  git init -b main")
    print("  git lfs install")
    print("  git add .")
    print("  git commit -m \"deploy: 云端镜像\"")
    print("  git remote add space https://huggingface.co/spaces/<用户名>/<空间名>")
    print("  git push space main --force")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
