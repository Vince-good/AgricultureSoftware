"""云端部署这条路的契约测试。

要守住的四件事：
  1. 端口与监听地址的解析口径：平台注入的 $PORT 优先，默认值就是 HF Spaces
     的约定端口，绝不写死成开发机上的 8080；
  2. build_app() 出来的是"评委点开链接就能用"的应用——健康检查不要口令就能
     200，设了口令就 401；
  3. Dockerfile / README 头 / requirements-cloud 三处口径必须和代码一致。
     改一处忘另一处的症状分别是"平台一直重启容器"和"镜像白白大 2GB"，
     两个都很难从日志里看出来；
  4. tools/make_cloud_bundle.py 生成的部署目录带着模型、不带本机识别记录。
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from heyan.server import cloud
from heyan.server import guard as guard_mod

REPO = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO / "Dockerfile"
DOCKERIGNORE = REPO / ".dockerignore"
CLOUD_REQS = REPO / "requirements-cloud.txt"
# 打包结果放测试沙箱里（artifacts 已被 .gitignore 和绝对路径守卫双双排除）。
# 不用 tmp_path_factory：本机系统 TEMP 目录权限异常，conftest 就是为此才把
# tmp_path 重定向到仓库内的。
STAGE_ROOT = REPO / "artifacts" / "_pytest_home" / "cloud-stage"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """每个用例从干净的环境变量开始，别让外面 shell 里残留的 PORT 串味。"""
    for name in ("PORT", "HEYAN_PORT", "HEYAN_HOST", "HOST",
                 "HEYAN_ACCESS_TOKEN", "HEYAN_RATE_RECOGNIZE", "HEYAN_RATE_DEFAULT",
                 "HEYAN_THREADS", "HEYAN_INFER_THREADS", "HEYAN_BUNDLE", "HEYAN_LANG"):
        monkeypatch.delenv(name, raising=False)


# ------------------------------------------------------------------ 端口与地址
def test_default_host_binds_all_interfaces():
    """云端必须绑 0.0.0.0：绑回环的话容器外面的负载均衡连不进来。"""
    assert cloud.DEFAULT_HOST == "0.0.0.0"
    assert cloud.resolve_host() == "0.0.0.0"


def test_host_env_override(monkeypatch):
    monkeypatch.setenv("HEYAN_HOST", "127.0.0.1")
    assert cloud.resolve_host() == "127.0.0.1"


def test_default_port_is_the_hf_convention():
    """没有 $PORT 时落到 7860，和 Dockerfile 的 EXPOSE、README 头三处一致。"""
    assert cloud.DEFAULT_PORT == 7860
    assert cloud.resolve_port() == 7860


def test_platform_port_wins(monkeypatch):
    monkeypatch.setenv("PORT", "5000")
    assert cloud.resolve_port() == 5000


def test_heyan_port_beats_platform_port(monkeypatch):
    """HEYAN_PORT 是我们自己的开关，压过平台注入的 PORT。"""
    monkeypatch.setenv("PORT", "5000")
    monkeypatch.setenv("HEYAN_PORT", "6000")
    assert cloud.resolve_port() == 6000


def test_explicit_port_beats_everything(monkeypatch):
    monkeypatch.setenv("HEYAN_PORT", "6000")
    assert cloud.resolve_port(9001) == 9001


def test_blank_port_env_is_ignored(monkeypatch):
    """平台有时会把 PORT 设成空串，那等同于没设，不能当成 0 端口。"""
    monkeypatch.setenv("PORT", "   ")
    assert cloud.resolve_port() == cloud.DEFAULT_PORT


@pytest.mark.parametrize("raw", ["abc", "70000", "0", "-1"])
def test_bad_port_fails_loudly(monkeypatch, raw):
    """端口不合法就退出。悄悄顺延端口在云端是最坏的结局：日志说起来了，
    平台健康检查的却是它分配的那个端口，链接永远打不开。"""
    monkeypatch.setenv("PORT", raw)
    with pytest.raises(SystemExit):
        cloud.resolve_port()


# ------------------------------------------------------------------ 限流额度
def test_cloud_limits_are_looser_than_local():
    """比赛现场评委常挤在会场同一个出口 IP 后面，按本机口径限流会集体挡在门外。"""
    local = guard_mod.DEFAULT_LIMITS["recognize"][0]
    assert cloud.CLOUD_LIMITS["recognize"][0] > local
    assert cloud.CLOUD_LIMITS["default"][0] >= guard_mod.DEFAULT_LIMITS["default"][0]


def test_resolve_limits_env_override(monkeypatch):
    monkeypatch.setenv("HEYAN_RATE_RECOGNIZE", "5")
    limits = cloud.resolve_limits()
    assert limits["recognize"] == (5, 60.0)
    # 没设的那档保持云端默认
    assert limits["default"] == cloud.CLOUD_LIMITS["default"]


@pytest.mark.parametrize("name", ["HEYAN_RATE_RECOGNIZE", "HEYAN_RATE_DEFAULT"])
def test_resolve_limits_rejects_garbage(monkeypatch, name):
    monkeypatch.setenv(name, "0")
    with pytest.raises(SystemExit):
        cloud.resolve_limits()
    monkeypatch.setenv(name, "lots")
    with pytest.raises(SystemExit):
        cloud.resolve_limits()


# ------------------------------------------------------------------ 应用口径
def test_build_app_is_open_for_judges(bundle_dir):
    """不设口令时：健康检查直接 200，公网加固开着，限流用云端口径。"""
    app = cloud.build_app(bundle=bundle_dir)
    try:
        cfg = app.config["HEYAN_GUARD"]
        assert cfg["public"] is True
        assert cfg["token"] is None
        assert cfg["limiter"].limits == cloud.resolve_limits()
        payload = app.test_client().get("/api/health").get_json()
        assert payload["ok"] is True
        assert payload["engine"] is True, "模型没预热成功，评委第一下就会看到报错"
        assert payload["guard"] == {"auth": False, "public": True, "rate_limited": True}
        # 公网模式不该把本机路径泄露给访问者
        assert "\\" not in str(payload.get("bundle", ""))
    finally:
        app.extensions["heyan"].close()


def test_build_app_honours_token(monkeypatch, bundle_dir):
    """设了 HEYAN_ACCESS_TOKEN 就多一道口令：裸访问 401，带口令头 200。"""
    monkeypatch.setenv("HEYAN_ACCESS_TOKEN", "s3cret")
    app = cloud.build_app(bundle=bundle_dir)
    try:
        client = app.test_client()
        assert client.get("/api/health").status_code == 401
        ok = client.get("/api/health", headers={"X-HeYan-Token": "s3cret"})
        assert ok.status_code == 200
        assert ok.get_json()["ok"] is True
    finally:
        app.extensions["heyan"].close()


def test_build_app_lands_on_resolved_port(monkeypatch, bundle_dir):
    monkeypatch.setenv("PORT", "4321")
    app = cloud.build_app(bundle=bundle_dir)
    try:
        assert app.config["HEYAN_HOST"] == "0.0.0.0"
        assert app.config["HEYAN_PORT"] == 4321
    finally:
        app.extensions["heyan"].close()


def test_banner_reports_the_port_it_will_use(bundle_dir, capsys, monkeypatch):
    """横幅里的端口必须就是实际监听的端口，否则排障时被自己的日志骗。"""
    monkeypatch.setenv("PORT", "4321")
    app = cloud.build_app(bundle=bundle_dir)
    try:
        cloud._banner(app, cloud.resolve_host(), cloud.resolve_port())
    finally:
        app.extensions["heyan"].close()
    out = capsys.readouterr().out
    assert "0.0.0.0:4321" in out
    assert "/api/health" in out


# ------------------------------------------------------------------ 构建口径一致
def test_dockerfile_exists_and_matches_code():
    assert DOCKERFILE.is_file(), "缺 Dockerfile，云端没法构建"
    lines = DOCKERFILE.read_text(encoding="utf-8").splitlines()
    exposed = [ln.split()[-1] for ln in lines if ln.startswith("EXPOSE")]
    assert exposed == [str(cloud.DEFAULT_PORT)], (
        f"EXPOSE {exposed} 与 cloud.DEFAULT_PORT={cloud.DEFAULT_PORT} 不一致，"
        "平台会连不上容器")
    cmd = next(ln for ln in lines if ln.startswith("CMD"))
    assert "heyan.server.cloud" in cmd, "CMD 没走云端入口，端口和绑定地址都会退回本机口径"


def test_dockerfile_installs_openmp_runtime():
    """onnxruntime 链接了 libgomp，slim 镜像里没有；漏装的话 import 失败，
    服务会静默退回慢一个数量级的纯 NumPy 后端——演示时看着就像卡住了。"""
    assert "libgomp1" in DOCKERFILE.read_text(encoding="utf-8")


def test_dockerfile_sets_a_writable_runtime_home():
    lines = DOCKERFILE.read_text(encoding="utf-8").splitlines()
    assert any(ln.startswith("ENV") or "HEYAN_HOME=" in ln for ln in lines)
    assert "HEYAN_HOME=" in "\n".join(lines), "没设 HEYAN_HOME，记录库会落到不可写的位置"
    assert any(ln.startswith("USER") for ln in lines), "HF Spaces 以 uid 1000 运行容器"


def test_dockerignore_keeps_bundle_but_drops_local_records():
    lines = [ln.strip() for ln in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    assert "artifacts/*" in lines, "artifacts 整体进镜像会带上本机识别记录"
    assert "!artifacts/bundles" in lines, "模型包必须留在镜像里"
    # artifacts/* + !artifacts/bundles 的组合下，records 仍然是被排除的
    assert "!artifacts/records" not in lines


def test_cloud_requirements_skip_training_deps():
    """torch 只在训练/导出时用。带上它镜像凭空大 2GB，HF 上要多构建好几分钟。"""
    text = CLOUD_REQS.read_text(encoding="utf-8").lower()
    packages = {ln.split(">")[0].split("=")[0].strip()
                for ln in text.splitlines()
                if ln.strip() and not ln.startswith("#")}
    for banned in ("torch", "torchvision", "flask-cors"):
        assert banned not in packages, f"{banned} 不该出现在云端依赖里"
    for needed in ("flask", "onnxruntime", "waitress", "numpy", "pillow"):
        assert needed in packages, f"云端跑不起来：缺 {needed}"


# ------------------------------------------------------------------ 打包脚本
def _load_tool():
    path = REPO / "tools" / "make_cloud_bundle.py"
    spec = importlib.util.spec_from_file_location("_make_cloud_bundle", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_make_cloud_bundle"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stage(bundle_dir):
    """跑一次真实打包（slim + 只带普通话，够快），后面的用例共用结果。"""
    tool = _load_tool()
    out = STAGE_ROOT
    shutil.rmtree(out, ignore_errors=True)
    saved = sys.argv
    try:
        sys.argv = [sys.argv[0], "--bundle", str(bundle_dir), "--out", str(out),
                    "--slim", "--voice-langs", "zh"]
        assert tool.main() == 0
    finally:
        sys.argv = saved
    return tool, out, Path(bundle_dir)


def test_stage_carries_code_and_model(stage):
    _tool, out, src = stage
    assert (out / "heyan" / "server" / "cloud.py").is_file()
    assert (out / "Dockerfile").is_file()
    assert (out / "requirements-cloud.txt").is_file()
    bundle = out / "artifacts" / "bundles" / src.name
    assert (bundle / "model_int8.onnx").is_file()
    assert (bundle / "manifest.json").is_file()
    assert (bundle / "labels.json").is_file()
    # benchmark.json 设置页要读，slim 也不能省
    assert (bundle / "benchmark.json").is_file()


def test_stage_slim_drops_the_fallbacks(stage):
    _tool, out, src = stage
    bundle = out / "artifacts" / "bundles" / src.name
    assert not (bundle / "model_fp32.onnx").exists()
    assert not (bundle / "model_portable.hgraph.npz").exists()


def test_stage_never_carries_local_runtime_data(stage):
    """本机攒下的识别记录一旦进镜像，等于公开给每一个点开链接的人。"""
    _tool, out, _src = stage
    artifacts = out / "artifacts"
    for banned in ("records", "exports", "outbox", "data", "models", "_pytest_home"):
        assert not (artifacts / banned).exists(), f"部署目录里不该有 artifacts/{banned}"
    assert not list(out.rglob("*.db")), "部署目录里不该有 sqlite 记录库"
    assert not list(out.rglob("__pycache__")), "缓存目录不该被拷上去"


def test_stage_voicepack_is_trimmed_and_index_stays_honest(stage):
    """裁掉 en 之后索引也要跟着裁：留着指向空气的条目，界面统计会说谎。"""
    _tool, out, src = stage
    pack = out / "artifacts" / "bundles" / src.name / "voicepack"
    assert (pack / "zh").is_dir()
    assert not (pack / "en").exists()
    index = json.loads((pack / "index.json").read_text(encoding="utf-8"))
    assert index["entries"], "语音索引被裁空了"
    assert all(e["lang"] == "zh" for e in index["entries"])
    for entry in index["entries"]:
        assert (pack / entry["file"]).is_file(), f"索引指向不存在的文件：{entry['file']}"


def test_stage_readme_is_a_docker_space(stage):
    """HF Spaces 靠 README.md 顶部的 YAML 头认 SDK；写错了构建直接失败。"""
    _tool, out, _src = stage
    text = (out / "README.md").read_text(encoding="utf-8")
    assert text.startswith("---"), "README 必须以 YAML 头开头"
    head = text.split("---")[1]
    assert "sdk: docker" in head
    assert f"app_port: {cloud.DEFAULT_PORT}" in head


def test_stage_gitattributes_uses_lfs(stage):
    """不走 lfs，push 会以 file exceeds 失败——之前推不上 GitHub 就是这个原因。"""
    _tool, out, _src = stage
    text = (out / ".gitattributes").read_text(encoding="utf-8")
    for pattern in ("*.onnx", "*.wav"):
        assert f"{pattern} filter=lfs" in text


def test_stage_gitignore_does_not_hide_the_model(stage):
    """部署目录的 .gitignore 必须和仓库根那份口径相反：这里模型是要提交的。"""
    _tool, out, _src = stage
    lines = (out / ".gitignore").read_text(encoding="utf-8").splitlines()
    active = {ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")}
    for banned in ("artifacts/", "*.onnx", "*.wav"):
        assert banned not in active, f"{banned} 被忽略的话模型根本推不上去"
    # 但运行期产物要挡住：在部署目录里本地试跑过一次，别把记录推上公网
    assert "artifacts/records/" in active


def test_stage_refuses_to_clobber_a_foreign_dir(stage, tmp_path):
    """清空输出目录前先看标记文件，别把用户自己的东西 rmtree 掉。"""
    tool, _out, src = stage
    foreign = tmp_path / "not-mine"
    foreign.mkdir()
    (foreign / "important.txt").write_text("keep me", encoding="utf-8")
    saved = sys.argv
    try:
        sys.argv = [sys.argv[0], "--bundle", str(src), "--out", str(foreign)]
        with pytest.raises(SystemExit):
            tool.main()
    finally:
        sys.argv = saved
    assert (foreign / "important.txt").is_file()


def test_staged_app_boots_from_the_stage_dir(stage):
    """最硬的一条：把部署目录当成项目根，用云端入口建应用，健康检查得绿。

    这能抓住"打包漏了文件"这类只在云端才现形的问题。

    不用 monkeypatch：这里要在用例内部就把 sys.path / sys.modules / HEYAN_HOME
    全部还原，否则 config.PATHS 会在重新导入时被冻结到部署目录上，
    后面所有用例（包括别的文件）都会读到那份假产物目录。
    """
    _tool, out, _src = stage
    def _heyan_mods():
        return [m for m in list(sys.modules) if m == "heyan" or m.startswith("heyan.")]

    saved_path = list(sys.path)
    saved_home = os.environ.get("HEYAN_HOME")
    saved_mods = {name: sys.modules[name] for name in _heyan_mods()}
    try:
        for name in saved_mods:
            del sys.modules[name]
        sys.path.insert(0, str(out))
        os.environ["HEYAN_HOME"] = str(out / "artifacts")
        import heyan.server.cloud as staged_cloud

        assert Path(staged_cloud.__file__).parent == out / "heyan" / "server"
        app = staged_cloud.build_app()
        try:
            payload = app.test_client().get("/api/health").get_json()
            assert payload["ok"] is True
            assert payload["engine"] is True
        finally:
            app.extensions["heyan"].close()
    finally:
        for name in _heyan_mods():
            del sys.modules[name]
        sys.modules.update(saved_mods)   # 原封不动还回去，连对象身份都不变
        sys.path[:] = saved_path
        if saved_home is None:
            os.environ.pop("HEYAN_HOME", None)
        else:
            os.environ["HEYAN_HOME"] = saved_home


def test_stage_dir_is_outside_version_control():
    """打包结果落在 artifacts/ 下：既不进 git，也不会被绝对路径守卫扫到。"""
    assert STAGE_ROOT.parts[: -2] == (REPO / "artifacts").parts
