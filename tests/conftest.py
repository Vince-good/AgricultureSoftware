"""pytest 全局夹具。

两件事必须在 import heyan 之前完成，否则测试会写到真实数据目录：
  1. 用 HEYAN_HOME 把 PATHS 重定向到临时目录（config.PATHS 是导入期冻结的）；
  2. 把真实模型包"借"进临时目录（junction 优先，失败再拷贝），
     这样识别/语音/预算测试跑的是真产物，却不碰 artifacts/。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REAL_BUNDLE = REPO / "artifacts" / "bundles" / "heyan-mnv3s-int8-v1.0.0"
DEMO_IMAGE = (REPO / "artifacts" / "data" / "demo_dataset"
              / "rice_blast" / "rice_blast_0000.jpg")

_TMP_HOME = Path(os.environ.get("HEYAN_TEST_HOME")
                 or (REPO / "artifacts" / "_pytest_home"))


def _prepare_home() -> Path:
    _TMP_HOME.mkdir(parents=True, exist_ok=True)
    bundles = _TMP_HOME / "bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    link = bundles / REAL_BUNDLE.name
    if not link.exists() and REAL_BUNDLE.exists():
        try:
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(REAL_BUNDLE)],
                check=True, capture_output=True,
            )
        except (subprocess.CalledProcessError, OSError):
            shutil.copytree(REAL_BUNDLE, link)
    return _TMP_HOME


os.environ["HEYAN_HOME"] = str(_prepare_home())
sys.path.insert(0, str(REPO))


@pytest.fixture()
def tmp_path(request):
    """覆盖内置 tmp_path。

    本机的系统 TEMP 目录权限异常（新建目录 0700，随后 WinError 5），
    内置 tmp_path 会踩坑；改到仓库内的测试沙箱下，权限正常且事后可清理。
    """
    base = _TMP_HOME / "tmp" / request.node.name
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture(scope="session")
def bundle_dir():
    if not REAL_BUNDLE.exists():
        pytest.skip("没有构建模型包，先运行 python -m heyan.cli build")
    return REAL_BUNDLE


@pytest.fixture(scope="session")
def app(bundle_dir):
    from heyan.server.app import create_app

    application = create_app(bundle=bundle_dir, language="zh", warm=False)
    yield application
    application.extensions["heyan"].close()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture(scope="session")
def demo_image():
    if not DEMO_IMAGE.exists():
        pytest.skip("没有演示数据集，先运行 python -m heyan.cli demo-data")
    return DEMO_IMAGE
