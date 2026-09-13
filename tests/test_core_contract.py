"""文档硬指标的回归测试：体积 / 延迟 / 内存 / 精度损失 / 类别与话术完整性。

这些数字不是风格问题，是方案文档 3.2 节写死的验收线；
构建产物一旦越线，这里必须红。
"""

from __future__ import annotations

import json

import pytest

from heyan import config
from heyan.classes import load_taxonomy
from heyan.i18n import STRINGS, coverage, table
from heyan.tts import languages


@pytest.fixture(scope="module")
def benchmark(bundle_dir):
    path = bundle_dir / "benchmark.json"
    if not path.exists():
        pytest.skip("bundle 缺少 benchmark.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_model_size_budget(benchmark):
    size = benchmark["model_size"]
    # 20MB 卡的是"设备上实际要装的那一个文件"，不是包里所有产物的总和：
    # FP32 只是训练对照，便携 npz 只是退路，一台设备只会用其中一个。
    assert size["deployed_mb"] <= config.MODEL_SIZE_MAX_MB
    assert size["deployed_kind"] == "int8"
    assert size["all_models_mb"] >= size["deployed_mb"]


def test_latency_budget(benchmark):
    p95 = benchmark["latency_ms"]["total_ms"]["p95"]
    assert p95 / 1000.0 <= config.LATENCY_MAX_S


def test_memory_budget(benchmark):
    delta = benchmark["memory_mb"]["runtime_delta_mb"]
    assert delta <= config.RUNTIME_MEMORY_MAX_MB


def test_quantization_accuracy_drop(benchmark):
    checks = {c["name"]: c for c in benchmark["budgets"]["checks"]}
    drop = None
    for name, check in checks.items():
        if "accuracy" in name or "drop" in name:
            drop = check
    assert benchmark["budget_ok"] is True
    if drop is not None:
        assert drop["ok"] is True


def test_taxonomy_shape():
    tax = load_taxonomy()
    ids = [c.id for c in tax]
    assert len(ids) == 14
    assert "unusable" in ids
    # 展示序按作物分组、兜底类垫底；索引必须连续，模型 logits 靠它对位
    assert [c.index for c in tax] == list(range(14))
    assert ids[-1] == "unusable"
    assert len(set(ids)) == 14
    for crop in ("rice", "peanut", "vegetable"):
        assert any(c.crop == crop for c in tax)


def test_i18n_full_coverage():
    cov = coverage()
    incomplete = {lang: info for lang, info in cov.items() if not info["complete"]}
    assert not incomplete, f"翻译不齐：{incomplete}"
    for lang in languages.SPOKEN_ORDER:
        assert table(lang)["sys_no_model"]


def test_advice_voice_texts_exist_for_every_class():
    from heyan.advice import build_advice

    tax = load_taxonomy()
    for cls in tax:
        for lang in ("zh", "en"):
            advice = build_advice(cls.id, None, lang)
            assert advice.voice, f"{cls.id}/{lang}"
            assert advice.name, f"{cls.id}/{lang}"


def test_severity_palette_distinct():
    from heyan.advice import load_advisory

    colors = [s.color for s in load_advisory().severities.values()]
    assert len(colors) == len(set(colors))
    assert STRINGS["sys_dialect_fallback"]["zh"]
