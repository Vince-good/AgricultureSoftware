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
    assert len(ids) == 17
    assert "unusable" in ids
    # 展示序按作物分组、兜底类垫底；索引必须连续，模型 logits 靠它对位
    assert [c.index for c in tax] == list(range(17))
    assert ids[-1] == "unusable"
    assert len(set(ids)) == 17
    for crop in ("rice", "peanut", "vegetable", "maize"):
        assert any(c.crop == crop for c in tax)


def test_label_aliases_cover_field_folders():
    """田间采集目录名 -> class_id 的映射：微调数据进对的类全靠这张表。

    dateBase_Maize 那批照片的目录名是团队自己起的（Maize_spotDisease），和
    taxonomy 里的 id（maize_northern_leaf_blight）对不上。映射错了不会报错，
    只会安静地把大斑病训成锈病，所以这里把三条对应关系钉死。
    """
    from heyan.classes import load_label_aliases

    aliases = load_label_aliases()
    tax = load_taxonomy()
    expected = {
        "Maize_healthy": "maize_healthy",
        "Maize_RustDisease": "maize_rust",
        "Maize_spotDisease": "maize_northern_leaf_blight",
        "玉米健康": "maize_healthy",
        "玉米锈病": "maize_rust",
        "玉米大斑病": "maize_northern_leaf_blight",
    }
    for name, cid in expected.items():
        assert aliases.resolve_strict(name) == cid, name
    # 归一化：分隔符、大小写、首尾空白都不该影响命中
    for variant in ("maize rustdisease", "MAIZE-RUSTDISEASE", "  MaizeRustDisease  "):
        assert aliases.resolve(variant) == "maize_rust", variant
    # 认不出来时必须返回 None，绝不能猜一个类塞进去
    assert aliases.resolve("Maize_UnknownBatch") is None
    assert aliases.unknown(["Maize_healthy", "Maize_UnknownBatch"]) == ["Maize_UnknownBatch"]
    # 每条别名都要落在真实存在的 class_id 上，否则整批照片会被静默丢掉
    for cid in aliases.by_class:
        assert cid in tax.ids, cid


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
