import pytest
from prometheus_client import REGISTRY

from app.config import settings
from app.infrastructure import metrics

pytestmark = pytest.mark.unit


def _value(name: str, labels: dict[str, str]) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_llm_call_records_tokens_and_cost(monkeypatch):
    monkeypatch.setattr(settings, "llm_cost_per_1k_input", 1.0)
    monkeypatch.setattr(settings, "llm_cost_per_1k_output", 2.0)
    labels = {"model": "test-model"}
    before_cost = _value("pm_llm_cost_usd_total", labels)
    before_prompt = _value("pm_llm_tokens_total", {**labels, "kind": "prompt"})

    metrics.record_llm_call("test-model", 0.5, 1000, 500)

    assert _value("pm_llm_tokens_total", {**labels, "kind": "prompt"}) - before_prompt == 1000
    assert _value("pm_llm_cost_usd_total", labels) - before_cost == pytest.approx(2.0)


def test_missing_usage_is_tolerated():
    metrics.record_llm_call("no-usage-model", 0.1, None, None)
    assert _value("pm_llm_seconds_count", {"model": "no-usage-model"}) == 1.0
