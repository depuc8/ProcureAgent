from datetime import date

import pytest

from observability.metrics import LlmCall, RequestMetrics
from observability.pricing import cost_usd, rate_for

# ---- pricing ---------------------------------------------------------------------------
def test_rate_changes_on_effective_date():
    assert (rate_for("gemini-3.8-flash", date(2026, 12, 31)).input_per_m,
            rate_for("gemini-3.8-flash", date(2026, 12, 31)).output_per_m) == (0.75, 3.75)
    assert (rate_for("gemini-3.8-flash", date(2027, 1, 1)).input_per_m,
            rate_for("gemini-3.8-flash", date(2027, 1, 1)).output_per_m) == (1.5, 7.5)


def test_cost_formula_with_cached_tokens():
    # 1M input of which 400k cached, 100k output on the 2026 rate card
    cost, _ = cost_usd("gemini-3.8-flash", 1_000_000, 100_000, 400_000, on=date(2026, 11, 18))
    assert cost == pytest.approx(0.6 * 0.75 + 0.4 * 0.075 + 0.1 * 3.75)


def test_unknown_model_falls_back_but_is_flagged():
    cost, rate = cost_usd("some-future-model", 1000, 1000, on=date(2026, 11, 18))
    assert rate.fallback and cost > 0 and rate.model == "gemini-3.8-flash"
    assert not rate_for("gemini-3.8-flash").fallback


# ---- metrics maths -----------------------------------------------------------------------
def test_request_metrics_math():
    m = RequestMetrics()
    m.started = 100.0
    m.first_chunk = 100.4
    m.ended = 102.0
    m.llm_calls = [
        LlmCall("a", "gemini-3.8-flash", started=100.2, ended=101.0, input_tokens=1000, output_tokens=80, thinking_tokens=20, cost_usd=0.001),
        LlmCall("b", "gemini-3.8-flash", started=101.0, ended=101.5, input_tokens=500, output_tokens=40, cost_usd=0.0005),
    ]
    d = m.to_dict()
    assert d["input_tokens"] == 1500 and d["output_tokens"] == 140 and d["total_tokens"] == 1640
    assert d["latency_s"] == 2.0 and d["ttft_s"] == 0.4
    assert d["llm_time_s"] == pytest.approx(1.3)
    assert d["tokens_per_sec"] == pytest.approx(140 / 1.3, abs=0.01)     # output tokens / time spent in the model
    assert d["cost_usd"] == pytest.approx(0.0015)
    assert d["agent_breakdown"]["a"]["output_tokens"] == 100
