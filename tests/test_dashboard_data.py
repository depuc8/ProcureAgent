import json

import pandas as pd
import pytest

from dashboard.metrics_data import bucketed, load_jsonl, summary


def rec(i, ts, latency, cost=0.01, status="ok"):
    return {"request_id": f"r{i}", "ts": ts, "latency_s": latency, "cost_usd": cost, "ttft_s": 0.5,
            "tokens_per_sec": 100.0, "total_tokens": 1000, "status": status}


@pytest.fixture
def path(tmp_path):
    rows = [rec(i, f"2026-11-18T10:{i:02d}:00+00:00", float(i + 1)) for i in range(10)]
    rows += [rec(10, "2026-11-18T11:00:00+00:00", 20.0, status="error")]
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n{truncated")
    return str(p)


def test_load_skips_half_written_lines(path):
    assert len(load_jsonl(path)) == 11


def test_missing_file_is_empty(tmp_path):
    assert load_jsonl(str(tmp_path / "nope.jsonl")).empty and summary(load_jsonl(str(tmp_path / "nope.jsonl"))) == {}


def test_hourly_percentiles(path):
    b = bucketed(load_jsonl(path), "1h")
    first = b.iloc[0]
    assert first["requests"] == 10 and first["p50_latency_s"] == pytest.approx(5.5)
    assert first["p95_latency_s"] == pytest.approx(9.55)
    assert first["cost_usd"] == pytest.approx(0.10)
    assert len(b) == 2


def test_summary_excludes_errors_from_latency_percentiles(path):
    k = summary(load_jsonl(path))
    assert k["requests"] == 11 and k["errors"] == 1
    assert k["p95_latency_s"] == pytest.approx(9.55)        # the 20s failure is not in the latency stats
    assert k["total_cost_usd"] == pytest.approx(0.11)


def test_empty_bucketing():
    assert bucketed(pd.DataFrame()).empty
