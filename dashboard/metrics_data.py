"""Data layer for the dashboard: load per-request metrics and compute percentiles over time.
Kept free of Streamlit so it is unit-testable."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

NUMERIC = ["input_tokens", "output_tokens", "total_tokens", "latency_s", "llm_time_s", "ttft_s",
           "tokens_per_sec", "cost_usd", "llm_calls", "tool_calls"]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("ts").reset_index(drop=True)


def load_jsonl(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        return pd.DataFrame()
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue                              # a half-written line must not blank the dashboard
    return _normalise(pd.DataFrame(rows))


def load_bigquery(table_id: str, days: int = 30, client=None) -> pd.DataFrame:
    if client is None:
        from google.cloud import bigquery

        client = bigquery.Client()
    from google.cloud import bigquery as bq

    sql = f"""SELECT * EXCEPT (agent_breakdown) FROM `{table_id}`
              WHERE ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY) ORDER BY ts"""
    job = client.query(sql, job_config=bq.QueryJobConfig(
        query_parameters=[bq.ScalarQueryParameter("days", "INT64", days)]))
    return _normalise(job.to_dataframe())


def bucketed(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Per time bucket: request count, cost, p50/p95 latency, mean TTFT and tokens/sec."""
    if df.empty:
        return pd.DataFrame(columns=["ts", "requests", "cost_usd", "p50_latency_s", "p95_latency_s",
                                     "mean_ttft_s", "mean_tokens_per_sec"])
    g = df.set_index("ts").resample(freq)
    out = pd.DataFrame({
        "requests": g["request_id"].count(),
        "cost_usd": g["cost_usd"].sum(),
        "p50_latency_s": g["latency_s"].quantile(0.50),
        "p95_latency_s": g["latency_s"].quantile(0.95),
        "mean_ttft_s": g["ttft_s"].mean(),
        "mean_tokens_per_sec": g["tokens_per_sec"].mean(),
    })
    return out[out["requests"] > 0].reset_index()


def summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    ok = df[df["status"] == "ok"] if "status" in df else df
    return {
        "requests": len(df),
        "errors": int((df["status"] != "ok").sum()) if "status" in df else 0,
        "total_cost_usd": float(df["cost_usd"].sum()),
        "mean_cost_usd": float(df["cost_usd"].mean()),
        "p50_latency_s": float(ok["latency_s"].quantile(0.5)) if len(ok) else None,
        "p95_latency_s": float(ok["latency_s"].quantile(0.95)) if len(ok) else None,
        "mean_ttft_s": float(df["ttft_s"].mean()),
        "mean_tokens_per_sec": float(df["tokens_per_sec"].mean()),
        "mean_tokens": float(df["total_tokens"].mean()),
    }
