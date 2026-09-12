"""Where finished per-request metrics go: structured log (Cloud Logging picks up
JSON on stdout), a local JSONL file (dashboard fallback) and, optionally,
BigQuery (dashboard / Looker Studio source)."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Protocol

from settings import get_settings

log = logging.getLogger("procure.metrics")

METRICS_SCHEMA: list[tuple[str, str, str]] = [
    ("request_id", "STRING", "REQUIRED"),
    ("ts", "TIMESTAMP", "REQUIRED"),
    ("session_id", "STRING", "NULLABLE"),
    ("user_id", "STRING", "NULLABLE"),
    ("root_agent", "STRING", "NULLABLE"),
    ("model", "STRING", "NULLABLE"),
    ("input_tokens", "INT64", "NULLABLE"),
    ("output_tokens", "INT64", "NULLABLE"),
    ("thinking_tokens", "INT64", "NULLABLE"),
    ("cached_tokens", "INT64", "NULLABLE"),
    ("total_tokens", "INT64", "NULLABLE"),
    ("llm_calls", "INT64", "NULLABLE"),
    ("tool_calls", "INT64", "NULLABLE"),
    ("latency_s", "FLOAT64", "NULLABLE"),
    ("llm_time_s", "FLOAT64", "NULLABLE"),
    ("ttft_s", "FLOAT64", "NULLABLE"),
    ("tokens_per_sec", "FLOAT64", "NULLABLE"),
    ("cost_usd", "FLOAT64", "NULLABLE"),
    ("status", "STRING", "NULLABLE"),
    ("error", "STRING", "NULLABLE"),
    ("pricing_fallback", "BOOL", "NULLABLE"),
    ("pricing_effective_from", "STRING", "NULLABLE"),
    ("agent_breakdown", "STRING", "NULLABLE"),  # JSON
]


class MetricsSink(Protocol):
    def emit(self, record: dict) -> None: ...


def ensure_metrics_table(client) -> None:
    from google.cloud import bigquery

    s = get_settings()
    table = bigquery.Table(
        s.bq_metrics_table_id,
        schema=[bigquery.SchemaField(n, t, mode=m) for n, t, m in METRICS_SCHEMA],
    )
    table.time_partitioning = bigquery.TimePartitioning(field="ts")
    client.create_table(table, exists_ok=True)


class DefaultSink:
    """Log line + JSONL + optional BigQuery. Never raises: telemetry must not break a request."""

    def __init__(self):
        self._lock = threading.Lock()
        self._bq = None

    def emit(self, record: dict) -> None:
        s = get_settings()
        try:
            log.info(json.dumps({"event": "request_metrics", **record}, default=str))
        except Exception:  # pragma: no cover
            log.exception("failed to log request metrics")
        path = s.metrics_jsonl_path
        if path and path != os.devnull:
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with self._lock, open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, default=str) + "\n")
            except OSError:
                log.exception("failed to write metrics jsonl")
        if s.metrics_to_bigquery and s.project:
            self._emit_bigquery(record)

    def _emit_bigquery(self, record: dict) -> None:
        try:
            if self._bq is None:
                from google.cloud import bigquery

                self._bq = bigquery.Client(project=get_settings().project, location=get_settings().bq_location)
            row = {**record, "agent_breakdown": json.dumps(record.get("agent_breakdown", {}))}
            errors = self._bq.insert_rows_json(get_settings().bq_metrics_table_id, [row])
            if errors:
                log.error("bigquery metrics insert errors: %s", errors)
        except Exception:
            log.exception("failed to insert request metrics into BigQuery")


class ListSink:
    """Collects records in memory (tests, benchmark)."""

    def __init__(self):
        self.records: list[dict] = []

    def emit(self, record: dict) -> None:
        self.records.append(record)
