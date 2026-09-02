"""Central configuration. Everything comes from environment variables; secrets
fall back to Secret Manager. Nothing sensitive lives in code or in the repo."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    project: str = field(default_factory=lambda: os.getenv("GOOGLE_CLOUD_PROJECT", ""))
    location: str = field(default_factory=lambda: os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))
    model: str = field(default_factory=lambda: os.getenv("PROCURE_MODEL", "gemini-3.8-flash"))

    vendor_backend: str = field(default_factory=lambda: os.getenv("VENDOR_BACKEND", "local"))
    bq_dataset: str = field(default_factory=lambda: os.getenv("BQ_DATASET", "procureagent"))
    bq_location: str = field(default_factory=lambda: os.getenv("BQ_LOCATION", "US"))
    bq_vendor_table: str = field(default_factory=lambda: os.getenv("BQ_VENDOR_TABLE", "vendors"))
    bq_metrics_table: str = field(default_factory=lambda: os.getenv("BQ_METRICS_TABLE", "request_metrics"))

    contracts_dir: Path = field(
        default_factory=lambda: Path(os.getenv("CONTRACTS_DIR", str(DATA_DIR / "contracts")))
    )
    policy_path: Path = field(
        default_factory=lambda: Path(os.getenv("POLICY_PATH", str(DATA_DIR / "policy.json")))
    )
    pricing_path: Path = field(
        default_factory=lambda: Path(os.getenv("PRICING_PATH", str(DATA_DIR / "pricing.json")))
    )

    otel_to_cloud_trace: bool = field(default_factory=lambda: _bool("OTEL_TO_CLOUD_TRACE"))
    metrics_jsonl_path: str = field(
        default_factory=lambda: os.getenv("METRICS_JSONL_PATH", "logs/request_metrics.jsonl")
    )
    metrics_to_bigquery: bool = field(default_factory=lambda: _bool("METRICS_TO_BIGQUERY"))

    secret_manager_enabled: bool = field(default_factory=lambda: _bool("SECRET_MANAGER_ENABLED"))

    @property
    def bq_vendor_table_id(self) -> str:
        return f"{self.project}.{self.bq_dataset}.{self.bq_vendor_table}"

    @property
    def bq_metrics_table_id(self) -> str:
        return f"{self.project}.{self.bq_dataset}.{self.bq_metrics_table}"


def get_settings() -> Settings:
    """Settings are re-read on every call so tests and `adk web` pick up env changes."""
    return Settings()


@lru_cache(maxsize=32)
def _secret_from_manager(project: str, secret_id: str) -> str | None:
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project}/secrets/{secret_id}/versions/latest"
    return client.access_secret_version(request={"name": name}).payload.data.decode("utf-8")


def get_secret(env_name: str) -> str | None:
    """Resolve a secret: environment variable first, then Secret Manager.

    The Secret Manager secret id is the env name lower-cased with `_` -> `-`
    (PROCURE_API_TOKEN -> procure-api-token).
    """
    value = os.getenv(env_name)
    if value:
        return value
    s = get_settings()
    if s.secret_manager_enabled and s.project:
        return _secret_from_manager(s.project, env_name.lower().replace("_", "-"))
    return None
