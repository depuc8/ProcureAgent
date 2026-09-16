from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

# Tests never touch GCP unless explicitly marked `live`.
os.environ.setdefault("VENDOR_BACKEND", "local")
os.environ.pop("OTEL_TO_CLOUD_TRACE", None)
os.environ.setdefault("METRICS_JSONL_PATH", os.devnull)


class FakeToolContext:
    """Minimal stand-in for ADK's ToolContext: tools only touch .state and .actions."""

    def __init__(self, state: dict | None = None):
        self.state = state if state is not None else {}
        self.actions = SimpleNamespace(escalate=False, skip_summarization=False)


@pytest.fixture
def ctx() -> FakeToolContext:
    return FakeToolContext()


@pytest.fixture(autouse=True)
def _reset_repo():
    from tools.vendor_repository import set_repository

    set_repository(None)
    yield
    set_repository(None)


@pytest.fixture
def laptop_request(ctx):
    from tools.state_tools import save_request

    save_request("Laptops", 50, 4_000_000, 21, ctx)
    return ctx


# ---- OpenTelemetry: one in-memory exporter for the whole session ------------------------
from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

_EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_provider)


@pytest.fixture
def spans() -> InMemorySpanExporter:
    _EXPORTER.clear()
    return _EXPORTER
