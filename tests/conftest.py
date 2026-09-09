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
