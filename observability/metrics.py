"""Per-request metrics: tokens, tokens/sec, time-to-first-token, cost.

Definitions (also in the README):
  output_tokens   candidates + thinking tokens (both billed as output)
  input_tokens    prompt tokens incl. cached, summed over every LLM call in the request
  llm_time_s      sum of the wall-clock duration of each LLM call
  tokens_per_sec  output_tokens / llm_time_s  (generation throughput while the model is working)
  ttft_s          request start -> first chunk of the first LLM call. Only a true
                  time-to-first-token when the run streams (StreamingMode.SSE); without
                  streaming the first chunk is the whole first response.
  cost_usd        per-call cost from data/pricing.json, summed
"""

from __future__ import annotations

import contextvars
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class LlmCall:
    agent: str
    model: str
    started: float
    first_chunk: float | None = None
    ended: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    finish_reason: str = ""
    error: str = ""

    @property
    def latency_s(self) -> float:
        return (self.ended or time.perf_counter()) - self.started

    @property
    def ttft_s(self) -> float | None:
        return None if self.first_chunk is None else self.first_chunk - self.started


@dataclass
class RequestMetrics:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str = ""
    user_id: str = ""
    root_agent: str = ""
    root_invocation_id: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started: float = field(default_factory=time.perf_counter)
    ended: float | None = None
    first_chunk: float | None = None
    llm_calls: list[LlmCall] = field(default_factory=list)
    tool_calls: int = 0
    tool_time_s: float = 0.0
    status: str = "ok"
    error: str = ""
    pricing_fallback: bool = False
    pricing_effective_from: str = ""
    finalized: bool = False

    # ---- running totals (spans use deltas of these) ------------------------------
    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.llm_calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens + c.thinking_tokens for c in self.llm_calls)

    @property
    def cached_tokens(self) -> int:
        return sum(c.cached_tokens for c in self.llm_calls)

    @property
    def thinking_tokens(self) -> int:
        return sum(c.thinking_tokens for c in self.llm_calls)

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.llm_calls)

    @property
    def llm_time_s(self) -> float:
        return sum(c.latency_s for c in self.llm_calls if c.ended is not None)

    @property
    def latency_s(self) -> float:
        return (self.ended or time.perf_counter()) - self.started

    @property
    def ttft_s(self) -> float | None:
        return None if self.first_chunk is None else self.first_chunk - self.started

    @property
    def tokens_per_sec(self) -> float | None:
        t = self.llm_time_s
        return round(self.output_tokens / t, 2) if t > 0 and self.output_tokens else None

    def per_agent(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for c in self.llm_calls:
            a = out.setdefault(c.agent, {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_time_s": 0.0})
            a["llm_calls"] += 1
            a["input_tokens"] += c.input_tokens
            a["output_tokens"] += c.output_tokens + c.thinking_tokens
            a["cost_usd"] = round(a["cost_usd"] + c.cost_usd, 8)
            a["llm_time_s"] = round(a["llm_time_s"] + (c.latency_s if c.ended else 0), 4)
        return out

    def to_dict(self) -> dict:
        models = sorted({c.model for c in self.llm_calls})
        return {
            "request_id": self.request_id,
            "ts": self.started_at.isoformat(),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "root_agent": self.root_agent,
            "model": ",".join(models),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "llm_calls": len(self.llm_calls),
            "tool_calls": self.tool_calls,
            "latency_s": round(self.latency_s, 4),
            "llm_time_s": round(self.llm_time_s, 4),
            "ttft_s": None if self.ttft_s is None else round(self.ttft_s, 4),
            "tokens_per_sec": self.tokens_per_sec,
            "cost_usd": round(self.cost_usd, 8),
            "status": self.status,
            "error": self.error,
            "pricing_fallback": self.pricing_fallback,
            "pricing_effective_from": self.pricing_effective_from,
            "agent_breakdown": self.per_agent(),
        }


_current: contextvars.ContextVar[RequestMetrics | None] = contextvars.ContextVar("procure_request", default=None)


def current_request() -> RequestMetrics | None:
    return _current.get()


class request_scope:
    """Pre-create the metrics object so a caller (the FastAPI wrapper) can pick the
    request id and read the finalized numbers after the run:

        with request_scope(request_id=rid) as m:
            async for event in runner.run_async(...): ...
        m.to_dict()
    """

    def __init__(self, request_id: str | None = None):
        self.metrics = RequestMetrics(request_id=request_id or uuid.uuid4().hex)
        self._token: contextvars.Token | None = None

    def __enter__(self) -> RequestMetrics:
        self._token = _current.set(self.metrics)
        return self.metrics

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            try:
                _current.reset(self._token)
            except ValueError:  # exited in a different context than entered
                _current.set(None)


def set_current(m: RequestMetrics | None) -> None:
    _current.set(m)
