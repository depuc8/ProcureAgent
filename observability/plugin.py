"""TelemetryPlugin: instruments every agent step, LLM call and tool call.

ADK already opens spans (`invoke_workflow`, `invoke_agent`, `execute_tool`,
`generate_content`). The plugin runs *inside* the workflow / agent / tool spans
and enriches them with `procure.*` attributes, and opens its own
`procure.llm_call` span around each model call (ADK's `generate_content` span
is not reachable from plugin callbacks). Every span therefore carries:

    procure.span_kind       workflow | agent | llm_call | tool
    procure.agent.name      agent that produced the span
    procure.tool.name       tool spans only
    procure.model           model name (agent / llm_call / workflow spans)
    procure.input_tokens    tokens in  (agent/tool spans: own LLM calls / calls made during the tool)
    procure.output_tokens   tokens out (incl. thinking)
    procure.latency_ms      wall-clock duration
    + llm_call: procure.ttft_ms, procure.cost_usd, gen_ai.usage.* ; workflow: request totals

It also feeds the per-request RequestMetrics (tokens, tokens/sec, TTFT, cost)
and hands the finished record to a sink when the root run ends. Because it
only sets attributes on whatever span is current, it works unchanged under
`adk web`, Agent Engine (AdkApp tracing) and the Cloud Run wrapper.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from google.adk.plugins.base_plugin import BasePlugin
from opentelemetry import trace

from observability.metrics import LlmCall, RequestMetrics, current_request, set_current
from observability.pricing import cost_usd
from observability.sink import DefaultSink, MetricsSink

log = logging.getLogger(__name__)
_tracer = trace.get_tracer("procure_agent.observability")


def _usage(resp) -> tuple[int, int, int, int]:
    """(input incl. cached, output, thinking, cached) from Gemini usage metadata."""
    u = getattr(resp, "usage_metadata", None)
    if u is None:
        return 0, 0, 0, 0
    prompt = (u.prompt_token_count or 0) + (getattr(u, "tool_use_prompt_token_count", 0) or 0)
    return (prompt, u.candidates_token_count or 0, getattr(u, "thoughts_token_count", 0) or 0,
            getattr(u, "cached_content_token_count", 0) or 0)


def _set(span, attrs: dict[str, Any]) -> None:
    if span is not None and span.is_recording():
        span.set_attributes({k: v for k, v in attrs.items() if v is not None})


class TelemetryPlugin(BasePlugin):
    def __init__(self, sink: MetricsSink | None = None, name: str = "procure_telemetry"):
        super().__init__(name=name)
        self.sink = sink or DefaultSink()
        self._llm: dict[tuple[str, str], tuple[LlmCall, Any]] = {}      # (invocation, agent) -> call, span
        self._agents: dict[tuple[str, str], tuple[float, int]] = {}      # -> start, call index at start
        self._tools: dict[str, tuple[float, int, int]] = {}              # function_call_id -> start, in, out

    # ---- request lifecycle -------------------------------------------------------
    async def before_run_callback(self, *, invocation_context) -> Optional[Any]:
        m = current_request()
        if m is None or m.finalized:
            m = RequestMetrics()
            set_current(m)
        if m.root_invocation_id is None:           # first run seen for this request = the root run
            m.root_invocation_id = invocation_context.invocation_id
            m.session_id = invocation_context.session.id
            m.user_id = invocation_context.user_id
            m.root_agent = invocation_context.agent.name
            span = trace.get_current_span()
            _set(span, {"procure.span_kind": "workflow", "procure.request_id": m.request_id,
                        "procure.agent.name": m.root_agent})
        return None

    async def after_run_callback(self, *, invocation_context) -> None:
        m = current_request()
        if m is None or m.root_invocation_id != invocation_context.invocation_id:
            return                                  # nested AgentTool run: not the request boundary
        await self._finalize(m)

    async def on_run_error_callback(self, *, invocation_context, error: Exception) -> None:
        m = current_request()
        if m is not None and m.root_invocation_id == invocation_context.invocation_id:
            m.status, m.error = "error", f"{type(error).__name__}: {error}"[:500]

    async def _finalize(self, m: RequestMetrics) -> None:
        if m.finalized:
            return
        m.ended = time.perf_counter()
        m.finalized = True
        record = m.to_dict()
        _set(trace.get_current_span(), {
            "procure.request_id": m.request_id, "procure.span_kind": "workflow",
            "procure.latency_ms": round(m.latency_s * 1000, 1),
            "procure.input_tokens": m.input_tokens, "procure.output_tokens": m.output_tokens,
            "procure.total_tokens": m.input_tokens + m.output_tokens,
            "procure.model": record["model"], "procure.cost_usd": record["cost_usd"],
            "procure.ttft_ms": None if m.ttft_s is None else round(m.ttft_s * 1000, 1),
            "procure.tokens_per_sec": m.tokens_per_sec,
            "procure.llm_calls": len(m.llm_calls), "procure.tool_calls": m.tool_calls,
        })
        # sinks may do network IO (BigQuery): keep it off the event loop, and never fail the request
        try:
            await asyncio.to_thread(self.sink.emit, record)
        except Exception:
            log.exception("metrics sink failed")
        set_current(None)

    # ---- agents ------------------------------------------------------------------
    async def before_agent_callback(self, *, agent, callback_context) -> Optional[Any]:
        m = current_request()
        self._agents[(callback_context.invocation_id, agent.name)] = (time.perf_counter(), len(m.llm_calls) if m else 0)
        _set(trace.get_current_span(), {"procure.span_kind": "agent", "procure.agent.name": agent.name,
                                        "procure.request_id": m.request_id if m else None})
        return None

    async def after_agent_callback(self, *, agent, callback_context) -> Optional[Any]:
        start, idx = self._agents.pop((callback_context.invocation_id, agent.name), (time.perf_counter(), 0))
        m = current_request()
        own = [c for c in (m.llm_calls[idx:] if m else []) if c.agent == agent.name]
        _set(trace.get_current_span(), {
            "procure.span_kind": "agent", "procure.agent.name": agent.name,
            "procure.latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "procure.input_tokens": sum(c.input_tokens for c in own),
            "procure.output_tokens": sum(c.output_tokens + c.thinking_tokens for c in own),
            "procure.model": ",".join(sorted({c.model for c in own})) or None,
            "procure.llm_calls": len(own),
        })
        return None

    # ---- model calls -------------------------------------------------------------
    async def before_model_callback(self, *, callback_context, llm_request) -> Optional[Any]:
        m = current_request()
        model = llm_request.model or ""
        call = LlmCall(agent=callback_context.agent_name, model=model, started=time.perf_counter())
        span = _tracer.start_span("procure.llm_call", attributes={
            "procure.span_kind": "llm_call", "procure.agent.name": call.agent, "procure.model": model,
            "gen_ai.request.model": model, "gen_ai.agent.name": call.agent,
            "procure.request_id": m.request_id if m else None,
        })
        self._llm[(callback_context.invocation_id, call.agent)] = (call, span)
        if m:
            m.llm_calls.append(call)
        return None

    async def after_model_callback(self, *, callback_context, llm_response) -> Optional[Any]:
        key = (callback_context.invocation_id, callback_context.agent_name)
        entry = self._llm.get(key)
        if entry is None:
            return None
        call, span = entry
        now = time.perf_counter()
        m = current_request()
        if call.first_chunk is None and (llm_response.content or llm_response.partial is not None):
            call.first_chunk = now
            if m and m.first_chunk is None:
                m.first_chunk = now             # first chunk of the first LLM call in the request
        in_t, out_t, think_t, cached_t = _usage(llm_response)
        if in_t or out_t:                      # usage on streamed chunks is cumulative: keep the latest
            call.input_tokens, call.output_tokens, call.thinking_tokens, call.cached_tokens = in_t, out_t, think_t, cached_t
        if llm_response.finish_reason:
            call.finish_reason = str(getattr(llm_response.finish_reason, "name", llm_response.finish_reason))
        if llm_response.partial:
            return None
        self._end_llm(key, m)
        return None

    async def on_model_error_callback(self, *, callback_context, llm_request, error: Exception) -> Optional[Any]:
        key = (callback_context.invocation_id, callback_context.agent_name)
        entry = self._llm.get(key)
        if entry:
            entry[0].error = f"{type(error).__name__}: {error}"[:300]
            self._end_llm(key, current_request(), error=entry[0].error)
        return None

    def _end_llm(self, key, m: RequestMetrics | None, error: str = "") -> None:
        call, span = self._llm.pop(key)
        call.ended = time.perf_counter()
        c, rate = cost_usd(call.model, call.input_tokens, call.output_tokens + call.thinking_tokens, call.cached_tokens)
        call.cost_usd = c
        if m:
            m.pricing_fallback = m.pricing_fallback or rate.fallback
            m.pricing_effective_from = rate.effective_from
        _set(span, {
            "procure.latency_ms": round(call.latency_s * 1000, 1),
            "procure.ttft_ms": None if call.ttft_s is None else round(call.ttft_s * 1000, 1),
            "procure.input_tokens": call.input_tokens,
            "procure.output_tokens": call.output_tokens + call.thinking_tokens,
            "procure.thinking_tokens": call.thinking_tokens,
            "procure.cached_tokens": call.cached_tokens,
            "procure.cost_usd": round(c, 8),
            "procure.pricing_fallback": rate.fallback,
            "gen_ai.usage.input_tokens": call.input_tokens,
            "gen_ai.usage.output_tokens": call.output_tokens + call.thinking_tokens,
            "gen_ai.response.finish_reasons": [call.finish_reason] if call.finish_reason else None,
        })
        if error and span is not None:
            span.set_status(trace.Status(trace.StatusCode.ERROR, error))
        if span is not None:
            span.end()

    # ---- tools -------------------------------------------------------------------
    async def before_tool_callback(self, *, tool, tool_args, tool_context) -> Optional[dict]:
        m = current_request()
        self._tools[tool_context.function_call_id or tool.name] = (
            time.perf_counter(), m.input_tokens if m else 0, m.output_tokens if m else 0)
        _set(trace.get_current_span(), {"procure.span_kind": "tool", "procure.tool.name": tool.name,
                                        "procure.agent.name": tool_context.agent_name})
        return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result) -> Optional[dict]:
        self._finish_tool(tool, tool_context)
        return None

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error: Exception) -> Optional[dict]:
        self._finish_tool(tool, tool_context, error=f"{type(error).__name__}: {error}"[:300])
        return None

    def _finish_tool(self, tool, tool_context, error: str = "") -> None:
        start, in0, out0 = self._tools.pop(tool_context.function_call_id or tool.name,
                                           (time.perf_counter(), 0, 0))
        m = current_request()
        elapsed = time.perf_counter() - start
        if m:
            m.tool_calls += 1
            m.tool_time_s += elapsed
        _set(trace.get_current_span(), {
            "procure.span_kind": "tool", "procure.tool.name": tool.name,
            "procure.agent.name": tool_context.agent_name,
            "procure.latency_ms": round(elapsed * 1000, 1),
            # plain tools make no LLM calls (0); an AgentTool wraps a whole sub-agent run
            "procure.input_tokens": (m.input_tokens - in0) if m else 0,
            "procure.output_tokens": (m.output_tokens - out0) if m else 0,
            "procure.error": error or None,
        })
