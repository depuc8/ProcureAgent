import asyncio

import pytest

from observability.metrics import current_request, request_scope
from observability.pricing import cost_usd
from tests.scripted import Harness

DEMO = "We need 50 laptops for new hires, budget ₹40L, delivery in 3 weeks"


# ---- plugin, end to end ------------------------------------------------------------------
async def test_request_record_is_consistent():
    h = Harness(latency_s=0.02)
    await h.ask(DEMO)
    assert len(h.sink.records) == 1
    r = h.sink.records[0]
    assert r["status"] == "ok" and r["root_agent"] == "procure_planner" and r["session_id"] == h.session_id
    # totals equal the sum of the per-agent breakdown
    bd = r["agent_breakdown"]
    assert r["input_tokens"] == sum(a["input_tokens"] for a in bd.values())
    assert r["output_tokens"] == sum(a["output_tokens"] for a in bd.values())
    assert r["llm_calls"] == sum(a["llm_calls"] for a in bd.values())
    assert set(bd) == {"procure_planner", "vendor_search_agent", "contract_review_agent", "approval_agent",
                       "draft_critic", "draft_reviser"} - {"draft_reviser"}    # approved first time: no revision
    assert r["total_tokens"] == r["input_tokens"] + r["output_tokens"] > 0
    # cost matches the pricing config, call by call
    assert r["cost_usd"] == pytest.approx(sum(a["cost_usd"] for a in bd.values()), rel=1e-6)
    expected, _ = cost_usd("gemini-3.8-flash", r["input_tokens"], r["output_tokens"])
    assert r["cost_usd"] == pytest.approx(expected, rel=1e-3)
    assert r["tool_calls"] > 10
    # latency ordering invariants
    assert 0 < r["ttft_s"] < r["latency_s"] and r["llm_time_s"] <= r["latency_s"]
    assert r["tokens_per_sec"] > 0 and r["pricing_fallback"] is False


async def test_ttft_is_shorter_than_call_latency_when_streaming():
    h = Harness(latency_s=0.06, stream=True)
    await h.ask(DEMO)
    # streamed text responses emit partial chunks before the final one
    assert h.sink.records[0]["ttft_s"] < h.sink.records[0]["latency_s"]


async def test_spans_carry_required_attributes(spans):
    h = Harness(latency_s=0.01)
    await h.ask(DEMO)
    finished = spans.get_finished_spans()
    by_kind: dict[str, list] = {}
    for s in finished:
        by_kind.setdefault(s.attributes.get("procure.span_kind", "other"), []).append(s)
    assert {"workflow", "agent", "llm_call", "tool"} <= set(by_kind)

    for s in by_kind["agent"]:
        a = s.attributes
        assert a["procure.agent.name"] and a["procure.latency_ms"] >= 0
        assert "procure.input_tokens" in a and "procure.output_tokens" in a
    for s in by_kind["llm_call"]:
        a = s.attributes
        assert a["procure.model"] == "gemini-3.8-flash" and a["procure.agent.name"]
        assert a["procure.input_tokens"] > 0 and a["procure.output_tokens"] > 0
        assert a["procure.latency_ms"] > 0 and a["procure.ttft_ms"] > 0 and a["procure.cost_usd"] > 0
    for s in by_kind["tool"]:
        a = s.attributes
        assert a["procure.tool.name"] and a["procure.agent.name"] and a["procure.latency_ms"] >= 0
        assert "procure.input_tokens" in a
    # a plain function tool makes no LLM call; an AgentTool wraps a whole sub-agent run
    tools = {s.attributes["procure.tool.name"]: s.attributes for s in by_kind["tool"]}
    assert tools["save_request"]["procure.input_tokens"] == 0
    assert tools["vendor_search_agent"]["procure.input_tokens"] > 0
    (wf,) = by_kind["workflow"]
    assert wf.attributes["procure.total_tokens"] > 0 and wf.attributes["procure.cost_usd"] > 0
    assert wf.attributes["procure.request_id"] == h.sink.records[0]["request_id"]
    # llm_call spans nest under an agent span, so traces read agent -> llm_call
    ids = {s.context.span_id: s for s in finished}
    assert all(ids[s.parent.span_id].attributes.get("gen_ai.operation.name") == "invoke_agent" for s in by_kind["llm_call"])


async def test_concurrent_requests_do_not_share_metrics():
    async def one():
        h = Harness(latency_s=0.01)
        await h.ask(DEMO)
        return h.sink.records

    a, b = await asyncio.gather(one(), one())
    assert len(a) == len(b) == 1
    ra, rb = a[0], b[0]
    assert ra["request_id"] != rb["request_id"]
    assert ra["llm_calls"] == rb["llm_calls"] and ra["input_tokens"] == rb["input_tokens"]


async def test_request_scope_lets_caller_choose_id_and_read_result():
    h = Harness()
    with request_scope(request_id="req-123") as m:
        await h.ask(DEMO)
    assert m.finalized and m.request_id == "req-123" and h.sink.records[0]["request_id"] == "req-123"
    assert current_request() is None


async def test_failing_sink_never_breaks_a_request():
    class Boom:
        def emit(self, record):
            raise RuntimeError("sink down")

    h = Harness()
    h.runner.app.plugins[0].sink = Boom()
    assert "Finance Director" in await h.ask(DEMO)


async def test_each_new_request_gets_new_metrics():
    h = Harness()
    await h.ask(DEMO)
    await h.ask("swap vendor 2 for a cheaper one")
    ids = [r["request_id"] for r in h.sink.records]
    assert len(ids) == 2 and ids[0] != ids[1]
    assert h.sink.records[1]["llm_calls"] < h.sink.records[0]["llm_calls"]
