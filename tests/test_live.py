"""Real Gemini / GCP. Skipped unless PROCURE_RUN_LIVE=1 (costs money, needs ADC + GOOGLE_CLOUD_PROJECT).

These assert the *business invariants* the scripted suite cannot: that the actual model follows the
planner's plan and that nothing it writes contradicts the policy engine. Wording is never asserted."""

import os

import pytest

from agents.planner import build_planner
from observability.sink import ListSink
from observability.plugin import TelemetryPlugin
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.genai import types

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(os.getenv("PROCURE_RUN_LIVE") != "1", reason="set PROCURE_RUN_LIVE=1 to call Gemini")]


async def ask(prompts):
    sink = ListSink()
    app = App(name="procure_live", root_agent=build_planner(), plugins=[TelemetryPlugin(sink=sink)])
    runner = InMemoryRunner(app=app)
    sid = (await runner.session_service.create_session(app_name=app.name, user_id="live")).id
    final = ""
    for p in prompts:
        async for ev in runner.run_async(user_id="live", session_id=sid,
                                         new_message=types.Content(role="user", parts=[types.Part(text=p)])):
            if ev.author == "procure_planner" and ev.is_final_response() and ev.content and ev.content.parts:
                final = "".join(x.text or "" for x in ev.content.parts)
    state = (await runner.session_service.get_session(app_name=app.name, user_id="live", session_id=sid)).state
    return final, dict(state), sink.records


async def test_live_demo_request():
    final, s, recs = await ask(["We need 50 laptops for new hires, budget ₹40L, delivery in 3 weeks"])
    assert s["request"]["quantity"] == 50 and s["request"]["budget_inr"] == 4_000_000 and s["request"]["max_delivery_days"] == 21
    assert len(s["shortlist"]) == 3 and s["contract_findings"] and s["approval_decisions"]
    rec = s.get("recommended_vendor_id")
    assert rec and s["approval_decisions"][rec]["decision"] != "reject"
    assert recs[0]["cost_usd"] > 0 and recs[0]["ttft_s"] > 0


async def test_live_over_budget_is_never_approved():
    final, s, _ = await ask(["We need 50 laptops, budget ₹10L, delivery in 3 weeks"])
    for d in (s.get("approval_decisions") or {}).values():
        assert d["decision"] == "reject"
    assert not s.get("recommended_vendor_id")


async def test_live_follow_up_swap():
    _, s1, _ = await ask(["We need 50 laptops for new hires, budget ₹40L, delivery in 3 weeks"])
    _, s2, _ = await ask(["We need 50 laptops for new hires, budget ₹40L, delivery in 3 weeks",
                          "swap vendor 2 for a cheaper one"])
    assert s2["shortlist"][1]["unit_price_inr"] <= s1["shortlist"][1]["unit_price_inr"]
