"""10 end-to-end scenarios through the real ADK runtime (Runner, AgentTool delegation,
LoopAgent critic, session state, real tools + policy engine, telemetry plugin) with
scripted models standing in for Gemini. Each asserts an expected business outcome."""

import re

import pytest

from tests.scripted import Harness
from tools.money import format_inr


async def run(text, **kw):
    h = Harness(**kw)
    final = await h.ask(text)
    return h, final, await h.state()


def decision_for(state, vid):
    return state["approval_decisions"][vid]


# 1 ---------------------------------------------------------------------------
async def test_01_demo_request_recommends_low_risk_vendor_needing_finance_director():
    h, final, s = await run("We need 50 laptops for new hires, budget ₹40L, delivery in 3 weeks")
    assert s["request"] == {"category": "Laptops", "quantity": 50, "budget_inr": 4_000_000.0,
                            "max_delivery_days": 21, "required_certifications": [], "min_rating": 3.5, "notes": ""}
    assert len(s["shortlist"]) == 3 and all(v["contract_on_file"] for v in s["shortlist"])
    rec = s["recommended_vendor_id"]
    assert s["contract_findings"][rec]["risk_score"] <= 3                  # picked a safe contract over the cheaper risky ones
    d = decision_for(s, rec)
    assert d["decision"] == "needs_approval" and "Finance Director" in d["required_approvers"]
    assert s["review_status"] == "approved"
    assert "Finance Director" in final and format_inr(4_000_000).split(" (")[0] in final


# 2 ---------------------------------------------------------------------------
async def test_02_over_budget_request_is_rejected():
    h, final, s = await run("We need 50 laptops, budget ₹10L, delivery in 3 weeks")
    assert s["shortlist"], "nearest over-budget options are surfaced so the rejection is auditable"
    assert all(decision_for(s, v["vendor_id"])["decision"] == "reject" for v in s["shortlist"])
    assert any("exceeds the stated budget" in r for v in s["shortlist"] for r in decision_for(s, v["vendor_id"])["reasons"])
    assert not s.get("recommended_vendor_id")
    assert "reject" in final.lower() and "No vendor is recommended" in final


# 3 ---------------------------------------------------------------------------
async def test_03_small_order_with_preferred_safe_vendor_is_auto_approved():
    h, final, s = await run("We need 2 laptops, budget ₹2L, delivery in 4 weeks")
    rec = s["recommended_vendor_id"]
    d = decision_for(s, rec)
    assert d["decision"] == "approve" and d["required_approvers"] == []
    assert d["total_price_inr"] <= 200_000
    assert s["shortlist"][[v["vendor_id"] for v in s["shortlist"]].index(rec)]["is_preferred"]


# 4 ---------------------------------------------------------------------------
async def test_04_very_large_order_escalates_to_cfo():
    h, final, s = await run("We need 400 laptops for a data-centre refresh, budget ₹3 crore, delivery in 4 weeks")
    d = decision_for(s, s["recommended_vendor_id"])
    assert d["decision"] == "needs_approval" and "CFO" in d["required_approvers"]
    assert 5_000_000 < d["total_price_inr"] <= 30_000_000 and "CFO" in final


# 5 ---------------------------------------------------------------------------
async def test_05_order_above_single_po_limit_is_rejected_even_if_within_budget():
    h, final, s = await run("We need 1000 laptops, budget ₹10 crore, delivery in 4 weeks")
    reasons = [r for v in s["shortlist"] for r in decision_for(s, v["vendor_id"])["reasons"]]
    assert any("single-PO limit" in r for r in reasons)
    assert not s.get("recommended_vendor_id") and "reject" in final.lower()


# 6 ---------------------------------------------------------------------------
async def test_06_impossible_delivery_window_is_rejected_with_reason():
    h, final, s = await run("We need 50 laptops, budget ₹40L, delivery in 2 days")
    assert all(decision_for(s, v["vendor_id"])["decision"] == "reject" for v in s["shortlist"])
    assert any("misses" in r for v in s["shortlist"] for r in decision_for(s, v["vendor_id"])["reasons"])
    assert not s.get("recommended_vendor_id")


# 7 ---------------------------------------------------------------------------
async def test_07_required_certification_is_enforced():
    h, final, s = await run("We need 20 laptops with ISO 27001 certification, budget ₹30L, delivery in 4 weeks")
    assert s["request"]["required_certifications"] == ["ISO 27001"]
    assert s["shortlist"] and all("ISO 27001" in v["certifications"] for v in s["shortlist"])
    assert decision_for(s, s["recommended_vendor_id"])["decision"] != "reject"


# 8 ---------------------------------------------------------------------------
async def test_08_vendor_without_contract_requires_legal_counsel():
    h, final, s = await run("We need 20 mobile phones, budget ₹8L, delivery in 2 weeks")
    rec = s["recommended_vendor_id"]
    assert s["contract_findings"][rec]["contract_found"] is False
    d = decision_for(s, rec)
    assert "Legal Counsel" in d["required_approvers"]
    assert "no contract on file" in final.lower()


# 9 ---------------------------------------------------------------------------
async def test_09_follow_up_swap_uses_persisted_session_state():
    h = Harness()
    await h.ask("We need 50 laptops for new hires, budget ₹40L, delivery in 3 weeks")
    before = await h.state()
    old = before["shortlist"][1]
    final = await h.ask("swap vendor 2 for a cheaper one")
    after = await h.state()
    new = after["shortlist"][1]
    assert after["request"] == before["request"]                              # request survived the turn
    assert new["vendor_id"] != old["vendor_id"] and new["unit_price_inr"] < old["unit_price_inr"]
    assert [v["vendor_id"] for v in after["shortlist"]][0] == before["shortlist"][0]["vendor_id"]
    assert old["vendor_id"] not in after["contract_findings"] and new["vendor_id"] in after["contract_findings"]
    assert new["vendor_id"] in after["approval_decisions"]                    # the NEW vendor was re-checked
    assert new["name"] in final and h.session_id is not None


# 10 --------------------------------------------------------------------------
async def test_10_critic_loop_sends_flawed_draft_back_and_it_is_fixed():
    h, final, s = await run("We need 50 laptops for new hires, budget ₹40L, delivery in 3 weeks",
                            planner_flaw="omit_approver_and_fake_figure")
    assert s["review_round"] == 2                       # round 1 rejected the draft, round 2 approved the revision
    assert s["review_status"] == "approved"
    assert "7,77,777" not in final                      # unsupported figure removed
    assert "Finance Director" in final                  # missing approver restored
    # the telemetry plugin saw every nested LLM call: 3 critic calls per round x 2 rounds, 1 reviser round
    breakdown = h.sink.records[-1]["agent_breakdown"]
    assert breakdown["draft_critic"]["llm_calls"] == 6
    assert breakdown["draft_reviser"]["llm_calls"] == 2
