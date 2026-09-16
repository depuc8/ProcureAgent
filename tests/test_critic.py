import pytest

from agents.critic import MAX_ITERATIONS, build_review_loop
from tests.scripted import Harness, build_draft
from tools.review_tools import check_draft_facts, save_revised_draft, submit_review
from tools.state_tools import save_draft
from tools.contract_tools import record_contract_findings
from tools.policy_tools import check_policy
from tools.state_tools import save_request, set_recommended_vendor
from tools.vendor_tools import search_vendors


def test_loop_is_capped_at_three_iterations():
    loop = build_review_loop()
    assert loop.max_iterations == MAX_ITERATIONS == 3
    assert [a.name for a in loop.sub_agents] == ["draft_critic", "draft_reviser"]


async def test_reviser_that_never_fixes_ends_after_exactly_three_rounds():
    h = Harness(planner_flaw="omit_approver_and_fake_figure", reviser_fixes=False)
    final = await h.ask("We need 50 laptops for new hires, budget ₹40L, delivery in 3 weeks")
    s = await h.state()
    assert s["review_round"] == 3                       # critic ran 3 times, then the loop stopped
    assert s["review_status"] != "approved"
    assert "open items" in final
    bd = h.sink.records[-1]["agent_breakdown"]
    assert bd["draft_critic"]["llm_calls"] == 9 and bd["draft_reviser"]["llm_calls"] == 6


async def test_clean_draft_exits_loop_after_first_round():
    h = Harness()
    await h.ask("We need 50 laptops for new hires, budget ₹40L, delivery in 3 weeks")
    s = await h.state()
    assert s["review_round"] == 1 and s["review_status"] == "approved"
    assert "draft_reviser" not in h.sink.records[-1]["agent_breakdown"]


# ---- the deterministic half of the critic ----------------------------------------------------
@pytest.fixture
def reviewed(ctx):
    save_request("Laptops", 50, 4_000_000, 21, ctx)
    sl = search_vendors(ctx, require_contract_on_file=True)["shortlist"]
    for v in sl:
        record_contract_findings(v["vendor_id"], "ok", ctx, risk_score=2 if v["vendor_id"] == "V0002" else 3)
        check_policy(v["vendor_id"], ctx)
    set_recommended_vendor("V0002", ctx)
    return ctx


def _draft(ctx, **kw):
    summary = {"request": ctx.state["request"], "shortlist": ctx.state["shortlist"],
               "contract_findings": ctx.state["contract_findings"], "approval_decisions": ctx.state["approval_decisions"],
               "recommended_vendor_id": "V0002"}
    return build_draft(summary, **kw)


def test_good_draft_has_no_issues(reviewed):
    save_draft(_draft(reviewed), reviewed)
    assert check_draft_facts(reviewed)["issues"] == []


def test_flags_unsupported_figure(reviewed):
    save_draft(_draft(reviewed) + "\nExpected saving: ₹7,77,777.", reviewed)
    assert any("Unsupported figure" in i for i in check_draft_facts(reviewed)["issues"])


def test_accepts_derived_headroom_figure(reviewed):
    headroom = 4_000_000 - 3_445_000                    # budget - V0002 total
    save_draft(_draft(reviewed) + f"\nBudget headroom: ₹{headroom:,}.", reviewed)
    assert check_draft_facts(reviewed)["issues"] == []


def test_flags_vendor_not_on_shortlist_and_missing_vendor(reviewed):
    draft = _draft(reviewed).replace("Nimbus Devices Ltd", "Nobody Ltd").replace("V0003", "V0099")  # name AND id gone
    save_draft(draft, reviewed)
    issues = " ".join(check_draft_facts(reviewed)["issues"])
    assert "V0099" in issues and "omits shortlisted vendor" in issues


def test_flags_missing_risk_score_and_budget(reviewed):
    save_draft("Recommend Bluepeak Systems India Pvt Ltd (V0002), Orion Digital Supply LLP, Nimbus Devices Ltd. Approved.", reviewed)
    issues = " ".join(check_draft_facts(reviewed)["issues"])
    assert "does not state the budget" in issues and "contract risk score" in issues


def test_flags_rejected_vendor_recommended_without_saying_so(reviewed):
    # force a reject decision on the recommended vendor, then a draft that hides it
    decisions = dict(reviewed.state["approval_decisions"])
    decisions["V0002"] = {**decisions["V0002"], "decision": "reject", "required_approvers": []}
    reviewed.state["approval_decisions"] = decisions
    save_draft(_draft(reviewed).replace("reject", "ok"), reviewed)
    assert any("REJECTED by policy" in i for i in check_draft_facts(reviewed)["issues"])


def test_submit_review_approves_and_escalates_on_empty_issues(ctx):
    out = submit_review(ctx, [])
    assert out["status"] == "approved" and ctx.actions.escalate is True
    assert ctx.state["review_status"] == "approved" and ctx.state["review_round"] == 1


def test_submit_review_with_issues_continues_loop(ctx):
    out = submit_review(ctx, ["missing approver"])
    assert out["status"] == "revise" and ctx.actions.escalate is False
    assert ctx.state["critique"] == "- missing approver" and ctx.state["review_round"] == 1
    assert save_revised_draft("new draft", ctx)["status"] == "ok" and ctx.state["review_status"] == "revised"
