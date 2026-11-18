from __future__ import annotations

import json

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import AgentTool

from agents.approval import build_approval_agent
from agents.common import ModelFactory, REACT_PREAMBLE, default_model_factory, gen_config
from agents.contract_review import build_contract_review_agent
from agents.critic import build_review_loop
from agents.vendor_search import build_vendor_search_agent
from tools.money import format_inr
from tools.state_tools import (
    CRITIQUE_KEY, DECISIONS_KEY, DRAFT_KEY, FINDINGS_KEY, RECOMMENDED_KEY, REQUEST_KEY, REVIEW_STATUS_KEY,
    SHORTLIST_KEY, get_final_draft, get_session_summary, parse_inr_amount, save_draft, save_request,
    set_recommended_vendor,
)

NAME = "procure_planner"

INSTRUCTION = f"""You are ProcureAgent's Planner: you turn a purchase need into an approval-ready
recommendation by delegating to specialist agents and assembling their verified results.
{REACT_PREAMBLE}

Your specialists (call them as tools; each takes a short plain-text request):
- vendor_search_agent    finds and ranks a shortlist of 3 vendors from the vendor database
- contract_review_agent  reviews contracts and scores risk 1-5 (pass it the vendor ids)
- approval_agent         checks policy and returns approve / needs-approval-from-X / reject (pass it the vendor ids)
- draft_review_loop      critic + reviser loop that reviews the saved draft (max 3 rounds)

NEW PURCHASE REQUEST - follow this plan, announcing it in one line first:
1. Parse the need. Use parse_inr_amount for every budget (never do lakh/crore maths yourself),
   convert delivery windows to days (1 week = 7 days), then call save_request. If quantity, budget
   or delivery is genuinely missing, ask the user one short question instead of guessing.
2. Call vendor_search_agent ("find a shortlist for the saved request").
3. Call contract_review_agent with the shortlisted vendor ids.
4. Call approval_agent with the shortlisted vendor ids.
5. Choose the recommendation: among vendors NOT rejected by policy, prefer lower contract risk, then
   fewer/lower approvals, then price. Call set_recommended_vendor. If every vendor is rejected (or
   none qualified), do NOT call set_recommended_vendor: recommend none and make the draft explain why
   and what would change the outcome. The tool refuses rejected vendors; a refusal is final.
6. Write the draft and call save_draft. Use these sections: Summary (decision up front) / Request /
   Shortlist comparison (all vendors: unit price, total, rating, delivery, contract risk /5, policy
   result) / Recommendation / Contract risks (exact clauses) / Policy and approvals (decision label,
   approvers, conditions) / Next steps. Quote figures only as they appear in tool results.
7. Call draft_review_loop ("review the saved draft").
8. Call get_final_draft and present that text verbatim as your answer. If review_status is not
   "approved", append one line noting the review ended with open items and list the critique.

FOLLOW-UP TURNS (the session state below is already loaded - do not start over):
- "swap vendor N for a cheaper one": call vendor_search_agent asking it to swap shortlist position N
  for a cheaper vendor; then contract_review_agent and approval_agent for the NEW vendor only;
  then continue from step 5 (re-recommend, re-draft, re-review).
- Changed budget/quantity/delivery: call save_request with the updated values, then redo steps 2-8.
- Questions about existing results: answer from the session state without calling agents.

RULES: the policy decision comes from approval_agent and is never yours to change. Do not recommend a
vendor the policy rejected. Do not state any figure you did not get from a tool."""


def _fmt_state(state) -> str:
    req = state.get(REQUEST_KEY)
    if not req:
        return "No purchase request yet in this session."
    lines = [
        "CURRENT SESSION STATE",
        f"Request: {req['quantity']} x {req['category']}, budget {format_inr(req['budget_inr'])}, "
        f"delivery within {req['max_delivery_days']} days, certifications {req.get('required_certifications') or 'none'}.",
    ]
    findings, decisions = state.get(FINDINGS_KEY) or {}, state.get(DECISIONS_KEY) or {}
    short = state.get(SHORTLIST_KEY) or []
    for v in short:
        f, d = findings.get(v["vendor_id"]), decisions.get(v["vendor_id"])
        risk = "not reviewed" if f is None else ("no contract on file" if f["risk_score"] is None else f"risk {f['risk_score']}/5")
        lines.append(
            f"  #{v['rank']} {v['vendor_id']} {v['name']}: {format_inr(v['unit_price_inr'])}/unit, "
            f"total {format_inr(v['total_price_inr'])}, rating {v['rating']}, {v['delivery_days']}d | "
            f"contract: {risk} | policy: {d['label'] if d else 'not checked'}")
    if state.get(RECOMMENDED_KEY):
        lines.append(f"Recommended vendor: {state[RECOMMENDED_KEY]}")
    if state.get(REVIEW_STATUS_KEY):
        lines.append(f"Draft review status: {state[REVIEW_STATUS_KEY]}")
    if state.get(CRITIQUE_KEY):
        lines.append("Open critique: " + state[CRITIQUE_KEY].replace("\n", " "))
    return "\n".join(lines)


def planner_instruction(ctx: ReadonlyContext) -> str:
    """Dynamic instruction: the static plan plus a compact view of persisted session state, so a
    follow-up turn ('swap vendor 2 for a cheaper one') starts from what is already known."""
    return INSTRUCTION + "\n\n" + _fmt_state(ctx.state)


def build_planner(model_for: ModelFactory = default_model_factory) -> LlmAgent:
    return LlmAgent(
        name=NAME,
        model=model_for(NAME),
        description="Root planner: breaks a purchase need into steps, delegates to specialist agents and assembles the final recommendation.",
        instruction=planner_instruction,
        tools=[
            parse_inr_amount, save_request, get_session_summary, set_recommended_vendor, save_draft, get_final_draft,
            AgentTool(build_vendor_search_agent(model_for)),
            AgentTool(build_contract_review_agent(model_for)),
            AgentTool(build_approval_agent(model_for)),
            AgentTool(build_review_loop(model_for)),
        ],
        generate_content_config=gen_config(0.3),
    )
