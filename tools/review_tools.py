"""Tools for the critic / reviser loop.

check_draft_facts is the deterministic half of the critic: it catches missing
fields, a decision that contradicts the policy engine, vendor ids that are not
on the shortlist, and rupee amounts that appear nowhere in the session data
(the 'unsupported claims' check). The LLM critic adds judgement on top.
"""

from __future__ import annotations

import re
from typing import Optional

from google.adk.tools import ToolContext

from tools.money import format_inr
from tools.policy_engine import load_policy
from tools.state_tools import (
    CRITIQUE_KEY, DECISIONS_KEY, DRAFT_KEY, FINDINGS_KEY, RECOMMENDED_KEY, REQUEST_KEY, REVIEW_STATUS_KEY,
    SHORTLIST_KEY,
)

ROUND_KEY = "review_round"
_AMOUNT = re.compile(r"₹\s?([0-9][0-9,]*(?:\.[0-9]+)?)\s*(Cr|L)?\b")


def _amounts_in(text: str) -> list[float]:
    out = []
    for m in _AMOUNT.finditer(text):
        n = float(m.group(1).replace(",", ""))
        out.append(n * {"Cr": 10_000_000, "L": 100_000}.get(m.group(2) or "", 1))
    return out


def _known_amounts(state) -> list[float]:
    req = state.get(REQUEST_KEY) or {}
    short = state.get(SHORTLIST_KEY) or []
    known: set[float] = set()
    if req:
        known.add(req["budget_inr"])
    totals = [v["total_price_inr"] for v in short]
    known.update(totals)
    known.update(v["unit_price_inr"] for v in short)
    policy = load_policy()
    known.update(t["up_to_inr"] for t in policy["approval_tiers"])
    known.update([policy["hard_limits"]["max_single_po_inr"],
                  policy["vendor_rules"]["non_preferred_requires_procurement_head_above_inr"]])
    # derived figures a good draft legitimately quotes: headroom, savings, deltas
    derived = set()
    for t in totals:
        if req:
            derived.add(abs(req["budget_inr"] - t))
        for u in totals:
            derived.add(abs(t - u))
    for a in short:
        for b in short:
            derived.add(abs(a["unit_price_inr"] - b["unit_price_inr"]) * (req.get("quantity", 1) if req else 1))
            derived.add(abs(a["unit_price_inr"] - b["unit_price_inr"]))
    for d in (state.get(DECISIONS_KEY) or {}).values():
        known.add(d["total_price_inr"])
    return sorted(known | derived)


def _close(x: float, pool: list[float]) -> bool:
    # amounts are often quoted rounded (₹29.95 L): allow 0.6% relative tolerance
    return any(abs(x - k) <= max(1.0, 0.006 * k) for k in pool)


def get_review_context(tool_context: ToolContext) -> dict:
    """Return the verified facts the draft must agree with: request, shortlist,
    contract findings, policy decisions, the recommended vendor, the current
    draft and the critique from the previous round."""
    s = tool_context.state
    return {
        "request": s.get(REQUEST_KEY), "shortlist": s.get(SHORTLIST_KEY),
        "contract_findings": s.get(FINDINGS_KEY), "approval_decisions": s.get(DECISIONS_KEY),
        "recommended_vendor_id": s.get(RECOMMENDED_KEY), "draft_recommendation": s.get(DRAFT_KEY),
        "previous_critique": s.get(CRITIQUE_KEY), "review_round": s.get(ROUND_KEY, 0),
    }


def check_draft_facts(tool_context: ToolContext) -> dict:
    """Run the deterministic consistency checks on the saved draft. Returns a
    list of `issues` (empty when none were found). Treat each as a finding unless
    you can show it is a false positive."""
    s = tool_context.state
    draft = s.get(DRAFT_KEY) or ""
    if not draft.strip():
        return {"status": "ok", "issues": ["No draft saved."]}
    low = draft.lower()
    req = s.get(REQUEST_KEY) or {}
    short = s.get(SHORTLIST_KEY) or []
    decisions = s.get(DECISIONS_KEY) or {}
    findings = s.get(FINDINGS_KEY) or {}
    rec_id = s.get(RECOMMENDED_KEY) or ""
    issues: list[str] = []

    if req and format_inr(req["budget_inr"]).split(" (")[0] not in draft and \
            not any(abs(a - req["budget_inr"]) < 1 for a in _amounts_in(draft)):
        issues.append("Draft does not state the budget.")

    on_list = {v["vendor_id"] for v in short}
    for vid in set(re.findall(r"\bV\d{4}\b", draft)) - on_list:
        issues.append(f"Draft references vendor {vid}, which is not on the shortlist.")

    mentioned = [v for v in short if v["name"].lower() in low or v["vendor_id"].lower() in low]
    if short and len(mentioned) < len(short):
        missing = [v["name"] for v in short if v not in mentioned]
        issues.append("Draft omits shortlisted vendor(s): " + ", ".join(missing) + " (comparison is incomplete).")

    if rec_id and rec_id in on_list:
        rec = next(v for v in short if v["vendor_id"] == rec_id)
        dec = decisions.get(rec_id)
        if dec is None:
            issues.append(f"Recommended vendor {rec['name']} has no policy decision recorded.")
        else:
            if dec["decision"] == "reject" and "reject" not in low:
                issues.append("Recommended vendor was REJECTED by policy but the draft does not say so.")
            if dec["decision"] == "needs_approval":
                for a in dec["required_approvers"]:
                    if a.lower() not in low:
                        issues.append(f"Draft omits required approver: {a}.")
            if dec["decision"] == "approve" and "approv" not in low:
                issues.append("Draft does not state the approval outcome.")
            if format_inr(dec["total_price_inr"]).split(" (")[0] not in draft:
                issues.append(f"Draft does not state the total cost {format_inr(dec['total_price_inr'])} for {rec['name']}.")
        f = findings.get(rec_id)
        if f is None:
            issues.append(f"No contract review recorded for {rec['name']}.")
        elif f.get("risk_score") is not None and not re.search(rf"\b{f['risk_score']}\s*/\s*5\b", draft):
            issues.append(f"Draft does not state the contract risk score ({f['risk_score']}/5) for {rec['name']}.")
        elif f.get("risk_score") is None and "no contract" not in low:
            issues.append(f"{rec['name']} has no contract on file; the draft must say so.")
    elif short and not rec_id and "reject" not in low and "no vendor" not in low and "not recommend" not in low:
        issues.append("No vendor is recommended, yet the draft does not explain why (rejection / no compliant vendor).")

    pool = _known_amounts(s)
    for amount in sorted(set(_amounts_in(draft))):
        if amount and not _close(amount, pool):
            issues.append(f"Unsupported figure: {format_inr(amount)} does not match any value in the session data.")

    return {"status": "ok", "issues": issues}


def submit_review(tool_context: ToolContext, issues: Optional[list[str]] = None) -> dict:
    """Submit your verdict on the draft. Pass an empty list (or omit `issues`) to
    APPROVE the draft and end the review loop. Pass a list of concrete, actionable
    problems to send it back to the reviser.

    Args:
        issues: each a specific problem: missing information, a policy violation,
            or a claim not supported by the session data.
    """
    state = tool_context.state
    issues = [i.strip() for i in (issues or []) if i and i.strip()]
    state[ROUND_KEY] = int(state.get(ROUND_KEY) or 0) + 1
    tool_context.actions.skip_summarization = True  # no extra critic turn; hand over to reviser / exit
    if not issues:
        state[REVIEW_STATUS_KEY] = "approved"
        state[CRITIQUE_KEY] = ""
        tool_context.actions.escalate = True        # exits the LoopAgent
        return {"status": "approved", "round": state[ROUND_KEY]}
    state[REVIEW_STATUS_KEY] = "revising"
    state[CRITIQUE_KEY] = "\n".join(f"- {i}" for i in issues)
    return {"status": "revise", "round": state[ROUND_KEY], "issues": issues}


def save_revised_draft(draft_markdown: str, tool_context: ToolContext) -> dict:
    """Replace the draft with your revised version.

    Args:
        draft_markdown: the complete revised recommendation (markdown).
    """
    if not draft_markdown.strip():
        return {"status": "error", "error": "Draft is empty."}
    tool_context.state[DRAFT_KEY] = draft_markdown
    tool_context.state[REVIEW_STATUS_KEY] = "revised"
    tool_context.actions.skip_summarization = True
    return {"status": "ok", "chars": len(draft_markdown)}
