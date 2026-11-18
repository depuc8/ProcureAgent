"""Session-state tools. State keys (persisted by the ADK session service):

    request             parsed purchase need
    shortlist           list of 3 vendor dicts (rank order)
    contract_findings   {vendor_id: findings}
    approval_decisions  {vendor_id: decision}
    recommended_vendor_id
    draft_recommendation, review_status, critique

Nested values are always re-assigned whole (never mutated in place) so every
change lands in the event's state_delta and survives a durable session service.
"""

from __future__ import annotations

from typing import Optional

from google.adk.tools import ToolContext

from tools.money import format_inr, parse_inr

REQUEST_KEY = "request"
SHORTLIST_KEY = "shortlist"
FINDINGS_KEY = "contract_findings"
DECISIONS_KEY = "approval_decisions"
RECOMMENDED_KEY = "recommended_vendor_id"
DRAFT_KEY = "draft_recommendation"
REVIEW_STATUS_KEY = "review_status"
CRITIQUE_KEY = "critique"


def parse_inr_amount(text: str) -> dict:
    """Convert an amount written in Indian notation to rupees.

    Use this for every budget or price the user states ("₹40L", "1.5 crore",
    "Rs 2,50,000") instead of doing lakh/crore arithmetic yourself.

    Args:
        text: the phrase containing the amount, e.g. "budget ₹40L".

    Returns:
        {"status": "ok", "amount_inr": 4000000.0} or {"status": "error", ...}.
    """
    amount = parse_inr(text)
    if amount is None:
        return {"status": "error", "error": f"No amount found in {text!r}."}
    return {"status": "ok", "amount_inr": amount, "formatted": format_inr(amount)}


def save_request(
    category: str,
    quantity: int,
    budget_inr: float,
    max_delivery_days: int,
    tool_context: ToolContext,
    required_certifications: Optional[list[str]] = None,
    min_rating: float = 3.5,
    notes: str = "",
) -> dict:
    """Persist the structured purchase request in session state.

    Call this first, once the user's need is parsed. Convert delivery windows to
    days (1 week = 7 days) and budgets to rupees (use parse_inr_amount).

    Args:
        category: product/service category, e.g. "Laptops".
        quantity: number of units.
        budget_inr: total budget in rupees.
        max_delivery_days: latest acceptable delivery, in days.
        required_certifications: certifications the vendor must hold, if any.
        min_rating: minimum vendor rating (default 3.5, the policy floor).
        notes: any other constraints stated by the user.
    """
    if quantity < 1 or budget_inr <= 0 or max_delivery_days < 1:
        return {"status": "error", "error": "quantity, budget_inr and max_delivery_days must be positive."}
    request = {
        "category": category.strip(),
        "quantity": int(quantity),
        "budget_inr": float(budget_inr),
        "max_delivery_days": int(max_delivery_days),
        "required_certifications": [c.strip() for c in (required_certifications or [])],
        "min_rating": float(min_rating),
        "notes": notes,
    }
    tool_context.state[REQUEST_KEY] = request
    return {"status": "ok", "request": request}


def get_session_summary(tool_context: ToolContext) -> dict:
    """Return everything stored so far for this conversation: the request, the
    shortlist, contract findings, approval decisions, recommended vendor and the
    current draft. Call this at the start of a follow-up turn."""
    s = tool_context.state
    return {
        "request": s.get(REQUEST_KEY),
        "shortlist": s.get(SHORTLIST_KEY),
        "contract_findings": s.get(FINDINGS_KEY),
        "approval_decisions": s.get(DECISIONS_KEY),
        "recommended_vendor_id": s.get(RECOMMENDED_KEY),
        "draft_recommendation": s.get(DRAFT_KEY),
        "review_status": s.get(REVIEW_STATUS_KEY),
    }


def set_recommended_vendor(vendor_id: str, tool_context: ToolContext) -> dict:
    """Record which shortlisted vendor the recommendation is built around.

    Refuses a vendor that policy rejected or that has not been policy-checked yet.

    Args:
        vendor_id: must be a shortlisted vendor that passed policy (approve or needs-approval).
    """
    shortlist = tool_context.state.get(SHORTLIST_KEY) or []
    if vendor_id not in {v["vendor_id"] for v in shortlist}:
        return {"status": "error", "error": f"{vendor_id} is not in the current shortlist."}
    # Enforced here, not just in the prompt: a policy-rejected (or unchecked) vendor can never be recommended.
    decision = (tool_context.state.get(DECISIONS_KEY) or {}).get(vendor_id)
    if decision is None:
        return {"status": "error", "error": f"{vendor_id} has not been policy-checked. Call approval_agent first."}
    if decision["decision"] == "reject":
        return {"status": "refused", "error": f"Policy REJECTED {vendor_id}; it cannot be recommended. "
                "If no shortlisted vendor passes policy, recommend none and explain why.",
                "reasons": decision["reasons"]}
    tool_context.state[RECOMMENDED_KEY] = vendor_id
    return {"status": "ok", "recommended_vendor_id": vendor_id}


def save_draft(draft_markdown: str, tool_context: ToolContext) -> dict:
    """Store the draft recommendation so the critic loop can review it.

    Args:
        draft_markdown: the complete draft recommendation (markdown).
    """
    if not draft_markdown.strip():
        return {"status": "error", "error": "Draft is empty."}
    tool_context.state[DRAFT_KEY] = draft_markdown
    tool_context.state[REVIEW_STATUS_KEY] = "pending"
    tool_context.state[CRITIQUE_KEY] = ""
    return {"status": "ok", "chars": len(draft_markdown)}


def get_final_draft(tool_context: ToolContext) -> dict:
    """Return the reviewed recommendation to present to the user verbatim."""
    return {
        "draft_recommendation": tool_context.state.get(DRAFT_KEY) or "",
        "review_status": tool_context.state.get(REVIEW_STATUS_KEY) or "pending",
    }
