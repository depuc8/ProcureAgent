"""Policy tools used by the approval_agent."""

from __future__ import annotations

from google.adk.tools import ToolContext

from tools.money import format_inr
from tools.policy_engine import evaluate, load_policy
from tools.state_tools import DECISIONS_KEY, FINDINGS_KEY, REQUEST_KEY, SHORTLIST_KEY
from tools.vendor_repository import get_repository


def get_policy_summary() -> dict:
    """Return the procurement policy: approval tiers by amount, hard limits and
    vendor rules. Use it to explain a decision; do not use it to compute one."""
    p = load_policy()
    return {
        "version": p["version"],
        "approval_tiers": [
            {"up_to": format_inr(t["up_to_inr"]), "approver": t["approver"] or "none (auto-approved)"}
            for t in p["approval_tiers"]
        ],
        "hard_limits": p["hard_limits"],
        "vendor_rules": p["vendor_rules"],
    }


def check_policy(vendor_id: str, tool_context: ToolContext) -> dict:
    """Check one vendor against procurement policy and record the decision.

    Reads the request, the vendor's real attributes and the recorded contract
    findings itself, so no figures need to be passed in. The decision
    (approve / needs_approval / reject) and the required approvers come from the
    policy engine and must not be overridden.

    Args:
        vendor_id: the vendor under evaluation, e.g. "V0002".
    """
    state = tool_context.state
    request = state.get(REQUEST_KEY)
    if not request:
        return {"status": "error", "error": "No saved request. Call save_request first."}
    vendor = get_repository().get(vendor_id)
    if vendor is None:
        return {"status": "error", "error": f"Unknown vendor {vendor_id}."}
    contract = (state.get(FINDINGS_KEY) or {}).get(vendor_id)
    result = evaluate(request, vendor, contract, state.get(SHORTLIST_KEY))
    result["contract_reviewed"] = contract is not None
    decisions = dict(state.get(DECISIONS_KEY) or {})
    decisions[vendor_id] = result
    state[DECISIONS_KEY] = decisions
    return {"status": "ok", **result}
