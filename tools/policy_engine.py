"""Procurement policy engine - pure functions, no I/O beyond loading the policy file.

Decisions are made here, never by the LLM: the approval agent explains them.

    reject            any hard rule fails (over budget, over PO limit, low rating,
                      unacceptable contract risk, delivery/cert/min-qty not met)
    needs_approval    no hard failure, but one or more approvers must sign off
    approve           within auto-approval limit and no escalation triggers
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from data.synthetic import Vendor
from tools.money import format_inr
from settings import get_settings


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_policy() -> dict:
    return _load(str(get_settings().policy_path))


def tier_for(amount: float, policy: dict) -> dict | None:
    for tier in policy["approval_tiers"]:
        if amount <= tier["up_to_inr"]:
            return tier
    return None


def evaluate(
    request: dict,
    vendor: Vendor,
    contract: dict | None,
    shortlist: list[dict] | None = None,
    policy: dict | None = None,
) -> dict[str, Any]:
    """Evaluate one vendor against the request and policy.

    contract: the recorded contract findings for this vendor, or None if not reviewed.
    shortlist: current shortlist cards, used for the preferred-vendor rule.
    """
    policy = policy or load_policy()
    hard, vr = policy["hard_limits"], policy["vendor_rules"]
    qty = request["quantity"]
    total = vendor.unit_price_inr * qty
    checks: list[dict] = []
    reasons: list[str] = []
    conditions: list[str] = []
    approvers: list[str] = []

    def check(rule: str, passed: bool, detail: str, blocking: bool = True) -> None:
        checks.append({"rule": rule, "passed": passed, "detail": detail, "blocking": blocking})
        if not passed and blocking:
            reasons.append(detail)

    # --- hard rules -------------------------------------------------------
    budget = request["budget_inr"]
    check("budget", total <= budget,
          f"Total {format_inr(total)} {'is within' if total <= budget else 'exceeds'} the stated budget of {format_inr(budget)}.")
    check("po_limit", total <= hard["max_single_po_inr"],
          f"Total {format_inr(total)} {'is within' if total <= hard['max_single_po_inr'] else 'exceeds'} the single-PO limit of {format_inr(hard['max_single_po_inr'])}.")
    check("vendor_rating", vendor.rating >= hard["min_vendor_rating"],
          f"Vendor rating {vendor.rating} {'meets' if vendor.rating >= hard['min_vendor_rating'] else 'is below'} the policy minimum of {hard['min_vendor_rating']}.")
    max_days = request["max_delivery_days"]
    check("delivery", vendor.delivery_days <= max_days,
          f"Delivery in {vendor.delivery_days} days {'meets' if vendor.delivery_days <= max_days else 'misses'} the {max_days}-day requirement.")
    needed = [c for c in request.get("required_certifications") or []]
    missing = [c for c in needed if c.lower() not in {x.lower() for x in vendor.certifications}]
    check("certifications", not missing,
          "Required certifications held." if not missing else f"Missing required certification(s): {', '.join(missing)}.")
    check("min_order_qty", qty >= vendor.min_order_qty,
          f"Quantity {qty} {'meets' if qty >= vendor.min_order_qty else 'is below'} the vendor minimum order of {vendor.min_order_qty}.")

    risk = contract.get("risk_score") if contract else None
    if risk is not None:
        check("contract_risk", risk < hard["reject_contract_risk_at_or_above"],
              f"Contract risk score {risk}/5 {'is acceptable' if risk < hard['reject_contract_risk_at_or_above'] else 'is unacceptable (renegotiate before proceeding)'}.")

    # --- escalation rules (only matter if nothing hard failed) -----------
    tier = tier_for(total, policy)
    if tier is None:
        check("approval_tier", False, f"Total {format_inr(total)} is beyond the highest delegated approval tier.")
    else:
        if tier["approver"]:
            approvers.append(tier["approver"])
        checks.append({"rule": "approval_tier", "passed": True, "blocking": False,
                       "detail": f"Amount {format_inr(total)} falls in tier '{tier['note']}' -> "
                                 f"{tier['approver'] or 'no approval required'}."})

    if not vendor.is_preferred:
        alt = None
        for c in shortlist or []:
            if (c["vendor_id"] != vendor.vendor_id and c.get("is_preferred") and
                    c["unit_price_inr"] <= vendor.unit_price_inr * (1 + vr["preferred_price_tolerance_pct"] / 100)):
                alt = c
                break
        over = total > vr["non_preferred_requires_procurement_head_above_inr"]
        if over or alt:
            approvers.append("Procurement Head")
            why = (f"preferred vendor {alt['name']} is within {vr['preferred_price_tolerance_pct']}% of this price"
                   if alt else f"order exceeds {format_inr(vr['non_preferred_requires_procurement_head_above_inr'])}")
            checks.append({"rule": "preferred_vendor", "passed": False, "blocking": False,
                           "detail": f"Non-preferred vendor: {why}; written justification required."})
            conditions.append("Attach written justification for choosing a non-preferred vendor.")
        else:
            checks.append({"rule": "preferred_vendor", "passed": True, "blocking": False,
                           "detail": "Non-preferred vendor, but below the justification threshold."})
    else:
        checks.append({"rule": "preferred_vendor", "passed": True, "blocking": False, "detail": "Preferred vendor."})

    if contract is None or not contract.get("contract_found", True):
        if vr["missing_contract_requires_legal"]:
            approvers.append("Legal Counsel")
            conditions.append("No contract on file: Legal must obtain and review a contract before a PO is raised.")
            checks.append({"rule": "contract_on_file", "passed": False, "blocking": False,
                           "detail": "No contract on file for this vendor."})
    else:
        if risk is not None and risk >= vr["contract_risk_requires_legal_at_or_above"]:
            approvers.append("Legal Counsel")
            conditions.append("Legal must sign off on the flagged contract clauses (or negotiate them) before PO.")
        elif risk is not None and risk >= vr["contract_risk_advisory_at"]:
            conditions.append("Recommended: Legal review of flagged contract clauses before PO.")

    failed = [c for c in checks if c["blocking"] and not c["passed"]]
    approvers = list(dict.fromkeys(approvers))  # de-dupe, keep order
    if failed:
        decision, label, approvers = "reject", "reject", []
    elif approvers:
        decision, label = "needs_approval", "needs-approval-from: " + ", ".join(approvers)
    else:
        decision, label = "approve", "approve"
    return {
        "vendor_id": vendor.vendor_id, "vendor_name": vendor.name,
        "decision": decision, "label": label,
        "required_approvers": approvers, "total_price_inr": total, "total_price": format_inr(total),
        "reasons": reasons if decision == "reject" else
                   [c["detail"] for c in checks if c["passed"] is False or c["rule"] == "approval_tier"],
        "conditions": [] if decision == "reject" else conditions, "checks": checks, "policy_version": policy["version"],
    }
