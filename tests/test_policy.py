import pytest

from data.synthetic import Vendor
from tools.policy_engine import evaluate, load_policy, tier_for
from tools.policy_tools import check_policy, get_policy_summary
from tools.state_tools import DECISIONS_KEY, FINDINGS_KEY, save_request
from tools.vendor_tools import search_vendors


def vendor(**kw) -> Vendor:
    base = dict(vendor_id="VT001", name="Test Vendor", category="Laptops", unit_price_inr=10_000.0,
                rating=4.5, delivery_days=10, certifications=("ISO 9001",), min_order_qty=1,
                is_preferred=True, contract_on_file=True, region="Pune")
    base.update(kw)
    return Vendor(**base)


def req(qty=10, budget=1_000_000, days=21, certs=()):
    return {"category": "Laptops", "quantity": qty, "budget_inr": budget, "max_delivery_days": days,
            "required_certifications": list(certs), "min_rating": 3.5}


SAFE = {"contract_found": True, "risk_score": 1}


def decide(request, v, contract=SAFE, shortlist=None):
    return evaluate(request, v, contract, shortlist)


# ---- hard rejects -------------------------------------------------------------

def test_over_budget_rejected():
    d = decide(req(qty=10, budget=50_000), vendor(unit_price_inr=10_000))      # 1L total > 50k
    assert d["decision"] == "reject" and any("exceeds the stated budget" in r for r in d["reasons"])


def test_exactly_on_budget_is_ok():
    assert decide(req(qty=10, budget=100_000), vendor(unit_price_inr=10_000))["decision"] != "reject"


def test_above_single_po_limit_rejected():
    d = decide(req(qty=1, budget=10**9), vendor(unit_price_inr=60_000_000))
    assert d["decision"] == "reject" and any("single-PO limit" in r for r in d["reasons"])


def test_low_rating_rejected():
    assert decide(req(), vendor(rating=3.4))["decision"] == "reject"
    assert decide(req(), vendor(rating=3.5))["decision"] != "reject"


def test_late_delivery_rejected():
    d = decide(req(days=14), vendor(delivery_days=15))
    assert d["decision"] == "reject" and any("misses" in r for r in d["reasons"])


def test_missing_certification_rejected():
    d = decide(req(certs=["ISO 27001"]), vendor())
    assert d["decision"] == "reject" and any("ISO 27001" in r for r in d["reasons"])
    assert decide(req(certs=["iso 9001"]), vendor())["decision"] != "reject"   # case-insensitive


def test_below_min_order_qty_rejected():
    assert decide(req(qty=5), vendor(min_order_qty=10))["decision"] == "reject"


def test_contract_risk_5_rejected():
    d = decide(req(), vendor(), {"contract_found": True, "risk_score": 5})
    assert d["decision"] == "reject" and d["conditions"] == []


# ---- approval tiers -----------------------------------------------------------

@pytest.mark.parametrize("total,approver", [
    (200_000, None), (200_001, "Department Head"), (1_000_000, "Department Head"),
    (1_000_001, "Finance Director"), (5_000_000, "Finance Director"),
    (5_000_001, "CFO"), (50_000_000, "CFO"),
])
def test_tier_boundaries(total, approver):
    assert tier_for(total, load_policy())["approver"] == approver


def test_auto_approve_small_preferred_order():
    d = decide(req(qty=10, budget=500_000), vendor(unit_price_inr=10_000))        # 1L
    assert d["decision"] == "approve" and d["required_approvers"] == []


def test_mid_order_needs_department_head():
    d = decide(req(qty=50, budget=2_000_000), vendor(unit_price_inr=10_000))      # 5L
    assert d["decision"] == "needs_approval" and d["required_approvers"] == ["Department Head"]
    assert d["label"] == "needs-approval-from: Department Head"


def test_large_order_needs_cfo():
    d = decide(req(qty=100, budget=10**8), vendor(unit_price_inr=100_000))        # 1 Cr
    assert d["required_approvers"] == ["CFO"]


# ---- preferred-vendor + contract escalations ----------------------------------

def test_non_preferred_large_order_needs_procurement_head():
    d = decide(req(qty=50, budget=2_000_000), vendor(unit_price_inr=30_000, is_preferred=False))   # 15L
    assert d["required_approvers"] == ["Finance Director", "Procurement Head"]
    assert any("justification" in c for c in d["conditions"])


def test_non_preferred_small_order_no_escalation():
    d = decide(req(qty=10, budget=500_000), vendor(unit_price_inr=10_000, is_preferred=False))
    assert d["decision"] == "approve"


def test_preferred_alternative_within_tolerance_escalates_small_order():
    short = [{"vendor_id": "VP", "name": "Pref Co", "is_preferred": True, "unit_price_inr": 10_900}]
    d = decide(req(qty=10, budget=500_000), vendor(unit_price_inr=10_000, is_preferred=False), shortlist=short)
    assert d["required_approvers"] == ["Procurement Head"]


def test_preferred_alternative_outside_tolerance_does_not_escalate():
    short = [{"vendor_id": "VP", "name": "Pref Co", "is_preferred": True, "unit_price_inr": 12_000}]
    d = decide(req(qty=10, budget=500_000), vendor(unit_price_inr=10_000, is_preferred=False), shortlist=short)
    assert d["decision"] == "approve"


def test_contract_risk_4_requires_legal():
    d = decide(req(), vendor(), {"contract_found": True, "risk_score": 4})
    assert "Legal Counsel" in d["required_approvers"]


def test_contract_risk_3_is_advisory_only():
    d = decide(req(qty=10, budget=500_000), vendor(), {"contract_found": True, "risk_score": 3})
    assert d["decision"] == "approve" and any("Legal review" in c for c in d["conditions"])


def test_missing_contract_requires_legal():
    d = decide(req(qty=10, budget=500_000), vendor(), None)
    assert d["required_approvers"] == ["Legal Counsel"]
    d = decide(req(qty=10, budget=500_000), vendor(), {"contract_found": False, "risk_score": None})
    assert d["required_approvers"] == ["Legal Counsel"]


def test_approvers_deduplicated_and_ordered():
    d = decide(req(qty=50, budget=10**7), vendor(unit_price_inr=30_000, is_preferred=False),
               {"contract_found": True, "risk_score": 4})
    assert d["required_approvers"] == ["Finance Director", "Procurement Head", "Legal Counsel"]


# ---- the tools wrapping the engine ---------------------------------------------

def test_check_policy_tool_reads_real_vendor_and_records_decision(laptop_request):
    sl = search_vendors(laptop_request, require_contract_on_file=True)["shortlist"]
    laptop_request.state[FINDINGS_KEY] = {sl[2]["vendor_id"]: {"contract_found": True, "risk_score": 2}}
    out = check_policy(sl[2]["vendor_id"], laptop_request)
    assert out["status"] == "ok" and out["total_price_inr"] == sl[2]["total_price_inr"]
    assert out["contract_reviewed"] is True
    assert laptop_request.state[DECISIONS_KEY][sl[2]["vendor_id"]]["decision"] == out["decision"]


def test_check_policy_unreviewed_contract_flagged(laptop_request):
    out = check_policy("V0002", laptop_request)
    assert out["contract_reviewed"] is False and "Legal Counsel" in out["required_approvers"]


def test_check_policy_errors(ctx):
    assert check_policy("V0001", ctx)["status"] == "error"          # no request yet
    save_request("Laptops", 5, 500_000, 21, ctx)
    assert check_policy("V9999", ctx)["status"] == "error"          # unknown vendor


def test_policy_summary_lists_tiers():
    s = get_policy_summary()
    assert [t["approver"] for t in s["approval_tiers"]][0].startswith("none")
    assert s["hard_limits"]["max_single_po_inr"] == 50_000_000
