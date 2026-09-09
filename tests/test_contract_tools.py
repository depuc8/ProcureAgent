import pytest

from data.generate_contracts import SPECS
from tools.contract_scanner import scan_contract
from tools.contract_tools import get_contract, record_contract_findings, scan_contract_clauses
from tools.state_tools import FINDINGS_KEY


def test_ten_contracts_on_file():
    found = [vid for vid in SPECS if get_contract(vid)["status"] == "ok"]
    assert len(found) == 10


@pytest.mark.parametrize("vendor_id", sorted(SPECS))
def test_scanner_matches_designed_risk(vendor_id):
    """Pins the rubric against the hand-assigned intended score, not against its own output."""
    out = scan_contract_clauses(vendor_id)
    assert out["status"] == "ok"
    assert out["baseline_score"] == SPECS[vendor_id]["intended"], out["flags"]


def test_scanner_flags_specific_clauses():
    v3 = {f["clause"]: f for f in scan_contract_clauses("V0003")["flags"]}
    assert v3["auto_renewal"]["severity"] == 5 and "36 months" in v3["auto_renewal"]["found"]
    assert v3["liability_cap"]["severity"] == 5
    assert v3["termination"]["severity"] == 4 and "cause only" in v3["termination"]["found"]
    assert v3["payment_terms"]["severity"] == 4            # Net 15 + 3%/month late interest
    v5 = {f["clause"]: f for f in scan_contract_clauses("V0005")["flags"]}
    assert v5["payment_terms"]["severity"] == 5 and "100% advance" in v5["payment_terms"]["found"]
    v7 = scan_contract_clauses("V0007")
    assert {w["item"] for w in v7["watch_items"]} == {"exclusivity"}
    assert {w["item"] for w in scan_contract_clauses("V0008")["watch_items"]} == {"unilateral_price_change"}
    assert scan_contract_clauses("V0001")["watch_items"] == []


def test_scanner_handles_unparseable_text():
    res = scan_contract("This agreement is silent on everything.")
    assert res.baseline_score >= 2
    sev = {f.clause: f.severity for f in res.flags}
    assert sev["liability_cap"] == 3 and sev["payment_terms"] == 3   # missing clauses are not 'safe'


@pytest.mark.parametrize("bad", ["../../etc/passwd", "V0001/../../x", "v0001", "", "V1", "V0001.txt"])
def test_path_traversal_rejected(bad):
    assert get_contract(bad)["status"] == "not_found"
    assert scan_contract_clauses(bad)["status"] == "not_found"


def test_no_contract_on_file():
    assert get_contract("V0100")["status"] == "not_found"


def test_record_findings_clamps_to_baseline_plus_minus_one(ctx):
    # V0003 baseline is 5; an agent trying to wave it through with a 1 is clamped to 4
    out = record_contract_findings("V0003", "looks fine to me", ctx, risk_score=1)
    f = out["findings"]
    assert f["baseline_score"] == 5 and f["risk_score"] == 4 and f["score_adjusted_from"] == 1
    # raising by one (exclusivity the scanner cannot score) is allowed
    out = record_contract_findings("V0007", "exclusivity clause", ctx, risk_score=5)
    assert out["findings"]["risk_score"] == 5 and out["findings"]["score_adjusted_from"] is None
    assert set(ctx.state[FINDINGS_KEY]) == {"V0003", "V0007"}


def test_record_findings_validates_score(ctx):
    assert record_contract_findings("V0001", "x", ctx, risk_score=9)["status"] == "error"
    assert record_contract_findings("V0001", "x", ctx)["status"] == "error"


def test_record_findings_for_missing_contract(ctx):
    out = record_contract_findings("V0100", "none on file", ctx)
    assert out["findings"]["contract_found"] is False and out["findings"]["risk_score"] is None


def test_findings_accumulate_across_calls(ctx):
    record_contract_findings("V0001", "ok", ctx, risk_score=1)
    record_contract_findings("V0002", "ok", ctx, risk_score=2)
    assert set(ctx.state[FINDINGS_KEY]) == {"V0001", "V0002"}
