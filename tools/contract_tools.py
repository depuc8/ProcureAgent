"""Contract tools used by the contract_review_agent."""

from __future__ import annotations

import re
from typing import Optional

from google.adk.tools import ToolContext

from settings import get_settings
from tools.contract_scanner import scan_contract
from tools.state_tools import FINDINGS_KEY

_VENDOR_ID = re.compile(r"^V\d{4}$")


def _read_contract(vendor_id: str) -> str | None:
    # vendor_id is validated against a strict pattern: it is used to build a path.
    if not _VENDOR_ID.match(vendor_id or ""):
        return None
    path = get_settings().contracts_dir / f"{vendor_id}.txt"
    return path.read_text(encoding="utf-8") if path.is_file() else None


def get_contract(vendor_id: str) -> dict:
    """Read the full contract text on file for a vendor.

    Args:
        vendor_id: vendor id such as "V0003".

    Returns:
        {"status": "ok", "text": ...} or {"status": "not_found"} when no contract is on file.
    """
    text = _read_contract(vendor_id)
    if text is None:
        return {"status": "not_found", "vendor_id": vendor_id,
                "message": "No contract on file for this vendor."}
    return {"status": "ok", "vendor_id": vendor_id, "text": text}


def scan_contract_clauses(vendor_id: str) -> dict:
    """Run the deterministic clause scanner on a vendor's contract. It rates
    auto-renewal, liability cap, payment terms and termination from 1 (safe) to
    5 (very risky), lists watch items (exclusivity, unilateral price changes) and
    returns a baseline overall score. Use it as the starting point, then read the
    contract text yourself for anything the scanner cannot see.

    Args:
        vendor_id: vendor id such as "V0003".
    """
    text = _read_contract(vendor_id)
    if text is None:
        return {"status": "not_found", "vendor_id": vendor_id, "message": "No contract on file for this vendor."}
    return {"status": "ok", "vendor_id": vendor_id, **scan_contract(text).to_dict()}


def record_contract_findings(
    vendor_id: str,
    summary: str,
    tool_context: ToolContext,
    risk_score: Optional[int] = None,
    risky_clauses: Optional[list[str]] = None,
    recommendation: str = "",
) -> dict:
    """Save your contract review for one vendor to session state.

    Your risk_score may differ from the scanner baseline by at most 1 (raise it
    for issues the scanner missed, e.g. exclusivity; lower it only with clear
    justification in `summary`). If the vendor has no contract on file, leave
    risk_score empty.

    Args:
        vendor_id: vendor id such as "V0003".
        summary: 1-3 sentence plain-English assessment.
        risk_score: overall risk 1 (safe) to 5 (very risky); omit if no contract on file.
        risky_clauses: short descriptions of the risky clauses found.
        recommendation: e.g. "Acceptable", "Negotiate liability cap before signing", "Do not proceed".
    """
    text = _read_contract(vendor_id)
    findings: dict
    if text is None:
        findings = {"vendor_id": vendor_id, "contract_found": False, "risk_score": None,
                    "summary": summary or "No contract on file.", "risky_clauses": [],
                    "recommendation": recommendation or "Obtain a contract before proceeding."}
    else:
        if risk_score is None or not 1 <= int(risk_score) <= 5:
            return {"status": "error", "error": "risk_score must be an integer 1-5 when a contract is on file."}
        scan = scan_contract(text)
        baseline = scan.baseline_score
        final = max(baseline - 1, min(baseline + 1, int(risk_score)))
        findings = {
            "vendor_id": vendor_id, "contract_found": True, "risk_score": final,
            "baseline_score": baseline, "score_adjusted_from": int(risk_score) if final != int(risk_score) else None,
            "summary": summary, "risky_clauses": risky_clauses or [],
            "clause_severity": {f.clause: f.severity for f in scan.flags},
            "watch_items": [w["item"] for w in scan.watch_items],
            "recommendation": recommendation,
        }
    current = dict(tool_context.state.get(FINDINGS_KEY) or {})
    current[vendor_id] = findings
    tool_context.state[FINDINGS_KEY] = current
    return {"status": "ok", "findings": findings}
