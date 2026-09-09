"""Vendor-search tools used by the vendor_search_agent."""

from __future__ import annotations

import difflib
from dataclasses import replace
from typing import Optional

from google.adk.tools import ToolContext

from data.synthetic import Vendor
from tools.money import format_inr
from tools.ranking import Ranked, rank_vendors
from tools.state_tools import (
    DECISIONS_KEY, DRAFT_KEY, FINDINGS_KEY, RECOMMENDED_KEY, REQUEST_KEY, REVIEW_STATUS_KEY, SHORTLIST_KEY,
)
from tools.vendor_repository import VendorFilter, get_repository

SHORTLIST_SIZE = 3


def resolve_category(text: str, categories: list[str]) -> str | None:
    t = (text or "").strip().lower()
    if not t:
        return None
    for c in categories:
        if c.lower() == t:
            return c
    for c in categories:
        if t in c.lower() or c.lower() in t:
            return c
    close = difflib.get_close_matches(t, [c.lower() for c in categories], n=1, cutoff=0.6)
    return next((c for c in categories if c.lower() == close[0]), None) if close else None


def vendor_card(r: Ranked, rank: int, quantity: int) -> dict:
    v = r.vendor
    return {
        "rank": rank,
        "vendor_id": v.vendor_id,
        "name": v.name,
        "category": v.category,
        "unit_price_inr": v.unit_price_inr,
        "total_price_inr": v.unit_price_inr * quantity,
        "rating": v.rating,
        "delivery_days": v.delivery_days,
        "certifications": list(v.certifications),
        "is_preferred": v.is_preferred,
        "contract_on_file": v.contract_on_file,
        "region": v.region,
        "score": r.score,
        "reasons": r.reasons,
    }


def _filter_from(request: dict, **overrides) -> VendorFilter:
    qty = request["quantity"]
    base = dict(
        category=request["category"],
        quantity=qty,
        max_unit_price_inr=request["budget_inr"] / qty,
        max_delivery_days=request["max_delivery_days"],
        min_rating=request.get("min_rating", 3.5),
        required_certifications=tuple(request.get("required_certifications") or ()),
    )
    base.update(overrides)
    return VendorFilter(**base)


def list_categories() -> dict:
    """List the vendor categories available in the vendor database. Use this when
    the user's wording does not obviously match a category."""
    return {"status": "ok", "categories": get_repository().categories()}


def search_vendors(
    tool_context: ToolContext,
    category: Optional[str] = None,
    max_unit_price_inr: Optional[float] = None,
    max_delivery_days: Optional[int] = None,
    min_rating: Optional[float] = None,
    required_certifications: Optional[list[str]] = None,
    exclude_vendor_ids: Optional[list[str]] = None,
    require_contract_on_file: bool = False,
) -> dict:
    """Query the vendor database and return a ranked shortlist of up to 3 vendors.

    Any argument left out defaults to the saved purchase request (price ceiling =
    budget / quantity). Pass an argument only to deliberately tighten or relax a
    constraint. If fewer than 3 vendors qualify, `diagnostics` shows how many
    would qualify if each constraint were dropped, so you can decide what to
    relax and tell the user.

    Args:
        category: vendor category (fuzzy-matched; see list_categories).
        max_unit_price_inr: price ceiling per unit in rupees.
        max_delivery_days: latest delivery in days.
        min_rating: minimum vendor rating.
        required_certifications: certifications the vendor must hold.
        exclude_vendor_ids: vendor ids to leave out.
        require_contract_on_file: only vendors whose contract text is on file, so
            the contract-review step has something to read. Try True first and
            relax it if too few vendors qualify.
    """
    request = tool_context.state.get(REQUEST_KEY)
    if not request:
        return {"status": "error", "error": "No saved request. Call save_request first."}
    repo = get_repository()
    resolved = resolve_category(category or request["category"], repo.categories())
    if resolved is None:
        return {"status": "error", "error": f"Unknown category {category!r}.",
                "categories": repo.categories()}

    f = _filter_from(
        request,
        category=resolved,
        **{k: v for k, v in {
            "max_unit_price_inr": max_unit_price_inr,
            "max_delivery_days": max_delivery_days,
            "min_rating": min_rating,
            "required_certifications": tuple(required_certifications) if required_certifications is not None else None,
            "exclude_ids": tuple(exclude_vendor_ids or ()),
            "contract_on_file_only": require_contract_on_file,
        }.items() if v is not None},
    )
    candidates = repo.search(f)
    ranked = rank_vendors(candidates, f.max_delivery_days, SHORTLIST_SIZE)
    shortlist = [vendor_card(r, i + 1, request["quantity"]) for i, r in enumerate(ranked)]

    # Fresh search -> everything downstream is stale.
    tool_context.state[SHORTLIST_KEY] = shortlist
    for key in (FINDINGS_KEY, DECISIONS_KEY):
        tool_context.state[key] = {}
    tool_context.state[RECOMMENDED_KEY] = ""
    tool_context.state[DRAFT_KEY] = ""
    tool_context.state[REVIEW_STATUS_KEY] = ""

    result = {
        "status": "ok" if len(shortlist) == SHORTLIST_SIZE else ("insufficient_results" if shortlist else "no_results"),
        "filters": {
            "category": f.category, "max_unit_price_inr": f.max_unit_price_inr,
            "max_delivery_days": f.max_delivery_days, "min_rating": f.min_rating,
            "required_certifications": list(f.required_certifications),
            "contract_on_file_only": f.contract_on_file_only,
        },
        "candidates_considered": len(candidates),
        "shortlist": shortlist,
    }
    if len(shortlist) < SHORTLIST_SIZE:
        relaxed = {
            "without_price_ceiling": replace(f, max_unit_price_inr=None),
            "without_delivery_limit": replace(f, max_delivery_days=None),
            "without_min_rating": replace(f, min_rating=None),
            "without_certification_requirement": replace(f, required_certifications=()),
            "without_contract_requirement": replace(f, contract_on_file_only=False),
        }
        result["diagnostics"] = {name: len(repo.search(rf)) for name, rf in relaxed.items()}
    return result


def find_cheaper_alternative(position: int, tool_context: ToolContext) -> dict:
    """Swap the shortlist vendor at `position` (1-3) for the best-scoring vendor
    that is strictly cheaper per unit and still meets every request constraint.
    Updates the shortlist in session state and clears that vendor's contract
    findings and approval decision (they no longer apply).

    Args:
        position: 1-based shortlist position to replace.
    """
    state = tool_context.state
    request, shortlist = state.get(REQUEST_KEY), list(state.get(SHORTLIST_KEY) or [])
    if not request or not shortlist:
        return {"status": "error", "error": "No shortlist in this session yet."}
    if not 1 <= position <= len(shortlist):
        return {"status": "error", "error": f"position must be between 1 and {len(shortlist)}."}

    old = shortlist[position - 1]
    f = _filter_from(
        request,
        category=old["category"],
        max_unit_price_inr=min(old["unit_price_inr"] - 0.01, request["budget_inr"] / request["quantity"]),
        exclude_ids=tuple(v["vendor_id"] for v in shortlist),
        contract_on_file_only=all(v["contract_on_file"] for v in shortlist),
    )
    repo = get_repository()
    ranked = rank_vendors(repo.search(f), f.max_delivery_days, 1)
    relaxed_contract = False
    if not ranked and f.contract_on_file_only:
        ranked = rank_vendors(repo.search(replace(f, contract_on_file_only=False)), f.max_delivery_days, 1)
        relaxed_contract = bool(ranked)
    if not ranked:
        return {"status": "no_cheaper_vendor", "replaced_vendor_id": old["vendor_id"],
                "message": f"No cheaper vendor meets all constraints (below {format_inr(old['unit_price_inr'])} per unit)."}

    new = vendor_card(ranked[0], position, request["quantity"])
    shortlist[position - 1] = new
    state[SHORTLIST_KEY] = shortlist
    for key in (FINDINGS_KEY, DECISIONS_KEY):
        state[key] = {k: v for k, v in (state.get(key) or {}).items() if k != old["vendor_id"]}
    if state.get(RECOMMENDED_KEY) == old["vendor_id"]:
        state[RECOMMENDED_KEY] = ""
    state[DRAFT_KEY] = ""
    state[REVIEW_STATUS_KEY] = ""
    saving = (old["unit_price_inr"] - new["unit_price_inr"]) * request["quantity"]
    return {
        "status": "ok", "position": position, "replaced": old, "replacement": new,
        "total_saving_inr": saving, "total_saving": format_inr(saving),
        "needs_review": new["vendor_id"],
        **({"warning": "Cheaper vendor has no contract on file; a contract must be obtained before PO."}
           if relaxed_contract else {}),
    }
