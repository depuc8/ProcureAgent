from tools.state_tools import (
    get_final_draft, get_session_summary, save_draft, save_request, set_recommended_vendor,
)
from tools.vendor_tools import search_vendors


def test_save_request_validates(ctx):
    assert save_request("Laptops", 0, 100, 5, ctx)["status"] == "error"
    assert save_request("Laptops", 5, -1, 5, ctx)["status"] == "error"
    assert "request" not in ctx.state


def test_save_request_normalises(ctx):
    out = save_request(" Laptops ", 50, 4_000_000, 21, ctx, required_certifications=[" ISO 27001 "])
    assert out["request"]["category"] == "Laptops"
    assert out["request"]["required_certifications"] == ["ISO 27001"]
    assert ctx.state["request"] == out["request"]


def test_session_summary_round_trip(laptop_request):
    search_vendors(laptop_request, require_contract_on_file=True)
    s = get_session_summary(laptop_request)
    assert s["request"]["quantity"] == 50 and len(s["shortlist"]) == 3


def test_set_recommended_vendor_must_be_on_shortlist(laptop_request):
    sl = search_vendors(laptop_request, require_contract_on_file=True)["shortlist"]
    assert set_recommended_vendor("V9999", laptop_request)["status"] == "error"
    assert set_recommended_vendor(sl[0]["vendor_id"], laptop_request)["status"] == "ok"


def test_draft_lifecycle(ctx):
    assert save_draft("   ", ctx)["status"] == "error"
    save_draft("# Recommendation", ctx)
    assert get_final_draft(ctx) == {"draft_recommendation": "# Recommendation", "review_status": "pending"}
