from collections import Counter

from data.synthetic import generate_vendors
from tools.state_tools import save_request
from tools.vendor_repository import BigQueryVendorRepository, LocalVendorRepository, VendorFilter, set_repository
from tools.vendor_tools import find_cheaper_alternative, list_categories, resolve_category, search_vendors


def test_synthetic_universe_shape():
    vendors = generate_vendors()
    assert len(vendors) == 200
    assert len({v.vendor_id for v in vendors}) == 200
    assert len(Counter(v.category for v in vendors)) == 10
    assert generate_vendors() == vendors  # deterministic


def test_list_categories():
    cats = list_categories()["categories"]
    assert "Laptops" in cats and len(cats) == 10


def test_resolve_category_fuzzy():
    cats = list_categories()["categories"]
    assert resolve_category("laptop", cats) == "Laptops"
    assert resolve_category("LAPTOPS", cats) == "Laptops"
    assert resolve_category("office chairs furniture", cats) == "Office Furniture"
    assert resolve_category("zzzz", cats) is None


def test_search_requires_saved_request(ctx):
    assert search_vendors(ctx)["status"] == "error"


def test_search_returns_ranked_shortlist_of_three(laptop_request):
    out = search_vendors(laptop_request, require_contract_on_file=True)
    assert out["status"] == "ok"
    sl = out["shortlist"]
    assert [v["rank"] for v in sl] == [1, 2, 3]
    assert [v["score"] for v in sl] == sorted((v["score"] for v in sl), reverse=True)
    for v in sl:
        assert v["unit_price_inr"] <= 80_000          # budget / quantity ceiling
        assert v["delivery_days"] <= 21 and v["rating"] >= 3.5
        assert v["contract_on_file"] and v["reasons"]
        assert v["total_price_inr"] == v["unit_price_inr"] * 50
    assert laptop_request.state["shortlist"] == sl


def test_search_clears_stale_downstream_state(laptop_request):
    laptop_request.state.update(contract_findings={"V0001": {}}, approval_decisions={"V0001": {}},
                                draft_recommendation="old", recommended_vendor_id="V0001")
    search_vendors(laptop_request)
    s = laptop_request.state
    assert s["contract_findings"] == {} and s["approval_decisions"] == {}
    assert s["draft_recommendation"] == "" and s["recommended_vendor_id"] == ""


def test_search_diagnostics_when_too_few(ctx):
    save_request("Laptops", 50, 4_000_000, 2, ctx)  # nobody delivers in 2 days
    out = search_vendors(ctx)
    assert out["status"] == "no_results"
    assert out["diagnostics"]["without_delivery_limit"] > 0


def test_search_unknown_category(laptop_request):
    out = search_vendors(laptop_request, category="submarines")
    assert out["status"] == "error" and "Laptops" in out["categories"]


def test_search_certification_filter(laptop_request):
    out = search_vendors(laptop_request, required_certifications=["ISO 27001"])
    assert out["shortlist"] and all("ISO 27001" in v["certifications"] for v in out["shortlist"])


def test_exclude_vendor_ids(laptop_request):
    first = search_vendors(laptop_request, require_contract_on_file=True)["shortlist"]
    second = search_vendors(laptop_request, require_contract_on_file=True,
                            exclude_vendor_ids=[first[0]["vendor_id"]])["shortlist"]
    assert first[0]["vendor_id"] not in {v["vendor_id"] for v in second}


def test_min_order_qty_respected(ctx):
    save_request("Laptops", 2, 400_000, 30, ctx)
    out = search_vendors(ctx)
    assert all(v["total_price_inr"] / 2 == v["unit_price_inr"] for v in out["shortlist"])
    repo = LocalVendorRepository()
    assert all(v.min_order_qty <= 2 for v in repo.search(VendorFilter("Laptops", quantity=2)))


def test_find_cheaper_alternative_swaps_and_invalidates(laptop_request):
    sl = search_vendors(laptop_request, require_contract_on_file=True)["shortlist"]
    laptop_request.state["contract_findings"] = {v["vendor_id"]: {"risk_score": 1} for v in sl}
    laptop_request.state["approval_decisions"] = {v["vendor_id"]: {"decision": "approve"} for v in sl}
    out = find_cheaper_alternative(2, laptop_request)
    assert out["status"] == "ok"
    old, new = out["replaced"], out["replacement"]
    assert new["unit_price_inr"] < old["unit_price_inr"] and out["total_saving_inr"] > 0
    state = laptop_request.state
    assert state["shortlist"][1]["vendor_id"] == new["vendor_id"]
    assert old["vendor_id"] not in state["contract_findings"] and old["vendor_id"] not in state["approval_decisions"]
    assert len(state["contract_findings"]) == 2          # the other two are untouched
    assert len({v["vendor_id"] for v in state["shortlist"]}) == 3


def test_find_cheaper_alternative_none_available(ctx):
    save_request("Laptops", 50, 4_000_000, 21, ctx)
    search_vendors(ctx)  # unrestricted by contract -> rank 1 is the cheapest-ish qualifying vendor
    cheapest_pos = min(range(3), key=lambda i: ctx.state["shortlist"][i]["unit_price_inr"]) + 1
    out = find_cheaper_alternative(cheapest_pos, ctx)
    # either nothing cheaper exists, or a strictly cheaper vendor was found; never a pricier one
    if out["status"] == "ok":
        assert out["replacement"]["unit_price_inr"] < out["replaced"]["unit_price_inr"]
    else:
        assert out["status"] == "no_cheaper_vendor"


def test_find_cheaper_alternative_bad_input(ctx):
    assert find_cheaper_alternative(1, ctx)["status"] == "error"
    save_request("Laptops", 50, 4_000_000, 21, ctx)
    search_vendors(ctx)
    assert find_cheaper_alternative(9, ctx)["status"] == "error"


# ---- BigQuery backend: verify query construction without a network ----------------

def test_bigquery_query_is_parameterised():
    f = VendorFilter("Laptops'; DROP TABLE x; --", quantity=5, max_unit_price_inr=80_000.0,
                     max_delivery_days=21, min_rating=3.5, required_certifications=("ISO 27001",),
                     exclude_ids=("V0001",), contract_on_file_only=True)
    sql, params = BigQueryVendorRepository.build_search_query("p.d.vendors", f)
    assert "DROP TABLE" not in sql                       # user input never reaches the SQL text
    by_name = {p.name: p for p in params}
    assert by_name["category"].value == "Laptops'; DROP TABLE x; --"
    assert by_name["required_certs"].values == ["iso 27001"]
    assert by_name["contract_only"].value is True
    for placeholder in ("@category", "@quantity", "@max_price", "@max_days", "@min_rating",
                        "@exclude_ids", "@required_certs", "@contract_only"):
        assert placeholder in sql


def test_bigquery_repo_maps_rows_with_fake_client():
    class Job:
        def result(self):
            return [{"vendor_id": "V9", "name": "N", "category": "Laptops", "unit_price_inr": 1000,
                     "rating": 4.0, "delivery_days": 3, "certifications": ["BIS"], "min_order_qty": 1,
                     "is_preferred": True, "contract_on_file": False, "region": None}]

    class Client:
        queries = []

        def query(self, sql, job_config=None):
            self.queries.append((sql, job_config))
            return Job()

    client = Client()
    repo = BigQueryVendorRepository(client=client, table_id="p.d.vendors")
    set_repository(repo)
    got = repo.search(VendorFilter("Laptops"))
    assert got[0].vendor_id == "V9" and got[0].certifications == ("BIS",) and got[0].region == ""
    assert "`p.d.vendors`" in client.queries[0][0]
