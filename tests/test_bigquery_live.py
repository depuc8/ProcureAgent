"""Live BigQuery parity: the BigQuery repository must return exactly what the local one does.

Needs ADC + a seeded table (python -m data.seed_bigquery). Skipped unless PROCURE_RUN_LIVE=1 and
GOOGLE_CLOUD_PROJECT is set. The unit tests use a fake client, so only this catches SQL semantics
(it found the empty-array-parameter bug)."""

import itertools
import os
import random

import pytest

from data.synthetic import generate_vendors
from tools.vendor_repository import BigQueryVendorRepository, LocalVendorRepository, VendorFilter

pytestmark = [pytest.mark.live, pytest.mark.skipif(
    os.getenv("PROCURE_RUN_LIVE") != "1" or not os.getenv("GOOGLE_CLOUD_PROJECT"),
    reason="set PROCURE_RUN_LIVE=1 and GOOGLE_CLOUD_PROJECT to query BigQuery")]

LOCAL = LocalVendorRepository()


@pytest.fixture(scope="module")
def bq():
    return BigQueryVendorRepository()


def test_no_certification_filter_returns_vendors(bq):
    assert len(bq.search(VendorFilter("Laptops"))) == len(LOCAL.search(VendorFilter("Laptops"))) > 0


def test_categories_and_get_match_local(bq):
    assert bq.categories() == LOCAL.categories()
    for v in random.Random(1).sample(generate_vendors(), 5):
        assert bq.get(v.vendor_id) == v
    assert bq.get("V9999") is None


def test_random_filter_combinations_match_local(bq):
    grid = list(itertools.product(LOCAL.categories(), [None, 80_000.0], [None, 14], [None, 4.2],
                                  [(), ("ISO 27001",)], [(), ("V0003",)], [False, True], [1, 50]))
    for cat, price, days, rating, certs, ex, contract, qty in random.Random(7).sample(grid, 25):
        f = VendorFilter(cat, qty, price, days, rating, certs, ex, contract)
        assert {v.vendor_id for v in bq.search(f)} == {v.vendor_id for v in LOCAL.search(f)}, f


def test_hostile_category_is_data_not_sql(bq):
    assert bq.search(VendorFilter("Laptops'; DROP TABLE x; --")) == []
    assert len(bq.search(VendorFilter("Laptops"))) > 0          # table still there
