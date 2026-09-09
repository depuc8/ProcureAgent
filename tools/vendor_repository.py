"""Vendor data access. One interface, two backends:

* BigQueryVendorRepository - production; parameterised SQL (never string-built).
* LocalVendorRepository    - deterministic in-memory synthetic vendors for
  tests, `adk web` without GCP, and offline benchmarks.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from data.synthetic import Vendor, generate_vendors
from settings import get_settings


@dataclass(frozen=True)
class VendorFilter:
    category: str
    quantity: int = 1
    max_unit_price_inr: float | None = None
    max_delivery_days: int | None = None
    min_rating: float | None = None
    required_certifications: tuple[str, ...] = ()
    exclude_ids: tuple[str, ...] = ()
    contract_on_file_only: bool = False


class VendorRepository(Protocol):
    def search(self, f: VendorFilter) -> list[Vendor]: ...
    def get(self, vendor_id: str) -> Vendor | None: ...
    def categories(self) -> list[str]: ...


def _matches(v: Vendor, f: VendorFilter) -> bool:
    if v.category.lower() != f.category.lower():
        return False
    if v.vendor_id in f.exclude_ids:
        return False
    if f.quantity < v.min_order_qty:
        return False
    if f.contract_on_file_only and not v.contract_on_file:
        return False
    if f.max_unit_price_inr is not None and v.unit_price_inr > f.max_unit_price_inr:
        return False
    if f.max_delivery_days is not None and v.delivery_days > f.max_delivery_days:
        return False
    if f.min_rating is not None and v.rating < f.min_rating:
        return False
    have = {c.lower() for c in v.certifications}
    return all(c.lower() in have for c in f.required_certifications)


class LocalVendorRepository:
    def __init__(self, vendors: list[Vendor] | None = None):
        self._vendors = vendors if vendors is not None else generate_vendors()
        self._by_id = {v.vendor_id: v for v in self._vendors}

    def search(self, f: VendorFilter) -> list[Vendor]:
        return [v for v in self._vendors if _matches(v, f)]

    def get(self, vendor_id: str) -> Vendor | None:
        return self._by_id.get(vendor_id)

    def categories(self) -> list[str]:
        return sorted({v.category for v in self._vendors})


_COLUMNS = (
    "vendor_id, name, category, unit_price_inr, rating, delivery_days, certifications, "
    "min_order_qty, is_preferred, contract_on_file, region"
)


def _row_to_vendor(row) -> Vendor:
    return Vendor(
        vendor_id=row["vendor_id"], name=row["name"], category=row["category"],
        unit_price_inr=float(row["unit_price_inr"]), rating=float(row["rating"]),
        delivery_days=int(row["delivery_days"]), certifications=tuple(row["certifications"] or ()),
        min_order_qty=int(row["min_order_qty"]), is_preferred=bool(row["is_preferred"]),
        contract_on_file=bool(row["contract_on_file"]), region=row["region"] or "",
    )


class BigQueryVendorRepository:
    def __init__(self, client=None, table_id: str | None = None):
        s = get_settings()
        self._table = table_id or s.bq_vendor_table_id
        self._client = client
        self._lock = threading.Lock()

    @property
    def client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from google.cloud import bigquery

                    s = get_settings()
                    self._client = bigquery.Client(project=s.project, location=s.bq_location)
        return self._client

    def _run(self, sql: str, params: list):
        from google.cloud import bigquery

        job = self.client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
        return list(job.result())

    @staticmethod
    def build_search_query(table: str, f: VendorFilter):
        """Returns (sql, params). Public so tests can assert nothing is interpolated."""
        from google.cloud import bigquery as bq

        sql = f"""
            SELECT {_COLUMNS}
            FROM `{table}`
            WHERE LOWER(category) = LOWER(@category)
              AND min_order_qty <= @quantity
              AND (@max_price IS NULL OR unit_price_inr <= @max_price)
              AND (@max_days IS NULL OR delivery_days <= @max_days)
              AND (@min_rating IS NULL OR rating >= @min_rating)
              AND (NOT @contract_only OR contract_on_file)
              AND vendor_id NOT IN UNNEST(@exclude_ids)
              AND (SELECT COUNT(DISTINCT LOWER(c)) FROM UNNEST(certifications) AS c
                   WHERE LOWER(c) IN UNNEST(@required_certs)) = ARRAY_LENGTH(@required_certs)
            ORDER BY unit_price_inr
            LIMIT 200
        """
        params = [
            bq.ScalarQueryParameter("category", "STRING", f.category),
            bq.ScalarQueryParameter("quantity", "INT64", f.quantity),
            bq.ScalarQueryParameter("max_price", "FLOAT64", f.max_unit_price_inr),
            bq.ScalarQueryParameter("max_days", "INT64", f.max_delivery_days),
            bq.ScalarQueryParameter("min_rating", "FLOAT64", f.min_rating),
            bq.ScalarQueryParameter("contract_only", "BOOL", f.contract_on_file_only),
            bq.ArrayQueryParameter("exclude_ids", "STRING", list(f.exclude_ids)),
            bq.ArrayQueryParameter("required_certs", "STRING", [c.lower() for c in f.required_certifications]),
        ]
        return sql, params

    def search(self, f: VendorFilter) -> list[Vendor]:
        sql, params = self.build_search_query(self._table, f)
        return [_row_to_vendor(r) for r in self._run(sql, params)]

    def get(self, vendor_id: str) -> Vendor | None:
        from google.cloud import bigquery as bq

        rows = self._run(
            f"SELECT {_COLUMNS} FROM `{self._table}` WHERE vendor_id = @id LIMIT 1",
            [bq.ScalarQueryParameter("id", "STRING", vendor_id)],
        )
        return _row_to_vendor(rows[0]) if rows else None

    def categories(self) -> list[str]:
        rows = self._run(f"SELECT DISTINCT category FROM `{self._table}` ORDER BY category", [])
        return [r["category"] for r in rows]


_override: VendorRepository | None = None


@lru_cache(maxsize=1)
def _local() -> LocalVendorRepository:
    return LocalVendorRepository()


@lru_cache(maxsize=1)
def _bigquery() -> BigQueryVendorRepository:
    return BigQueryVendorRepository()


def get_repository() -> VendorRepository:
    if _override is not None:
        return _override
    return _bigquery() if get_settings().vendor_backend.lower() == "bigquery" else _local()


def set_repository(repo: VendorRepository | None) -> None:
    """Test hook: inject a repository (None restores env-based selection)."""
    global _override
    _override = repo
