"""Create and seed the BigQuery vendor table with ~200 synthetic vendors.

    python -m data.seed_bigquery                 # uses GOOGLE_CLOUD_PROJECT / BQ_* env vars
    python -m data.seed_bigquery --dry-run       # print schema + sample rows, touch nothing

The load is idempotent (WRITE_TRUNCATE), so re-running resets the table.
"""

from __future__ import annotations

import argparse
import json

from data.synthetic import vendor_rows
from settings import get_settings

VENDOR_SCHEMA: list[tuple[str, str, str]] = [
    ("vendor_id", "STRING", "REQUIRED"),
    ("name", "STRING", "REQUIRED"),
    ("category", "STRING", "REQUIRED"),
    ("unit_price_inr", "FLOAT64", "REQUIRED"),
    ("rating", "FLOAT64", "REQUIRED"),
    ("delivery_days", "INT64", "REQUIRED"),
    ("certifications", "STRING", "REPEATED"),
    ("min_order_qty", "INT64", "REQUIRED"),
    ("is_preferred", "BOOL", "REQUIRED"),
    ("contract_on_file", "BOOL", "REQUIRED"),
    ("region", "STRING", "NULLABLE"),
]


def bq_schema():
    from google.cloud import bigquery

    return [bigquery.SchemaField(n, t, mode=m) for n, t, m in VENDOR_SCHEMA]


def seed(dry_run: bool = False) -> int:
    s = get_settings()
    rows = vendor_rows()
    if dry_run:
        print(f"table: {s.bq_vendor_table_id}")
        print("schema:", ", ".join(f"{n}:{t}({m})" for n, t, m in VENDOR_SCHEMA))
        print(f"rows: {len(rows)}")
        print(json.dumps(rows[:2], indent=2))
        return len(rows)

    if not s.project:
        raise SystemExit("Set GOOGLE_CLOUD_PROJECT before seeding BigQuery.")

    from google.cloud import bigquery

    client = bigquery.Client(project=s.project, location=s.bq_location)
    dataset = bigquery.Dataset(f"{s.project}.{s.bq_dataset}")
    dataset.location = s.bq_location
    client.create_dataset(dataset, exists_ok=True)
    job = client.load_table_from_json(
        rows,
        s.bq_vendor_table_id,
        job_config=bigquery.LoadJobConfig(
            schema=bq_schema(),
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        ),
    )
    job.result()
    table = client.get_table(s.bq_vendor_table_id)
    print(f"seeded {table.num_rows} vendors into {s.bq_vendor_table_id}")

    # Create the per-request metrics table now so the dashboard has something to query.
    from observability.sink import ensure_metrics_table

    ensure_metrics_table(client)
    print(f"ensured metrics table {s.bq_metrics_table_id}")
    return table.num_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    seed(parser.parse_args().dry_run)


if __name__ == "__main__":
    main()
