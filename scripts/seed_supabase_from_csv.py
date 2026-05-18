"""
One-time / repeatable seed: upload local Phase-2 CSVs into Supabase tables.

Prerequisites:
  - Run supabase/migrations/0001_phase2_tables.sql in the Supabase SQL editor (or migrations pipeline).
  - pip install supabase pandas

Environment:
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY

Usage (from project root):
  python scripts/seed_supabase_from_csv.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# Project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_modules.phase2_repository import Phase2Paths  # noqa: E402


def _chunked(rows: list, size: int):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def _upsert_table(client, table: str, df: pd.DataFrame, chunk_size: int = 500) -> None:
    if df.empty:
        print(f"  skip {table}: empty dataframe")
        return
    # Replace NaN with None for JSON compatibility
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    total = len(records)
    for batch in _chunked(records, chunk_size):
        client.table(table).upsert(batch).execute()
    print(f"  upserted {table}: {total} rows")


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        raise SystemExit(
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (e.g. in a gitignored .env at project root).\n"
            "SUPABASE_URL must be the API URL: https://<project-ref>.supabase.co — not the dashboard URL.\n"
            "Use the service_role key from Supabase Settings → API (keep it server-side only)."
        )

    try:
        from supabase import create_client
    except ImportError as e:
        raise SystemExit("pip install supabase") from e

    client = create_client(url, key)
    paths = Phase2Paths()
    base = ROOT

    tables = [
        ("city_area_standards", paths.city_area_standards),
        ("materials_master", paths.materials_master),
        ("pricing_data", paths.pricing_data),
        ("construction_rates", paths.construction_rates),
        ("labor_rates", paths.labor_rates),
        ("phase_labor_mapping", paths.phase_labor_mapping),
    ]

    print("Seeding Supabase from CSV...")
    for table, rel in tables:
        csv_path = base / rel
        if not csv_path.exists():
            raise SystemExit(f"Missing CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        _upsert_table(client, table, df)
    print("Done.")


if __name__ == "__main__":
    main()
