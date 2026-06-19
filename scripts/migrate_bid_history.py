#!/usr/bin/env python3
"""Create vfx_bid_history table in Xata Postgres."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vfx_estimator.config import get_settings
from vfx_estimator.integrations.xata import CREATE_BID_HISTORY_TABLE_SQL, get_postgres_connection


def main() -> None:
    get_settings.cache_clear()
    settings = get_settings()
    url = settings.resolved_xata_postgres_url()
    if not url:
        print("FAIL  Set XATA_POSTGRES_URL or XATA_DATABASE_URL (postgresql://...) in .env")
        sys.exit(1)

    print("Xata bid history migration")
    print("=" * 60)

    try:
        with get_postgres_connection(settings) as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_BID_HISTORY_TABLE_SQL)
            conn.commit()
        print("OK    Table vfx_bid_history ready (CREATE TABLE IF NOT EXISTS + indexes)")
        print("Done. Safe to run this script again.")
    except Exception as e:
        print(f"FAIL  Could not create table: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
