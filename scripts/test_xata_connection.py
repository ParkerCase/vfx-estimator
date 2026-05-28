#!/usr/bin/env python3
"""Verify Xata API key + database URL (REST HTTPS or Postgres)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx

from vfx_estimator.config import get_settings
from vfx_estimator.integrations.xata import XataShotSearch, resolve_xata_rest_base


def _print_dept_columns(url: str, table: str) -> None:
    """List comp/camera/dept-related columns on the historical shots table."""
    try:
        import psycopg2
    except ImportError:
        print("WARN  psycopg2 not installed")
        return

    print(f"\nDept columns on '{table}' (comp / cam / mandays / _days):")
    try:
        conn = psycopg2.connect(url, connect_timeout=15)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = %s
              AND (
                column_name ILIKE %s OR column_name ILIKE %s
                OR column_name ILIKE %s OR column_name ILIKE %s
              )
            ORDER BY column_name
            """,
            (table, "%comp%", "%cam%", "%mandays%", "%_days"),
        )
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        print(f"FAIL  {e}")
        return

    if not rows:
        print("  (none found)")
        return
    for (name,) in rows:
        print(f"  {name}")


def _check_postgres(url: str, table: str) -> bool:
    print("\nPostgres connection string detected")
    try:
        import psycopg2
    except ImportError:
        print("WARN  psycopg2 not installed — pip install psycopg2-binary for Postgres checks")
        return False

    try:
        conn = psycopg2.connect(url, connect_timeout=15)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        print("OK    Postgres TCP connection")
    except Exception as e:
        print(f"FAIL  Postgres connection: {e}")
        return False

    cur.execute(
        """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
          AND table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY 1, 2
        """
    )
    tables = cur.fetchall()
    if not tables:
        print("WARN  Database is empty (no user tables).")
        print("      Live Xata search will return nothing until you import shots.")
        print("      Training still works from local retraining_bundle.json.")
        conn.close()
        return False

    print(f"OK    Found {len(tables)} table(s):")
    for schema, name in tables[:15]:
        print(f"      {schema}.{name}")
    if len(tables) > 15:
        print(f"      … and {len(tables) - 15} more")

    found = any(name == table for _, name in tables)
    if found:
        cur.execute(f'SELECT COUNT(*) FROM "{table}"')
        n = cur.fetchone()[0]
        cur.execute(
            f'SELECT COUNT(*) FROM "{table}" WHERE COALESCE(mandays,0) > 0 OR COALESCE(cost,0) > 0'
        )
        labeled = cur.fetchone()[0]
        print(f"OK    Table '{table}' exists ({n} rows, {labeled} with cost/mandays)")
    else:
        print(f"WARN  Table '{table}' not found — set XATA_TABLE or create/import the table")

    conn.close()
    return found


def _check_rest(key: str, rest_base: str, table: str) -> bool:
    print(f"\nREST base: {rest_base}")
    url = f"{rest_base}/tables/{table}/query"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json={"page": {"size": 1}})
    except Exception as e:
        print(f"FAIL  Network error: {e}")
        return False

    if resp.status_code == 401:
        print("FAIL  HTTP 401 — invalid XATA_API_KEY for this workspace.")
        return False
    if resp.status_code == 404:
        print("FAIL  HTTP 404 — wrong database URL, branch, or table name.")
        print("      Copy the HTTPS database URL from Xata → your DB → Connect (not Postgres URL).")
        print("      Example: https://YOUR-WORKSPACE.us-east-1.xata.sh/db/YOUR-DB:main")
        return False
    if resp.status_code >= 400:
        print(f"FAIL  HTTP {resp.status_code}: {resp.text[:400]}")
        return False

    data = resp.json()
    n = len(data.get("records") or [])
    print(f"OK    REST query succeeded (sample records: {n})")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify Xata connection")
    ap.add_argument(
        "--schema-depts",
        action="store_true",
        help="Print comp/camera/dept column names and exit",
    )
    args = ap.parse_args()

    get_settings.cache_clear()
    s = get_settings()
    key = (s.xata_api_key or "").strip()
    raw_url = (s.resolved_xata_postgres_url() or s.xata_database_url or "").strip()
    table = s.xata_table

    print("Xata connection check")
    print("=" * 60)

    if key:
        print(f"OK    API key present ({len(key)} chars)")
    else:
        print("INFO  XATA_API_KEY empty — REST mode off; Postgres can still work.")

    if not raw_url:
        print("FAIL  Set XATA_POSTGRES_URL or XATA_DATABASE_URL (postgresql://...).")
        sys.exit(1)

    if args.schema_depts:
        if raw_url.startswith("postgresql://") or raw_url.startswith("postgres://"):
            _print_dept_columns(raw_url, table)
            sys.exit(0)
        print("FAIL  --schema-depts requires a postgresql:// URL")
        sys.exit(1)

    ok = False
    if raw_url.startswith("postgresql://") or raw_url.startswith("postgres://"):
        ok = _check_postgres(raw_url, table)
        if key:
            rest = resolve_xata_rest_base(raw_url, s.xata_branch)
            if rest:
                _check_rest(key, rest, table)
    elif raw_url.startswith("http"):
        ok = _check_rest(key, resolve_xata_rest_base(raw_url, s.xata_branch), table)
    else:
        print(f"FAIL  Unrecognized XATA_DATABASE_URL scheme: {urlparse(raw_url).scheme}")
        sys.exit(1)

    x = XataShotSearch(s)
    print(f"\nSearch mode: {x.mode}")
    if x.enabled:
        hits = x.search("wire removal comp greenscreen", top_k=3)
        print(f"Shot search: {len(hits)} row(s)")
        if hits:
            print(f"  Example: {hits[0]['description'][:60]}… → {hits[0]['mandays']} mandays")
        elif ok:
            print("  WARN  table has rows but keyword search returned 0 — try a richer description")

    if ok:
        print("\nRESULT: Xata connected (data present)")
        sys.exit(0)

    print("\nRESULT: Credentials work, but live shot search is not ready yet.")
    print("  → Import historical shots into Xata, OR")
    print("  → Use the HTTPS REST database URL from your OLD workspace (if shots live there), OR")
    print("  → Keep using local data/retraining_bundle.json (already working).")
    sys.exit(2)


if __name__ == "__main__":
    main()
