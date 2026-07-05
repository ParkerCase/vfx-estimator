#!/usr/bin/env python3
"""Create vfx_corrections table in Xata Postgres and migrate local JSONL."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vfx_estimator.config import get_settings
from vfx_estimator.integrations.xata import (
    CREATE_CORRECTIONS_TABLE_SQL,
    CREATE_PRESETS_TABLE_SQL,
    XataCorrectionsStore,
    get_postgres_connection,
)
from vfx_estimator.types import UserCorrection


CREATE_DASHBOARD_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS vfx_users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    name TEXT DEFAULT '',
    picture TEXT DEFAULT '',
    role TEXT DEFAULT 'vendor',
    org_name TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS vfx_projects (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    client TEXT DEFAULT '',
    status TEXT DEFAULT 'active',
    stage TEXT DEFAULT 'estimating',
    due_date DATE,
    owner_id TEXT REFERENCES vfx_users(id),
    org_name TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS vfx_tasks (
    id SERIAL PRIMARY KEY,
    user_id TEXT REFERENCES vfx_users(id),
    project_id INTEGER REFERENCES vfx_projects(id),
    title TEXT NOT NULL,
    done BOOLEAN DEFAULT FALSE,
    due_date DATE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE vfx_bid_history
    ADD COLUMN IF NOT EXISTS project_id INTEGER REFERENCES vfx_projects(id),
    ADD COLUMN IF NOT EXISTS user_id TEXT;

CREATE INDEX IF NOT EXISTS idx_projects_owner ON vfx_projects(owner_id);
CREATE INDEX IF NOT EXISTS idx_tasks_user ON vfx_tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_bid_history_project_id ON vfx_bid_history(project_id);
"""


def _load_jsonl(path: Path) -> list[UserCorrection]:
    if not path.is_file():
        return []
    out: list[UserCorrection] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            data.pop("timestamp", None)
            out.append(UserCorrection.model_validate(data))
    return out


def main() -> None:
    get_settings.cache_clear()
    settings = get_settings()
    url = settings.resolved_xata_postgres_url()
    if not url:
        print("FAIL  Set XATA_POSTGRES_URL or XATA_DATABASE_URL (postgresql://...) in .env")
        sys.exit(1)

    print("Xata corrections migration")
    print("=" * 60)

    try:
        with get_postgres_connection(settings) as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_CORRECTIONS_TABLE_SQL)
                cur.execute(CREATE_PRESETS_TABLE_SQL)
                cur.execute(CREATE_DASHBOARD_TABLES_SQL)
            conn.commit()
        print("OK    Table vfx_corrections ready (CREATE TABLE IF NOT EXISTS)")
        print("OK    Table vfx_presets ready (CREATE TABLE IF NOT EXISTS + seed presets)")
        print("OK    Dashboard auth/project/task tables ready")
    except Exception as e:
        print(f"FAIL  Could not create table: {e}")
        sys.exit(1)

    store = XataCorrectionsStore(settings=settings)
    count_before = store.count()

    jsonl_path = settings.corrections_path()
    local_rows = _load_jsonl(jsonl_path)
    migrated = 0

    if count_before == 0 and local_rows:
        for c in local_rows:
            store.append(c)
            migrated += 1
        print(f"OK    Migrated {migrated} correction(s) from {jsonl_path.name}")
    elif local_rows and count_before > 0:
        print(
            f"SKIP  JSONL has {len(local_rows)} row(s) but Xata already has {count_before}; "
            "not re-importing (idempotent)."
        )
    elif local_rows:
        print(f"INFO  JSONL has {len(local_rows)} row(s); table was empty, migrated {migrated}")
    else:
        print(f"INFO  No local {jsonl_path.name} to migrate")

    count_after = store.count()
    print(f"\nSummary: {count_after} correction(s) in Xata vfx_corrections")
    print("Done. Safe to run this script again.")


if __name__ == "__main__":
    main()
