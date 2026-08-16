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

CHE_SHOT_PRESETS_PATH = ROOT / "vfx_estimator" / "data" / "che_shot_presets.json"


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
    ADD COLUMN IF NOT EXISTS user_id TEXT,
    ADD COLUMN IF NOT EXISTS currency TEXT DEFAULT 'USD',
    ADD COLUMN IF NOT EXISTS display_currency TEXT DEFAULT 'USD',
    ADD COLUMN IF NOT EXISTS fx_rate FLOAT DEFAULT 1,
    ADD COLUMN IF NOT EXISTS day_rate_usd FLOAT DEFAULT 700,
    ADD COLUMN IF NOT EXISTS variant_label TEXT DEFAULT '',
    ADD COLUMN IF NOT EXISTS cover_snapshot JSONB DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS assets JSONB DEFAULT '[]';

UPDATE vfx_bid_history
SET display_currency = COALESCE(NULLIF(currency, ''), 'USD')
WHERE display_currency IS NULL
   OR display_currency = 'USD';

CREATE INDEX IF NOT EXISTS idx_projects_owner ON vfx_projects(owner_id);
CREATE INDEX IF NOT EXISTS idx_tasks_user ON vfx_tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_bid_history_project_id ON vfx_bid_history(project_id);
"""

CREATE_ASSET_PRESETS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS vfx_asset_presets (
    id SERIAL PRIMARY KEY,
    asset_type TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    modelling FLOAT DEFAULT 0,
    texturing FLOAT DEFAULT 0,
    rigging FLOAT DEFAULT 0,
    cfx FLOAT DEFAULT 0,
    fx FLOAT DEFAULT 0,
    lookdev FLOAT DEFAULT 0,
    dmp FLOAT DEFAULT 0,
    comp_dev FLOAT DEFAULT 0,
    total FLOAT DEFAULT 0,
    created_by TEXT DEFAULT 'system',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO vfx_asset_presets
  (asset_type, description, modelling, texturing, rigging,
   cfx, fx, lookdev, dmp, comp_dev, total)
VALUES
  ('hero_creature','Hero creature, full pipeline',12,6,12,5,3,4,0,2,44),
  ('background_creature','Background/distant creature',6,3,6,1,0,2,0,1,19),
  ('cg_vehicle_hero','Hero CG vehicle (car, ship, aircraft)',10,5,3,0,1,4,0,2,25),
  ('cg_vehicle_background','Background CG vehicle',4,2,1,0,0,1,0,1,9),
  ('cg_environment_hero','Hero CG environment (castle, city)',14,6,0,0,0,5,2,2,29),
  ('cg_environment_background','Background CG environment',7,3,0,0,0,3,1,1,15),
  ('digital_double','Digital double of actor',10,6,14,6,0,5,0,3,44),
  ('hero_prop','Hero CG prop (featured, close camera)',4,2,0,0,0,2,0,1,9),
  ('background_prop','Background CG prop',2,1,0,0,0,1,0,1,4)
ON CONFLICT (asset_type) DO NOTHING;
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


def _load_reference_presets(path: Path = CHE_SHOT_PRESETS_PATH) -> list[dict]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        row
        for row in data
        if isinstance(row, dict) and float(row.get("total") or 0) > 0
    ]


def _insert_reference_presets(cur, presets: list[dict]) -> int:
    if not presets:
        return 0
    from psycopg2.extras import execute_values

    columns = (
        "camera_track",
        "matchmove",
        "layout",
        "animation",
        "cfx",
        "fx",
        "lighting",
        "dmp",
        "comp_paint",
        "comp_roto",
        "compositing",
    )
    values = [
        (
            row["shot_type"],
            row.get("description") or row["shot_type"],
            *(float(row.get(column) or 0) for column in columns),
            float(row.get("total") or 0),
            "reference",
        )
        for row in presets
    ]
    execute_values(
        cur,
        f"""
        INSERT INTO vfx_presets
          (shot_type, description, {", ".join(columns)}, total, created_by)
        VALUES %s
        ON CONFLICT (shot_type) DO NOTHING
        """,
        values,
        page_size=1000,
    )
    return max(0, cur.rowcount)


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
        reference_presets = _load_reference_presets()
        inserted_presets = 0
        preset_count = 0
        with get_postgres_connection(settings) as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_CORRECTIONS_TABLE_SQL)
                cur.execute(CREATE_PRESETS_TABLE_SQL)
                inserted_presets = _insert_reference_presets(cur, reference_presets)
                cur.execute(CREATE_ASSET_PRESETS_TABLE_SQL)
                cur.execute(CREATE_DASHBOARD_TABLES_SQL)
                cur.execute("SELECT COUNT(*) FROM vfx_presets")
                preset_count = int(cur.fetchone()[0])
            conn.commit()
        print("OK    Table vfx_corrections ready (CREATE TABLE IF NOT EXISTS)")
        print("OK    Table vfx_presets ready (CREATE TABLE IF NOT EXISTS + seed presets)")
        print(
            f"OK    Shot reference presets: {inserted_presets} inserted, "
            f"{preset_count} total ({len(reference_presets)} valid source rows)"
        )
        print("OK    Table vfx_asset_presets ready (CREATE TABLE IF NOT EXISTS + seed presets)")
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
