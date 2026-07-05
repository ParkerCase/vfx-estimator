"""Optional Xata: Postgres shot search + corrections persistence."""

from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional
from urllib.parse import urlparse

import httpx

from vfx_estimator.config import Settings, get_settings
from vfx_estimator.types import UserCorrection

logger = logging.getLogger(__name__)

CORRECTIONS_TABLE = "vfx_corrections"
CORRECTIONS_LOAD_LIMIT = 500

# internal dept key -> record column(s), first non-zero wins.
# Live Xata: *_days columns hold data; comp_mandays/comp_days are always empty.
_DEPT_COL_MAP: Dict[str, tuple[str, ...]] = {
    "camera_track": ("cam_track_days", "obj_track_days", "cam_track_mandays"),
    "compositing": ("comp_days", "comp_mandays"),
    "comp_paint": ("comp_paint_days", "paint_days", "paint_mandays"),
    "comp_roto": ("comp_roto_days", "roto_days", "roto_mandays"),
    "layout": ("layout_days", "layout_mandays"),
    "lighting": ("lighting_days", "lgt_mandays"),
    "animation": ("animation_days", "anim_mandays"),
    "fx": ("fx_days", "fx_mandays"),
    "matchmove": ("matchmove_days", "matchmove_mandays"),
    "dmp": ("dmp_days", "dmp_mandays"),
    "cfx": ("cfx_days", "cfx_mandays"),
    "prep": ("prep_days", "prep_mandays"),
    "ai": ("ai_days", "ai_mandays"),
}


def _record_total_mandays(rec: Dict[str, Any], day_rate: float) -> float:
    for key in ("total_mandays", "mandays", "per_shot_mandays"):
        try:
            v = float(rec.get(key) or 0)
            if v > 0:
                return v
        except (TypeError, ValueError):
            continue
    cost = 0.0
    for key in ("cost", "cost_per_shot", "total_cost"):
        try:
            cost = float(rec.get(key) or 0)
            if cost > 0:
                break
        except (TypeError, ValueError):
            continue
    if cost > 0 and day_rate > 0:
        return cost / day_rate
    return 0.0


def _derive_compositing_residual(rec: Dict[str, Any], dept: Dict[str, float], *, day_rate: float) -> float:
    """Infer COMP integration days when comp_days/comp_mandays are unset in source data."""
    total = _record_total_mandays(rec, day_rate)
    if total <= 0:
        return 0.0
    residual = total - sum(float(v) for v in dept.values())
    return max(0.0, residual)


def extract_dept_days_from_record(rec: Dict[str, Any], *, day_rate: float) -> Dict[str, float]:
    dept: Dict[str, float] = {}
    for dept_key, cols in _DEPT_COL_MAP.items():
        for col in cols:
            try:
                v = float(rec.get(col) or 0)
            except (TypeError, ValueError):
                continue
            if v > 0:
                dept[dept_key] = v
                break
    if dept.get("compositing", 0) <= 0:
        residual = _derive_compositing_residual(rec, dept, day_rate=day_rate)
        if residual > 0:
            dept["compositing"] = residual
    return dept

_DEPT_SELECT_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(
        col
        for cols in _DEPT_COL_MAP.values()
        for col in cols
    )
)

CREATE_CORRECTIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS vfx_corrections (
    id SERIAL PRIMARY KEY,
    description TEXT NOT NULL,
    final_total_days FLOAT NOT NULL,
    final_departments JSONB DEFAULT '{}',
    user_id TEXT DEFAULT 'default',
    notes TEXT DEFAULT '',
    ai_total_days FLOAT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);
"""

BID_HISTORY_TABLE = "vfx_bid_history"

CREATE_BID_HISTORY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS vfx_bid_history (
    id SERIAL PRIMARY KEY,
    project_name TEXT NOT NULL,
    user_id TEXT DEFAULT 'supervisor',
    shot_count INTEGER DEFAULT 0,
    total_mandays FLOAT DEFAULT 0,
    shots JSONB DEFAULT '[]',
    pre_qual JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    notes TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_bid_history_project ON vfx_bid_history(project_name);
CREATE INDEX IF NOT EXISTS idx_bid_history_user ON vfx_bid_history(user_id);
CREATE INDEX IF NOT EXISTS idx_bid_history_created ON vfx_bid_history(created_at DESC);
"""

PRESET_DEPT_COLUMNS = (
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

CREATE_PRESETS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS vfx_presets (
    id SERIAL PRIMARY KEY,
    shot_type TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    camera_track FLOAT DEFAULT 0,
    matchmove FLOAT DEFAULT 0,
    layout FLOAT DEFAULT 0,
    animation FLOAT DEFAULT 0,
    cfx FLOAT DEFAULT 0,
    fx FLOAT DEFAULT 0,
    lighting FLOAT DEFAULT 0,
    dmp FLOAT DEFAULT 0,
    comp_paint FLOAT DEFAULT 0,
    comp_roto FLOAT DEFAULT 0,
    compositing FLOAT DEFAULT 0,
    total FLOAT DEFAULT 0,
    created_by TEXT DEFAULT 'system',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO vfx_presets (shot_type, description, matchmove, fx, compositing, total)
VALUES
  ('muzzle_flash', 'Gunfire muzzle flash', 0.25, 0.4, 0.75, 1.7),
  ('wire_removal', 'Safety wire removal', 0, 0, 0.6, 1.9),
  ('sky_replacement', 'Sky plate replacement', 0.3, 0, 1.0, 2.3)
ON CONFLICT (shot_type) DO NOTHING;
"""


def resolve_xata_rest_base(database_url: str, branch: str = "main") -> str:
    """Turn database URL into REST base .../db/{name}:{branch} (Postgres or HTTPS)."""
    url = (database_url or "").strip().rstrip("/")
    if not url:
        return ""

    if url.startswith("postgresql://") or url.startswith("postgres://"):
        parsed = urlparse(url)
        host = parsed.hostname or ""
        dbname = (parsed.path or "").strip("/").split("/")[0] or "xata"
        return f"https://{host}/db/{dbname}:{branch}"

    if url.startswith("http://") or url.startswith("https://"):
        if re.search(r"/db/[^/]+:[^/]+$", url):
            return url
        if re.search(r"/db/[^/]+$", url):
            return f"{url}:{branch}"
        return url

    return url


def postgres_url_from_settings(settings: Optional[Settings] = None) -> str:
    settings = settings or get_settings()
    return settings.resolved_xata_postgres_url()


@contextmanager
def get_postgres_connection(
    settings: Optional[Settings] = None,
    *,
    connect_timeout: int = 20,
) -> Generator[Any, None, None]:
    """Yield a psycopg2 connection, or raise if URL/driver unavailable."""
    url = postgres_url_from_settings(settings)
    if not url:
        raise RuntimeError("Xata Postgres URL not configured")
    try:
        import psycopg2
    except ImportError as e:
        raise RuntimeError("psycopg2 not installed") from e

    conn = psycopg2.connect(url, connect_timeout=connect_timeout)
    try:
        yield conn
    finally:
        conn.close()


def _parse_departments_jsonb(raw: Any) -> Dict[str, float]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    else:
        return {}
    out: Dict[str, float] = {}
    for k, v in data.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def _record_from_row(r: Dict[str, Any], day_rate: float) -> Optional[Dict[str, Any]]:
    desc = str(r.get("shot_description") or r.get("description") or "").strip()
    if not desc:
        return None
    cost = 0.0
    for key in ("cost", "cost_per_shot", "total_cost"):
        try:
            cost = float(r.get(key) or 0)
            if cost > 0:
                break
        except (TypeError, ValueError):
            pass
    md = _record_total_mandays(r, day_rate)
    if md <= 0:
        return None
    dept = extract_dept_days_from_record(r, day_rate=day_rate)
    return {
        "description": desc,
        "mandays": md,
        "cost": cost or md * day_rate,
        "source": "xata",
        "weight": 1.25,
        "project": str(r.get("project") or ""),
        "dept_days": dept,
    }


class XataCorrectionsStore:
    """Persist supervisor corrections in Xata Postgres."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.table = (getattr(self.settings, "xata_corrections_table", None) or CORRECTIONS_TABLE).strip()
        self.postgres_url = self.settings.resolved_xata_postgres_url()

    @property
    def enabled(self) -> bool:
        return bool(self.postgres_url)

    def count(self) -> int:
        if not self.enabled:
            return 0
        try:
            from psycopg2 import sql

            with get_postgres_connection(self.settings) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(self.table))
                    )
                    row = cur.fetchone()
                    return int(row[0]) if row else 0
        except Exception as e:
            logger.warning("Xata corrections count failed: %s", e)
            return 0

    def load(self) -> List[UserCorrection]:
        if not self.enabled:
            return []
        try:
            from psycopg2 import sql
            from psycopg2.extras import RealDictCursor

            query = sql.SQL(
                """
                SELECT description, final_total_days, final_departments,
                       user_id, notes, ai_total_days, timestamp
                FROM {table}
                ORDER BY timestamp DESC
                LIMIT {limit}
                """
            ).format(
                table=sql.Identifier(self.table),
                limit=sql.Literal(CORRECTIONS_LOAD_LIMIT),
            )
            with get_postgres_connection(self.settings) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query)
                    rows = cur.fetchall()
        except Exception as e:
            logger.warning("Xata corrections load failed: %s", e)
            return []

        out: List[UserCorrection] = []
        for row in rows:
            d = dict(row)
            try:
                out.append(
                    UserCorrection(
                        description=str(d["description"]),
                        final_total_days=float(d["final_total_days"]),
                        final_departments=_parse_departments_jsonb(d.get("final_departments")),
                        user_id=str(d.get("user_id") or "default"),
                        notes=str(d.get("notes") or ""),
                        ai_total_days=float(d["ai_total_days"])
                        if d.get("ai_total_days") is not None
                        else None,
                    )
                )
            except (KeyError, TypeError, ValueError) as e:
                logger.warning("Skipping invalid correction row: %s", e)
        return out

    def append(self, correction: UserCorrection) -> None:
        if not self.enabled:
            return
        try:
            from psycopg2 import sql
            from psycopg2.extras import Json

            dept = correction.final_departments or {}
            query = sql.SQL(
                """
                INSERT INTO {table}
                    (description, final_total_days, final_departments,
                     user_id, notes, ai_total_days, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
            ).format(table=sql.Identifier(self.table))
            ts = datetime.now(timezone.utc)
            with get_postgres_connection(self.settings) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        query,
                        (
                            correction.description.strip(),
                            float(correction.final_total_days),
                            Json(dept),
                            correction.user_id or "default",
                            correction.notes or "",
                            correction.ai_total_days,
                            ts,
                        ),
                    )
                conn.commit()
        except Exception as e:
            logger.warning("Xata corrections append failed: %s", e)

    def as_training_rows(self) -> List[dict]:
        rows = []
        dr = self.settings.day_rate
        for c in self.load():
            rows.append(
                {
                    "description": c.description,
                    "mandays": c.final_total_days,
                    "cost": c.final_total_days * dr,
                    "source": "correction",
                    "dept_days": c.final_departments,
                    "user_id": c.user_id,
                }
            )
        return rows


class XataShotSearch:
    """
    Query Xata for similar shots via Postgres (preferred when URL is postgresql://)
    or REST table query (when HTTPS URL + API key are set).
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.api_key = (self.settings.xata_api_key or "").strip()
        self.branch = (self.settings.xata_branch or "main").strip()
        self.table = (self.settings.xata_table or "vfx_historical_shots").strip()
        self.postgres_url = self.settings.resolved_xata_postgres_url()
        self.rest_base = self.settings.resolved_xata_rest_base()

    @property
    def enabled(self) -> bool:
        if self.postgres_url:
            return True
        return bool(self.api_key and self.rest_base.startswith("https://"))

    @property
    def mode(self) -> str:
        if self.postgres_url:
            return "postgres"
        if self.api_key and self.rest_base.startswith("https://"):
            return "rest"
        return "off"

    def count(self) -> int:
        """Return the live row count for the historical shot table."""
        if not self.enabled:
            return 0
        if self.postgres_url:
            try:
                from psycopg2 import sql

                with get_postgres_connection(self.settings) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(self.table))
                        )
                        row = cur.fetchone()
                        return int(row[0]) if row else 0
            except Exception as e:
                logger.warning("Xata shot count failed: %s", e)
                return 0
        if self.api_key and self.rest_base.startswith("https://"):
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            url = f"{self.rest_base}/tables/{self.table}/query"
            try:
                with httpx.Client(timeout=15.0) as client:
                    resp = client.post(url, headers=headers, json={"page": {"size": 1}})
                    if resp.status_code >= 400:
                        return 0
                    data = resp.json()
            except Exception as e:
                logger.warning("Xata REST shot count failed: %s", e)
                return 0

            page = ((data.get("meta") or {}).get("page") or {})
            try:
                return int(page.get("total") or page.get("totalCount") or 0)
            except (TypeError, ValueError):
                return 0
        return 0

    def _search_postgres(self, description: str, *, top_k: int) -> List[Dict[str, Any]]:
        try:
            from psycopg2 import sql
            from psycopg2.extras import RealDictCursor
        except ImportError:
            return []

        words = [w for w in description.split() if len(w) > 3][:4]
        if not words:
            words = [w for w in description.split() if len(w) > 2][:2]
        if not words:
            return []

        table = sql.Identifier(self.table)
        clauses = []
        params: List[str] = []
        for w in words:
            like = f"%{w}%"
            clauses.append(
                sql.SQL("(shot_description ILIKE {} OR COALESCE(shot_summary, '') ILIKE {})").format(
                    sql.Placeholder(), sql.Placeholder()
                )
            )
            params.extend([like, like])
        where = sql.SQL(" OR ").join(clauses)
        dr = self.settings.day_rate
        dept_cols_sql = sql.SQL(", ").join(
            sql.Identifier(c) for c in _DEPT_SELECT_COLUMNS
        )
        query = sql.SQL(
            """
            SELECT shot_description, shot_summary, cost, mandays, project,
                   {dept_cols},
                   COALESCE(mandays, CASE WHEN cost > 0 THEN cost / {day_rate} ELSE 0 END) AS effective_mandays
            FROM {table}
            WHERE ({where})
              AND (COALESCE(mandays, 0) > 0 OR COALESCE(cost, 0) > 0)
            ORDER BY effective_mandays DESC
            LIMIT {limit}
            """
        ).format(
            dept_cols=dept_cols_sql,
            table=table,
            where=where,
            limit=sql.Literal(top_k),
            day_rate=sql.Literal(dr),
        )

        try:
            with get_postgres_connection(self.settings) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    raw = cur.fetchall()
        except Exception:
            return []

        out: List[Dict[str, Any]] = []
        dr = self.settings.day_rate
        for row in raw:
            d = dict(row)
            if d.get("effective_mandays"):
                d["mandays"] = float(d["effective_mandays"])
            rec = _record_from_row(d, dr)
            if rec:
                out.append(rec)
        return out

    def _search_rest(self, description: str, *, top_k: int) -> List[Dict[str, Any]]:
        words = [w for w in description.split() if len(w) > 3][:6]
        payload: Dict[str, Any] = {"page": {"size": top_k}}
        if words:
            payload["filter"] = {
                "$any": [{"shot_description": {"$contains": w}} for w in words]
                + [{"description": {"$contains": w}} for w in words]
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.rest_base}/tables/{self.table}/query"
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code >= 400:
                    return []
                data = resp.json()
        except Exception:
            return []

        out: List[Dict[str, Any]] = []
        dr = self.settings.day_rate
        for rec in data.get("records") or []:
            row = _record_from_row(rec if isinstance(rec, dict) else {}, dr)
            if row:
                out.append(row)
        return out

    def search(self, description: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        if self.postgres_url:
            rows = self._search_postgres(description, top_k=top_k)
            if rows:
                return rows
        if self.api_key and self.rest_base.startswith("https://"):
            return self._search_rest(description, top_k=top_k)
        return []


def _shot_total_mandays(shot: Dict[str, Any]) -> float:
    for key in ("total_mandays", "total_days"):
        try:
            v = float(shot.get(key) or 0)
            if v > 0:
                return v
        except (TypeError, ValueError):
            continue
    dept = shot.get("dept_days") or {}
    if isinstance(dept, dict):
        try:
            return sum(float(v or 0) for v in dept.values())
        except (TypeError, ValueError):
            pass
    return 0.0


def save_bid_history(
    pg_url: str,
    project_name: str,
    user_id: str,
    shots: List[Dict[str, Any]],
    pre_qual: Optional[Dict[str, Any]] = None,
    notes: str = "",
    project_id: Optional[int] = None,
) -> int:
    """Save a completed batch estimate to history."""
    if not pg_url:
        raise RuntimeError("Postgres URL not configured")
    try:
        import psycopg2
        from psycopg2.extras import Json
    except ImportError as e:
        raise RuntimeError("psycopg2 not installed") from e

    total_mandays = sum(_shot_total_mandays(s) for s in shots)
    conn = psycopg2.connect(pg_url, connect_timeout=20)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO vfx_bid_history
                  (project_name, user_id, shot_count, total_mandays,
                   shots, pre_qual, notes, project_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    project_name,
                    user_id or "supervisor",
                    len(shots),
                    total_mandays,
                    Json(shots),
                    Json(pre_qual or {}),
                    notes or "",
                    project_id,
                ),
            )
            bid_id = int(cur.fetchone()[0])
        conn.commit()
        return bid_id
    finally:
        conn.close()


def list_bid_history(
    pg_url: str,
    user_id: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """List recent bids, optionally filtered by user."""
    if not pg_url:
        return []
    try:
        import psycopg2
    except ImportError:
        return []

    conn = psycopg2.connect(pg_url, connect_timeout=20)
    try:
        with conn.cursor() as cur:
            if user_id:
                cur.execute(
                    """
                    SELECT id, project_name, user_id, shot_count,
                           total_mandays, created_at, notes
                    FROM vfx_bid_history
                    WHERE user_id = %s
                    ORDER BY created_at DESC LIMIT %s
                    """,
                    (user_id, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT id, project_name, user_id, shot_count,
                           total_mandays, created_at, notes
                    FROM vfx_bid_history
                    ORDER BY created_at DESC LIMIT %s
                    """,
                    (limit,),
                )

            cols = [
                "id",
                "project_name",
                "user_id",
                "shot_count",
                "total_mandays",
                "created_at",
                "notes",
            ]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as e:
        logger.warning("list_bid_history failed: %s", e)
        return []
    finally:
        conn.close()


def get_bid_history(pg_url: str, bid_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific saved bid with full shots data."""
    if not pg_url:
        return None
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        return None

    conn = psycopg2.connect(pg_url, connect_timeout=20)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, project_name, user_id, shot_count,
                       total_mandays, shots, pre_qual,
                       created_at, notes
                FROM vfx_bid_history WHERE id = %s
                """,
                (bid_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return dict(row)
    except Exception as e:
        logger.warning("get_bid_history failed: %s", e)
        return None
    finally:
        conn.close()


def delete_bid_history(pg_url: str, bid_id: int) -> bool:
    """Delete a saved bid by id."""
    if not pg_url:
        return False
    try:
        import psycopg2
    except ImportError:
        return False

    conn = psycopg2.connect(pg_url, connect_timeout=20)
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM vfx_bid_history WHERE id = %s", (bid_id,))
            deleted = cur.rowcount > 0
        conn.commit()
        return deleted
    except Exception as e:
        logger.warning("delete_bid_history failed: %s", e)
        return False
    finally:
        conn.close()


def _preset_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "description": str(row.get("description") or ""),
        "source": "studio" if str(row.get("created_by") or "") != "system" else "system",
    }
    total = 0.0
    for col in PRESET_DEPT_COLUMNS:
        try:
            value = float(row.get(col) or 0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            out[col] = value
            total += value
    try:
        db_total = float(row.get("total") or 0)
    except (TypeError, ValueError):
        db_total = 0.0
    out["total"] = db_total if db_total > 0 else total
    return out


def load_presets(pg_url: str) -> Dict[str, Dict[str, Any]]:
    """Load studio shot presets from vfx_presets table.

    Returns an empty dict if the table doesn't exist yet.
    """
    if not pg_url:
        return {}
    try:
        import psycopg2

        conn = psycopg2.connect(pg_url, sslmode="require")
        result: Dict[str, Dict[str, Any]] = {}
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'vfx_presets'
                    )
                    """
                )
                if not cur.fetchone()[0]:
                    return {}
                cur.execute(
                    """
                    SELECT shot_type, description, camera_track,
                           matchmove, layout, animation, cfx, fx,
                           lighting, dmp, comp_paint, comp_roto,
                           compositing, total
                    FROM vfx_presets
                    """
                )
                cols = [
                    "description",
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
                    "total",
                ]
                for row in cur.fetchall():
                    result[str(row[0])] = dict(zip(cols, row[1:]))
        finally:
            conn.close()
        return result
    except Exception as e:
        logger.warning("[presets] load_presets failed: %s", e)
        return {}


def upsert_preset(pg_url: str, preset: Any) -> None:
    """Save or update a shot type preset."""
    if not pg_url:
        raise RuntimeError("Postgres URL not configured")
    try:
        import psycopg2
    except ImportError as e:
        raise RuntimeError("psycopg2 not installed") from e

    data = preset.model_dump() if hasattr(preset, "model_dump") else dict(preset)
    shot_type = str(data.get("shot_type") or "").strip()
    if not shot_type:
        raise ValueError("shot_type required")

    cols = [
        "shot_type",
        "description",
        *PRESET_DEPT_COLUMNS,
        "total",
        "created_by",
    ]
    values = [data.get(col, "") if col in ("shot_type", "description", "created_by") else float(data.get(col) or 0) for col in cols]
    updates = ", ".join(f"{col} = EXCLUDED.{col}" for col in cols if col != "shot_type")

    conn = psycopg2.connect(pg_url, connect_timeout=20)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO vfx_presets ({", ".join(cols)})
                VALUES ({", ".join(["%s"] * len(cols))})
                ON CONFLICT (shot_type) DO UPDATE SET {updates}
                """,
                values,
            )
        conn.commit()
    finally:
        conn.close()


def delete_preset_from_db(pg_url: str, shot_type: str) -> bool:
    """Delete a studio preset override."""
    if not pg_url:
        return False
    try:
        import psycopg2
    except ImportError:
        return False

    conn = psycopg2.connect(pg_url, connect_timeout=20)
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM vfx_presets WHERE shot_type = %s", (shot_type,))
            deleted = cur.rowcount > 0
        conn.commit()
        return deleted
    except Exception as e:
        logger.warning("delete_preset_from_db failed: %s", e)
        return False
    finally:
        conn.close()
