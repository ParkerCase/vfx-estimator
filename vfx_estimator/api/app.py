"""FastAPI for estimates, human corrections, and supervisor flags."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from typing import Any, Dict, List, Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from vfx_estimator.api.bid_batch import (
    BatchEstimateRequest,
    attach_csv_export,
    iter_batch_estimate_events,
    parse_bid_csv_upload,
    run_bid_batch_estimate,
    validate_batch_shots,
)
from vfx_estimator.config import get_settings
from vfx_estimator.estimate.service import EstimatorService
from vfx_estimator.fx import get_usd_fx_rates
from vfx_estimator.integrations.xata import (
    XataShotSearch,
    delete_bid_history,
    delete_preset_from_db,
    get_bid_history,
    list_bid_history,
    load_presets,
    save_bid_history,
    get_postgres_connection,
    upsert_preset,
)
from vfx_estimator.learning.flags import FlagsStore
from vfx_estimator.llm.asset_rag import ASSET_BASELINES
from vfx_estimator.llm.mandays_rag import SHOT_BASELINES
from vfx_estimator.rates import build_dept_rates
from vfx_estimator.types import (
    AssetPresetRequest,
    FLAG_TYPES,
    PresetRequest,
    SaveBidRequest,
    UserCorrection,
    UserFlag,
    bid_departments_to_internal,
)
from vfx_estimator.types import BidPreQual

_service: Optional[EstimatorService] = None
_flags: Optional[FlagsStore] = None


def get_service() -> EstimatorService:
    global _service
    if _service is None:
        _service = EstimatorService()
    return _service


def get_flags() -> FlagsStore:
    global _flags
    if _flags is None:
        _flags = FlagsStore(settings=get_settings())
    return _flags


DEPT_MAX_DAYS: Dict[str, float] = {
    "animation": 25,
    "fx": 20,
    "lighting": 15,
    "compositing": 15,
    "comp_paint": 12,
    "comp_roto": 10,
    "layout": 10,
    "dmp": 10,
    "camera_track": 8,
    "matchmove": 8,
    "cfx": 10,
    "prep": 5,
    "ai": 10,
    "crowds": 25,
    "enviro": 15,
}


def _compute_adjustment_ranges(dept_days: Dict[str, float], overall_confidence: float) -> Dict[str, Dict]:
    """Compute wide slider min/max/step per department for supervisor HITL."""
    out: Dict[str, Dict] = {}
    for dept, predicted in dept_days.items():
        predicted = float(predicted)
        if predicted <= 0:
            continue
        dept_cap = DEPT_MAX_DAYS.get(dept, 15)
        hi = max(predicted * 3.0, dept_cap)
        hi = round(hi * 2) / 2
        out[dept] = {
            "predicted": predicted,
            "min": 0.0,
            "max": hi,
            "step": 0.5,
            "confidence": overall_confidence,
        }
    return out


# ── Request / response models ──────────────────────────────────────────────


class EstimateRequest(BaseModel):
    description: str
    pre_qual: Optional[BidPreQual] = None
    mode: Optional[str] = None
    dept_rates: Optional[Dict[str, float]] = None


class CorrectionRequest(BaseModel):
    description: str
    final_total_days: float
    final_departments: Dict[str, float] = Field(default_factory=dict)
    user_id: str = "default"
    notes: str = ""
    ai_total_days: Optional[float] = None


class BatchCorrectionsRequest(BaseModel):
    corrections: List[CorrectionRequest]


class AssetCorrectionRequest(BaseModel):
    asset_name: str
    final_dept_days: Dict[str, float] = Field(default_factory=dict)
    ai_total_days: Optional[float] = None
    final_total_days: float = 0
    user_id: str = "supervisor"
    notes: str = ""


class FlagRequest(BaseModel):
    description: str
    flag_type: str
    notes: str = ""
    user_id: str = "default"
    ai_total_days: Optional[float] = None
    ai_shot_type: Optional[str] = None
    ai_departments: Dict[str, float] = Field(default_factory=dict)


class SuggestMethodologyRequest(BaseModel):
    description: str
    shot_code: Optional[str] = None
    notes: Optional[str] = None


class GoogleAuthRequest(BaseModel):
    credential: str


class UserUpdateRequest(BaseModel):
    role: Optional[str] = None
    org_name: Optional[str] = None


class ProjectCreate(BaseModel):
    name: str
    client: str = ""
    stage: str = "estimating"
    due_date: Optional[str] = None


class TaskCreate(BaseModel):
    title: str
    project_id: Optional[int] = None
    due_date: Optional[str] = None


class AssetEstimateItem(BaseModel):
    asset_name: str
    description: str
    variations: int = 1


class AssetContextRequest(BaseModel):
    tier: Optional[str] = None
    notes: Optional[str] = None
    day_rate: float = 500


class AssetEstimateRequest(BaseModel):
    assets: List[AssetEstimateItem]
    asset_context: Optional[AssetContextRequest] = None


def _prepare_batch_service(svc: EstimatorService) -> None:
    if svc.gemini:
        svc.gemini.flags = get_flags()


def _live_training_count(svc: EstimatorService) -> Dict[str, Any]:
    xata = XataShotSearch(svc.settings)
    correction_count = svc.corrections.count()
    xata_count = xata.count()
    if xata_count > 0:
        return {
            "training_shots": xata_count + correction_count,
            "base_training_shots": xata_count,
            "corrections": correction_count,
            "training_count_source": "xata",
        }
    return {
        "training_shots": len(svc.training) + correction_count,
        "base_training_shots": len(svc.training),
        "corrections": correction_count,
        "training_count_source": "local",
    }


def _merged_presets(pg_url: str = "") -> Dict[str, Dict[str, Any]]:
    presets = {
        key: {**value, "source": "system"}
        for key, value in SHOT_BASELINES.items()
    }
    if pg_url:
        for key, value in load_presets(pg_url).items():
            presets[key] = {**presets.get(key, {}), **value}
    return presets


ASSET_PRESET_COLUMNS = (
    "modelling",
    "texturing",
    "rigging",
    "cfx",
    "fx",
    "lookdev",
    "dmp",
    "comp_dev",
)
ASSET_PRESET_FIELDS = [*ASSET_PRESET_COLUMNS, "total"]


def _load_asset_presets(pg_url: str) -> Dict[str, Dict[str, Any]]:
    if not pg_url:
        return {}
    try:
        import psycopg2

        with psycopg2.connect(pg_url, sslmode="require") as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'vfx_asset_presets'
                    )
                    """
                )
                if not cur.fetchone()[0]:
                    return {}
                cur.execute(
                    """
                    SELECT asset_type, description, modelling, texturing,
                           rigging, cfx, fx, lookdev, dmp, comp_dev,
                           total, created_by
                    FROM vfx_asset_presets
                    ORDER BY asset_type
                    """
                )
                cols = [
                    "description",
                    *ASSET_PRESET_COLUMNS,
                    "total",
                    "created_by",
                ]
                presets = {}
                for row in cur.fetchall():
                    data = dict(zip(cols, row[1:]))
                    data["source"] = (
                        "studio" if data.pop("created_by", "system") != "system" else "system"
                    )
                    presets[str(row[0])] = data
                return presets
    except Exception:
        return {}


def _merged_asset_presets(pg_url: str = "") -> Dict[str, Dict[str, Any]]:
    presets = {
        key: {**value, "source": "system"}
        for key, value in ASSET_BASELINES.items()
    }
    if pg_url:
        for key, value in _load_asset_presets(pg_url).items():
            presets[key] = {**presets.get(key, {}), **value}
    return presets


def _upsert_asset_preset(pg_url: str, preset: AssetPresetRequest) -> None:
    if not pg_url:
        raise RuntimeError("Postgres URL not configured")
    import psycopg2

    data = preset.model_dump()
    data["created_by"] = "studio"
    cols = ["asset_type", "description", *ASSET_PRESET_COLUMNS, "total", "created_by"]
    values = [
        data.get(col, "") if col in ("asset_type", "description", "created_by") else float(data.get(col) or 0)
        for col in cols
    ]
    updates = ", ".join(f"{col} = EXCLUDED.{col}" for col in cols if col != "asset_type")
    with psycopg2.connect(pg_url, sslmode="require", connect_timeout=20) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO vfx_asset_presets ({", ".join(cols)})
                VALUES ({", ".join(["%s"] * len(cols))})
                ON CONFLICT (asset_type) DO UPDATE SET {updates}
                """,
                values,
            )
        conn.commit()


def _delete_asset_preset(pg_url: str, asset_type: str) -> None:
    if not pg_url:
        raise RuntimeError("Postgres URL not configured")
    import psycopg2

    with psycopg2.connect(pg_url, sslmode="require", connect_timeout=20) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM vfx_asset_presets WHERE asset_type = %s AND created_by = 'studio'",
                (asset_type,),
            )
        conn.commit()


def _iso_row(row: Dict[str, Any], keys: tuple[str, ...]) -> Dict[str, Any]:
    for key in keys:
        value = row.get(key)
        if value is not None and hasattr(value, "isoformat"):
            row[key] = value.isoformat()
    return row


def _require_auth(authorization: Optional[str]) -> Dict[str, Any]:
    """Extract and verify Google token from Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Authentication required")
    credential = authorization.split(" ", 1)[1]
    try:
        from vfx_estimator.integrations.auth import verify_google_token

        return verify_google_token(credential)
    except ValueError as exc:
        raise HTTPException(401, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, f"Google auth verifier unavailable: {exc}") from exc


def _current_user_record(user_id: str) -> Dict[str, Any]:
    settings = get_service().settings
    with get_postgres_connection(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, email, name, picture, role, org_name, created_at
                FROM vfx_users WHERE id = %s
                """,
                (user_id,),
            )
            row = cur.fetchone()
    if not row:
        raise HTTPException(404, "User not found — sign in first")
    cols = ["id", "email", "name", "picture", "role", "org_name", "created_at"]
    return _iso_row(dict(zip(cols, row)), ("created_at",))


# ── App factory ────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    from vfx_estimator.retrieval.index import build_index

    build_index()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="VFX Estimator", version="0.4.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        s = get_settings()
        svc = get_service()
        flags = get_flags().load()
        counts = _live_training_count(svc)
        return {
            "ok": True,
            **counts,
            "flags": len(flags),
            "mode_default": s.estimate_mode,
            "gemini_configured": bool(s.resolved_gemini_key()),
            "legacy_numeric": s.use_legacy_numeric,
            "xata_mode": XataShotSearch(s).mode,
            "xata_corrections": svc.corrections.storage_backend,
        }

    @app.get("/ping")
    def ping() -> Dict[str, bool]:
        return {"ok": True}

    @app.get("/fx/rates")
    def fx_rates() -> Dict[str, Any]:
        """USD-base FX for display (Frankfurter, cached 1h)."""
        return get_usd_fx_rates()

    @app.get("/auth/config")
    def auth_config() -> Dict[str, Any]:
        return {"google_client_id": get_settings().google_client_id}

    @app.post("/auth/google")
    def google_auth(req: GoogleAuthRequest) -> Dict[str, Any]:
        """Verify Google token, upsert user, return user record."""
        try:
            from vfx_estimator.integrations.auth import verify_google_token

            user_info = verify_google_token(req.credential)
        except ValueError as exc:
            raise HTTPException(401, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, f"Google auth verifier unavailable: {exc}") from exc

        try:
            with get_postgres_connection(get_service().settings) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO vfx_users (id, email, name, picture, last_seen)
                        VALUES (%s, %s, %s, %s, NOW())
                        ON CONFLICT (id) DO UPDATE SET
                            email = EXCLUDED.email,
                            name = EXCLUDED.name,
                            picture = EXCLUDED.picture,
                            last_seen = NOW()
                        RETURNING id, email, name, picture, role, org_name, created_at
                        """,
                        (
                            user_info["id"],
                            user_info["email"],
                            user_info["name"],
                            user_info["picture"],
                        ),
                    )
                    row = cur.fetchone()
                conn.commit()
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                503,
                "User database unavailable — run scripts/migrate_xata_corrections on Postgres",
            ) from exc
        cols = ["id", "email", "name", "picture", "role", "org_name", "created_at"]
        user = _iso_row(dict(zip(cols, row)), ("created_at",))
        return {"ok": True, "user": user, "token": req.credential}

    @app.get("/auth/me")
    def get_me(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        user_info = _require_auth(authorization)
        return _current_user_record(user_info["id"])

    @app.put("/auth/me")
    def update_me(
        req: UserUpdateRequest,
        authorization: Optional[str] = Header(None),
    ) -> Dict[str, Any]:
        user = _require_auth(authorization)
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        if "role" in updates and updates["role"] not in {"vendor", "production"}:
            raise HTTPException(400, "role must be 'vendor' or 'production'")
        if not updates:
            return {"ok": True}
        set_clause = ", ".join(f"{k} = %s" for k in updates)
        values = list(updates.values()) + [user["id"]]
        with get_postgres_connection(get_service().settings) as conn:
            with conn.cursor() as cur:
                cur.execute(f"UPDATE vfx_users SET {set_clause} WHERE id = %s", values)
            conn.commit()
        return {"ok": True}

    @app.post("/projects")
    def create_project(
        req: ProjectCreate,
        authorization: Optional[str] = Header(None),
    ) -> Dict[str, Any]:
        user = _require_auth(authorization)
        if not req.name.strip():
            raise HTTPException(400, "name required")
        user_record = _current_user_record(user["id"])
        with get_postgres_connection(get_service().settings) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO vfx_projects
                      (name, client, stage, due_date, owner_id, org_name)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id, name, client, status, stage, due_date,
                              owner_id, created_at
                    """,
                    (
                        req.name.strip(),
                        req.client.strip(),
                        req.stage,
                        req.due_date or None,
                        user["id"],
                        user_record.get("org_name", ""),
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        cols = ["id", "name", "client", "status", "stage", "due_date", "owner_id", "created_at"]
        return _iso_row(dict(zip(cols, row)), ("due_date", "created_at"))

    @app.get("/projects")
    def list_projects(
        status: str = "active",
        authorization: Optional[str] = Header(None),
    ) -> Dict[str, Any]:
        user = _require_auth(authorization)
        with get_postgres_connection(get_service().settings) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT p.id, p.name, p.client, p.status, p.stage,
                           p.due_date, p.created_at, p.updated_at,
                           COUNT(b.id) AS bid_count,
                           COALESCE(SUM(b.total_mandays), 0) AS total_mandays
                    FROM vfx_projects p
                    LEFT JOIN vfx_bid_history b ON b.project_id = p.id
                    WHERE p.owner_id = %s AND p.status = %s
                    GROUP BY p.id
                    ORDER BY p.updated_at DESC
                    """,
                    (user["id"], status),
                )
                cols = [
                    "id",
                    "name",
                    "client",
                    "status",
                    "stage",
                    "due_date",
                    "created_at",
                    "updated_at",
                    "bid_count",
                    "total_mandays",
                ]
                projects = [dict(zip(cols, row)) for row in cur.fetchall()]
        for project in projects:
            _iso_row(project, ("due_date", "created_at", "updated_at"))
            project["bid_count"] = int(project.get("bid_count") or 0)
            project["total_mandays"] = float(project.get("total_mandays") or 0)
        return {"projects": projects, "count": len(projects)}

    @app.put("/projects/{project_id}")
    def update_project(
        project_id: int,
        req: Dict[str, Any],
        authorization: Optional[str] = Header(None),
    ) -> Dict[str, Any]:
        user = _require_auth(authorization)
        allowed = {"name", "client", "stage", "status", "due_date"}
        updates = {k: v for k, v in req.items() if k in allowed}
        if not updates:
            return {"ok": True}
        set_clause = ", ".join(f"{k} = %s" for k in updates)
        set_clause += ", updated_at = NOW()"
        values = list(updates.values()) + [project_id, user["id"]]
        with get_postgres_connection(get_service().settings) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE vfx_projects SET {set_clause} WHERE id = %s AND owner_id = %s",
                    values,
                )
            conn.commit()
        return {"ok": True}

    @app.get("/dashboard/stats")
    def dashboard_stats(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        user = _require_auth(authorization)
        with get_postgres_connection(get_service().settings) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM vfx_projects WHERE owner_id = %s AND status = 'active'",
                    (user["id"],),
                )
                active_projects = int(cur.fetchone()[0])
                cur.execute(
                    """
                    SELECT COUNT(*), COALESCE(SUM(total_mandays), 0),
                           COALESCE(AVG(shot_count), 0)
                    FROM vfx_bid_history
                    WHERE user_id = %s
                    """,
                    (user["id"],),
                )
                bid_count, total_mandays, avg_shots = cur.fetchone()
                cur.execute(
                    "SELECT COUNT(*) FROM vfx_corrections WHERE user_id = %s",
                    (user["id"],),
                )
                corrections = int(cur.fetchone()[0])
        return {
            "active_projects": active_projects,
            "total_bids": int(bid_count or 0),
            "total_mandays": round(float(total_mandays or 0), 1),
            "avg_shots_per_bid": round(float(avg_shots or 0), 1),
            "corrections_made": corrections,
        }

    @app.get("/tasks")
    def list_tasks(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        user = _require_auth(authorization)
        with get_postgres_connection(get_service().settings) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT t.id, t.title, t.done, t.due_date,
                           t.created_at, p.name AS project_name
                    FROM vfx_tasks t
                    LEFT JOIN vfx_projects p ON p.id = t.project_id
                    WHERE t.user_id = %s
                    ORDER BY t.done ASC, t.due_date ASC NULLS LAST
                    """,
                    (user["id"],),
                )
                cols = ["id", "title", "done", "due_date", "created_at", "project_name"]
                tasks = [dict(zip(cols, row)) for row in cur.fetchall()]
        for task in tasks:
            _iso_row(task, ("due_date", "created_at"))
        return {"tasks": tasks}

    @app.post("/tasks")
    def create_task(
        req: TaskCreate,
        authorization: Optional[str] = Header(None),
    ) -> Dict[str, Any]:
        user = _require_auth(authorization)
        if not req.title.strip():
            raise HTTPException(400, "title required")
        with get_postgres_connection(get_service().settings) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO vfx_tasks (user_id, project_id, title, due_date)
                    VALUES (%s, %s, %s, %s) RETURNING id
                    """,
                    (user["id"], req.project_id, req.title.strip(), req.due_date or None),
                )
                task_id = cur.fetchone()[0]
            conn.commit()
        return {"ok": True, "id": task_id}

    @app.put("/tasks/{task_id}")
    def update_task(
        task_id: int,
        req: Dict[str, Any],
        authorization: Optional[str] = Header(None),
    ) -> Dict[str, Any]:
        user = _require_auth(authorization)
        allowed = {"title", "done", "due_date"}
        updates = {k: v for k, v in req.items() if k in allowed}
        if not updates:
            return {"ok": True}
        set_clause = ", ".join(f"{k} = %s" for k in updates)
        values = list(updates.values()) + [task_id, user["id"]]
        with get_postgres_connection(get_service().settings) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE vfx_tasks SET {set_clause} WHERE id = %s AND user_id = %s",
                    values,
                )
            conn.commit()
        return {"ok": True}

    @app.post("/estimate")
    def estimate(req: EstimateRequest) -> Dict[str, Any]:
        if not req.description.strip():
            raise HTTPException(400, "description required")
        svc = get_service()
        _prepare_batch_service(svc)

        est = svc.estimate(
            req.description,
            pre_qual=req.pre_qual,
            mode=req.mode,
            dept_rates=req.dept_rates,
        )
        result = est.model_dump()

        dept = {k: float(v) for k, v in (est.dept_days or {}).items() if float(v or 0) > 0}
        result["adjustment_ranges"] = _compute_adjustment_ranges(dept, est.confidence)
        result["dept_confidence"] = {}
        result["dept_rates"] = svc.settings.resolved_dept_rates(overrides=req.dept_rates)

        return result

    @app.post("/estimate/batch")
    def estimate_batch(req: BatchEstimateRequest) -> Dict[str, Any]:
        svc = get_service()
        _prepare_batch_service(svc)
        return run_bid_batch_estimate(
            svc,
            req.shots,
            project=req.project,
            day_rate=req.day_rate,
            dept_rates=req.dept_rates,
            mode=req.mode,
            pre_qual=req.pre_qual,
            compute_ranges=_compute_adjustment_ranges,
        )

    @app.post("/estimate/batch/stream")
    async def batch_estimate_stream(req: BatchEstimateRequest) -> StreamingResponse:
        validate_batch_shots(req.shots)
        svc = get_service()
        _prepare_batch_service(svc)
        rate = float(req.day_rate if req.day_rate is not None else svc.settings.day_rate)
        rates = svc.settings.resolved_dept_rates(overrides=req.dept_rates)
        if not isinstance(rates, dict):
            rates = build_dept_rates(fallback=rate, overrides=req.dept_rates)
        proj = req.project or (req.pre_qual.project if req.pre_qual else None) or "BID"

        async def generate():
            event_q: queue.Queue = queue.Queue()

            def run_batch() -> None:
                try:
                    for event in iter_batch_estimate_events(
                        req.shots,
                        svc=svc,
                        project=proj,
                        day_rate=rate,
                        dept_rates=rates,
                        mode=req.mode,
                        pre_qual=req.pre_qual,
                        compute_ranges=_compute_adjustment_ranges,
                    ):
                        event_q.put(event)
                except Exception as exc:
                    event_q.put({"type": "fatal", "error": str(exc)})
                finally:
                    event_q.put(None)

            threading.Thread(target=run_batch, daemon=True).start()

            while True:
                try:
                    while True:
                        item = event_q.get_nowait()
                        if item is None:
                            return
                        yield f"data: {json.dumps(item)}\n\n"
                except queue.Empty:
                    await asyncio.sleep(0.05)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/estimate/batch/csv")
    async def estimate_batch_csv(
        file: UploadFile = File(...),
        mode: Optional[str] = Form(None),
        day_rate: Optional[float] = Form(750),
        project: Optional[str] = Form(None),
        pre_qual_json: Optional[str] = Form(None),
    ) -> Dict[str, Any]:
        if not file.filename:
            raise HTTPException(400, "file required")
        raw = await file.read()
        if not raw:
            raise HTTPException(400, "uploaded file is empty")

        shots, artifact = parse_bid_csv_upload(raw)
        pre_qual: Optional[BidPreQual] = None
        if pre_qual_json and pre_qual_json.strip():
            try:
                pre_qual = BidPreQual.model_validate(json.loads(pre_qual_json))
            except (json.JSONDecodeError, ValueError) as exc:
                raise HTTPException(400, f"invalid pre_qual_json: {exc}") from exc

        svc = get_service()
        _prepare_batch_service(svc)
        proj = project or (pre_qual.project if pre_qual else None) or "BID"
        rate = float(day_rate if day_rate is not None else svc.settings.day_rate)

        payload = run_bid_batch_estimate(
            svc,
            shots,
            project=proj,
            day_rate=rate,
            mode=mode,
            pre_qual=pre_qual,
            compute_ranges=_compute_adjustment_ranges,
        )
        return attach_csv_export(payload, artifact=artifact, project=proj, day_rate=rate)

    @app.post("/suggest-methodology")
    def suggest_methodology(req: SuggestMethodologyRequest) -> Dict[str, Any]:
        svc = get_service()
        prompt = f"""You are a senior VFX supervisor.
Given this shot, suggest 2-3 possible VFX methodologies showing how the shot would be executed.

SHOT: "{req.description}"
SHOT CODE: "{req.shot_code or ''}"
NOTES: "{req.notes or ''}"

Each methodology should be 2-3 sentences describing the technical approach and which VFX departments are involved.
Make them meaningfully different options (e.g. full 3D vs 2.5D DMP vs 2D comp-only approaches where applicable).

Return JSON only:
{{
  "suggestions": [
    {{
      "label": "Full CG Integration",
      "methodology": "Full CG element rendered in 3D and integrated with live-action plate. Camera track required. Lighting matched to practical plate.",
      "departments": ["camera_track", "layout", "lighting", "compositing"]
    }},
    {{
      "label": "2D Composite Only",
      "methodology": "Plate-based 2D composite using roto, paint, and comp integration only.",
      "departments": ["compositing"]
    }}
  ]
}}"""
        from vfx_estimator.llm.gemini_client import generate_json

        try:
            print("[suggest_methodology] Prompt being sent:", flush=True)
            print(prompt, flush=True)
            result = generate_json(
                prompt,
                settings=svc.settings,
                debug_label="suggest_methodology",
                timeout_sec=45,
            )
        except Exception as exc:
            print(
                f"[suggest_methodology] generate_json raised {type(exc).__name__}: {exc}",
                flush=True,
            )
            result = None
        if not result:
            print(
                f"[suggest_methodology] generate_json returned None for: {req.description[:80]}",
                flush=True,
            )
        elif "suggestions" not in result:
            print(f"[suggest_methodology] Got result but no 'suggestions' key: {result}", flush=True)
        elif not result["suggestions"]:
            print(
                f"[suggest_methodology] suggestions key present but EMPTY. Raw result: {result}",
                flush=True,
            )
        else:
            print(
                f"[suggest_methodology] SUCCESS — {len(result['suggestions'])} suggestions",
                flush=True,
            )
        return result or {"suggestions": []}

    @app.post("/estimate/assets/stream")
    async def estimate_assets_stream(req: AssetEstimateRequest) -> StreamingResponse:
        if not req.assets:
            raise HTTPException(400, "assets required")
        svc = get_service()

        async def generate():
            from vfx_estimator.llm.asset_rag import estimate_single_asset

            total = len(req.assets)
            for i, asset in enumerate(req.assets):
                yield f"data: {json.dumps({'type': 'progress', 'current': i + 1, 'total': total, 'asset_name': asset.asset_name})}\n\n"
                try:
                    result = estimate_single_asset(asset, req.asset_context, svc.settings)
                    yield f"data: {json.dumps({'type': 'result', 'index': i, 'data': result})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'index': i, 'error': str(e)})}\n\n"
                await asyncio.sleep(0)
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/corrections")
    def add_correction(req: CorrectionRequest) -> Dict[str, Any]:
        svc = get_service()
        data = req.model_dump()
        data["final_departments"] = bid_departments_to_internal(data.get("final_departments") or {})
        svc.record_correction(UserCorrection.model_validate(data))
        total = svc.corrections.count()
        return {
            "ok": True,
            "count": total,
            "message": f"Correction saved. {total} total — retrieval index updated.",
        }

    @app.post("/corrections/batch")
    def add_corrections_batch(req: BatchCorrectionsRequest) -> Dict[str, Any]:
        if not req.corrections:
            raise HTTPException(400, "corrections required")
        svc = get_service()
        saved = 0
        for item in req.corrections:
            if not item.description.strip():
                raise HTTPException(400, "each correction requires description")
            data = item.model_dump()
            data["final_departments"] = bid_departments_to_internal(data.get("final_departments") or {})
            svc.record_correction(UserCorrection.model_validate(data))
            saved += 1
        total = svc.corrections.count()
        return {"ok": True, "saved": saved, "count": total}

    @app.get("/corrections")
    def list_corrections() -> Dict[str, Any]:
        svc = get_service()
        corrections = svc.corrections.load()
        return {
            "count": len(corrections),
            "corrections": [c.model_dump() for c in corrections[-20:]],
        }

    @app.get("/corrections/stats")
    def corrections_stats() -> Dict[str, Any]:
        svc = get_service()
        corrections = svc.corrections.load()
        if not corrections:
            return {"count": 0, "dept_frequency": {}, "avg_ai_days": None, "avg_final_days": None}

        dept_freq: Dict[str, int] = {}
        ai_days_list: List[float] = []
        final_days_list: List[float] = []

        for c in corrections:
            for dept in (c.final_departments or {}):
                dept_freq[dept] = dept_freq.get(dept, 0) + 1
            if c.ai_total_days is not None:
                ai_days_list.append(float(c.ai_total_days))
            final_days_list.append(float(c.final_total_days))

        sorted_depts = sorted(dept_freq.items(), key=lambda x: -x[1])
        avg_ai = sum(ai_days_list) / len(ai_days_list) if ai_days_list else None
        avg_final = sum(final_days_list) / len(final_days_list) if final_days_list else None
        avg_delta = (avg_final - avg_ai) if (avg_ai and avg_final) else None

        return {
            "count": len(corrections),
            "dept_frequency": dict(sorted_depts[:10]),
            "avg_ai_days": round(avg_ai, 2) if avg_ai is not None else None,
            "avg_final_days": round(avg_final, 2) if avg_final else None,
            "avg_delta_days": round(avg_delta, 2) if avg_delta is not None else None,
        }

    @app.post("/corrections/assets")
    def save_asset_correction(req: AssetCorrectionRequest) -> Dict[str, Any]:
        pg_url = get_service().settings.resolved_xata_postgres_url()
        if not pg_url:
            raise HTTPException(503, "Asset corrections require XATA_POSTGRES_URL")
        try:
            import psycopg2

            conn = psycopg2.connect(pg_url, sslmode="require")
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS vfx_asset_corrections (
                        id SERIAL PRIMARY KEY,
                        asset_name TEXT NOT NULL,
                        final_dept_days JSONB DEFAULT '{}',
                        ai_total_days FLOAT,
                        final_total_days FLOAT NOT NULL,
                        user_id TEXT DEFAULT 'supervisor',
                        notes TEXT DEFAULT '',
                        timestamp TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute("""
                    INSERT INTO vfx_asset_corrections
                      (asset_name, final_dept_days, ai_total_days,
                       final_total_days, user_id, notes)
                    VALUES (%s, %s::jsonb, %s, %s, %s, %s)
                """, (
                    req.asset_name,
                    json.dumps(req.final_dept_days),
                    req.ai_total_days,
                    req.final_total_days,
                    req.user_id,
                    req.notes,
                ))
                conn.commit()
            conn.close()
        except Exception as e:
            raise HTTPException(500, str(e)) from e
        return {"ok": True}

    @app.get("/corrections/assets/stats")
    def asset_correction_stats() -> Dict[str, Any]:
        pg_url = get_service().settings.resolved_xata_postgres_url()
        if not pg_url:
            return {"count": 0, "dept_frequency": {}}
        try:
            import psycopg2

            conn = psycopg2.connect(pg_url, sslmode="require")
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'vfx_asset_corrections'
                    )
                """)
                if not cur.fetchone()[0]:
                    return {"count": 0, "dept_frequency": {}}

                cur.execute("""
                    SELECT COUNT(*),
                           AVG(ai_total_days),
                           AVG(final_total_days),
                           AVG(final_total_days - COALESCE(ai_total_days, final_total_days))
                    FROM vfx_asset_corrections
                """)
                row = cur.fetchone()
                count = row[0] or 0
                avg_ai = float(row[1]) if row[1] else None
                avg_final = float(row[2]) if row[2] else None
                avg_delta = float(row[3]) if row[3] else None

                cur.execute("""
                    SELECT final_dept_days FROM vfx_asset_corrections
                """)
                dept_freq: Dict[str, int] = {}
                for (dept_json,) in cur.fetchall():
                    if dept_json:
                        for dept, days in dept_json.items():
                            if days and float(days) > 0:
                                dept_freq[dept] = dept_freq.get(dept, 0) + 1

            conn.close()
            return {
                "count": count,
                "avg_ai_days": avg_ai,
                "avg_final_days": avg_final,
                "avg_delta_days": avg_delta,
                "dept_frequency": dict(
                    sorted(dept_freq.items(), key=lambda x: -x[1])[:6]
                ),
            }
        except Exception as e:
            return {"count": 0, "dept_frequency": {}, "error": str(e)}

    @app.post("/flags")
    def add_flag(req: FlagRequest) -> Dict[str, Any]:
        if req.flag_type not in FLAG_TYPES:
            raise HTTPException(400, f"flag_type must be one of: {FLAG_TYPES}")
        store = get_flags()
        store.append(UserFlag.model_validate(req.model_dump()))
        stats = store.stats()
        return {
            "ok": True,
            "count": stats["count"],
            "message": f"Flag saved. {stats['count']} total flags — will inform future estimates.",
        }

    @app.get("/flags")
    def list_flags() -> Dict[str, Any]:
        store = get_flags()
        flags = store.load()
        return {
            "count": len(flags),
            "flags": [f.model_dump() for f in flags[-20:]],
        }

    @app.get("/flags/stats")
    def flags_stats() -> Dict[str, Any]:
        return get_flags().stats()

    @app.get("/flags/types")
    def flag_types() -> Dict[str, Any]:
        return {"flag_types": FLAG_TYPES}

    @app.get("/tuning")
    def get_tuning() -> Dict[str, Any]:
        s = get_settings()
        return {
            "estimate_mode": s.estimate_mode,
            "blend_numeric_weight": s.blend_numeric_weight,
            "blend_gemini_weight": s.blend_gemini_weight,
            "correction_boost": s.correction_boost,
            "retrieval_top_k": s.retrieval_top_k,
            "day_rate": s.day_rate,
            "dept_rates": s.resolved_dept_rates(),
            "use_legacy_numeric": s.use_legacy_numeric,
            "overrides_file": str(s.tuning_path()),
            "overrides": s.load_tuning_overrides(),
        }

    @app.put("/tuning")
    def put_tuning(body: Dict[str, Any]) -> Dict[str, Any]:
        path = get_settings().tuning_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(body, f, indent=2)
        get_settings.cache_clear()
        global _service
        from vfx_estimator.retrieval.index import invalidate_index

        invalidate_index()
        _service = EstimatorService(get_settings())
        return {"ok": True, "path": str(path)}

    @app.post("/bid-history")
    def save_bid(req: SaveBidRequest) -> Dict[str, Any]:
        """Save a completed batch estimate."""
        svc = get_service()
        pg_url = svc.settings.resolved_xata_postgres_url()
        if not pg_url:
            raise HTTPException(503, "Bid history requires XATA_POSTGRES_URL")
        if not req.project_name.strip():
            raise HTTPException(400, "project_name required")
        if not req.shots:
            raise HTTPException(400, "shots required")
        bid_id = save_bid_history(
            pg_url,
            project_name=req.project_name.strip(),
            user_id=req.user_id or "supervisor",
            shots=req.shots,
            pre_qual=req.pre_qual,
            notes=req.notes or "",
            project_id=req.project_id,
        )
        return {"ok": True, "bid_id": bid_id}

    @app.get("/bid-history")
    def list_bids(user_id: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        """List saved bids."""
        svc = get_service()
        pg_url = svc.settings.resolved_xata_postgres_url()
        if not pg_url:
            return {"bids": [], "count": 0}
        bids = list_bid_history(pg_url, user_id=user_id, limit=limit)
        for b in bids:
            created = b.get("created_at")
            if created is not None and hasattr(created, "isoformat"):
                b["created_at"] = created.isoformat()
        return {"bids": bids, "count": len(bids)}

    @app.get("/bid-history/{bid_id}")
    def get_bid(bid_id: int) -> Dict[str, Any]:
        """Get a specific saved bid."""
        svc = get_service()
        pg_url = svc.settings.resolved_xata_postgres_url()
        if not pg_url:
            raise HTTPException(503, "Bid history requires XATA_POSTGRES_URL")
        bid = get_bid_history(pg_url, bid_id)
        if not bid:
            raise HTTPException(404, "Bid not found")
        created = bid.get("created_at")
        if created is not None and hasattr(created, "isoformat"):
            bid["created_at"] = created.isoformat()
        return bid

    @app.delete("/bid-history/{bid_id}")
    def remove_bid(bid_id: int) -> Dict[str, Any]:
        """Delete a saved bid."""
        svc = get_service()
        pg_url = svc.settings.resolved_xata_postgres_url()
        if not pg_url:
            raise HTTPException(503, "Bid history requires XATA_POSTGRES_URL")
        if not delete_bid_history(pg_url, bid_id):
            raise HTTPException(404, "Bid not found")
        return {"ok": True, "deleted": bid_id}

    @app.get("/presets")
    def list_presets() -> Dict[str, Any]:
        """List all shot type presets."""
        svc = get_service()
        pg_url = svc.settings.resolved_xata_postgres_url()
        presets = _merged_presets(pg_url)
        return {"presets": presets, "count": len(presets)}

    @app.post("/presets")
    def save_preset(req: PresetRequest) -> Dict[str, Any]:
        """Save or update a shot type preset."""
        svc = get_service()
        pg_url = svc.settings.resolved_xata_postgres_url()
        if not pg_url:
            raise HTTPException(503, "Presets require XATA_POSTGRES_URL")
        if not req.shot_type.strip():
            raise HTTPException(400, "shot_type required")
        upsert_preset(pg_url, req)
        return {"ok": True, "shot_type": req.shot_type}

    @app.delete("/presets/{shot_type}")
    def delete_preset(shot_type: str) -> Dict[str, Any]:
        """Reset a preset back to system default."""
        svc = get_service()
        pg_url = svc.settings.resolved_xata_postgres_url()
        if not pg_url:
            raise HTTPException(503, "Presets require XATA_POSTGRES_URL")
        delete_preset_from_db(pg_url, shot_type)
        return {"ok": True}

    @app.get("/presets/assets")
    def list_asset_presets() -> Dict[str, Any]:
        """List all asset type presets."""
        svc = get_service()
        pg_url = svc.settings.resolved_xata_postgres_url()
        presets = _merged_asset_presets(pg_url)
        return {"presets": presets, "count": len(presets)}

    @app.post("/presets/assets")
    def save_asset_preset(req: AssetPresetRequest) -> Dict[str, Any]:
        """Save or update an asset type preset."""
        svc = get_service()
        pg_url = svc.settings.resolved_xata_postgres_url()
        if not pg_url:
            raise HTTPException(503, "Asset presets require XATA_POSTGRES_URL")
        if not req.asset_type.strip():
            raise HTTPException(400, "asset_type required")
        _upsert_asset_preset(pg_url, req)
        return {"ok": True, "asset_type": req.asset_type}

    @app.delete("/presets/assets/{asset_type}")
    def delete_asset_preset(asset_type: str) -> Dict[str, Any]:
        """Reset an asset preset back to system default."""
        svc = get_service()
        pg_url = svc.settings.resolved_xata_postgres_url()
        if not pg_url:
            raise HTTPException(503, "Asset presets require XATA_POSTGRES_URL")
        _delete_asset_preset(pg_url, asset_type)
        return {"ok": True}

    return app
