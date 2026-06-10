"""FastAPI for estimates, human corrections, and supervisor flags."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from typing import Any, Dict, List, Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
from vfx_estimator.integrations.xata import XataShotSearch
from vfx_estimator.learning.flags import FlagsStore
from vfx_estimator.types import (
    FLAG_TYPES,
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


class FlagRequest(BaseModel):
    description: str
    flag_type: str
    notes: str = ""
    user_id: str = "default"
    ai_total_days: Optional[float] = None
    ai_shot_type: Optional[str] = None
    ai_departments: Dict[str, float] = Field(default_factory=dict)


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

    return app
