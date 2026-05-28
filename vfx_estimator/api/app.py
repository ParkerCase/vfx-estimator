"""FastAPI for estimates, human corrections, and supervisor flags."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from vfx_estimator.config import get_settings
from vfx_estimator.estimate.service import EstimatorService
from vfx_estimator.integrations.xata import XataShotSearch
from vfx_estimator.learning.flags import FlagsStore
from vfx_estimator.types import BidPreQual, FLAG_TYPES, UserCorrection, UserFlag

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


def _compute_adjustment_ranges(dept_days: Dict[str, float], overall_confidence: float) -> Dict[str, Dict]:
    """Compute slider min/max/step/predicted per department based on confidence.

    Confidence bands:
      >= 0.8  -> ±50% range
      >= 0.6  -> ±75% range
       < 0.6  -> ±100% range
    Min always clamped to 0.
    """
    out: Dict[str, Dict] = {}
    for dept, predicted in dept_days.items():
        predicted = float(predicted)
        if predicted <= 0:
            continue
        if overall_confidence >= 0.8:
            factor = 0.50
        elif overall_confidence >= 0.6:
            factor = 0.75
        else:
            factor = 1.00
        lo = max(0.0, round((predicted * (1 - factor)) * 2) / 2)
        hi = round((predicted * (1 + factor)) * 2) / 2
        out[dept] = {
            "predicted": predicted,
            "min": lo,
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


class CorrectionRequest(BaseModel):
    description: str
    final_total_days: float
    final_departments: Dict[str, float] = Field(default_factory=dict)
    user_id: str = "default"
    notes: str = ""
    ai_total_days: Optional[float] = None


class FlagRequest(BaseModel):
    description: str
    flag_type: str
    notes: str = ""
    user_id: str = "default"
    ai_total_days: Optional[float] = None
    ai_shot_type: Optional[str] = None
    ai_departments: Dict[str, float] = Field(default_factory=dict)


# ── App factory ────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(title="VFX Estimator", version="0.3.0")

    # Allow the local HTML file to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ──────────────────────────────────────────────────────────────

    @app.get("/health")
    def health() -> Dict[str, Any]:
        s = get_settings()
        svc = get_service()
        corrections = svc.corrections.load()
        flags = get_flags().load()
        return {
            "ok": True,
            "training_shots": len(svc.training),
            "corrections": len(corrections),
            "flags": len(flags),
            "mode_default": s.estimate_mode,
            "gemini_configured": bool(s.resolved_gemini_key()),
            "legacy_numeric": s.use_legacy_numeric,
            "day_rate": s.day_rate,
            "xata_mode": XataShotSearch(s).mode,
            "xata_corrections": svc.corrections.storage_backend,
        }

    # ── Estimate ────────────────────────────────────────────────────────────

    @app.post("/estimate")
    def estimate(req: EstimateRequest) -> Dict[str, Any]:
        if not req.description.strip():
            raise HTTPException(400, "description required")
        svc = get_service()

        # Reload flags into Gemini estimator so they're fresh
        if svc.gemini:
            svc.gemini.flags = get_flags()

        est = svc.estimate(req.description, pre_qual=req.pre_qual, mode=req.mode)
        result = est.model_dump()

        # Compute per-department slider ranges for HITL UI
        dept = {k: float(v) for k, v in (est.dept_days or {}).items() if float(v or 0) > 0}
        result["adjustment_ranges"] = _compute_adjustment_ranges(dept, est.confidence)
        result["dept_confidence"] = {}

        return result

    # ── Corrections ─────────────────────────────────────────────────────────

    @app.post("/corrections")
    def add_correction(req: CorrectionRequest) -> Dict[str, Any]:
        svc = get_service()
        svc.record_correction(UserCorrection.model_validate(req.model_dump()))
        total = len(svc.corrections.load())
        return {
            "ok": True,
            "count": total,
            "message": f"Correction saved. {total} total — retrieval index updated.",
        }

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

    # ── Flags ───────────────────────────────────────────────────────────────

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

    # ── Tuning ──────────────────────────────────────────────────────────────

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
        _service = EstimatorService(get_settings())
        return {"ok": True, "path": str(path)}

    return app
