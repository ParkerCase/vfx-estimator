"""Retrieval-based numeric predictor when legacy pipeline is not vendored."""

from __future__ import annotations

from typing import Any, Dict, Optional

from vfx_estimator.retrieval.index import ShotRetrievalIndex
from vfx_estimator.types import BidPreQual

_TIER_MULT = {
    "low": 0.85,
    "indie_low": 0.85,
    "mid": 1.0,
    "medium": 1.0,
    "mid_tier_tv": 1.0,
    "high": 1.15,
    "premium_tv": 1.15,
    "hero": 1.35,
    "feature_film": 1.35,
}
_COMPLEX_MULT = {
    "low": 0.9,
    "medium": 1.0,
    "mid": 1.0,
    "high": 1.2,
    "very_high": 1.35,
    "hero": 1.4,
}


def _round_half(x: float) -> float:
    return round(float(x) * 2) / 2


def _prequal_multiplier(pre_qual: Optional[Dict[str, Any]]) -> float:
    if not pre_qual:
        return 1.0
    m = 1.0
    tier = str(pre_qual.get("bid_scale_tier") or "").lower()
    if tier in _TIER_MULT:
        m *= _TIER_MULT[tier]
    band = str(pre_qual.get("complexity_band") or "").lower()
    if band in _COMPLEX_MULT:
        m *= _COMPLEX_MULT[band]
    return m


def predict_with_index(
    index: ShotRetrievalIndex,
    description: str,
    *,
    pre_qual: Optional[Dict[str, Any]] = None,
    day_rate: float = 700.0,
) -> Dict[str, Any]:
    """Weighted k-NN mandays + dept split from neighbors (fallback for legacy pipeline)."""
    hits = index.query(description.strip(), top_k=10)
    if not hits:
        base = 2.0
    else:
        weights = [max(0.01, h.similarity) for h in hits]
        wsum = sum(weights)
        base = sum(h.mandays * w / wsum for h, w in zip(hits, weights))

    mult = _prequal_multiplier(pre_qual)
    per_shot = max(0.25, _round_half(base * mult))

    dept: Dict[str, float] = {}
    if hits:
        for key in (
            "comp_paint",
            "comp_roto",
            "layout",
            "lighting",
            "animation",
            "fx",
            "cam_track",
            "matchmove",
        ):
            vals = []
            weights_d = []
            for h in hits:
                row = next((r for r in index.rows if r["description"] == h.description), None)
                if not row:
                    continue
                d = (row.get("dept_days") or {}).get(key)
                if d and float(d) > 0:
                    vals.append(float(d))
                    weights_d.append(max(0.01, h.similarity))
            if vals and weights_d:
                dept[key] = _round_half(
                    sum(v * w for v, w in zip(vals, weights_d)) / sum(weights_d) * mult
                )

    allot = 1
    if pre_qual:
        try:
            allot = max(1, int(pre_qual.get("allotment_n") or 1))
        except (TypeError, ValueError):
            allot = 1

    total = _round_half(per_shot * allot)
    return {
        "per_shot_mandays": per_shot,
        "total_mandays": total,
        "cost": round(total * day_rate, 2),
        "dept_days": dept,
        "screenplay_scene_matches": [],
        "predictor": "retrieval_fallback",
    }


def make_predictor(index: ShotRetrievalIndex):
    def _predict(
        description: str,
        pre_qual: Optional[Dict[str, Any]] = None,
        day_rate: float = 700.0,
    ) -> Dict[str, Any]:
        return predict_with_index(index, description, pre_qual=pre_qual, day_rate=day_rate)

    return _predict
