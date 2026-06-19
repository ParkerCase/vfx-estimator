"""Per-department day rates and shot cost calculation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from vfx_estimator.types import BID_DEPT_MAP, bid_departments_to_internal

# Internal department keys with typical VFX facility rates ($/day).
DEFAULT_DEPT_RATES: Dict[str, float] = {
    "camera_track": 700.0,
    "matchmove": 700.0,
    "layout": 800.0,
    "animation": 900.0,
    "cfx": 850.0,
    "fx": 850.0,
    "lighting": 800.0,
    "dmp": 750.0,
    "comp_paint": 750.0,
    "comp_roto": 700.0,
    "compositing": 750.0,
    "crowds": 900.0,
    "enviro": 800.0,
    "ai": 750.0,
    "prep": 650.0,
    "obj_track": 700.0,
}

RATE_DEPARTMENTS: tuple[str, ...] = tuple(DEFAULT_DEPT_RATES.keys())


def _to_float(val: Any) -> Optional[float]:
    try:
        f = float(val)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def normalize_dept_key(key: str) -> str:
    """Map bid column names or aliases to internal department keys."""
    k = str(key or "").strip()
    if not k:
        return k
    ku = k.upper()
    if ku in BID_DEPT_MAP:
        return BID_DEPT_MAP[ku]
    if k in BID_DEPT_MAP.values():
        return k
    return k


def build_dept_rates(
    *,
    fallback: float = 700.0,
    defaults: Optional[Mapping[str, float]] = None,
    tuning: Optional[Mapping[str, Any]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Merge default, tuning, and request overrides into a full rate table."""
    base = dict(defaults or DEFAULT_DEPT_RATES)
    fb = _to_float(fallback) or 700.0
    out = {dept: float(base.get(dept, fb)) for dept in RATE_DEPARTMENTS}

    for source in (tuning or {}, overrides or {}):
        if not source:
            continue
        normalized = bid_departments_to_internal(
            {str(k): float(v) for k, v in source.items() if _to_float(v) is not None}
        )
        for dept, rate in normalized.items():
            dept = normalize_dept_key(dept)
            r = _to_float(rate)
            if r is not None and dept in out:
                out[dept] = r
    return out


def rate_for_dept(dept: str, rates: Mapping[str, float], *, fallback: float) -> float:
    key = normalize_dept_key(dept)
    r = _to_float(rates.get(key))
    if r is not None:
        return r
    r = _to_float(rates.get(dept))
    if r is not None:
        return r
    return float(fallback)


def compute_dept_costs(
    dept_days: Mapping[str, float],
    rates: Mapping[str, float],
    *,
    fallback_rate: float = 700.0,
) -> Dict[str, float]:
    """Per-department labor cost (days × rate). Keys are internal dept names."""
    internal = bid_departments_to_internal(dict(dept_days))
    costs: Dict[str, float] = {}
    for dept, days in internal.items():
        d = float(days or 0)
        if d <= 0:
            continue
        key = normalize_dept_key(dept)
        line = round(d * rate_for_dept(key, rates, fallback=fallback_rate), 2)
        costs[key] = costs.get(key, 0.0) + line
    return costs


def compute_shot_cost(
    dept_days: Mapping[str, float],
    rates: Mapping[str, float],
    *,
    fallback_rate: float = 700.0,
    mandays_fallback: float = 0.0,
    allot: int = 1,
) -> float:
    """
    Shot labor cost from department mandays × per-dept rates.

    If no department breakdown exists, falls back to mandays_fallback × fallback_rate.
    """
    dept_costs = compute_dept_costs(dept_days, rates, fallback_rate=fallback_rate)
    if dept_costs:
        per_shot = sum(dept_costs.values())
    else:
        per_shot = float(mandays_fallback or 0) * float(fallback_rate)
    return round(per_shot * max(1, int(allot or 1)), 2)
