"""Evaluation metrics for mandays predictions."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def within_band(preds: Sequence[float], actuals: Sequence[float], low: float, high: float) -> float:
    p = np.asarray(preds, dtype=float)
    a = np.asarray(actuals, dtype=float)
    ok = (a > 1e-9) & (p >= a * low) & (p <= a * high)
    return float(np.mean(ok))


def metrics_dict(preds: Sequence[float], actuals: Sequence[float]) -> Dict[str, float]:
    p = np.asarray(preds, dtype=float)
    a = np.asarray(actuals, dtype=float)
    n = len(a)
    if n == 0:
        return {"n": 0, "mae": 0.0, "within_10pct": 0.0, "within_20pct": 0.0, "within_1day": 0.0}
    mae = float(np.mean(np.abs(p - a)))
    return {
        "n": n,
        "mae": round(mae, 4),
        "within_10pct": round(within_band(p, a, 0.9, 1.1) * 100, 2),
        "within_20pct": round(within_band(p, a, 0.8, 1.2) * 100, 2),
        "within_1day": round(float(np.mean(np.abs(p - a) <= 1.0)) * 100, 2),
    }
