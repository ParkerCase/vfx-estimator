"""Load training and holdout rows from JSON / CSV."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from vfx_estimator.config import Settings, get_settings


@dataclass
class TrainingShot:
    description: str
    mandays: float
    project: str = "unknown"
    cost: float = 0.0
    dept_days: Dict[str, float] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.dept_days is None:
            self.dept_days = {}
        if self.cost <= 0 and self.mandays > 0:
            self.cost = self.mandays * 700.0


def _mandays_from_record(rec: Dict[str, Any], day_rate: float) -> float:
    for key in ("total_mandays", "mandays", "per_shot_mandays"):
        v = rec.get(key)
        if v is not None:
            try:
                f = float(v)
                if f > 0:
                    return f
            except (TypeError, ValueError):
                pass
    for key in ("cost_per_shot", "cost", "total_cost"):
        v = rec.get(key)
        if v is not None:
            try:
                c = float(v)
                if c > 0 and day_rate > 0:
                    return c / day_rate
            except (TypeError, ValueError):
                pass
    return 0.0


def _description_from_record(rec: Dict[str, Any]) -> str:
    for key in (
        "shot_description",
        "technical_description",
        "description",
        "vfx_assumptions",
        "VFX_ASSUMPTIONS",
    ):
        v = rec.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def load_training_shots(settings: Optional[Settings] = None) -> List[TrainingShot]:
    settings = settings or get_settings()
    path = settings.resolved_training_json()
    if not path.exists():
        raise FileNotFoundError(f"Training JSON not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    records: List[Dict[str, Any]]
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict):
        records = raw.get("shots") or raw.get("records") or raw.get("data") or []
        if not records and "projects" in raw:
            records = []
            for proj, rows in raw["projects"].items():
                if isinstance(rows, list):
                    for r in rows:
                        if isinstance(r, dict):
                            r = {**r, "project": r.get("project") or proj}
                            records.append(r)
    else:
        records = []

    out: List[TrainingShot] = []
    dr = settings.day_rate
    for rec in records:
        if not isinstance(rec, dict):
            continue
        desc = _description_from_record(rec)
        md = _mandays_from_record(rec, dr)
        if not desc or md <= 0:
            continue
        dept = {}
        for col, key in (
            ("comp_paint_days", "comp_paint"),
            ("comp_roto_days", "comp_roto"),
            ("animation_days", "animation"),
            ("layout_days", "layout"),
            ("cam_track_days", "cam_track"),
            ("matchmove_days", "matchmove"),
        ):
            try:
                v = float(rec.get(col) or 0)
                if v > 0:
                    dept[key] = v
            except (TypeError, ValueError):
                pass
        out.append(
            TrainingShot(
                description=desc,
                mandays=md,
                project=str(rec.get("project") or "unknown"),
                cost=md * dr,
                dept_days=dept,
            )
        )
    return out


def load_byzantine_holdout(settings: Optional[Settings] = None) -> List[Dict[str, Any]]:
    """Rows with description + actual_per_shot_mandays."""
    settings = settings or get_settings()
    path = settings.resolved_byzantine_csv()
    df = pd.read_csv(path)
    desc_col = next(
        (c for c in ("technical_description", "shot_description", "description") if c in df.columns),
        None,
    )
    if not desc_col:
        raise ValueError(f"No description column in {path}")

    rows: List[Dict[str, Any]] = []
    dr = settings.day_rate
    for _, r in df.iterrows():
        desc = str(r.get(desc_col, "")).strip()
        if not desc:
            continue
        total_md = r.get("total_mandays")
        try:
            total_md = float(total_md) if total_md is not None and str(total_md).strip() else None
        except (TypeError, ValueError):
            total_md = None
        n_shots = 1
        if "number_of_shots" in df.columns:
            try:
                n_shots = max(1, int(float(r.get("number_of_shots") or 1)))
            except (TypeError, ValueError):
                n_shots = 1
        if total_md is not None and total_md > 0:
            actual = total_md / n_shots
        else:
            cost = 0.0
            for cc in ("cost_per_shot", "total_cost", "actual_cost", "cost"):
                if cc in df.columns:
                    try:
                        cost = float(r.get(cc) or 0)
                        if cost > 0:
                            break
                    except (TypeError, ValueError):
                        pass
            if cost <= 0:
                continue
            actual = (cost / dr) / n_shots
        rows.append({"description": desc, "actual_per_shot_mandays": actual, "n_shots": n_shots})
    if not rows:
        raise ValueError(f"No labeled rows in {path}")
    return rows
