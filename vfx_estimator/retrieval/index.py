"""TF-IDF retrieval over training + user corrections (+ optional Xata hits)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from vfx_estimator.config import Settings, get_settings
from vfx_estimator.data.loaders import TrainingShot, load_training_shots
from vfx_estimator.learning.corrections import CorrectionsStore
from vfx_estimator.types import SimilarShot

_cached_index: Optional["ShotRetrievalIndex"] = None

PRESET_DEPT_TO_BID = {
    "camera_track": "CAMERA",
    "matchmove": "MATCHMOVE",
    "layout": "LAYOUT",
    "animation": "ANIM",
    "cfx": "CFX",
    "fx": "FX",
    "lighting": "LGT",
    "dmp": "DMP",
    "comp_paint": "COMP PAINT",
    "comp_roto": "COMP ROTO",
    "compositing": "COMP",
}


def load_preset_training_rows(settings: Settings) -> List[Dict[str, Any]]:
    """Convert DB shot presets into retrieval rows weighted like training data."""
    pg_url = settings.resolved_xata_postgres_url()
    if not pg_url:
        return []
    try:
        from vfx_estimator.integrations.xata import load_presets

        presets = load_presets(pg_url)
    except Exception:
        return []

    rows: List[Dict[str, Any]] = []
    for key, preset in presets.items():
        total = float(preset.get("total") or 0)
        description = str(preset.get("description") or "").strip()
        if total <= 0 or not description:
            continue
        dept_days = {
            bid_key: float(preset.get(internal_key) or 0)
            for internal_key, bid_key in PRESET_DEPT_TO_BID.items()
            if float(preset.get(internal_key) or 0) > 0
        }
        rows.append(
            {
                "description": description,
                "mandays": total,
                "cost": total * float(settings.day_rate),
                "source": "preset",
                "weight": 1.0,
                "project": key,
                "dept_days": dept_days,
            }
        )
    return rows


def build_index(
    settings: Optional[Settings] = None,
    corrections: Optional[CorrectionsStore] = None,
) -> "ShotRetrievalIndex":
    """Load training + corrections and fit TF-IDF once into the module cache."""
    global _cached_index
    settings = settings or get_settings()
    training = load_training_shots(settings)
    store = corrections or CorrectionsStore(settings=settings)
    preset_rows = load_preset_training_rows(settings)
    _cached_index = ShotRetrievalIndex(
        training,
        settings=settings,
        corrections=store,
        extra_rows=preset_rows,
    )
    return _cached_index


def get_index(
    settings: Optional[Settings] = None,
    corrections: Optional[CorrectionsStore] = None,
) -> "ShotRetrievalIndex":
    """Return the cached index, rebuilding only if empty."""
    global _cached_index
    if _cached_index is None:
        build_index(settings=settings, corrections=corrections)
    return _cached_index


def invalidate_index() -> None:
    """Clear the cached index so the next access rebuilds."""
    global _cached_index
    _cached_index = None


def _rows_from_sources(
    training: Sequence[TrainingShot],
    *,
    settings: Settings,
    corrections: Optional[CorrectionsStore] = None,
    extra_rows: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    correction_boost = float(settings.correction_boost)

    for s in training:
        rows.append(
            {
                "description": s.description,
                "mandays": float(s.mandays),
                "cost": float(s.cost or s.mandays * settings.day_rate),
                "source": "training",
                "weight": 1.0,
                "project": s.project,
                "dept_days": dict(s.dept_days or {}),
            }
        )

    if corrections is not None:
        for c in corrections.as_training_rows():
            rows.append(
                {
                    "description": c["description"],
                    "mandays": float(c["mandays"]),
                    "cost": float(c["cost"]),
                    "source": "correction",
                    "weight": correction_boost,
                    "project": "user_correction",
                    "dept_days": c.get("dept_days") or {},
                }
            )

    for r in extra_rows or []:
        rows.append(
            {
                "description": r.get("description", ""),
                "mandays": float(r.get("mandays") or 0),
                "cost": float(r.get("cost") or 0),
                "source": r.get("source", "xata"),
                "weight": float(r.get("weight", 1.2)),
                "project": r.get("project"),
                "dept_days": r.get("dept_days") or {},
            }
        )

    return rows


class ShotRetrievalIndex:
    def __init__(
        self,
        training: Sequence[TrainingShot],
        *,
        settings: Optional[Settings] = None,
        corrections: Optional[CorrectionsStore] = None,
        extra_rows: Optional[List[Dict[str, Any]]] = None,
    ):
        self.settings = settings or get_settings()
        self.correction_boost = float(self.settings.correction_boost)
        self.rows = _rows_from_sources(
            training,
            settings=self.settings,
            corrections=corrections,
            extra_rows=extra_rows,
        )
        self._fit_vectors()

    def _fit_vectors(self) -> None:
        descriptions = [r["description"] for r in self.rows if r["description"]]
        self._vectorizer = TfidfVectorizer(max_features=800, ngram_range=(1, 2), min_df=1)
        if descriptions:
            self._matrix = self._vectorizer.fit_transform(descriptions)
        else:
            self._matrix = None

    def with_extra_rows(self, extra_rows: List[Dict[str, Any]]) -> "ShotRetrievalIndex":
        """Return an index overlay with extra rows transformed via the fitted vectorizer."""
        if not extra_rows:
            return self

        overlay = object.__new__(ShotRetrievalIndex)
        overlay.settings = self.settings
        overlay.correction_boost = self.correction_boost
        overlay._vectorizer = self._vectorizer
        overlay.rows = list(self.rows)
        for r in extra_rows:
            overlay.rows.append(
                {
                    "description": r.get("description", ""),
                    "mandays": float(r.get("mandays") or 0),
                    "cost": float(r.get("cost") or 0),
                    "source": r.get("source", "xata"),
                    "weight": float(r.get("weight", 1.2)),
                    "project": r.get("project"),
                    "dept_days": r.get("dept_days") or {},
                }
            )

        new_descs = [r.get("description", "") for r in extra_rows if r.get("description")]
        if new_descs and self._vectorizer is not None:
            extra_matrix = self._vectorizer.transform(new_descs)
            overlay._matrix = (
                vstack([self._matrix, extra_matrix]) if self._matrix is not None else extra_matrix
            )
        else:
            overlay._matrix = self._matrix
        return overlay

    def __len__(self) -> int:
        return len(self.rows)

    def query(self, description: str, top_k: Optional[int] = None) -> List[SimilarShot]:
        k = top_k or self.settings.retrieval_top_k
        if not self.rows or self._matrix is None:
            return []
        qv = self._vectorizer.transform([description])
        sims = cosine_similarity(qv, self._matrix)[0]
        weighted = [float(s) * float(self.rows[i]["weight"]) for i, s in enumerate(sims)]
        order = np.argsort(weighted)[-k:][::-1]
        out: List[SimilarShot] = []
        for idx in order:
            r = self.rows[int(idx)]
            out.append(
                SimilarShot(
                    description=r["description"],
                    mandays=float(r["mandays"]),
                    cost=float(r["cost"]),
                    similarity=float(sims[int(idx)]),
                    source=r["source"],
                    project=r.get("project"),
                    dept_days=r.get("dept_days") or {},
                )
            )
        return out

    def median_mandays(self, description: str, top_k: int = 5) -> float:
        hits = self.query(description, top_k=top_k)
        if not hits:
            return 0.0
        vals = [h.mandays for h in hits if h.mandays > 0]
        return float(np.median(vals)) if vals else 0.0
