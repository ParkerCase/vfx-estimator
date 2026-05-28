"""TF-IDF retrieval over training + user corrections (+ optional Xata hits)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from vfx_estimator.config import Settings, get_settings
from vfx_estimator.data.loaders import TrainingShot
from vfx_estimator.learning.corrections import CorrectionsStore
from vfx_estimator.types import SimilarShot


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
        self.rows: List[Dict[str, Any]] = []

        for s in training:
            self.rows.append(
                {
                    "description": s.description,
                    "mandays": float(s.mandays),
                    "cost": float(s.cost or s.mandays * self.settings.day_rate),
                    "source": "training",
                    "weight": 1.0,
                    "project": s.project,
                    "dept_days": dict(s.dept_days or {}),
                }
            )

        store = corrections or CorrectionsStore(settings=self.settings)
        for c in store.as_training_rows():
            self.rows.append(
                {
                    "description": c["description"],
                    "mandays": float(c["mandays"]),
                    "cost": float(c["cost"]),
                    "source": "correction",
                    "weight": self.correction_boost,
                    "project": "user_correction",
                    "dept_days": c.get("dept_days") or {},
                }
            )

        for r in extra_rows or []:
            self.rows.append(
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

        descriptions = [r["description"] for r in self.rows if r["description"]]
        self._vectorizer = TfidfVectorizer(max_features=800, ngram_range=(1, 2), min_df=1)
        if descriptions:
            self._matrix = self._vectorizer.fit_transform(descriptions)
        else:
            self._matrix = None

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
                )
            )
        return out

    def median_mandays(self, description: str, top_k: int = 5) -> float:
        hits = self.query(description, top_k=top_k)
        if not hits:
            return 0.0
        vals = [h.mandays for h in hits if h.mandays > 0]
        return float(np.median(vals)) if vals else 0.0
