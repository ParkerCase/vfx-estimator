"""Orchestrates numeric + retrieval + Gemini + screenplay + corrections."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from vfx_estimator.config import Settings, get_settings
from vfx_estimator.data.loaders import load_training_shots
from vfx_estimator.integrations.xata import XataShotSearch
from vfx_estimator.learning.corrections import CorrectionsStore
from vfx_estimator.llm.mandays_rag import GeminiMandaysEstimator
from vfx_estimator.numeric.legacy_bridge import LegacyNumericEstimator
from vfx_estimator.retrieval.index import ShotRetrievalIndex
from vfx_estimator.screenplay.scene_match import fdx_xml_to_plaintext, screenplay_augment_with_metadata
from vfx_estimator.types import BidPreQual, ShotEstimate, UserCorrection


def _round_half(x: float) -> float:
    return round(float(x) * 2) / 2


class EstimatorService:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.training = load_training_shots(self.settings)
        self._xata = XataShotSearch(self.settings)
        self.corrections = CorrectionsStore(settings=self.settings)
        self._rebuild_index()

        self.legacy = LegacyNumericEstimator(self.settings)
        self.legacy.set_retrieval_index(self.index)
        self.gemini: Optional[GeminiMandaysEstimator] = None
        if self.settings.resolved_gemini_key():
            self.gemini = GeminiMandaysEstimator(self.index, self.settings)

    def _rebuild_index(self) -> None:
        self.index = ShotRetrievalIndex(self.training, settings=self.settings, corrections=self.corrections)

    def reload_corrections(self) -> None:
        self._rebuild_index()
        self.legacy.set_retrieval_index(self.index)
        if self.gemini:
            self.gemini.index = self.index

    def _screenplay_text(self, pre_qual: Optional[BidPreQual]) -> str:
        if not pre_qual:
            return ""
        if pre_qual.screenplay_full_text and pre_qual.screenplay_full_text.strip():
            return pre_qual.screenplay_full_text.strip()
        for attr in ("screenplay_text_path", "screenplay_fdx_path"):
            pth = getattr(pre_qual, attr, None)
            if not pth:
                continue
            fp = Path(str(pth)).expanduser()
            if not fp.is_file() or fp.stat().st_size > 4_000_000:
                continue
            raw = fp.read_text(encoding="utf-8", errors="replace")
            if attr == "screenplay_fdx_path":
                return fdx_xml_to_plaintext(raw)
            return raw
        return ""

    def _augment_description(self, description: str, pre_qual: Optional[BidPreQual]) -> tuple[str, List[Dict[str, Any]]]:
        sp = self._screenplay_text(pre_qual)
        if not sp:
            return description.strip(), []
        return screenplay_augment_with_metadata(description, sp)

    def estimate(
        self,
        description: str,
        *,
        pre_qual: Optional[BidPreQual] = None,
        mode: Optional[str] = None,
    ) -> ShotEstimate:
        mode = mode or self.settings.estimate_mode
        desc_user = description.strip()
        desc_aug, sp_matches = self._augment_description(desc_user, pre_qual)

        xata_hits = self._xata.search(desc_user, top_k=5) if self._xata.enabled else []
        if xata_hits:
            self.index = ShotRetrievalIndex(
                self.training,
                settings=self.settings,
                corrections=self.corrections,
                extra_rows=xata_hits,
            )
            if self.gemini:
                self.gemini.index = self.index

        similar = self.index.query(desc_user)
        retrieval_med = self.index.median_mandays(desc_user)

        numeric_md: Optional[float] = None
        gemini_md: Optional[float] = None
        reasoning = ""
        confidence = 0.5
        dept: Dict[str, float] = {}

        pq_aug = pre_qual
        if pre_qual and sp_matches:
            pq_aug = pre_qual.model_copy(deep=True)
            pq_aug.screenplay_full_text = None
            pq_aug.screenplay_text_path = None
            pq_aug.screenplay_fdx_path = None

        if mode in ("numeric_only", "hybrid") and self.settings.use_legacy_numeric:
            leg = self.legacy.predict(desc_aug if pq_aug else desc_user, pq_aug)
            numeric_md = float(leg.get("per_shot_mandays") or 0)
            dept = {k: float(v) for k, v in (leg.get("dept_days") or {}).items()}
            sp_matches = leg.get("screenplay_scene_matches") or sp_matches

        if mode in ("gemini_rag", "hybrid") and self.gemini:
            g = self.gemini.estimate(desc_aug, pre_qual=pre_qual)
            gemini_md = float(g.get("total_days") or 0)
            reasoning = str(g.get("reasoning") or "")
            confidence = float(g.get("confidence") or 0.5)
            if not dept and isinstance(g.get("departments"), dict):
                for k, v in g["departments"].items():
                    if isinstance(v, dict):
                        dept[k] = float(v.get("days") or 0)

        if mode == "numeric_only":
            final = numeric_md or retrieval_med or 1.0
        elif mode == "gemini_rag":
            final = gemini_md or retrieval_med or numeric_md or 1.0
        else:  # hybrid
            # Numeric owns the TOTAL (66% accurate on Byzantine, zero leakage).
            # Gemini owns the DEPT BREAKDOWN — shot classification + which depts apply.
            # Gemini's day estimates are consistently too low (strong wrong priors),
            # so we scale its dept proportions to the numeric total rather than
            # blending totals (which dragged hybrid below numeric_only).
            final = numeric_md or retrieval_med or 1.0
            if dept and gemini_md and gemini_md > 0 and final > 0:
                scale = final / gemini_md
                dept = {
                    k: max(0.5, _round_half(v * scale))
                    for k, v in dept.items()
                    if v > 0
                }

        final = max(0.25, _round_half(final))
        allot = max(1, int((pre_qual.allotment_n if pre_qual else 1) or 1))
        dr = self.settings.day_rate

        return ShotEstimate(
            description=desc_user,
            per_shot_mandays=final,
            total_mandays=_round_half(final * allot),
            cost=round(final * allot * dr, 2),
            confidence=confidence,
            mode=mode,
            reasoning=reasoning,
            dept_days=dept,
            similar_shots=similar,
            screenplay_scene_matches=sp_matches,
            numeric_mandays=numeric_md,
            gemini_mandays=gemini_md,
            retrieval_median_mandays=retrieval_med if retrieval_med > 0 else None,
        )

    def record_correction(self, correction: UserCorrection) -> None:
        self.corrections.append(correction)
        self.reload_corrections()
