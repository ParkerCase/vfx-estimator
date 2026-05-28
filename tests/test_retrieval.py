"""
Tests for ShotRetrievalIndex:
  - Retrieval returns results in correct order
  - Correction weighting (2x boost)
  - Orphaned corrections (corrections for shots not in training)
  - Cosine similarity ordering
  - Empty corpus edge case
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from vfx_estimator.config import get_settings
from vfx_estimator.data.loaders import TrainingShot
from vfx_estimator.learning.corrections import CorrectionsStore
from vfx_estimator.retrieval.index import ShotRetrievalIndex
from vfx_estimator.types import UserCorrection


def _make_shots(descriptions_and_days: list) -> list:
    return [
        TrainingShot(description=d, mandays=m, project="test", cost=m * 700)
        for d, m in descriptions_and_days
    ]


class TestRetrieval:
    def test_returns_requested_k(self):
        shots = _make_shots([
            ("wire removal from stunt", 5.0),
            ("CG castle establishing", 18.0),
            ("hero creature closeup", 22.0),
            ("greenscreen comp", 7.0),
            ("background crowd", 12.0),
        ])
        idx = ShotRetrievalIndex(shots)
        hits = idx.query("wire removal cleanup", top_k=3)
        assert len(hits) == 3

    def test_most_similar_ranked_first(self):
        shots = _make_shots([
            ("wire removal from stunt double", 5.0),
            ("CG environment castle establishing shot", 18.0),
            ("particle FX fire explosion", 14.0),
        ])
        idx = ShotRetrievalIndex(shots)
        hits = idx.query("wire removal stunt cleanup")
        assert hits[0].description == "wire removal from stunt double"

    def test_similarity_scores_descending(self):
        shots = _make_shots([
            ("CG creature animation hero closeup", 20.0),
            ("wire removal simple shot", 4.0),
            ("background environment plate", 6.0),
        ])
        idx = ShotRetrievalIndex(shots)
        hits = idx.query("CG creature hero animation lighting comp")
        sims = [h.similarity for h in hits]
        assert sims == sorted(sims, reverse=True), "Hits must be sorted descending by similarity"

    def test_empty_corpus_returns_empty(self):
        idx = ShotRetrievalIndex([])
        hits = idx.query("CG establishing shot")
        assert hits == []

    def test_median_mandays_returns_positive(self):
        shots = _make_shots([
            ("CG castle shot", 18.0),
            ("CG castle wide shot", 20.0),
            ("CG environment establishing", 22.0),
        ])
        idx = ShotRetrievalIndex(shots)
        med = idx.median_mandays("CG castle establishing shot")
        assert med > 0.0

    def test_median_mandays_empty_returns_zero(self):
        idx = ShotRetrievalIndex([])
        assert idx.median_mandays("anything") == 0.0


class TestCorrectionWeighting:
    def test_correction_source_labeled(self):
        shots = _make_shots([("wire removal background", 5.0)])
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            corr_path = Path(f.name)

        s = get_settings()
        store = CorrectionsStore(path=corr_path, settings=s)
        store.append(UserCorrection(
            description="wire removal stunt hero shot",
            final_total_days=7.0,
            ai_total_days=5.0,
        ))
        idx = ShotRetrievalIndex(shots, settings=s, corrections=store)
        hits = idx.query("wire removal stunt hero shot", top_k=5)
        sources = [h.source for h in hits]
        assert "correction" in sources, "Correction shot should appear in retrieval results"
        corr_path.unlink(missing_ok=True)

    def test_correction_outranks_training_for_exact_match(self):
        """A user correction for a near-identical description should rank highest."""
        shots = _make_shots([
            ("CG environment background", 8.0),
            ("particle FX explosion", 15.0),
        ])
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            corr_path = Path(f.name)

        s = get_settings()
        store = CorrectionsStore(path=corr_path, settings=s)
        store.append(UserCorrection(
            description="CG castle establishing shot wide",
            final_total_days=21.0,
        ))
        training = _make_shots([("CG castle establishing shot wide", 18.0)])
        idx = ShotRetrievalIndex(training + shots, settings=s, corrections=store)
        hits = idx.query("CG castle establishing shot wide", top_k=5)
        top = hits[0]
        assert top.source == "correction", (
            f"Correction should outrank training for identical description; got source={top.source}"
        )
        corr_path.unlink(missing_ok=True)

    def test_corrections_count_increases_index(self):
        shots = _make_shots([("wire removal", 5.0)])
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            corr_path = Path(f.name)

        s = get_settings()
        store = CorrectionsStore(path=corr_path, settings=s)
        idx_before = ShotRetrievalIndex(shots, settings=s)
        n_before = len(idx_before)

        store.append(UserCorrection(description="new correction shot", final_total_days=9.0))
        idx_after = ShotRetrievalIndex(shots, settings=s, corrections=store)
        n_after = len(idx_after)

        assert n_after == n_before + 1, "Adding a correction should grow the index by 1"
        corr_path.unlink(missing_ok=True)


class TestOrphanedCorrections:
    """
    Orphaned corrections: user corrections referencing descriptions not in
    training data. These should still be indexed and used for retrieval —
    the system should not silently drop them.
    """

    def test_orphaned_correction_still_indexed(self):
        shots = _make_shots([("CG environment background plate", 8.0)])
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            corr_path = Path(f.name)

        s = get_settings()
        store = CorrectionsStore(path=corr_path, settings=s)
        # This description does NOT appear in training — it's "orphaned"
        store.append(UserCorrection(
            description="entirely novel shot type with no training analog",
            final_total_days=12.0,
        ))
        idx = ShotRetrievalIndex(shots, settings=s, corrections=store)
        assert len(idx) == 2, "Orphaned correction should be indexed even with no training match"
        corr_path.unlink(missing_ok=True)

    def test_orphaned_correction_retrievable(self):
        shots = _make_shots([("wire removal", 5.0)])
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            corr_path = Path(f.name)

        s = get_settings()
        store = CorrectionsStore(path=corr_path, settings=s)
        store.append(UserCorrection(
            description="hero digi-double animation crowd battle sequence",
            final_total_days=30.0,
        ))
        idx = ShotRetrievalIndex(shots, settings=s, corrections=store)
        hits = idx.query("hero digi-double animation crowd battle sequence")
        sources = {h.source for h in hits}
        assert "correction" in sources
        corr_path.unlink(missing_ok=True)
