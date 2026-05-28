"""
Tests for CorrectionsStore JSONL persistence.

Covers:
  - Write + read roundtrip
  - Multiple corrections accumulate correctly
  - as_training_rows produces correct cost calculation (mandays * day_rate)
  - Missing file returns empty list (no crash)
  - Timestamp is written
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from vfx_estimator.config import get_settings
from vfx_estimator.learning.corrections import CorrectionsStore
from vfx_estimator.types import UserCorrection


def _tmp_store() -> tuple[CorrectionsStore, Path]:
    s = get_settings()
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        p = Path(f.name)
    p.unlink()  # delete so store starts fresh
    return CorrectionsStore(path=p, settings=s), p


class TestCorrectionsStore:
    def test_empty_when_file_missing(self):
        store, p = _tmp_store()
        assert store.load() == []
        p.unlink(missing_ok=True)

    def test_append_and_load_one(self):
        store, p = _tmp_store()
        c = UserCorrection(
            description="CG castle establishing shot",
            final_total_days=21.0,
            final_departments={"lighting": 9.0, "layout": 3.0},
            user_id="che",
            notes="Hero work",
            ai_total_days=17.0,
        )
        store.append(c)
        loaded = store.load()
        assert len(loaded) == 1
        assert loaded[0].description == "CG castle establishing shot"
        assert loaded[0].final_total_days == 21.0
        p.unlink(missing_ok=True)

    def test_append_multiple_accumulates(self):
        store, p = _tmp_store()
        for i in range(5):
            store.append(UserCorrection(description=f"shot {i}", final_total_days=float(i + 5)))
        assert len(store.load()) == 5
        p.unlink(missing_ok=True)

    def test_timestamp_written_to_jsonl(self):
        store, p = _tmp_store()
        store.append(UserCorrection(description="test shot", final_total_days=10.0))
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw.strip())
        assert "timestamp" in data, "timestamp must be written to JSONL"
        p.unlink(missing_ok=True)

    def test_as_training_rows_cost_calculation(self):
        """Cost in training rows must equal final_total_days * day_rate."""
        store, p = _tmp_store()
        s = get_settings()
        store.append(UserCorrection(description="wire removal", final_total_days=5.0))
        rows = store.as_training_rows()
        assert len(rows) == 1
        expected_cost = 5.0 * s.day_rate
        assert rows[0]["cost"] == pytest.approx(expected_cost), (
            f"Expected cost {expected_cost}, got {rows[0]['cost']}"
        )
        p.unlink(missing_ok=True)

    def test_as_training_rows_includes_dept_days(self):
        store, p = _tmp_store()
        store.append(UserCorrection(
            description="CG environment wide",
            final_total_days=14.0,
            final_departments={"lighting": 6.0, "layout": 3.0, "comp_paint": 5.0},
        ))
        rows = store.as_training_rows()
        assert rows[0]["dept_days"]["lighting"] == 6.0
        p.unlink(missing_ok=True)

    def test_as_training_rows_source_is_correction(self):
        store, p = _tmp_store()
        store.append(UserCorrection(description="any shot", final_total_days=8.0))
        rows = store.as_training_rows()
        assert rows[0]["source"] == "correction"
        p.unlink(missing_ok=True)

    def test_reload_after_append_reflects_new_data(self):
        """Simulates service.reload_corrections() after a new correction."""
        store, p = _tmp_store()
        store.append(UserCorrection(description="shot A", final_total_days=10.0))
        # Second store instance pointing at same path (simulates reload)
        store2 = CorrectionsStore(path=p, settings=get_settings())
        store2.append(UserCorrection(description="shot B", final_total_days=15.0))
        all_corrections = store2.load()
        assert len(all_corrections) == 2
        descs = [c.description for c in all_corrections]
        assert "shot A" in descs and "shot B" in descs
        p.unlink(missing_ok=True)
