"""Tests for Xata department column mapping."""

from __future__ import annotations

from types import SimpleNamespace

from vfx_estimator.api.app import _live_training_count
from vfx_estimator.integrations.xata import _record_from_row


class TestRecordFromRow:
    def test_maps_compositing_and_camera_track(self):
        row = {
            "shot_description": "Hero comp shot",
            "mandays": 10,
            "comp_mandays": 4.0,
            "cam_track_days": 2.0,
            "paint_mandays": 1.0,
            "lgt_mandays": 3.0,
        }
        rec = _record_from_row(row, day_rate=700.0)
        assert rec is not None
        assert rec["dept_days"]["compositing"] == 4.0
        assert rec["dept_days"]["camera_track"] == 2.0
        assert rec["dept_days"]["comp_paint"] == 1.0
        assert rec["dept_days"]["lighting"] == 3.0

    def test_fallback_comp_days_when_mandays_zero(self):
        row = {
            "shot_description": "Comp from days col",
            "mandays": 5,
            "comp_mandays": 0,
            "comp_days": 3.5,
        }
        rec = _record_from_row(row, day_rate=700.0)
        assert rec["dept_days"]["compositing"] == 3.5

    def test_derives_compositing_residual_when_comp_columns_empty(self):
        row = {
            "shot_description": "Hero comp with paint and roto",
            "total_mandays": 10.0,
            "comp_paint_days": 2.0,
            "comp_roto_days": 1.0,
            "layout_days": 2.0,
        }
        rec = _record_from_row(row, day_rate=700.0)
        assert rec is not None
        assert rec["dept_days"]["compositing"] == 5.0

    def test_prefers_comp_paint_days_over_empty_paint_mandays(self):
        row = {
            "shot_description": "Paint from days col",
            "mandays": 4,
            "comp_paint_days": 1.5,
            "paint_mandays": 0,
        }
        rec = _record_from_row(row, day_rate=700.0)
        assert rec["dept_days"]["comp_paint"] == 1.5


class TestLiveTrainingCount:
    def test_uses_live_xata_count_plus_corrections(self, monkeypatch):
        class FakeXata:
            mode = "postgres"

            def __init__(self, _settings):
                pass

            def count(self):
                return 4287

        svc = SimpleNamespace(
            settings=object(),
            training=[object(), object()],
            corrections=SimpleNamespace(count=lambda: 3),
        )
        monkeypatch.setattr("vfx_estimator.api.app.XataShotSearch", FakeXata)

        counts = _live_training_count(svc)

        assert counts["training_shots"] == 4290
        assert counts["base_training_shots"] == 4287
        assert counts["corrections"] == 3
        assert counts["training_count_source"] == "xata"

    def test_falls_back_to_local_training_plus_corrections(self, monkeypatch):
        class FakeXata:
            mode = "off"

            def __init__(self, _settings):
                pass

            def count(self):
                return 0

        svc = SimpleNamespace(
            settings=object(),
            training=[object(), object()],
            corrections=SimpleNamespace(count=lambda: 3),
        )
        monkeypatch.setattr("vfx_estimator.api.app.XataShotSearch", FakeXata)

        counts = _live_training_count(svc)

        assert counts["training_shots"] == 5
        assert counts["base_training_shots"] == 2
        assert counts["corrections"] == 3
        assert counts["training_count_source"] == "local"
