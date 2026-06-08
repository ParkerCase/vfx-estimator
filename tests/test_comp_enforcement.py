"""Tests for compositing minimum enforcement."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vfx_estimator.estimate.service import EstimatorService, enforce_department_minimums


class TestEnforceDepartmentMinimums:
    @pytest.mark.parametrize(
        "description,dept_in,total,min_comp",
        [
            ("Wire removal from stunt", {"comp_roto": 1.0, "comp_paint": 1.0}, 4.0, 2.0),
            ("CG castle establishing shot", {"layout": 3.0, "lighting": 6.0, "dmp": 2.0}, 18.0, 5.0),
            ("CG creature hero shot with fire", {"animation": 6.0, "fx": 5.0, "lighting": 4.0}, 20.0, 6.0),
        ],
    )
    def test_required_shots_get_comp_minimum(self, description, dept_in, total, min_comp):
        dept = enforce_department_minimums(dept_in, total, description=description)
        assert dept.get("compositing", 0) >= min_comp


class TestEstimateCompMinimums:
    def _make_service(self) -> EstimatorService:
        svc = EstimatorService.__new__(EstimatorService)
        svc.settings = MagicMock(day_rate=750, estimate_mode="numeric_only", use_legacy_numeric=True)
        svc.settings.resolved_gemini_key.return_value = ""
        svc.corrections = MagicMock()
        svc.index = MagicMock()
        svc.index.query.return_value = []
        svc.index.median_mandays.return_value = 8.0
        svc._xata = MagicMock(enabled=False)
        svc.gemini = None
        svc.legacy = MagicMock()
        return svc

    @pytest.mark.parametrize(
        "description,min_comp",
        [
            ("Wire removal from stunt", 2.0),
            ("CG castle establishing shot", 5.0),
            ("CG creature hero shot with fire", 6.0),
        ],
    )
    def test_estimate_always_returns_comp(self, description, min_comp):
        svc = self._make_service()
        if "castle" in description:
            svc.legacy.predict.return_value = {
                "per_shot_mandays": 18.0,
                "dept_days": {"layout": 3.0, "lighting": 6.0, "dmp": 2.0},
                "screenplay_scene_matches": [],
            }
        elif "creature" in description:
            svc.legacy.predict.return_value = {
                "per_shot_mandays": 20.0,
                "dept_days": {"animation": 6.0, "fx": 5.0, "lighting": 4.0},
                "screenplay_scene_matches": [],
            }
        else:
            svc.legacy.predict.return_value = {
                "per_shot_mandays": 4.0,
                "dept_days": {"comp_roto": 1.0, "comp_paint": 1.0},
                "screenplay_scene_matches": [],
            }
        est = svc.estimate(description, mode="numeric_only")
        assert est.dept_days.get("compositing", 0) >= min_comp
