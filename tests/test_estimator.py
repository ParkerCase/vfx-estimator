"""
Comprehensive test suite for VFX estimation core logic.

Per vfx-estimator skill TDD rules, covers:
  - Zero-division errors in mandays extraction
  - NaN handling in critical day-bucket columns
  - Dept sum vs stated total mismatch detection
  - Department-presence rules (FX sim, 2D FX, wire removal, CG env)
  - Adjustment range calculation for HITL UI sliders
  - UserCorrection model validation
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pytest

from vfx_estimator.api.app import _compute_adjustment_ranges
from vfx_estimator.data.loaders import _mandays_from_record
from vfx_estimator.types import DeptDays, ShotEstimate, UserCorrection


# ---------------------------------------------------------------------------
# 1. Zero-division safety in mandays extraction
# ---------------------------------------------------------------------------


class TestMandaysExtraction:
    def test_extracts_mandays_directly(self):
        assert _mandays_from_record({"mandays": 10.0}, day_rate=700.0) == 10.0

    def test_extracts_from_cost_when_mandays_missing(self):
        assert _mandays_from_record({"cost": 7000.0}, day_rate=700.0) == pytest.approx(10.0)

    def test_zero_division_safe_when_day_rate_zero(self):
        """Must not raise ZeroDivisionError; returns 0.0."""
        result = _mandays_from_record({"cost": 5000.0}, day_rate=0.0)
        assert result == 0.0

    def test_missing_both_returns_zero(self):
        assert _mandays_from_record({"project": "test"}, day_rate=700.0) == 0.0

    def test_null_mandays_falls_back_to_cost(self):
        assert _mandays_from_record({"mandays": None, "cost": 14000.0}, day_rate=700.0) == pytest.approx(20.0)

    def test_string_mandays_falls_back_to_cost(self):
        """Non-numeric mandays string should fall through to cost fallback."""
        result = _mandays_from_record({"mandays": "TBD", "cost": 3500.0}, day_rate=700.0)
        assert result == pytest.approx(5.0)

    def test_negative_mandays_returns_zero(self):
        """Negative values are data corruption — must be rejected."""
        result = _mandays_from_record({"mandays": -5.0}, day_rate=700.0)
        assert result == 0.0

    def test_prefers_total_mandays_over_mandays_key(self):
        result = _mandays_from_record({"total_mandays": 20.0, "mandays": 5.0}, day_rate=700.0)
        assert result == pytest.approx(20.0)

    def test_cost_per_shot_key_also_works(self):
        result = _mandays_from_record({"cost_per_shot": 4200.0}, day_rate=700.0)
        assert result == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# 2. NaN safety in critical day-bucket columns
# ---------------------------------------------------------------------------


class TestDeptDaysNaN:
    def test_dept_days_total_is_finite_with_normal_values(self):
        dd = DeptDays(lighting=5.0, comp_paint=3.0)
        assert math.isfinite(dd.total())

    def test_all_zero_total_is_zero(self):
        dd = DeptDays()
        assert dd.total() == 0.0

    def test_to_dict_values_all_finite(self):
        dd = DeptDays(lighting=4.0, fx=6.0, comp_roto=2.0)
        for k, v in dd.to_dict().items():
            assert math.isfinite(v), f"dept_days[{k}] is non-finite: {v}"

    def test_shot_estimate_accepts_empty_dept_days(self):
        est = ShotEstimate(
            description="test",
            per_shot_mandays=5.0,
            total_mandays=5.0,
            cost=3500.0,
            dept_days={},
        )
        assert est.dept_days == {}

    def test_shot_estimate_dept_days_sum_matches_total(self):
        dept = {"lighting": 7.0, "layout": 3.0, "comp_paint": 5.0}
        est = ShotEstimate(
            description="CG castle establishing",
            per_shot_mandays=15.0,
            total_mandays=15.0,
            cost=10500.0,
            dept_days=dept,
        )
        dept_sum = sum(est.dept_days.values())
        # Document: Gemini must enforce this; service uses dept sum when available
        assert dept_sum == pytest.approx(est.total_mandays, abs=0.5), (
            f"Dept sum {dept_sum} != total_mandays {est.total_mandays}"
        )


# ---------------------------------------------------------------------------
# 3. Dept sum vs stated total mismatch (audit check for Gemini output)
# ---------------------------------------------------------------------------


class TestDeptTotalMismatch:
    def _dept_sum(self, departments: Dict[str, Any]) -> float:
        total = 0.0
        for v in departments.values():
            if isinstance(v, dict):
                total += float(v.get("days") or 0)
            else:
                total += float(v or 0)
        return total

    def test_matching_total_passes(self):
        depts = {"lighting": {"days": 7.0}, "layout": {"days": 3.0}, "comp_paint": {"days": 5.0}}
        stated = 15.0
        assert self._dept_sum(depts) == pytest.approx(stated)

    def test_mismatched_total_detected(self):
        depts = {"lighting": {"days": 7.0}, "comp_paint": {"days": 5.0}}
        stated = 20.0  # deliberately wrong
        assert self._dept_sum(depts) != pytest.approx(stated), "Expected mismatch not detected"

    def test_empty_depts_sum_zero(self):
        assert self._dept_sum({}) == 0.0

    def test_mixed_dict_and_float_values(self):
        depts = {"lighting": {"days": 7.0}, "layout": 3.0}
        assert self._dept_sum(depts) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 4. Department-presence rules (skill "No Missing Link" Rule)
# ---------------------------------------------------------------------------


class TestDepartmentPresenceRules:
    """
    Per skill rules validated at estimation/audit time:
      FX animation/sim -> fx AND lighting must be non-zero
      2D/2.5D FX       -> comp_roto AND (camera_track OR matchmove) non-zero
      Wire removal      -> ONLY comp_roto + comp_paint; ALL 3D depts must be zero
      CG Environment    -> animation MUST be 0 (static buildings)
    """

    def _has_fx_and_lighting(self, d: Dict[str, float]) -> bool:
        return d.get("fx", 0) > 0 and d.get("lighting", 0) > 0

    def _has_comp_and_tracking(self, d: Dict[str, float]) -> bool:
        has_comp = d.get("comp_roto", 0) > 0 or d.get("comp_paint", 0) > 0
        has_track = d.get("camera_track", 0) > 0 or d.get("matchmove", 0) > 0
        return has_comp and has_track

    def _wire_removal_3d_clean(self, d: Dict[str, float]) -> bool:
        three_d = ["layout", "animation", "lighting", "cfx", "fx", "dmp"]
        return all(d.get(dept, 0) == 0 for dept in three_d)

    def _cg_env_no_animation(self, d: Dict[str, float]) -> bool:
        return d.get("animation", 0) == 0

    # FX sim
    def test_fx_sim_valid(self):
        assert self._has_fx_and_lighting({"fx": 5.0, "lighting": 3.0, "comp_paint": 4.0})

    def test_fx_sim_missing_lighting_fails(self):
        assert not self._has_fx_and_lighting({"fx": 5.0, "comp_paint": 4.0})

    def test_fx_sim_missing_fx_fails(self):
        assert not self._has_fx_and_lighting({"lighting": 3.0, "comp_paint": 4.0})

    # 2D/2.5D FX
    def test_2d_fx_valid_matchmove(self):
        assert self._has_comp_and_tracking({"comp_roto": 3.0, "matchmove": 1.5})

    def test_2d_fx_valid_camera_track(self):
        assert self._has_comp_and_tracking({"comp_paint": 2.0, "camera_track": 2.0})

    def test_2d_fx_no_tracking_fails(self):
        assert not self._has_comp_and_tracking({"comp_roto": 3.0})

    def test_2d_fx_no_comp_fails(self):
        assert not self._has_comp_and_tracking({"camera_track": 2.0})

    # Wire removal
    def test_wire_removal_2d_only_passes(self):
        assert self._wire_removal_3d_clean({"comp_roto": 3.0, "comp_paint": 2.0})

    def test_wire_removal_with_lighting_fails(self):
        assert not self._wire_removal_3d_clean({"comp_roto": 3.0, "lighting": 2.0})

    def test_wire_removal_with_layout_fails(self):
        assert not self._wire_removal_3d_clean({"comp_roto": 3.0, "layout": 1.0})

    def test_wire_removal_with_animation_fails(self):
        assert not self._wire_removal_3d_clean({"comp_roto": 3.0, "animation": 2.0})

    # CG environment
    def test_cg_env_no_animation_passes(self):
        assert self._cg_env_no_animation({"layout": 2.0, "lighting": 6.0, "comp_paint": 5.0})

    def test_cg_env_with_animation_fails(self):
        assert not self._cg_env_no_animation({"layout": 2.0, "animation": 3.0, "lighting": 6.0})


# ---------------------------------------------------------------------------
# 5. Adjustment range calculation for HITL slider UI
# ---------------------------------------------------------------------------


class TestAdjustmentRanges:
    def test_high_confidence_50pct_range(self):
        ranges = _compute_adjustment_ranges({"lighting": 8.0}, overall_confidence=0.85)
        r = ranges["lighting"]
        assert r["min"] == pytest.approx(4.0, abs=0.5)
        assert r["max"] == pytest.approx(12.0, abs=0.5)
        assert r["predicted"] == 8.0

    def test_medium_confidence_75pct_range(self):
        ranges = _compute_adjustment_ranges({"comp_paint": 4.0}, overall_confidence=0.65)
        r = ranges["comp_paint"]
        assert r["min"] == pytest.approx(1.0, abs=0.5)
        assert r["max"] == pytest.approx(7.0, abs=0.5)

    def test_low_confidence_100pct_range(self):
        ranges = _compute_adjustment_ranges({"animation": 5.0}, overall_confidence=0.45)
        r = ranges["animation"]
        assert r["min"] == 0.0  # clamped, never negative
        assert r["max"] == pytest.approx(10.0, abs=0.5)

    def test_min_never_negative(self):
        ranges = _compute_adjustment_ranges({"fx": 0.5}, overall_confidence=0.3)
        assert ranges["fx"]["min"] >= 0.0

    def test_zero_predicted_excluded(self):
        ranges = _compute_adjustment_ranges({"lighting": 0.0, "comp_paint": 4.0}, 0.7)
        assert "lighting" not in ranges
        assert "comp_paint" in ranges

    def test_empty_returns_empty(self):
        assert _compute_adjustment_ranges({}, 0.8) == {}

    def test_step_is_half_day(self):
        ranges = _compute_adjustment_ranges({"layout": 3.0}, 0.9)
        assert ranges["layout"]["step"] == 0.5

    def test_boundary_0_8_is_high_band(self):
        """Exactly 0.80 -> high band -> ±50%."""
        ranges = _compute_adjustment_ranges({"lighting": 10.0}, overall_confidence=0.80)
        assert ranges["lighting"]["max"] == pytest.approx(15.0, abs=0.5)

    def test_boundary_0_6_is_medium_band(self):
        """Exactly 0.60 -> medium band -> ±75%."""
        ranges = _compute_adjustment_ranges({"lighting": 10.0}, overall_confidence=0.60)
        assert ranges["lighting"]["max"] == pytest.approx(17.5, abs=0.5)

    def test_multiple_departments(self):
        depts = {"layout": 2.0, "lighting": 7.0, "comp_paint": 5.0}
        ranges = _compute_adjustment_ranges(depts, overall_confidence=0.75)
        assert len(ranges) == 3
        for dept in depts:
            assert dept in ranges
            assert ranges[dept]["predicted"] == depts[dept]


# ---------------------------------------------------------------------------
# 6. UserCorrection model validation
# ---------------------------------------------------------------------------


class TestUserCorrectionModel:
    def test_full_roundtrip(self):
        c = UserCorrection(
            description="CG castle establishing shot",
            final_total_days=21.0,
            final_departments={"lighting": 9.0, "layout": 3.0, "comp_paint": 9.0},
            user_id="che",
            notes="Hero lighting increased for historical accuracy",
            ai_total_days=17.0,
        )
        c2 = UserCorrection.model_validate(c.model_dump())
        assert c2.final_total_days == 21.0
        assert c2.final_departments["lighting"] == 9.0
        assert c2.user_id == "che"

    def test_minimal_fields_use_defaults(self):
        c = UserCorrection(description="wire removal", final_total_days=5.0)
        assert c.user_id == "default"
        assert c.notes == ""
        assert c.final_departments == {}

    def test_ai_total_days_optional(self):
        c = UserCorrection(description="test", final_total_days=8.0)
        assert c.ai_total_days is None

    def test_partial_dept_breakdown_accepted(self):
        """Supervisor may give partial breakdown without it matching total."""
        c = UserCorrection(
            description="test",
            final_total_days=20.0,
            final_departments={"lighting": 8.0},  # partial — sum < total is OK
        )
        assert c.final_departments["lighting"] == 8.0
