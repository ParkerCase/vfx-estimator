"""Tests for practical vs CG slider scaling."""

from __future__ import annotations

from vfx_estimator.estimate.practical_cg import (
    apply_practical_cg_ratio,
    build_practical_cg_prompt_rules,
    cg_ratio_from_pre_qual,
    practical_cg_total_multiplier,
)
from vfx_estimator.estimate.service import enforce_department_minimums
from vfx_estimator.types import BidPreQual


def test_cg_ratio_from_pre_qual_defaults_to_full_cg():
    assert cg_ratio_from_pre_qual(None) == 100
    assert cg_ratio_from_pre_qual(BidPreQual()) == 100


def test_cg_ratio_from_pre_qual_reads_slider():
    assert cg_ratio_from_pre_qual(BidPreQual(practical_cg_ratio=0)) == 0
    assert cg_ratio_from_pre_qual(BidPreQual(practical_cg_ratio=40)) == 40


def test_apply_practical_cg_zeros_pipeline_at_0():
    dept = {
        "layout": 4.0,
        "animation": 8.0,
        "lighting": 6.0,
        "fx": 5.0,
        "compositing": 4.0,
        "comp_paint": 2.0,
    }
    out = apply_practical_cg_ratio(dept, 0)
    assert "layout" not in out
    assert "animation" not in out
    assert "lighting" not in out
    assert "fx" not in out
    assert out["compositing"] == 4.0
    assert out["comp_paint"] == 2.0


def test_apply_practical_cg_scales_pipeline_at_50():
    dept = {"animation": 10.0, "lighting": 6.0, "compositing": 4.0}
    out = apply_practical_cg_ratio(dept, 50)
    assert out["animation"] == 5.0
    assert out["lighting"] == 3.0
    assert out["compositing"] == 4.0


def test_apply_practical_cg_unchanged_at_100():
    dept = {"animation": 10.0, "compositing": 4.0}
    out = apply_practical_cg_ratio(dept, 100)
    assert out == dept


def test_practical_total_multiplier_range():
    assert practical_cg_total_multiplier(100) == 1.0
    assert practical_cg_total_multiplier(0) < 0.4
    assert practical_cg_total_multiplier(50) > practical_cg_total_multiplier(0)


def test_enforce_minimums_respects_zero_cg():
    dept = enforce_department_minimums(
        {"animation": 2.0, "compositing": 2.0},
        4.0,
        description="Digital creature hero shot",
        cg_ratio=0,
    )
    assert "animation" not in dept
    assert "lighting" not in dept
    assert dept["compositing"] >= 2.0


def test_prompt_rules_material_at_extremes():
    practical_rules = build_practical_cg_prompt_rules(0)
    mixed_rules = build_practical_cg_prompt_rules(50)
    assert "layout=0" in practical_rules
    assert "MANDATORY" in practical_rules
    assert "50%" in mixed_rules
    assert build_practical_cg_prompt_rules(100) == ""
