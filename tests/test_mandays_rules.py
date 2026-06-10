"""Tests for Gemini VFX rule enforcement."""

from __future__ import annotations

from vfx_estimator.llm.mandays_rag import GeminiMandaysEstimator, _enforce_vfx_rules
from vfx_estimator.types import BidPreQual


def test_compositing_floor_when_missing():
    data = {"departments": {"comp_roto": {"days": 1.0}}, "total_days": 1.0}
    _enforce_vfx_rules("Hero beauty comp", data)
    assert data["departments"]["compositing"]["days"] == 2.0


def test_wire_removal_strips_fx_and_adds_comp():
    data = {
        "departments": {
            "fx": {"days": 5.0},
            "animation": {"days": 2.0},
            "comp_roto": {"days": 1.0},
        },
        "total_days": 8.0,
    }
    _enforce_vfx_rules("Wire removal on telephone wires", data)
    assert "fx" not in data["departments"]
    assert "animation" not in data["departments"]
    assert data["departments"]["compositing"]["days"] == 2.0


def test_build_project_context_block():
    est = GeminiMandaysEstimator.__new__(GeminiMandaysEstimator)
    pq = BidPreQual(
        bid_scale_tier="premium_tv",
        complexity_band="high",
        practical_cg_ratio=85,
        director_brief="Photoreal result",
        vfx_assumptions="Background is full CG",
    )
    block = est._build_project_context(pq)
    assert "PROJECT CONTEXT (applies to all shots):" in block
    assert "Scale: premium_tv" in block
    assert "Complexity: high" in block
    assert "CG ratio: 85% CG" in block
    assert "Director intent: Photoreal result" in block
    assert "VFX assumptions: Background is full CG" in block
    assert "overrides generic assumptions" in block


def test_fire_triggers_fx():
    data = {"departments": {"compositing": {"days": 3.0}}, "total_days": 3.0}
    _enforce_vfx_rules("Dragon breathing fire", data)
    assert data["departments"]["fx"]["days"] == 3.0
    assert data["departments"]["lighting"]["days"] == 3.0


def test_cg_department_triggers_lighting():
    data = {"departments": {"animation": {"days": 4.0}}, "total_days": 4.0}
    _enforce_vfx_rules("Creature animation pass", data)
    assert data["departments"]["lighting"]["days"] == 3.0


def test_cg_description_triggers_lighting():
    data = {"departments": {"compositing": {"days": 2.0}}, "total_days": 2.0}
    _enforce_vfx_rules("Small CG prop integration", data)
    assert data["departments"]["lighting"]["days"] == 3.0
