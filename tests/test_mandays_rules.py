"""Tests for Gemini VFX rule enforcement."""

from __future__ import annotations

from vfx_estimator.llm.mandays_rag import _enforce_vfx_rules


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


def test_fire_triggers_fx():
    data = {"departments": {"compositing": {"days": 3.0}}, "total_days": 3.0}
    _enforce_vfx_rules("Dragon breathing fire", data)
    assert data["departments"]["fx"]["days"] == 3.0
