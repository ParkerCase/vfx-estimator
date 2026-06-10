"""Tests for per-department day rates and cost calculation."""

from __future__ import annotations

from vfx_estimator.rates import (
    build_dept_rates,
    compute_dept_costs,
    compute_shot_cost,
)


class TestDeptRates:
    def test_build_dept_rates_merges_overrides(self):
        rates = build_dept_rates(
            fallback=700.0,
            overrides={"animation": 900, "compositing": 750, "FX": 850},
        )
        assert rates["animation"] == 900.0
        assert rates["compositing"] == 750.0
        assert rates["fx"] == 850.0
        assert rates["layout"] == 800.0

    def test_compute_dept_costs_from_internal_keys(self):
        rates = build_dept_rates(overrides={"animation": 900, "compositing": 750, "fx": 850})
        costs = compute_dept_costs(
            {"animation": 4.0, "compositing": 2.0, "fx": 3.0},
            rates,
        )
        assert costs["animation"] == 3600.0
        assert costs["compositing"] == 1500.0
        assert costs["fx"] == 2550.0

    def test_compute_dept_costs_accepts_bid_keys(self):
        rates = build_dept_rates(overrides={"compositing": 750})
        costs = compute_dept_costs({"COMP": 4.0, "ANIM": 2.0}, rates)
        assert costs["compositing"] == 3000.0
        assert costs["animation"] == 1800.0

    def test_compute_shot_cost_sums_departments(self):
        rates = build_dept_rates(overrides={"animation": 900, "compositing": 750})
        cost = compute_shot_cost(
            {"animation": 2.0, "compositing": 3.0},
            rates,
            fallback_rate=700.0,
        )
        assert cost == 4050.0

    def test_compute_shot_cost_falls_back_to_mandays(self):
        rates = build_dept_rates(fallback=700.0)
        cost = compute_shot_cost({}, rates, fallback_rate=700.0, mandays_fallback=6.0)
        assert cost == 4200.0

    def test_compute_shot_cost_scales_by_allotment(self):
        rates = build_dept_rates(overrides={"compositing": 750})
        cost = compute_shot_cost(
            {"compositing": 2.0},
            rates,
            fallback_rate=700.0,
            allot=3,
        )
        assert cost == 4500.0
