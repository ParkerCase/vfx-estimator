"""Unit tests for bid history Postgres helpers."""

from __future__ import annotations

from vfx_estimator.integrations.xata import _shot_total_mandays


class TestShotTotalMandays:
    def test_prefers_total_mandays(self):
        assert _shot_total_mandays({"total_mandays": 6.0, "total_days": 4.0}) == 6.0

    def test_falls_back_to_total_days(self):
        assert _shot_total_mandays({"total_days": 5.0}) == 5.0

    def test_sums_dept_days_when_no_total(self):
        assert _shot_total_mandays({"dept_days": {"LGT": 2.0, "COMP": 3.0}}) == 5.0

    def test_empty_shot(self):
        assert _shot_total_mandays({}) == 0.0
