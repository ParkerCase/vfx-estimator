"""Tests for metrics module."""
import pytest
from vfx_estimator.metrics import metrics_dict, within_band


def test_within_20():
    m = metrics_dict([10.0, 15.0], [10.0, 10.0])
    assert m["n"] == 2
    assert m["within_20pct"] == 50.0


def test_all_within_10():
    m = metrics_dict([10.0, 10.5], [10.0, 10.0])
    assert m["within_10pct"] == 100.0


def test_empty_sequence():
    m = metrics_dict([], [])
    assert m["n"] == 0
    assert m["mae"] == 0.0


def test_mae_calculation():
    m = metrics_dict([12.0], [10.0])
    assert m["mae"] == 2.0


def test_within_1day():
    m = metrics_dict([10.0, 11.0, 15.0], [10.0, 10.0, 10.0])
    # 10.0 and 11.0 are within 1 day of 10.0; 15.0 is not
    assert m["within_1day"] == pytest.approx(100 * 2 / 3, abs=0.1)


def test_within_band_all_inside():
    assert within_band([10.0, 9.5], [10.0, 10.0], 0.8, 1.2) == 1.0


def test_within_band_none_inside():
    assert within_band([5.0], [10.0], 0.9, 1.1) == 0.0
