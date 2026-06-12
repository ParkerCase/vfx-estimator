"""Tests for USD-base FX rates."""

from __future__ import annotations

from vfx_estimator.fx import _FALLBACK_RATES, get_usd_fx_rates


def test_get_usd_fx_rates_includes_usd_base():
    data = get_usd_fx_rates(force_refresh=True)
    assert data["base"] == "USD"
    assert data["rates"]["USD"] == 1.0
    for code in ("CAD", "GBP", "EUR", "AUD"):
        assert code in data["rates"]
        assert data["rates"][code] > 0


def test_get_usd_fx_rates_uses_cache():
    first = get_usd_fx_rates(force_refresh=True)
    second = get_usd_fx_rates()
    assert second["cached"] is True
    assert second["rates"] == first["rates"]


def test_fallback_rates_are_sane():
    for code, rate in _FALLBACK_RATES.items():
        assert rate > 0
