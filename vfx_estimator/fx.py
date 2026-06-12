"""USD-base FX rates via Frankfurter (free, no API key)."""

from __future__ import annotations

import json
import time
from typing import Any, Dict
from urllib.error import URLError
from urllib.request import urlopen

FRANKFURTER_URL = "https://api.frankfurter.dev/v1/latest?from=USD&to=CAD,GBP,EUR,AUD"
CACHE_TTL_SEC = 3600

# Last-resort fallbacks if Frankfurter is unreachable.
_FALLBACK_RATES: Dict[str, float] = {
    "USD": 1.0,
    "CAD": 1.36,
    "GBP": 0.79,
    "EUR": 0.92,
    "AUD": 1.52,
}

_cache: Dict[str, Any] = {}


def get_usd_fx_rates(*, force_refresh: bool = False) -> Dict[str, Any]:
    """Return { base, rates, date, source, cached, fallback? }."""
    now = time.time()
    if (
        not force_refresh
        and _cache.get("rates")
        and now - float(_cache.get("ts") or 0) < CACHE_TTL_SEC
    ):
        out = dict(_cache)
        out["cached"] = True
        return out

    try:
        with urlopen(FRANKFURTER_URL, timeout=12) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        rates: Dict[str, float] = {"USD": 1.0}
        for code, val in (payload.get("rates") or {}).items():
            rates[str(code).upper()] = float(val)
        _cache.clear()
        _cache.update(
            {
                "base": "USD",
                "rates": rates,
                "date": payload.get("date"),
                "source": "frankfurter",
                "ts": now,
                "cached": False,
                "fallback": False,
            }
        )
        return dict(_cache)
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError):
        if _cache.get("rates"):
            stale = dict(_cache)
            stale["cached"] = True
            stale["stale"] = True
            return stale
        return {
            "base": "USD",
            "rates": dict(_FALLBACK_RATES),
            "date": None,
            "source": "fallback",
            "ts": now,
            "cached": False,
            "fallback": True,
        }
