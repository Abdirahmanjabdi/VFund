"""Bar-interval metadata.

Crypto trades 24/7/365, so annualisation factors differ from equities. These
constants are used both for paginating data requests and for annualising
Sharpe/volatility in analytics.
"""

from __future__ import annotations

# Interval string -> duration in milliseconds.
INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}

# Number of bars in a (crypto) year, used to annualise return statistics.
_MS_PER_YEAR = 365 * 24 * 60 * 60_000


def bars_per_year(interval: str) -> float:
    """Bars per year for a given interval (assuming a 24/7 market)."""
    if interval not in INTERVAL_MS:
        raise ValueError(
            f"unknown interval {interval!r}; known: {sorted(INTERVAL_MS)}"
        )
    return _MS_PER_YEAR / INTERVAL_MS[interval]
