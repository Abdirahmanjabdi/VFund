"""Synthetic price data via geometric Brownian motion.

Why ship a fake-data generator in a trading platform? Two reasons:

1. The demo and the test suite run fully offline and deterministically — no
   network, no API keys, no flakiness.
2. It teaches an essential habit: before trusting a backtest, run your strategy
   on data with *known* properties. A momentum strategy should make money on a
   trending series and bleed on a mean-reverting one. If it doesn't, the bug is
   in your engine, not the market.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl

from vfund.data.intervals import INTERVAL_MS
from vfund.data.models import validate_bars


def generate_gbm_bars(
    n: int,
    *,
    start_price: float = 100.0,
    mu: float = 0.10,
    sigma: float = 0.60,
    interval: str = "1h",
    start: datetime | None = None,
    seed: int | None = 42,
) -> pl.DataFrame:
    """Generate ``n`` OHLCV bars following geometric Brownian motion.

    ``mu`` and ``sigma`` are annualised drift and volatility. Intrabar high/low
    are synthesised from a small noise band around the open/close so that
    fills and slippage have something realistic to bite on.
    """
    if n < 2:
        raise ValueError("need at least 2 bars")
    if interval not in INTERVAL_MS:
        raise ValueError(f"unknown interval {interval!r}")

    rng = np.random.default_rng(seed)
    dt = INTERVAL_MS[interval] / (365 * 24 * 60 * 60_000)  # year fraction per bar

    # Closing prices via the exact GBM discretisation.
    shocks = rng.standard_normal(n)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    close = start_price * np.exp(np.cumsum(log_returns))

    # Opens are the previous close (first open == start_price).
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]

    # High/low: extend beyond the open/close range by a small random wick.
    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    wick = np.abs(rng.standard_normal((2, n))) * sigma * np.sqrt(dt) * close
    high = body_hi + wick[0]
    low = np.maximum(body_lo - wick[1], 1e-9)

    volume = rng.lognormal(mean=6.0, sigma=0.5, size=n)

    start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    step_ms = INTERVAL_MS[interval]
    start_ms = int(start.timestamp() * 1000)
    ts = pl.Series(
        "timestamp",
        [start_ms + i * step_ms for i in range(n)],
    ).cast(pl.Datetime("ms", time_zone="UTC"))

    df = pl.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    return validate_bars(df)


def generate_gbm_panel(
    n_assets: int,
    n_bars: int,
    *,
    interval: str = "1h",
    market_vol: float = 0.4,
    idio_vol: float = 0.6,
    reversion: float = 0.0,
    start: datetime | None = None,
    seed: int | None = 42,
) -> "pl.DataFrame":
    """Generate a multi-asset panel for offline cross-sectional research.

    Each asset's return = a shared *market* factor + an idiosyncratic part.
    ``reversion`` in ``(0, 1)`` injects negative autocorrelation into the
    idiosyncratic part (an AR(1) with coefficient ``-reversion``): a coin that
    pops tends to give it back. That is a *real* cross-sectional reversal edge,
    by construction — so a correct engine should profit on it, and earn ~nothing
    (after costs) when ``reversion == 0``. This is how you test the microscope
    before you trust it on the market.
    """
    from vfund.data.intervals import INTERVAL_MS
    from vfund.data.panel import validate_panel

    if n_assets < 2:
        raise ValueError("need at least 2 assets for a cross-section")
    if interval not in INTERVAL_MS:
        raise ValueError(f"unknown interval {interval!r}")

    rng = np.random.default_rng(seed)
    dt = INTERVAL_MS[interval] / (365 * 24 * 60 * 60_000)
    sqrt_dt = np.sqrt(dt)

    market = market_vol * sqrt_dt * rng.standard_normal(n_bars)

    frames = []
    step_ms = INTERVAL_MS[interval]
    start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    start_ms = int(start.timestamp() * 1000)
    ts = pl.Series(
        "timestamp", [start_ms + i * step_ms for i in range(n_bars)]
    ).cast(pl.Datetime("ms", time_zone="UTC"))

    for a in range(n_assets):
        shocks = idio_vol * sqrt_dt * rng.standard_normal(n_bars)
        idio = np.empty(n_bars)
        idio[0] = shocks[0]
        for t in range(1, n_bars):
            idio[t] = -reversion * idio[t - 1] + shocks[t]

        rets = market + idio
        close = 100.0 * np.exp(np.cumsum(rets))
        open_ = np.empty(n_bars)
        open_[0] = 100.0
        open_[1:] = close[:-1]
        wick = np.abs(rng.standard_normal(n_bars)) * idio_vol * sqrt_dt * close
        high = np.maximum(open_, close) + wick
        low = np.maximum(np.minimum(open_, close) - wick, 1e-9)

        frames.append(
            pl.DataFrame(
                {
                    "timestamp": ts,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": rng.lognormal(6.0, 0.5, n_bars),
                }
            ).with_columns(pl.lit(f"SYN{a:02d}").alias("symbol"))
        )

    return validate_panel(pl.concat(frames))
