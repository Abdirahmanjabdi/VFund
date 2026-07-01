"""Canonical market-data model used across VFund.

Everything in VFund — ingestion, backtesting, analytics — speaks the same OHLCV
bar schema. Keeping one schema is what lets a strategy written against synthetic
data run unchanged against a live Binance feed later.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl

# The canonical OHLCV bar schema. Timestamps are UTC, millisecond precision,
# and mark the *open* time of the bar.
BAR_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("ms", time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}

REQUIRED_COLUMNS = tuple(BAR_SCHEMA.keys())


@dataclass(frozen=True, slots=True)
class Bar:
    """A single OHLCV bar — the atom the backtester replays."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def validate_bars(df: pl.DataFrame) -> pl.DataFrame:
    """Validate and normalise a bar DataFrame to the canonical schema.

    Raises ``ValueError`` on missing columns or non-monotonic timestamps — the
    kind of silent data bug that quietly poisons a backtest. Returns a frame
    sorted by timestamp with exactly the canonical columns, in order.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"bars missing required columns: {missing}; got {df.columns}"
        )

    df = df.select(REQUIRED_COLUMNS).sort("timestamp")

    if df.height == 0:
        raise ValueError("bars frame is empty")

    # Duplicate or out-of-order timestamps are a classic source of look-ahead
    # and double-counting bugs — fail loudly rather than trade on them.
    n_unique = df["timestamp"].n_unique()
    if n_unique != df.height:
        raise ValueError(
            f"bars contain {df.height - n_unique} duplicate timestamp(s)"
        )

    return df
