"""Multi-asset panel data: many symbols, one aligned table.

Cross-sectional research needs a *universe*, not a single coin. A panel is the
long-format table (timestamp, symbol, OHLCV) that holds it. ``pivot_to_wide``
turns that into the time x symbol matrix the cross-sectional engine consumes.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from vfund.data.models import BAR_SCHEMA

# A panel is canonical bars plus a `symbol` column.
PANEL_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")


def validate_panel(df: pl.DataFrame) -> pl.DataFrame:
    """Validate a long-format multi-asset panel."""
    missing = [c for c in PANEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"panel missing columns: {missing}; got {df.columns}")

    df = df.select(PANEL_COLUMNS).sort(["symbol", "timestamp"])
    if df.height == 0:
        raise ValueError("panel is empty")

    dupes = df.height - df.select(["symbol", "timestamp"]).n_unique()
    if dupes:
        raise ValueError(f"panel has {dupes} duplicate (symbol, timestamp) rows")
    return df


def pivot_to_wide(
    panel: pl.DataFrame, value: str = "close", *, drop_incomplete: bool = True
) -> pl.DataFrame:
    """Pivot a long panel to a wide ``timestamp x symbol`` matrix of ``value``.

    ``drop_incomplete=True`` (default) keeps only timestamps present for *every*
    symbol — a dense matrix, but it discards early history for late-listed coins
    and truncates when any coin delists. That is a survivorship trap.

    ``drop_incomplete=False`` keeps the full (ragged) timeline with nulls where a
    coin isn't yet listed or has delisted, so coins can enter and exit the
    cross-section as they actually traded. The engine handles the nulls.
    """
    wide = panel.pivot(values=value, index="timestamp", on="symbol").sort("timestamp")
    if drop_incomplete:
        wide = wide.drop_nulls()
    if wide.height == 0:
        raise ValueError(
            "no timestamps survive — check intervals/date ranges (or set "
            "drop_incomplete=False for a ragged panel)"
        )
    return wide


FUNDING_COLUMNS = ("timestamp", "symbol", "funding_rate")


def validate_funding(df: pl.DataFrame) -> pl.DataFrame:
    """Validate a long-format funding panel (timestamp, symbol, funding_rate)."""
    missing = [c for c in FUNDING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"funding panel missing columns: {missing}; got {df.columns}")
    return df.select(FUNDING_COLUMNS).sort(["symbol", "timestamp"])


def save_funding(df: pl.DataFrame, path: str | Path) -> Path:
    df = validate_funding(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def load_funding(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no funding file at {path}")
    return validate_funding(pl.read_parquet(path))


def align_funding(
    funding: pl.DataFrame,
    timestamps: pl.Series,
    symbols: list[str],
) -> tuple["object", "object"]:
    """Align an 8-hourly funding panel onto a (finer) price timeline.

    Returns two ``(T, N)`` NumPy matrices, columns ordered like ``symbols``:

    * ``event`` — the funding rate on the exact bar it is paid, 0 elsewhere
      (used for P&L accrual);
    * ``prevailing`` — the last known rate forward-filled to every bar (used as
      the trading signal, since that's the rate you can actually see).
    """
    import numpy as np

    funding = validate_funding(funding)
    base = pl.DataFrame({"timestamp": timestamps})
    event_cols, prevailing_cols = [], []
    for sym in symbols:
        f = funding.filter(pl.col("symbol") == sym).select(
            ["timestamp", "funding_rate"]
        )
        joined = base.join(f, on="timestamp", how="left").sort("timestamp")
        rate = joined["funding_rate"]
        event_cols.append(rate.fill_null(0.0).to_numpy())
        prevailing_cols.append(rate.fill_null(strategy="forward").fill_null(0.0).to_numpy())
    event = np.column_stack(event_cols) if event_cols else np.zeros((timestamps.len(), 0))
    prevailing = (
        np.column_stack(prevailing_cols) if prevailing_cols else np.zeros((timestamps.len(), 0))
    )
    return event, prevailing


def save_panel(panel: pl.DataFrame, path: str | Path) -> Path:
    panel = validate_panel(panel)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(path)
    return path


def load_panel(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no panel file at {path}")
    return validate_panel(pl.read_parquet(path))
