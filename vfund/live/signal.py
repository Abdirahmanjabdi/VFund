"""Generate the combined book's target weights for the latest bar.

This is the same trend+size portfolio validated in research, evaluated at the
*end* of the panel to answer: "what should I be holding right now?" The output
is a dict of symbol -> target weight (fraction of equity; negative = short).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from vfund.backtest.construct import (
    scores_to_weights,
    short_liquidity_mask,
    trailing_dollar_volume,
    vol_scale_weights,
)
from vfund.data.intervals import bars_per_year as _bars_per_year
from vfund.data.panel import pivot_to_wide, validate_panel
from vfund.strategy import CrossSectionalSize, TimeSeriesTrendEnsemble
from vfund.strategy.cross_sectional import PanelContext


@dataclass
class Book:
    weights: dict[str, float]   # symbol -> target weight
    asof: object                # timestamp of the bar the signal is based on
    gross: float                # sum of |weights|
    net: float                  # sum of weights (net long/short tilt)


def combined_book(
    panel: pl.DataFrame,
    *,
    size_lookback: int = 20,
    vol_target: float = 0.30,
    top_k: int = 5,
    interval: str = "1d",
    vol_lookback: int = 30,
    max_leverage: float = 3.0,
    trend_weight: float = 0.5,
    min_short_dollar_volume: float | None = None,
    short_dv_lookback: int = 30,
) -> Book:
    """Compute the trend+size target book from the last bar of ``panel``.

    Pass ``min_short_dollar_volume`` to enforce the same hard-to-short constraint
    used in validation (only short names with enough recent liquidity).
    """
    panel = validate_panel(panel)
    # Ragged: use the true latest bar and whatever coins are tradable then, so the
    # book reflects the current universe (delisted coins are NaN -> excluded).
    wide = pivot_to_wide(panel, "close", drop_incomplete=False)
    symbols = [c for c in wide.columns if c != "timestamp"]
    closes = wide.select(symbols).to_numpy()

    vwide = wide.select("timestamp").join(
        pivot_to_wide(panel, "volume", drop_incomplete=False), on="timestamp", how="left"
    )
    volumes = vwide.select(symbols).fill_null(0.0).to_numpy()

    i = closes.shape[0] - 1
    ctx = PanelContext(i, closes, symbols, volumes=volumes)

    # Trend sleeve: directional, vol-targeted.
    w_trend = scores_to_weights(
        TimeSeriesTrendEnsemble().scores(ctx), leverage=1.0, neutralize=False
    )
    rets = np.zeros_like(closes)
    rets[1:] = closes[1:] / closes[:-1] - 1.0
    rets = np.where(np.isfinite(rets), rets, 0.0)  # ragged panel may hold NaN
    lo = max(1, i - vol_lookback + 1)
    w_trend = vol_scale_weights(
        w_trend, rets[lo : i + 1],
        vol_target=vol_target, bars_per_year=_bars_per_year(interval),
        max_leverage=max_leverage,
    )

    # Size sleeve: market-neutral, concentrated.
    w_size = scores_to_weights(
        CrossSectionalSize(size_lookback).scores(ctx),
        leverage=1.0, top_k=top_k, neutralize=True,
    )

    # Hard-to-short: forbid shorts on illiquid names (can't borrow them).
    if min_short_dollar_volume is not None:
        dv = trailing_dollar_volume(closes, volumes, i, short_dv_lookback)
        w_trend = short_liquidity_mask(w_trend, dv, min_short_dollar_volume)
        w_size = short_liquidity_mask(w_size, dv, min_short_dollar_volume)

    combined = trend_weight * w_trend + (1 - trend_weight) * w_size
    weights = {s: float(w) for s, w in zip(symbols, combined) if abs(w) > 1e-6}
    return Book(
        weights=weights,
        asof=wide["timestamp"][i],
        gross=float(np.abs(combined).sum()),
        net=float(combined.sum()),
    )


def format_book(book: Book) -> str:
    longs = sorted(((s, w) for s, w in book.weights.items() if w > 0), key=lambda x: -x[1])
    shorts = sorted(((s, w) for s, w in book.weights.items() if w < 0), key=lambda x: x[1])
    lines = [
        f"Target book as of {book.asof}  (gross {book.gross:.2f}, net {book.net:+.2f})",
        "-" * 44,
        f"  LONG ({len(longs)}):",
    ]
    lines += [f"    {s:<10} {w*100:>6.1f}%" for s, w in longs] or ["    (none)"]
    lines += [f"  SHORT ({len(shorts)}):"]
    lines += [f"    {s:<10} {w*100:>6.1f}%" for s, w in shorts] or ["    (none)"]
    return "\n".join(lines)
