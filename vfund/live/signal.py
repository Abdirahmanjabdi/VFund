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


def _sleeve_current_weights(bt) -> tuple[dict[str, float], float]:
    """Latest-bar target weights + return volatility of one sleeve's backtester."""
    C = bt.closes
    rets = np.zeros_like(C)
    rets[1:] = C[1:] / C[:-1] - 1.0
    rets = np.where(np.isfinite(rets), rets, 0.0)
    i = C.shape[0] - 1
    w = bt._target_weights(i, rets)  # reuse the exact engine logic
    weights = {s: float(x) for s, x in zip(bt.symbols, w) if abs(x) > 1e-9}
    eq = bt.run().equity_curve["equity"].to_numpy()
    r = eq[1:] / eq[:-1] - 1.0
    return weights, float(r.std()) or 1e-9


def alpha_book(
    broad_panel: pl.DataFrame,
    defi_panel: pl.DataFrame,
    tvl: pl.DataFrame,
    *,
    fees: pl.DataFrame | None = None,
    min_short_dollar_volume: float = 5_000_000,
) -> Book:
    """The diversified alpha book (trend + size + on-chain), as live weights.

    Each sleeve runs on its own universe via the real backtester (so the live
    book matches the validated strategy), then they're combined at equal risk
    (inverse-vol) across sleeves — coins appearing in more than one sleeve get
    the summed allocation.

    Args:
        broad_panel: broad spot universe for the trend and size sleeves.
        defi_panel: DeFi price panel for the on-chain sleeves.
        tvl: TVL panel (parked capital) for the TVL-divergence sleeve.
        fees: optional protocol fee-revenue panel, stored in the TVL schema. When
            supplied a fourth sleeve is added — revenue grew but price lagged —
            which is the 4-sleeve book benchmarked in ``examples/four_sleeve.py``.
        min_short_dollar_volume: hard-to-short gate for the broad sleeves.
    """
    from vfund.backtest.cross_sectional import CrossSectionalBacktester
    from vfund.strategy import CrossSectionalSize, TimeSeriesTrendEnsemble, TVLDivergence

    # The on-chain sleeve aligns prices to TVL by symbol; TVL uses bare symbols
    # (AAVE), so normalise the DeFi price panel (AAVEUSDT -> AAVE) to match.
    defi_panel = defi_panel.with_columns(pl.col("symbol").str.replace(r"USDT$", ""))
    common = dict(interval="1d", cost_bps=10, short_cost_bps_annual=1000)
    sleeves = [
        CrossSectionalBacktester(broad_panel, TimeSeriesTrendEnsemble(),
                                 rebalance_every=7, neutralize=False, vol_target=0.30,
                                 vol_lookback=30, max_leverage=3.0,
                                 min_short_dollar_volume=min_short_dollar_volume, **common),
        CrossSectionalBacktester(broad_panel, CrossSectionalSize(20), rebalance_every=7,
                                 top_k=5, min_short_dollar_volume=min_short_dollar_volume,
                                 **common),
        CrossSectionalBacktester(defi_panel, TVLDivergence(60), rebalance_every=7,
                                 top_k=5, tvl=tvl, **common),
    ]
    if fees is not None:
        # Same divergence logic, shorter window, driven by revenue not TVL.
        sleeves.append(
            CrossSectionalBacktester(defi_panel, TVLDivergence(30), rebalance_every=7,
                                     top_k=5, tvl=fees, **common)
        )
    weights, vols, asof = [], [], None
    for bt in sleeves:
        w, v = _sleeve_current_weights(bt)
        weights.append(w)
        vols.append(v)
        asof = bt.timestamps[-1]

    rw = np.array([1.0 / v for v in vols])
    rw /= rw.sum()  # inverse-vol risk allocation across sleeves

    combined: dict[str, float] = {}
    for alloc, sleeve in zip(rw, weights):
        for s, x in sleeve.items():
            key = s[:-4] if s.endswith("USDT") else s  # normalise BTCUSDT<->BTC
            combined[key] = combined.get(key, 0.0) + alloc * x
    combined = {s: w for s, w in combined.items() if abs(w) > 1e-6}
    arr = np.array(list(combined.values()))
    return Book(weights=combined, asof=asof, gross=float(np.abs(arr).sum()),
                net=float(arr.sum()))


def three_sleeve_book(
    broad_panel: pl.DataFrame,
    defi_panel: pl.DataFrame,
    tvl: pl.DataFrame,
    *,
    min_short_dollar_volume: float = 5_000_000,
) -> Book:
    """The 3-sleeve alpha book — :func:`alpha_book` without the fees sleeve.

    Kept as a named entry point because the original forward paper account was
    started on exactly this configuration and must keep running it unchanged.
    """
    return alpha_book(broad_panel, defi_panel, tvl,
                      min_short_dollar_volume=min_short_dollar_volume)


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
