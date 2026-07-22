"""Today's target book — the weights the validated strategy says to hold now.

Research/live parity
--------------------
Every book here is produced by the **backtest engine itself**, by asking
:class:`~vfund.backtest.cross_sectional.CrossSectionalBacktester` for its target
weights at the final bar. Nothing re-implements the weight pipeline.

That rule exists because it was once broken. ``combined_book`` used to hand-roll
the overlays, and drifted from the engine in two ways that a reader would never
spot: it omitted the capacity cap entirely, and it applied vol-targeting *before*
the shortability mask instead of after. The result was a live book up to 1.75x
too large with a capacity limit configured, and ~9% too small on the vol-targeted
trend sleeve — every single name wrong. The backtest was honest and the live
signal quietly disagreed with it.

The engine owns the overlay chain (`scores -> shortability -> vol-target ->
capacity`) and its order. Callers here choose *parameters*, never *steps*.
``tests/test_parity.py`` asserts this equivalence and fails CI if it drifts again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class Book:
    weights: dict[str, float]   # symbol -> target weight
    asof: object                # timestamp of the bar the signal is based on
    gross: float                # sum of |weights|
    net: float                  # sum of weights (net long/short tilt)


def _engine_weights(bt) -> dict[str, float]:
    """Target weights at the engine's final bar — the single source of truth.

    Calls the engine's own ``_target_weights``, so every overlay (shortability,
    vol-targeting, capacity) is applied exactly as it was in the backtest, in
    the engine's order. This is the whole parity guarantee in one function.
    """
    C = bt.closes
    rets = np.zeros_like(C)
    rets[1:] = C[1:] / C[:-1] - 1.0
    rets = np.where(np.isfinite(rets), rets, 0.0)   # ragged panels carry NaN
    w = bt._target_weights(C.shape[0] - 1, rets)
    return {s: float(x) for s, x in zip(bt.symbols, w) if abs(x) > 1e-9}


def _to_book(weights: dict[str, float], asof) -> Book:
    arr = np.array(list(weights.values())) if weights else np.zeros(0)
    return Book(weights=weights, asof=asof,
                gross=float(np.abs(arr).sum()), net=float(arr.sum()))


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
    capacity_aum: float | None = None,
    max_participation: float = 0.02,
) -> Book:
    """The trend+size target book at the last bar of ``panel``.

    Both sleeves are evaluated by the backtest engine, so the book is exactly
    what the engine would have targeted on this bar with these parameters — see
    the module docstring on parity.

    Args:
        panel: long-format OHLCV panel.
        size_lookback: dollar-volume window for the size sleeve.
        vol_target: annualised vol target for the (directional) trend sleeve.
        top_k: names per side in the size sleeve.
        interval: bar interval, for annualising the vol target.
        vol_lookback: trailing window used to estimate realised vol.
        max_leverage: gross cap applied by vol-targeting.
        trend_weight: capital split; 1.0 = trend only, 0.0 = size only.
        min_short_dollar_volume: hard-to-short gate; ``None`` disables it.
        short_dv_lookback: trailing window for the shortability gate.
        capacity_aum: book size in USD for the participation cap. Must match the
            value used in the backtest, or the live book will be sized for a
            different amount of money than the one that was validated.
        max_participation: max share of a name's daily volume the book may take.
    """
    from vfund.backtest.cross_sectional import CrossSectionalBacktester
    from vfund.strategy import CrossSectionalSize, TimeSeriesTrendEnsemble

    if not 0.0 <= trend_weight <= 1.0:
        raise ValueError("trend_weight must be between 0 and 1")

    common = dict(
        rebalance_every=7, interval=interval, cost_bps=10,
        min_short_dollar_volume=min_short_dollar_volume,
        short_dv_lookback=short_dv_lookback,
        capacity_aum=capacity_aum, max_participation=max_participation,
    )
    trend_bt = CrossSectionalBacktester(
        panel, TimeSeriesTrendEnsemble(), neutralize=False, vol_target=vol_target,
        vol_lookback=vol_lookback, max_leverage=max_leverage, **common,
    )
    size_bt = CrossSectionalBacktester(
        panel, CrossSectionalSize(size_lookback), top_k=top_k, **common,
    )

    w_trend = _engine_weights(trend_bt)
    w_size = _engine_weights(size_bt)

    combined: dict[str, float] = {}
    for alloc, sleeve in ((trend_weight, w_trend), (1.0 - trend_weight, w_size)):
        for s, x in sleeve.items():
            combined[s] = combined.get(s, 0.0) + alloc * x
    combined = {s: w for s, w in combined.items() if abs(w) > 1e-6}
    return _to_book(combined, trend_bt.timestamps[-1])


def _sleeve_current_weights(bt) -> tuple[dict[str, float], float]:
    """Latest-bar target weights + return volatility of one sleeve's backtester."""
    weights = _engine_weights(bt)
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
    return _to_book(combined, asof)


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
