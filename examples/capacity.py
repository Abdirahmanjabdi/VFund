"""How much money can this strategy actually run? And can we tame the drawdown?

Capacity: re-runs the combined book at increasing account sizes. A position is
capped so you're never more than ``max_participation`` of a coin's daily dollar
volume — so at large AUM the small-cap positions get clipped and the edge decays.
The point where Sharpe falls off is the strategy's realistic capacity.

Circuit-breaker: shows the drawdown de-risking overlay cutting the worst drawdown.

Requires data/uni_broad.parquet + data/delisted.parquet.
    python examples/capacity.py
"""

import numpy as np
import polars as pl

from vfund.analytics.performance import max_drawdown, sharpe_ratio
from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.panel import load_panel
from vfund.data.universe import clean_universe
from vfund.strategy import CrossSectionalSize, TimeSeriesTrendEnsemble

PPY = 365
broad = clean_universe(load_panel("data/uni_broad.parquet"))
panel = pl.concat([broad, load_panel("data/delisted.parquet")])


def combo(aum=None, derisk=None):
    common = dict(interval="1d", cost_bps=10, short_cost_bps_annual=1000,
                  min_short_dollar_volume=5_000_000, capacity_aum=aum,
                  max_participation=0.02)
    if derisk:
        common.update(dd_derisk_start=derisk[0], dd_derisk_full=derisk[1], dd_derisk_floor=derisk[2])

    def rets(res):
        eq = res.equity_curve["equity"].to_numpy()
        return eq[1:] / eq[:-1] - 1.0

    t = rets(CrossSectionalBacktester(panel, TimeSeriesTrendEnsemble(), rebalance_every=7,
             neutralize=False, vol_target=0.30, vol_lookback=30, max_leverage=3.0, **common).run())
    s = rets(CrossSectionalBacktester(panel, CrossSectionalSize(20), rebalance_every=7,
             top_k=5, **common).run())
    n = min(t.size, s.size)
    t, s = t[-n:], s[-n:]
    wa, wb = 1 / t.std(), 1 / s.std()
    wa, wb = wa / (wa + wb), wb / (wa + wb)
    return wa * t + wb * s


def stats(r):
    eq = np.concatenate([[1.0], np.cumprod(1 + r)])
    mdd, _, _ = max_drawdown(eq)
    return sharpe_ratio(r, PPY), (eq[-1] - 1) * 100, mdd * 100


print("CAPACITY CURVE - how the edge decays as account size grows\n")
print(f"{'AUM':>12} | {'Sharpe':>6} {'TotRet':>8} {'MaxDD':>7}")
print("-" * 40)
for aum in [None, 1e5, 1e6, 1e7, 3e7, 1e8, 3e8]:
    sh, ret, mdd = stats(combo(aum=aum))
    label = "unlimited" if aum is None else f"${aum:,.0f}"
    print(f"{label:>12} | {sh:>6.2f} {ret:>7.0f}% {mdd:>6.0f}%")

print("\nDRAWDOWN CIRCUIT-BREAKER (at $10M AUM)\n")
print(f"{'setting':>16} | {'Sharpe':>6} {'TotRet':>8} {'MaxDD':>7}")
print("-" * 44)
for label, dr in [("off", None), ("derisk 15->35%", (0.15, 0.35, 0.3))]:
    sh, ret, mdd = stats(combo(aum=1e7, derisk=dr))
    print(f"{label:>16} | {sh:>6.2f} {ret:>7.0f}% {mdd:>6.0f}%")

print("\nCapacity: where Sharpe collapses is roughly the most money this can hold.")
print("Circuit-breaker: trades some return for a shallower worst-case drawdown.")
