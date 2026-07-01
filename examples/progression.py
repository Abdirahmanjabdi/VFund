"""How the $100k / 2021-today result changed as we added honesty, then improved it.

Three stages, all from a $100,000 start over the same 2021-2026 window:

  A) NAIVE      — survivor universe, free shorts, no hard-to-short (flattering)
  B) HONEST 2   — broad + delisted coins, short costs, hard-to-short gate
  C) HONEST 3   — B plus the uncorrelated on-chain sleeve (diversification)

A->B shows de-biasing (the number gets more truthful). B->C shows a genuine
improvement (diversification). Splits out-of-sample (2025-26) too.

    python examples/progression.py
"""

import numpy as np
import polars as pl

from vfund.analytics.performance import max_drawdown, sharpe_ratio
from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.onchain import load_tvl
from vfund.data.panel import load_panel
from vfund.data.universe import clean_universe
from vfund.strategy import CrossSectionalSize, TimeSeriesTrendEnsemble, TVLDivergence

PPY = 365
survivor = load_panel("data/uni_2026.parquet")
broad_dead = pl.concat([clean_universe(load_panel("data/uni_broad.parquet")),
                        load_panel("data/delisted.parquet")])
defi = load_panel("data/tvl_prices.parquet")
tvl = load_tvl("data/tvl.parquet")


def sleeve(panel, strat, **kw):
    res = CrossSectionalBacktester(panel, strat, rebalance_every=7, interval="1d",
                                   cost_bps=10, **kw).run()
    ec = res.equity_curve
    eq = ec["equity"].to_numpy()
    return pl.DataFrame({"timestamp": ec["timestamp"][1:], "r": eq[1:] / eq[:-1] - 1.0})


def blend(dfs):
    df = dfs[0].rename({"r": "r0"})
    for i, d in enumerate(dfs[1:], 1):
        df = df.join(d.rename({"r": f"r{i}"}), on="timestamp", how="inner")
    df = df.sort("timestamp")
    cols = [c for c in df.columns if c.startswith("r")]
    R = np.column_stack([df[c].to_numpy() for c in cols])
    w = 1 / R.std(axis=0)
    w /= w.sum()
    return R @ w, np.array([int(str(t)[:4]) for t in df["timestamp"].to_numpy()])


def report(label, r, yr):
    eq = 100_000 * np.cumprod(1 + r)
    years = r.size / PPY
    mdd, _, _ = max_drawdown(np.concatenate([[100_000.0], eq]))
    oos = yr >= 2025
    oos_sh = sharpe_ratio(r[oos], PPY) if oos.sum() > 5 else float("nan")
    print(f"{label:>26} | ${eq[-1]:>10,.0f} {((eq[-1]/1e5)**(1/years)-1)*100:>6.1f}% "
          f"{sharpe_ratio(r, PPY):>6.2f} {mdd*100:>6.0f}% | {oos_sh:>6.2f}")


SHORT = dict(short_cost_bps_annual=1000, min_short_dollar_volume=5_000_000)
TREND = dict(neutralize=False, vol_target=0.30, vol_lookback=30, max_leverage=3.0)

# A) naive: survivor universe, free shorts, no hard-to-short gate
a_t = sleeve(survivor, TimeSeriesTrendEnsemble(), **TREND)
a_s = sleeve(survivor, CrossSectionalSize(20), top_k=5)
# B) honest 2-sleeve
b_t = sleeve(broad_dead, TimeSeriesTrendEnsemble(), **TREND, **SHORT)
b_s = sleeve(broad_dead, CrossSectionalSize(20), top_k=5, **SHORT)
# C) + on-chain
c_o = sleeve(defi, TVLDivergence(60), top_k=5, tvl=tvl, **SHORT)

print(f"{'stage':>26} | {'final $':>11} {'CAGR':>6} {'Shrp':>6} {'MaxDD':>6} | {'OOS Shrp':>8}")
print("-" * 74)
report("A) naive (survivor)", *blend([a_t, a_s]))
report("B) honest (broad+dead)", *blend([b_t, b_s]))
report("C) honest + on-chain (3)", *blend([b_t, b_s, c_o]))
print("-" * 74)
print("A->B = de-biasing (more truthful). B->C = real improvement (diversification).")
