"""Where does the 3-sleeve edge decay as capital grows past a few hundred k?

Re-runs each sleeve with a capacity cap (a position can't exceed
`max_participation` of a coin's daily dollar volume), blends, and reports the
full-period Sharpe and CAGR at each account size. The edge lives in small-caps,
so it fades as size forces those positions to shrink.

    python examples/capacity_curve.py
"""

import numpy as np
import polars as pl

from vfund.analytics.performance import sharpe_ratio
from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.onchain import load_tvl
from vfund.data.panel import load_panel
from vfund.data.universe import clean_universe
from vfund.strategy import CrossSectionalSize, TimeSeriesTrendEnsemble, TVLDivergence

PPY = 365
broad_dead = pl.concat([clean_universe(load_panel("data/uni_broad.parquet")),
                        load_panel("data/delisted.parquet")])
defi = load_panel("data/tvl_prices.parquet")
tvl = load_tvl("data/tvl.parquet")
SHORT = dict(short_cost_bps_annual=1000, min_short_dollar_volume=5_000_000)
TREND = dict(neutralize=False, vol_target=0.30, vol_lookback=30, max_leverage=3.0)


def sleeve(panel, strat, aum, **kw):
    res = CrossSectionalBacktester(
        panel, strat, rebalance_every=7, interval="1d", cost_bps=10,
        capacity_aum=aum, max_participation=0.02, **kw).run()
    eq = res.equity_curve["equity"].to_numpy()
    ts = res.equity_curve["timestamp"][1:]
    return pl.DataFrame({"timestamp": ts, "r": eq[1:] / eq[:-1] - 1.0})


def combo_sharpe(aum):
    dfs = [
        sleeve(broad_dead, TimeSeriesTrendEnsemble(), aum, **TREND, **SHORT),
        sleeve(broad_dead, CrossSectionalSize(20), aum, top_k=5, **SHORT),
        sleeve(defi, TVLDivergence(60), aum, top_k=5, tvl=tvl, **SHORT),
    ]
    df = dfs[0].rename({"r": "r0"})
    for i, d in enumerate(dfs[1:], 1):
        df = df.join(d.rename({"r": f"r{i}"}), on="timestamp", how="inner")
    R = np.column_stack([df[c].to_numpy() for c in df.columns if c.startswith("r")])
    w = 1 / R.std(axis=0); w /= w.sum()
    r = R @ w
    eq = np.cumprod(1 + r)
    years = r.size / PPY
    return sharpe_ratio(r, PPY), (eq[-1] ** (1 / years) - 1) * 100


base, _ = combo_sharpe(None)
print(f"Unlimited-size Sharpe (the small-money edge): {base:.2f}\n")
print(f"{'account size':>14} | {'Sharpe':>6} {'CAGR':>6} | {'% of edge left':>14}")
print("-" * 52)
for aum in [250_000, 500_000, 750_000, 1_000_000, 2_000_000,
            5_000_000, 10_000_000, 25_000_000]:
    sh, cagr = combo_sharpe(aum)
    print(f"${aum:>12,.0f} | {sh:>6.2f} {cagr:>5.0f}% | {sh/base*100:>12.0f}%")
print("-" * 52)
print("Rule of thumb: the edge is ~gone once Sharpe falls below ~half the")
print("small-money value. (max_participation=2% of daily volume.)")
