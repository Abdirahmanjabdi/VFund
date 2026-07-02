"""Fold the on-chain FEES sleeve into the book: does a 4th sleeve help?

Standalone measurement — no changes to the live signal or paper tracker. Four
weakly-related sleeves, blended at equal risk (inverse-vol):

  1 trend    (broad + delisted)      directional time-series trend
  2 size     (broad + delisted)      long small / short large
  3 tvl-div  (DeFi coins)            usage grew but price lagged
  4 fees-div (DeFi coins)            revenue grew but price lagged  <- NEW

We print the sleeve correlation matrix, then compare the 3-sleeve book to the
4-sleeve book, in-sample and out-of-sample.

Prereqs: uni_broad, delisted, tvl_prices, tvl, fees parquet files.
    python examples/four_sleeve.py
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
broad_dead = pl.concat([clean_universe(load_panel("data/uni_broad.parquet")),
                        load_panel("data/delisted.parquet")])
defi = load_panel("data/tvl_prices.parquet")
tvl = load_tvl("data/tvl.parquet")
fees = load_tvl("data/fees.parquet")     # on-chain column stored as 'tvl'
SHORT = dict(short_cost_bps_annual=1000, min_short_dollar_volume=5_000_000)
TREND = dict(neutralize=False, vol_target=0.30, vol_lookback=30, max_leverage=3.0)


def sleeve(name, panel, strat, **kw):
    res = CrossSectionalBacktester(panel, strat, rebalance_every=7, interval="1d",
                                   cost_bps=10, **kw).run()
    ec = res.equity_curve
    eq = ec["equity"].to_numpy()
    return pl.DataFrame({"timestamp": ec["timestamp"][1:], name: eq[1:] / eq[:-1] - 1.0})


s = [sleeve("trend", broad_dead, TimeSeriesTrendEnsemble(), **TREND, **SHORT),
     sleeve("size", broad_dead, CrossSectionalSize(20), top_k=5, **SHORT),
     sleeve("tvl", defi, TVLDivergence(60), top_k=5, tvl=tvl),
     sleeve("fees", defi, TVLDivergence(30), top_k=5, tvl=fees)]
df = s[0]
for d in s[1:]:
    df = df.join(d, on="timestamp", how="inner")
df = df.sort("timestamp")
names = ["trend", "size", "tvl", "fees"]
R = {n: df[n].to_numpy() for n in names}
yr = np.array([int(str(t)[:4]) for t in df["timestamp"].to_numpy()])

print("SLEEVE CORRELATIONS\n")
print("       " + "".join(f"{n:>7}" for n in names))
for a in names:
    print(f"{a:>6} " + "".join(f"{np.corrcoef(R[a], R[b])[0,1]:>7.2f}" for b in names))


def blend(cols):
    w = np.array([1 / R[c].std() for c in cols]); w /= w.sum()
    return sum(wi * R[c] for wi, c in zip(w, cols))


def stat(label, r):
    eq = np.cumprod(1 + r); yrs = r.size / PPY
    mdd, _, _ = max_drawdown(np.concatenate([[1.0], eq]))
    ins, oos = yr < 2025, yr >= 2025
    print(f"{label:>26} | {sharpe_ratio(r,PPY):>6.2f} {(eq[-1]**(1/yrs)-1)*100:>6.0f}% "
          f"{mdd*100:>6.0f}% | {sharpe_ratio(r[ins],PPY):>6.2f} {sharpe_ratio(r[oos],PPY):>7.2f}")


print("\n" + "=" * 66)
print(f"{'book':>26} | {'Sharpe':>6} {'CAGR':>6} {'MaxDD':>6} | {'IS':>6} {'OOS':>7}")
print("-" * 66)
stat("3-sleeve (no fees)", blend(["trend", "size", "tvl"]))
stat("4-sleeve (+ fees)", blend(["trend", "size", "tvl", "fees"]))
print("=" * 66)
print("\nIf the 4-sleeve OOS Sharpe > 3-sleeve, the fees sleeve earns its place.")
