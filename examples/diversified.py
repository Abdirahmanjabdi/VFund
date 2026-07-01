"""Does adding the on-chain sleeve improve the combined book? (Diversification)

Three weakly-correlated real edges, each on its own universe, blended at equal
risk (inverse-vol):
  * trend  — directional time-series trend (broad + delisted coins)
  * size   — long small / short large (broad + delisted coins)
  * onchain— TVL divergence: usage grew but price lagged (DeFi coins w/ TVL)

Diversification is the one genuine free lunch: uncorrelated edges raise Sharpe
without a better single signal. We check the correlations, then compare the
2-sleeve (trend+size) book to the 3-sleeve book, in-sample and out-of-sample.

Prereq: data/uni_broad.parquet, data/delisted.parquet, data/tvl_prices.parquet,
data/tvl.parquet.   python examples/diversified.py
"""

import numpy as np
import polars as pl

from vfund.analytics.performance import max_drawdown, sharpe_ratio
from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.onchain import load_tvl
from vfund.data.panel import load_panel
from vfund.data.universe import clean_universe
from vfund.strategy import (
    CrossSectionalSize, TimeSeriesTrendEnsemble, TVLDivergence,
)

PPY = 365
broad_dead = pl.concat([clean_universe(load_panel("data/uni_broad.parquet")),
                        load_panel("data/delisted.parquet")])
tvl_prices = load_panel("data/tvl_prices.parquet")
tvl = load_tvl("data/tvl.parquet")
COST = dict(interval="1d", cost_bps=10, short_cost_bps_annual=1000)


def sleeve(name, panel, strat, **kw):
    res = CrossSectionalBacktester(panel, strat, rebalance_every=7, **COST, **kw).run()
    ec = res.equity_curve
    eq = ec["equity"].to_numpy()
    return pl.DataFrame({"timestamp": ec["timestamp"][1:],
                         name: eq[1:] / eq[:-1] - 1.0})


trend = sleeve("trend", broad_dead, TimeSeriesTrendEnsemble(), neutralize=False,
               vol_target=0.30, vol_lookback=30, max_leverage=3.0,
               min_short_dollar_volume=5_000_000)
size = sleeve("size", broad_dead, CrossSectionalSize(20), top_k=5,
              min_short_dollar_volume=5_000_000)
onchain = sleeve("onchain", tvl_prices, TVLDivergence(60), top_k=5, tvl=tvl)

df = trend.join(size, on="timestamp", how="inner").join(onchain, on="timestamp", how="inner").sort("timestamp")
yr = np.array([int(str(t)[:4]) for t in df["timestamp"].to_numpy()])
R = {c: df[c].to_numpy() for c in ("trend", "size", "onchain")}

print("SLEEVE CORRELATIONS (near 0 = good diversifier)\n")
names = ["trend", "size", "onchain"]
print("          " + "".join(f"{n:>9}" for n in names))
for a in names:
    print(f"{a:>9} " + "".join(f"{np.corrcoef(R[a], R[b])[0,1]:>9.2f}" for b in names))


def blend(cols):
    w = np.array([1 / R[c].std() for c in cols])
    w /= w.sum()
    return sum(wi * R[c] for wi, c in zip(w, cols))


def stats(r, mask):
    rr = r[mask]
    eq = np.concatenate([[1.0], np.cumprod(1 + rr)])
    mdd, _, _ = max_drawdown(eq)
    return sharpe_ratio(rr, PPY), (np.prod(1 + rr) - 1) * 100, mdd * 100


ins, oos = yr < 2025, yr >= 2025
print("\n" + "=" * 60)
print(f"{'book':>22} | {'IS Shrp':>7} {'OOS Shrp':>8} {'OOS ret':>8} {'OOS DD':>7}")
print("-" * 60)
for label, cols in [("trend+size (2 sleeves)", ["trend", "size"]),
                    ("+ on-chain (3 sleeves)", ["trend", "size", "onchain"])]:
    r = blend(cols)
    is_sh, _, _ = stats(r, ins)
    oos_sh, oos_ret, oos_dd = stats(r, oos)
    print(f"{label:>22} | {is_sh:>7.2f} {oos_sh:>8.2f} {oos_ret:>7.0f}% {oos_dd:>6.0f}%")
print("=" * 60)
print("\nIf the 3-sleeve OOS Sharpe > 2-sleeve, the on-chain edge is adding real,")
print("uncorrelated value - diversification improving returns for free.")
