"""Does on-chain protocol FEES (revenue) carry signal — and is it a new diversifier?

Fees = real protocol revenue, distinct from TVL (parked capital). We test:
  * fees momentum  — long protocols with growing revenue
  * fees divergence — revenue grew but price lagged (fundamental value)

selecting in-sample (2021-24), judging out-of-sample (2025-26); and we check the
correlation of the fees sleeve with the existing TVL-divergence sleeve — a low
correlation means fees is a genuinely new, stackable edge.

Reuses the TVL strategies by feeding fees as the on-chain input.
Prereqs: data/tvl_prices.parquet, data/fees.parquet, data/tvl.parquet.
    python examples/onchain_fees.py
"""

import numpy as np

from vfund.analytics.performance import sharpe_ratio
from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.onchain import load_tvl
from vfund.data.panel import load_panel
from vfund.strategy import TVLDivergence, TVLMomentum

PPY = 365
prices = load_panel("data/tvl_prices.parquet")
fees = load_tvl("data/fees.parquet")   # saved with the on-chain column named 'tvl'
tvl = load_tvl("data/tvl.parquet")


def sleeve_returns(strat, feature):
    res = CrossSectionalBacktester(prices, strat, tvl=feature, rebalance_every=7,
                                   top_k=5, cost_bps=10, interval="1d").run()
    ec = res.equity_curve
    eq = ec["equity"].to_numpy()
    ts = ec["timestamp"].to_numpy()[1:]
    return eq[1:] / eq[:-1] - 1.0, np.array([int(str(t)[:4]) for t in ts])


def evaluate(name, make, grid, feature):
    best = None
    for p in grid:
        r, yr = sleeve_returns(make(p), feature)
        ins, oos = yr < 2025, yr >= 2025
        if ins.sum() < 30 or oos.sum() < 30:
            continue
        row = dict(p=p, is_sh=sharpe_ratio(r[ins], PPY), oos_sh=sharpe_ratio(r[oos], PPY), r=r)
        if best is None or row["is_sh"] > best["is_sh"]:
            best = row
    v = ("CANDIDATE" if best["is_sh"] > 0.5 and best["oos_sh"] > 0.5 else
         "promising" if best["is_sh"] > 0.5 and best["oos_sh"] > 0.2 else
         "overfit" if best["is_sh"] > 0.5 else
         "noise?" if best["oos_sh"] > 0.5 else "dead")
    print(f"{name:>18} | {best['p']:>4} | {best['is_sh']:>7.2f} | {best['oos_sh']:>8.2f}  {v}")
    return best["r"]


print("On-chain FEES signal (select IS / judge OOS)\n")
print(f"{'signal':>18} | {'sel':>4} | {'IS Shrp':>7} | {'OOS Shrp':>8}")
print("-" * 56)
evaluate("fees momentum", lambda p: TVLMomentum(p), [30, 60, 90, 180], fees)
fees_div = evaluate("fees divergence", lambda p: TVLDivergence(p), [30, 60, 90, 180], fees)
tvl_div = evaluate("TVL divergence", lambda p: TVLDivergence(p), [30, 60, 90, 180], tvl)

n = min(fees_div.size, tvl_div.size)
corr = np.corrcoef(fees_div[-n:], tvl_div[-n:])[0, 1]
print("-" * 56)
print(f"\nfees-divergence vs TVL-divergence correlation: {corr:+.2f}")
print("Low correlation + positive both periods = a new, stackable on-chain sleeve.")
