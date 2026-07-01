"""Does on-chain TVL growth predict crypto returns? Honest IS/OOS test.

TVL momentum = long protocols whose total-value-locked is growing, short those
shrinking. We compare it to plain price momentum on the SAME coins — if TVL only
recovers what price momentum already knows, it isn't adding anything.

Select each lookback in-sample (2021-2024), judge out-of-sample (2025-2026).

Prereq: data/tvl_prices.parquet (prices) and data/tvl.parquet (TVL).
    python examples/onchain_tvl.py
"""

import numpy as np

from vfund.analytics.performance import sharpe_ratio
from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.onchain import load_tvl
from vfund.data.panel import load_panel
from vfund.strategy import CrossSectionalMomentum, TVLDivergence, TVLMomentum

PPY = 365
prices = load_panel("data/tvl_prices.parquet")
tvl = load_tvl("data/tvl.parquet")
n_coins = prices["symbol"].n_unique()


def evaluate(label, make, grid, use_tvl):
    best = None
    for p in grid:
        res = CrossSectionalBacktester(
            prices, make(p), tvl=tvl if use_tvl else None,
            rebalance_every=7, top_k=5, cost_bps=10, interval="1d",
        ).run()
        eq = res.equity_curve["equity"].to_numpy()
        ts = res.equity_curve["timestamp"].to_numpy()[1:]
        r = eq[1:] / eq[:-1] - 1.0
        yr = np.array([int(str(t)[:4]) for t in ts])
        is_m, oos_m = yr < 2025, yr >= 2025
        if is_m.sum() < 30 or oos_m.sum() < 30:
            continue
        is_sh = sharpe_ratio(r[is_m], PPY)
        row = dict(p=p, is_sh=is_sh, oos_sh=sharpe_ratio(r[oos_m], PPY),
                   oos_ret=float(np.prod(1 + r[oos_m]) - 1))
        if best is None or is_sh > best["is_sh"]:
            best = row
    is_sh, oos_sh = best["is_sh"], best["oos_sh"]
    verdict = ("CANDIDATE" if is_sh > 0.5 and oos_sh > 0.5 else
               "promising" if is_sh > 0.5 and oos_sh > 0.2 else  # positive both, modest
               "overfit" if is_sh > 0.5 and oos_sh <= 0 else
               "faded" if is_sh > 0.5 else
               "noise?" if oos_sh > 0.5 else "dead")
    print(f"{label:>16} | {best['p']:>4} | {is_sh:>7.2f} | {oos_sh:>8.2f} {best['oos_ret']*100:>7.0f}%  {verdict}")


print(f"On-chain TVL vs price momentum, {n_coins} DeFi coins (select IS, judge OOS)\n")
print(f"{'signal':>16} | {'sel':>4} | {'IS Shrp':>7} | {'OOS Shrp':>8} {'OOSret':>8}  verdict")
print("-" * 62)
evaluate("TVL momentum", lambda p: TVLMomentum(p), [14, 30, 60, 90], use_tvl=True)
evaluate("TVL divergence", lambda p: TVLDivergence(p), [30, 60, 90, 180], use_tvl=True)
evaluate("price momentum", lambda p: CrossSectionalMomentum(p), [14, 30, 60, 90], use_tvl=False)
print("\nTVL 'divergence' (usage grew, price lagged) positive in BOTH periods is")
print("the signal to watch - a fundamental value bet on differentiated data. Small")
print("universe + repeated OOS looks mean: promising lead, not a verdict.")
