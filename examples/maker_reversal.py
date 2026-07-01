"""Can maker execution revive the shelved reversal edge?

Short-term cross-sectional reversal had a real gross signal but died on ~10bp
TAKER costs (crossing the spread every hour). As a MAKER you post limit orders:
you earn a small rebate (or pay a tiny fee) instead of crossing — but you only
fill a fraction of your intended size, and the unfilled part means you hold a
laggier version of a fast signal. This tests whether that trade-off nets out
positive.

Data: data/uni.parquet (30 coins, hourly, ~2023-2024). In-sample only — treat as
a feasibility check, not a validated edge.
    python examples/maker_reversal.py
"""

import numpy as np

from vfund.analytics.performance import sharpe_ratio
from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.intervals import bars_per_year
from vfund.data.panel import load_panel
from vfund.strategy import CrossSectionalReversal

panel = load_panel("data/uni.parquet")
PPY = bars_per_year("1h")


def run(cost_bps, fill_rate):
    res = CrossSectionalBacktester(
        panel, CrossSectionalReversal(2), rebalance_every=1, top_k=5,
        interval="1h", cost_bps=cost_bps, fill_rate=fill_rate,
    ).run()
    eq = res.equity_curve["equity"].to_numpy()
    r = eq[1:] / eq[:-1] - 1.0
    return sharpe_ratio(r, PPY), (eq[-1] / eq[0] - 1) * 100


print("REVERSAL under different execution assumptions (hourly, in-sample)\n")
print(f"{'execution':>26} | {'cost':>6} {'fill':>5} | {'Sharpe':>7} {'return':>8}")
print("-" * 60)
configs = [
    ("taker (what killed it)", 10, 1.0),
    ("taker, cheap (2bp)",      2, 1.0),
    ("maker rebate, full fill", -1, 1.0),
    ("maker rebate, 70% fill",  -1, 0.7),
    ("maker rebate, 50% fill",  -1, 0.5),
    ("maker 1bp fee, 50% fill",  1, 0.5),
]
for label, cost, fill in configs:
    sh, ret = run(cost, fill)
    print(f"{label:>26} | {cost:>5}bp {fill:>5.2f} | {sh:>7.2f} {ret:>7.0f}%")

print("\nIf a realistic maker row (rebate/low fee, partial fill) is solidly")
print("positive, reversal is a market-making edge worth a proper execution build.")
