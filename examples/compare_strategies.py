"""Example: does a moving-average crossover actually beat buy-and-hold?

Run with:  python examples/compare_strategies.py

This is the question every active strategy must answer. Here we run both on the
same synthetic series and print their reports side by side. Swap in real data
with `vfund.data.storage.load_parquet(...)` once you've fetched some.
"""

from vfund.backtest import Backtester
from vfund.data.synthetic import generate_gbm_bars
from vfund.strategy import BuyAndHold, MACrossover

INTERVAL = "1h"
data = generate_gbm_bars(4000, interval=INTERVAL, mu=0.20, sigma=0.7, seed=11)

for name, strat in [
    ("Buy & Hold", BuyAndHold()),
    ("MA 20/50", MACrossover(fast=20, slow=50)),
]:
    result = Backtester(data, strat, interval=INTERVAL).run()
    m = result.metrics()
    print(result.summary())
    print()

print(
    "Reminder: beating buy-and-hold on ONE synthetic seed is not edge.\n"
    "Real validation = many seeds, out-of-sample data, and after costs."
)
