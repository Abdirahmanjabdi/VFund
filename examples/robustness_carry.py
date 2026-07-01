"""Pressure-test the funding-carry candidate from every angle.

Prerequisite data (see funding_carry_study.py):
    data/uni.parquet, data/funding.parquet

We (1) sweep configs and record every trial's Sharpe, (2) take the best, then
subject it to sub-period stability, a coin-dropping bootstrap, and the
Probabilistic / Deflated Sharpe ratios. The Deflated Sharpe is the honest one:
it asks "given I tried this many configs, is the winner still real?"
"""

from itertools import product

import numpy as np

from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.panel import load_funding, load_panel
from vfund.research.robustness import (
    RobustnessResult,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    subperiod_stability,
    universe_bootstrap,
)
from vfund.strategy import FundingCarry

panel = load_panel("data/uni.parquet")
funding = load_funding("data/funding.parquet")


def returns_of(res):
    eq = res.equity_curve["equity"].to_numpy()
    return eq[1:] / eq[:-1] - 1.0


# (1) Sweep every config; record each trial's Sharpe and keep the best. -------
REBAL = [8, 24]
TOPK = [3, 5, None]
SMOOTH = [1, 3, 8]

trial_sharpes, best = [], None
for rebalance, top_k, smooth in product(REBAL, TOPK, SMOOTH):
    res = CrossSectionalBacktester(
        panel, FundingCarry(smooth=smooth), funding=funding,
        rebalance_every=rebalance, top_k=top_k, cost_bps=5, interval="1h",
    ).run()
    r = returns_of(res)
    per_obs_sharpe = r.mean() / r.std() if r.std() > 0 else 0.0
    trial_sharpes.append(per_obs_sharpe)
    ann = res.metrics()["sharpe"]
    if best is None or ann > best["ann"]:
        best = {"ann": ann, "cfg": (rebalance, top_k, smooth), "returns": r}

rb, tk, sm = best["cfg"]
print(f"best config: rebalance={rb}h  top_k={tk}  smooth={sm}  "
      f"(annualised Sharpe {best['ann']:.2f}, {len(trial_sharpes)} configs tried)\n")

bt_kwargs = dict(rebalance_every=rb, top_k=tk, cost_bps=5, interval="1h")

# (2) Sub-period stability and universe bootstrap on the winning config. ------
sp = subperiod_stability(
    panel, lambda: FundingCarry(smooth=sm),
    n_periods=6, backtest_kwargs=bt_kwargs, funding=funding,
)
uni = universe_bootstrap(
    panel, lambda: FundingCarry(smooth=sm),
    n_draws=40, subset_size=20, backtest_kwargs=bt_kwargs, funding=funding, seed=1,
)

# (3) Probabilistic and Deflated Sharpe on the winner's returns. --------------
psr = probabilistic_sharpe_ratio(best["returns"])
dsr = deflated_sharpe_ratio(best["returns"], np.array(trial_sharpes))

print(RobustnessResult(sp, uni, psr, dsr, n_trials=len(trial_sharpes)).summary())
print(
    "\nRead: high sub-period + bootstrap agreement and a Deflated Sharpe well\n"
    "above 50% would mean a real edge. A DSR near/below 50% means the winner is\n"
    "likely just the luckiest of the configs tried - keep it on the watchlist,\n"
    "don't trade it."
)
