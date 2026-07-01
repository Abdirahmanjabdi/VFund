"""Run every hypothesis through the same honest gauntlet and rank them.

For each idea we report:
  * OOS Sharpe   — walk-forward (params chosen in-sample, judged out-of-sample)
  * Univ %pos    — fraction of random coin-subsets where it stays positive
  * PSR          — P(true Sharpe > 0) given sample length & fat tails
  * DSR          — Deflated Sharpe: PSR after adjusting for how many params tried

A result is only interesting if ALL of these are strong together. Expect most to
fail — that is the job. Requires data/uni.parquet (+ data/funding.parquet for carry).

    python examples/hypothesis_gauntlet.py
"""

import numpy as np

from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.panel import load_funding, load_panel
from vfund.research import walk_forward
from vfund.research.robustness import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    universe_bootstrap,
)
from vfund.strategy import (
    CrossSectionalLowVol,
    CrossSectionalMomentum,
    CrossSectionalReversal,
    CrossSectionalValue,
    FundingCarry,
    TimeSeriesTrend,
)
from vfund.strategy.cross_sectional import CrossSectionalStrategy

panel = load_panel("data/uni.parquet")
try:
    funding = load_funding("data/funding.parquet")
except FileNotFoundError:
    funding = None


class EqualWeightLong(CrossSectionalStrategy):
    """Benchmark: hold every coin, equal weight (market beta)."""

    def scores(self, ctx):
        return np.ones(ctx.closes.shape[1])


def _returns(res):
    eq = res.equity_curve["equity"].to_numpy()
    return eq[1:] / eq[:-1] - 1.0


# name -> (factory, param grid, engine kwargs, needs_funding)
HYPOTHESES = {
    "reversal":   (CrossSectionalReversal, [1, 2, 3, 6],       dict(rebalance_every=1, top_k=5), False),
    "momentum":   (CrossSectionalMomentum, [72, 168, 336, 720], dict(rebalance_every=24, top_k=5), False),
    "lowvol":     (CrossSectionalLowVol,   [72, 168, 336],     dict(rebalance_every=24, top_k=5), False),
    "value":      (CrossSectionalValue,    [336, 720, 1440],   dict(rebalance_every=24, top_k=5), False),
    "trend*":     (TimeSeriesTrend,        [48, 168, 336],     dict(rebalance_every=24, neutralize=False), False),
    "carry":      (lambda p: FundingCarry(smooth=p), [1, 3, 8], dict(rebalance_every=24, top_k=5), True),
}

COMMON = dict(cost_bps=5.0, interval="1h")
print(f"{'hypothesis':>11} | {'OOS Shrp':>8} {'Univ%pos':>8} {'PSR':>6} {'DSR':>6}  verdict")
print("-" * 62)

for name, (factory, grid, kw, needs_funding) in HYPOTHESES.items():
    make = (lambda p: factory(p)) if not isinstance(factory, type) else factory
    bt_kwargs = {**COMMON, **kw}
    fund = funding if needs_funding else None

    # Full-sample per-config Sharpes (for DSR) + pick best config.
    trial_sr, best = [], None
    for p in grid:
        res = CrossSectionalBacktester(panel, make(p), funding=fund, **bt_kwargs).run()
        r = _returns(res)
        trial_sr.append(r.mean() / r.std() if r.std() > 0 else 0.0)
        ann = res.metrics()["sharpe"]
        if best is None or ann > best[0]:
            best = (ann, p, r)
    best_p, best_r = best[1], best[2]

    # Walk-forward OOS Sharpe.
    wf = walk_forward(
        panel, lambda lookback: make(lookback), [{"lookback": p} for p in grid],
        train_size=4000, test_size=1500, interval="1h",
        backtest_kwargs=bt_kwargs, funding=fund,
    )
    oos = wf.oos_sharpe()

    # Universe bootstrap on the best config.
    uni = universe_bootstrap(
        panel, lambda: make(best_p), n_draws=30, subset_size=20,
        backtest_kwargs=bt_kwargs, funding=fund, seed=1,
    )
    upos = float(np.mean(uni > 0))
    psr = probabilistic_sharpe_ratio(best_r)
    dsr = deflated_sharpe_ratio(best_r, np.array(trial_sr))

    passed = oos > 0.5 and upos > 0.65 and dsr > 0.6
    verdict = "PASS" if passed else ("weak" if oos > 0 else "REJECT")
    print(f"{name:>11} | {oos:>8.2f} {upos*100:>7.0f}% {psr*100:>5.0f}% {dsr*100:>5.0f}%  {verdict}")

# Market benchmark for context (esp. for the directional trend*).
bench = CrossSectionalBacktester(
    panel, EqualWeightLong(), rebalance_every=24, neutralize=False, **COMMON
).run()
print("-" * 62)
print(f"{'buy&hold':>11} | {bench.metrics()['sharpe']:>8.2f}   (equal-weight long benchmark)")
print("\n* trend is directional (carries market beta) - compare it to buy&hold,\n"
      "  not to zero. The neutral strategies are already market-neutral.")
