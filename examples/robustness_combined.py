"""The decisive honesty test: put the COMBINED trend+size book through the same
gauntlet that killed funding carry - universe bootstrap, sub-period stability,
and a Deflated Sharpe that adjusts for how many combined configs we tried.

Requires data/uni_daily.parquet.  python examples/robustness_combined.py
"""

from itertools import product

import numpy as np

from vfund.analytics.performance import alpha_beta, sharpe_ratio
from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.panel import load_panel
from vfund.research.robustness import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from vfund.strategy import CrossSectionalSize, TimeSeriesTrendEnsemble
from vfund.strategy.cross_sectional import CrossSectionalStrategy

PPY = 365
panel = load_panel("data/uni_daily.parquet")


class EqualWeightLong(CrossSectionalStrategy):
    def scores(self, ctx):
        return np.ones(ctx.closes.shape[1])


def _rets(res):
    eq = res.equity_curve["equity"].to_numpy()
    return eq[1:] / eq[:-1] - 1.0


def combo_returns(pnl, size_lb=30, vt=0.40, rebalance=7):
    """Equal-risk blend of vol-targeted trend and the size factor."""
    t = CrossSectionalBacktester(
        pnl, TimeSeriesTrendEnsemble(), rebalance_every=rebalance, neutralize=False,
        cost_bps=10, interval="1d", vol_target=vt, vol_lookback=30, max_leverage=3.0).run()
    s = CrossSectionalBacktester(
        pnl, CrossSectionalSize(size_lb), rebalance_every=rebalance, top_k=5,
        cost_bps=10, interval="1d").run()
    tr, sr = _rets(t), _rets(s)
    n = min(tr.size, sr.size)
    tr, sr = tr[-n:], sr[-n:]
    wa, wb = 1 / tr.std(), 1 / sr.std()
    wa, wb = wa / (wa + wb), wb / (wa + wb)
    return wa * tr + wb * sr


# --- Deflated Sharpe over the combined-book config grid -----------------------
SIZE_LB, VT, REB = [20, 30, 50], [0.30, 0.40, 0.50], [7, 14]
trial_sr, best = [], None
for sl, vt, rb in product(SIZE_LB, VT, REB):
    r = combo_returns(panel, sl, vt, rb)
    per_obs = r.mean() / r.std() if r.std() > 0 else 0.0
    trial_sr.append(per_obs)
    ann = sharpe_ratio(r, PPY)
    if best is None or ann > best[0]:
        best = (ann, (sl, vt, rb), r)

sl, vt, rb = best[1]
best_r = best[2]
print(f"best combined config: size_lb={sl} vol_target={vt} rebalance={rb}d "
      f"(Sharpe {best[0]:.2f}, {len(trial_sr)} configs tried)\n")

# --- Universe (coin-dropping) bootstrap ---------------------------------------
symbols = panel["symbol"].unique().to_list()
rng = np.random.default_rng(1)
uni = []
for _ in range(40):
    pick = list(rng.choice(symbols, size=15, replace=False))
    sub = panel.filter(panel["symbol"].is_in(pick))
    uni.append(sharpe_ratio(combo_returns(sub, sl, vt, rb), PPY))
uni = np.array(uni)

# --- Sub-period stability -----------------------------------------------------
import polars as pl
ts = panel.select("timestamp").unique().sort("timestamp")["timestamp"]
n = ts.len(); size = n // 6
subs = []
for k in range(6):
    chunk = ts.slice(k * size, size if k < 5 else n - k * size)
    sub = panel.filter(pl.col("timestamp").is_in(chunk.implode()))
    subs.append(sharpe_ratio(combo_returns(sub, sl, vt, rb), PPY))
subs = np.array(subs)

# --- Significance -------------------------------------------------------------
bench = CrossSectionalBacktester(panel, EqualWeightLong(), rebalance_every=7,
                                 neutralize=False, cost_bps=10, interval="1d").run()
br = _rets(bench)[-best_r.size:]
ab = alpha_beta(best_r, br, PPY)
psr = probabilistic_sharpe_ratio(best_r)
dsr = deflated_sharpe_ratio(best_r, np.array(trial_sr))

print("=" * 58)
print("  COMBINED trend+size - robustness verdict")
print("=" * 58)
print(f"  Full-sample Sharpe          {sharpe_ratio(best_r, PPY):>8.2f}")
print(f"  Alpha (ann) / t-stat        {ab['alpha_ann']*100:>6.0f}% / {ab['alpha_t']:.1f}")
print("-" * 58)
print(f"  Sub-periods positive        {int((subs>0).sum())}/6")
print(f"  Universe bootstrap %pos     {float(np.mean(uni>0))*100:>7.0f}%")
print(f"  Universe median / 5th pct   {np.median(uni):>5.2f} / {np.percentile(uni,5):.2f}")
print("-" * 58)
print(f"  Probabilistic Sharpe (>0)   {psr*100:>7.1f}%")
print(f"  Deflated Sharpe ({len(trial_sr)} cfgs)  {dsr*100:>6.1f}%")
print("=" * 58)
print("\nBar: Univ %pos > ~70%, DSR > ~60%, alpha t > ~2 => worth paper-trading.")
