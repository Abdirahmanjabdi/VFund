"""Build an edge: vol-targeted trend + a size factor, then combine them.

Three research-backed moves, evaluated over the full 2021-2024 cycle:

  1. Vol-target the trend ensemble (managed-futures risk control).
  2. Add the size factor (Liu-Tsyvinski-Wu: small crypto outperforms).
  3. Combine the two at equal risk — two weakly-correlated edges beat one.

Everything is judged beta-adjusted (alpha vs an equal-weight buy&hold), because
a directional book that's just long a bull market is beta, not skill.

Requires data/uni_daily.parquet.  python examples/build_edge.py
"""

from datetime import datetime, timezone

import numpy as np
import polars as pl

from vfund.analytics.performance import alpha_beta, max_drawdown, sharpe_ratio
from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.panel import load_panel
from vfund.strategy import CrossSectionalSize, TimeSeriesTrendEnsemble
from vfund.strategy.cross_sectional import CrossSectionalStrategy

PPY = 365
panel = load_panel("data/uni_daily.parquet")


class EqualWeightLong(CrossSectionalStrategy):
    def scores(self, ctx):
        return np.ones(ctx.closes.shape[1])


def rets_of(res):
    eq = res.equity_curve["equity"].to_numpy()
    return eq[1:] / eq[:-1] - 1.0, res.equity_curve["timestamp"].to_numpy()[1:]


def stats(r):
    eq = 10_000 * np.cumprod(1 + r)
    mdd, _, _ = max_drawdown(np.concatenate([[10_000], eq]))
    return dict(sharpe=sharpe_ratio(r, PPY), maxdd=mdd, ret=float(eq[-1] / 10_000 - 1))


bench = CrossSectionalBacktester(panel, EqualWeightLong(), rebalance_every=7,
                                 neutralize=False, cost_bps=10, interval="1d").run()
br, bts = rets_of(bench)

runs = {"buy&hold": (br, 1.0)}

# 1. Trend ensemble, plain and vol-targeted.
trend_plain = CrossSectionalBacktester(
    panel, TimeSeriesTrendEnsemble(), rebalance_every=7, neutralize=False,
    cost_bps=10, interval="1d").run()
runs["trend-ens"] = (rets_of(trend_plain)[0], None)

trend_vt = CrossSectionalBacktester(
    panel, TimeSeriesTrendEnsemble(), rebalance_every=7, neutralize=False,
    cost_bps=10, interval="1d", vol_target=0.40, vol_lookback=30, max_leverage=3.0).run()
tvt_r = rets_of(trend_vt)[0]
runs["trend+volT"] = (tvt_r, None)

# 2. Size factor (market-neutral).
size = CrossSectionalBacktester(
    panel, CrossSectionalSize(lookback=30), rebalance_every=7, top_k=5,
    cost_bps=10, interval="1d").run()
size_r = rets_of(size)[0]
runs["size"] = (size_r, None)

# 3. Combine trend+volT and size at equal risk (inverse-vol weights).
n = min(tvt_r.size, size_r.size)
a, b = tvt_r[-n:], size_r[-n:]
wa, wb = 1 / a.std(), 1 / b.std()
wa, wb = wa / (wa + wb), wb / (wa + wb)
combo_r = wa * a + wb * b
runs["COMBINED"] = (combo_r, None)

print("FULL CYCLE 2021-2024 (daily, 10bp, beta-adjusted vs buy&hold)\n")
print(f"{'strategy':>12} | {'Sharpe':>6} {'MaxDD':>7} {'TotRet':>8} {'beta':>5} {'alpha/yr':>8} {'a_t':>5}")
print("-" * 62)
for name, (r, _) in runs.items():
    s = stats(r)
    if name == "buy&hold":
        print(f"{name:>12} | {s['sharpe']:>6.2f} {s['maxdd']*100:>6.0f}% {s['ret']*100:>7.0f}%"
              f"  1.00      -    -")
        continue
    ab = alpha_beta(r, br[-len(r):], PPY)
    print(f"{name:>12} | {s['sharpe']:>6.2f} {s['maxdd']*100:>6.0f}% {s['ret']*100:>7.0f}%"
          f" {ab['beta']:>5.2f} {ab['alpha_ann']*100:>7.0f}% {ab['alpha_t']:>5.1f}")

# Regime breakdown for the combined book.
def regime_ret(r, ts, lo, hi):
    mask = (ts >= np.datetime64(lo)) & (ts < np.datetime64(hi))
    rr = r[mask]
    return float(np.prod(1 + rr) - 1) if rr.size else 0.0

_, cts = rets_of(trend_vt)
cts = cts[-n:]
print("\nREGIME BREAKDOWN - total return\n")
print(f"{'regime':>12} | {'buy&hold':>10} {'COMBINED':>10}")
print("-" * 38)
for name, lo, hi in [("2021 bull", "2021-01-01", "2022-01-01"),
                     ("2022 BEAR", "2022-01-01", "2023-01-01"),
                     ("2023-24 up", "2023-01-01", "2025-01-01")]:
    print(f"{name:>12} | {regime_ret(br[-n:], cts, lo, hi)*100:>9.0f}% "
          f"{regime_ret(combo_r, cts, lo, hi)*100:>9.0f}%")

corr = np.corrcoef(a, b)[0, 1]
print(f"\ntrend/size return correlation: {corr:+.2f}  "
      f"(low correlation is why combining helps)")
