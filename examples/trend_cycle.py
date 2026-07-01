"""Does time-series trend earn real alpha, or just ride market beta?

Runs directional trend against an equal-weight buy&hold benchmark over a full
2021-2024 cycle (bull -> 2022 bear -> recovery), then:

  * regresses trend on the benchmark to isolate ALPHA from BETA, and
  * breaks performance down by regime — the real question is whether trend
    sidesteps the 2022 bear drawdown that buy&hold ate.

Requires data/uni_daily.parquet (daily, spanning 2022).
    python examples/trend_cycle.py
"""

from datetime import datetime, timezone

import numpy as np
import polars as pl

from vfund.analytics.performance import alpha_beta, max_drawdown, sharpe_ratio
from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.panel import load_panel
from vfund.strategy import TimeSeriesTrend, TimeSeriesTrendEnsemble
from vfund.strategy.cross_sectional import CrossSectionalStrategy

PPY = 365  # daily bars
panel = load_panel("data/uni_daily.parquet")


class EqualWeightLong(CrossSectionalStrategy):
    def scores(self, ctx):
        return np.ones(ctx.closes.shape[1])


def run(strategy):
    return CrossSectionalBacktester(
        panel, strategy, rebalance_every=7, neutralize=False,
        cost_bps=10, interval="1d",
    ).run()


def curve_stats(ec, lo=None, hi=None):
    df = ec
    if lo is not None:
        df = df.filter(pl.col("timestamp") >= lo)
    if hi is not None:
        df = df.filter(pl.col("timestamp") < hi)
    eq = df["equity"].to_numpy()
    if eq.size < 3:
        return None
    r = eq[1:] / eq[:-1] - 1.0
    mdd, _, _ = max_drawdown(eq)
    return dict(sharpe=sharpe_ratio(r, PPY), maxdd=mdd, ret=eq[-1] / eq[0] - 1.0, r=r)


bench = run(EqualWeightLong())
bench_stats = curve_stats(bench.equity_curve)

print("FULL CYCLE 2021-2024 (daily, directional, 10bp)\n")
print(f"{'strategy':>12} | {'Sharpe':>6} {'MaxDD':>7} {'TotRet':>9} {'beta':>5} {'alpha/yr':>8} {'a_t':>5}")
print("-" * 62)
print(f"{'buy&hold':>12} | {bench_stats['sharpe']:>6.2f} {bench_stats['maxdd']*100:>6.0f}% "
      f"{bench_stats['ret']*100:>8.0f}%   1.00      -    -")

trend30_ec = None
for lb in (20, 30, 50):
    res = run(TimeSeriesTrend(lookback=lb))
    st = curve_stats(res.equity_curve)
    ab = alpha_beta(st["r"], bench_stats["r"], PPY)
    print(f"{'trend-' + str(lb):>12} | {st['sharpe']:>6.2f} {st['maxdd']*100:>6.0f}% "
          f"{st['ret']*100:>8.0f}% {ab['beta']:>5.2f} {ab['alpha_ann']*100:>7.0f}% {ab['alpha_t']:>5.1f}")
    if lb == 30:
        trend30_ec = res.equity_curve

# Our own edge: the multi-horizon ensemble (no single lookback to overfit).
ens_res = run(TimeSeriesTrendEnsemble(lookbacks=(20, 30, 50, 100)))
ens_stats = curve_stats(ens_res.equity_curve)
ens_ab = alpha_beta(ens_stats["r"], bench_stats["r"], PPY)
print(f"{'trend-ENS':>12} | {ens_stats['sharpe']:>6.2f} {ens_stats['maxdd']*100:>6.0f}% "
      f"{ens_stats['ret']*100:>8.0f}% {ens_ab['beta']:>5.2f} {ens_ab['alpha_ann']*100:>7.0f}% "
      f"{ens_ab['alpha_t']:>5.1f}")

# Regime breakdown: does trend protect in the 2022 bear? -----------------------
def d(y, m):
    return datetime(y, m, 1, tzinfo=timezone.utc)

regimes = [
    ("2021 bull", d(2021, 1), d(2022, 1)),
    ("2022 BEAR", d(2022, 1), d(2023, 1)),
    ("2023-24 up", d(2023, 1), d(2025, 1)),
]
print("\nREGIME BREAKDOWN - total return by regime\n")
print(f"{'regime':>12} | {'buy&hold':>10} {'trend-30':>10} {'trend-ENS':>10}")
print("-" * 50)
for name, lo, hi in regimes:
    bh = curve_stats(bench.equity_curve, lo, hi)
    tr = curve_stats(trend30_ec, lo, hi)
    en = curve_stats(ens_res.equity_curve, lo, hi)
    print(f"{name:>12} | {bh['ret']*100:>9.0f}% {tr['ret']*100:>9.0f}% {en['ret']*100:>9.0f}%")

print("\nRead: if trend's alpha t-stat is > ~2 AND it loses far less than buy&hold\n"
      "in 2022, it's real (beta-adjusted) edge - crisis alpha. If alpha_t is small\n"
      "and it just tracks buy&hold scaled down, it's beta, not skill.")
