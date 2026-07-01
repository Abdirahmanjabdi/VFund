"""A $100k account running the combined book, 2021 -> now (2026).

Crucially split into:
  * IN-SAMPLE   (2021-2024): the period the strategy config was chosen on.
  * OUT-OF-SAMPLE (2025-2026): data the config NEVER saw - the honest test.

Note: this still uses a survivor universe, so survivorship bias remains even in
the OOS slice. What the OOS slice *does* test is whether the edge holds on new
time it wasn't tuned on.
"""

import numpy as np
import polars as pl

from vfund.analytics.performance import max_drawdown, sharpe_ratio
from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.panel import load_panel
from vfund.strategy import CrossSectionalSize, TimeSeriesTrendEnsemble
from vfund.strategy.cross_sectional import CrossSectionalStrategy

PPY = 365
START = 100_000.0
panel = load_panel("data/uni_2026.parquet")

# Drop coins that don't reach the end (e.g. EOS, delisted May 2025) so the
# inner-join alignment spans the full 2021->2026 window instead of truncating.
gmax = panel["timestamp"].max()
keep = (panel.group_by("symbol").agg(pl.col("timestamp").max().alias("m"))
        .filter(pl.col("m") == gmax)["symbol"].to_list())
panel = panel.filter(pl.col("symbol").is_in(keep))


class EqualWeightLong(CrossSectionalStrategy):
    def scores(self, ctx):
        return np.ones(ctx.closes.shape[1])


def rets_ts(res):
    eq = res.equity_curve["equity"].to_numpy()
    ts = res.equity_curve["timestamp"].to_numpy()[1:]
    return eq[1:] / eq[:-1] - 1.0, ts


def combo():
    t = CrossSectionalBacktester(
        panel, TimeSeriesTrendEnsemble(), rebalance_every=7, neutralize=False,
        cost_bps=10, interval="1d", vol_target=0.30, vol_lookback=30, max_leverage=3.0).run()
    s = CrossSectionalBacktester(
        panel, CrossSectionalSize(20), rebalance_every=7, top_k=5,
        cost_bps=10, interval="1d").run()
    (tr, ts), (sr, _) = rets_ts(t), rets_ts(s)
    n = min(tr.size, sr.size)
    tr, sr, ts = tr[-n:], sr[-n:], ts[-n:]
    wa, wb = 1 / tr.std(), 1 / sr.std()
    wa, wb = wa / (wa + wb), wb / (wa + wb)
    return wa * tr + wb * sr, ts


r, ts = combo()
bench_r, _ = rets_ts(CrossSectionalBacktester(
    panel, EqualWeightLong(), rebalance_every=7, neutralize=False,
    cost_bps=10, interval="1d").run())
bench_r = bench_r[-r.size:]

eq = START * np.cumprod(1 + r)
beq = START * np.cumprod(1 + bench_r)
years = r.size / PPY
mdd, _, _ = max_drawdown(np.concatenate([[START], eq]))
bmdd, _, _ = max_drawdown(np.concatenate([[START], beq]))

print("=" * 64)
print(f"  $100,000 account, {str(ts[0])[:10]} -> {str(ts[-1])[:10]}  ({years:.1f} yrs)")
print("=" * 64)
print(f"{'':22}{'COMBINED':>14}{'buy&hold':>14}")
print(f"  Final value        {'$'+format(eq[-1],',.0f'):>14}{'$'+format(beq[-1],',.0f'):>14}")
print(f"  Total return       {(eq[-1]/START-1)*100:>13,.0f}%{(beq[-1]/START-1)*100:>13,.0f}%")
print(f"  CAGR               {((eq[-1]/START)**(1/years)-1)*100:>13,.1f}%{((beq[-1]/START)**(1/years)-1)*100:>13,.1f}%")
print(f"  Sharpe             {sharpe_ratio(r,PPY):>14.2f}{sharpe_ratio(bench_r,PPY):>14.2f}")
print(f"  Max drawdown       {mdd*100:>13,.0f}%{bmdd*100:>13,.0f}%")
print(f"  Worst day ($)      {'$'+format(START*r.min(),'+,.0f'):>14}{'$'+format(START*bench_r.min(),'+,.0f'):>14}")

# Year-end equity milestones.
yr = np.array([int(str(t)[:4]) for t in ts])
print("\n  Equity by year-end (COMBINED):")
run_eq = START
prev = None
for y in range(2021, 2027):
    mask = yr == y
    if not mask.any():
        continue
    ye = (START * np.cumprod(1 + r))[mask][-1]
    print(f"    {y}: ${ye:>12,.0f}")

# In-sample vs out-of-sample split.
oos = yr >= 2025
ins = ~oos
def seg(mask):
    rr = r[mask]
    return sharpe_ratio(rr, PPY), float(np.prod(1 + rr) - 1)
is_sh, is_ret = seg(ins)
oos_sh, oos_ret = seg(oos)
print("\n" + "-" * 64)
print(f"  IN-SAMPLE   2021-2024 : return {is_ret*100:>7,.0f}%   Sharpe {is_sh:>5.2f}")
print(f"  OUT-OF-SAMPLE 2025-26 : return {oos_ret*100:>7,.0f}%   Sharpe {oos_sh:>5.2f}   <- config never saw this")
print("-" * 64)
print("""
  The OOS row is the one that matters. Strong there = the edge held on new
  time. Weak/negative = it was partly a fit to 2021-2024. Survivorship bias
  still inflates BOTH rows; a true verdict also needs delisted coins.""")
