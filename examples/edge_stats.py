"""Everything about the edge, in numbers: performance, risk, trading behaviour.

Runs the validated combined trend+size book on the survivorship-corrected
universe (broad current coins + delisted coins) with realistic costs and the
hard-to-short gate, then prints a full statistical profile.

    python examples/edge_stats.py
"""

import numpy as np
import polars as pl

from vfund.analytics.performance import max_drawdown, sharpe_ratio, sortino_ratio
from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.panel import load_panel
from vfund.data.universe import clean_universe
from vfund.strategy import CrossSectionalSize, TimeSeriesTrendEnsemble

PPY = 365
CFG = dict(interval="1d", cost_bps=10, short_cost_bps_annual=1000,
           min_short_dollar_volume=5_000_000)

broad = clean_universe(load_panel("data/uni_broad.parquet"))
dead = load_panel("data/delisted.parquet")
panel = pl.concat([broad, dead])
n_universe = panel["symbol"].n_unique()


def run(strat, **kw):
    return CrossSectionalBacktester(panel, strat, **CFG, **kw).run()


trend = run(TimeSeriesTrendEnsemble(), rebalance_every=7, neutralize=False,
            vol_target=0.30, vol_lookback=30, max_leverage=3.0)
size = run(CrossSectionalSize(20), rebalance_every=7, top_k=5)


def rets(res):
    eq = res.equity_curve["equity"].to_numpy()
    return eq[1:] / eq[:-1] - 1.0


tr, sr = rets(trend), rets(size)
n = min(tr.size, sr.size)
tr, sr = tr[-n:], sr[-n:]
wa, wb = 1 / tr.std(), 1 / sr.std()
wa, wb = wa / (wa + wb), wb / (wa + wb)
r = wa * tr + wb * sr
ts = trend.equity_curve["timestamp"].to_numpy()[1:][-n:]
yr = np.array([int(str(t)[:4]) for t in ts])
eq = 100_000 * np.cumprod(1 + r)
years = n / PPY

pos, neg = r[r > 0], r[r < 0]
mdd, _, _ = max_drawdown(np.concatenate([[100_000], eq]))

print("=" * 60)
print("  VFUND EDGE - full profile (2021 -> 2026, $100k, broad+dead)")
print("=" * 60)
print(f"  Universe                 {n_universe} coins (USDT spot, incl. delisted)")
print(f"  Period                   {str(ts[0])[:10]} -> {str(ts[-1])[:10]} ({years:.1f}y)")
print("  --- RETURN ---")
print(f"  Final value              ${eq[-1]:,.0f}  (from $100,000)")
print(f"  Total return             {(eq[-1]/1e5-1)*100:,.0f}%")
print(f"  CAGR                     {((eq[-1]/1e5)**(1/years)-1)*100:.1f}%")
print("  --- RISK ---")
print(f"  Annualised volatility    {r.std()*np.sqrt(PPY)*100:.0f}%")
print(f"  Max drawdown             {mdd*100:.0f}%")
print(f"  Sharpe ratio             {sharpe_ratio(r, PPY):.2f}")
print(f"  Sortino ratio            {sortino_ratio(r, PPY):.2f}")
print("  --- TRADE QUALITY (daily) ---")
print(f"  Win rate (days up)       {len(pos)/n*100:.0f}%")
print(f"  Profit factor            {pos.sum()/abs(neg.sum()):.2f}")
print(f"  Avg up day / down day    +{pos.mean()*100:.2f}% / {neg.mean()*100:.2f}%")
print(f"  Payoff ratio             {pos.mean()/abs(neg.mean()):.2f}")
print(f"  Best / worst day         +{r.max()*100:.1f}% / {r.min()*100:.1f}%")

# Trading behaviour: turnover, holding period, positions.
def turnover_stats(res):
    tv = res.trades["turnover"].to_numpy()
    one_way = 0.5 * tv.mean()                      # buy+sell counted -> halve
    gross = res.equity_curve["gross_exposure"].to_numpy().mean()
    frac_replaced = one_way / gross if gross else 0
    hold_days = 7 / frac_replaced if frac_replaced else 0  # rebalanced weekly
    return one_way, gross, hold_days, len(tv)


for name, res in [("trend sleeve", trend), ("size sleeve", size)]:
    ow, gross, hold, nreb = turnover_stats(res)
    print(f"  --- {name.upper()} ---")
    print(f"  Rebalances               {nreb} (weekly) = {nreb/years:.0f}/yr, ~{nreb/years/12:.1f}/mo")
    print(f"  Avg gross exposure       {gross*100:.0f}% of equity")
    print(f"  One-way turnover/rebal   {ow*100:.0f}% of book")
    print(f"  Avg holding period       ~{hold:.0f} days")

print("  --- BY YEAR (combined) ---")
for y in range(2021, 2027):
    m = yr == y
    if m.any():
        print(f"  {y}                     {(np.prod(1+r[m])-1)*100:+6.0f}%")
print("=" * 60)
