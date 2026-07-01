"""What would the combined book do to a real $100 account?

Two parts:
  1. The return *profile* scaled to $100 (best/worst day, drawdown in dollars).
  2. The brutal reality check: whether $100 can even run this strategy.
"""

import numpy as np

from vfund.analytics.performance import max_drawdown, sharpe_ratio
from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.panel import load_panel
from vfund.strategy import CrossSectionalSize, TimeSeriesTrendEnsemble

PPY = 365
START = 100.0
panel = load_panel("data/uni_daily.parquet")


def _rets(res):
    eq = res.equity_curve["equity"].to_numpy()
    return eq[1:] / eq[:-1] - 1.0


def combo_returns(size_lb=20, vt=0.30, rebalance=7):
    t = CrossSectionalBacktester(
        panel, TimeSeriesTrendEnsemble(), rebalance_every=rebalance, neutralize=False,
        cost_bps=10, interval="1d", vol_target=vt, vol_lookback=30, max_leverage=3.0).run()
    s = CrossSectionalBacktester(
        panel, CrossSectionalSize(size_lb), rebalance_every=rebalance, top_k=5,
        cost_bps=10, interval="1d").run()
    tr, sr = _rets(t), _rets(s)
    n = min(tr.size, sr.size)
    tr, sr = tr[-n:], sr[-n:]
    wa, wb = 1 / tr.std(), 1 / sr.std()
    wa, wb = wa / (wa + wb), wb / (wa + wb)
    return wa * tr + wb * sr


r = combo_returns()
eq = START * np.cumprod(1 + r)
days = r.size
years = days / PPY
mdd, _, trough = max_drawdown(np.concatenate([[START], eq]))
peak_before_trough = np.maximum.accumulate(np.concatenate([[START], eq]))[trough]

print("=" * 60)
print(f"  $100 account, combined trend+size, {years:.1f} years (2021-2024)")
print("=" * 60)
print(f"  Final value            ${eq[-1]:>10,.2f}")
print(f"  Total return           {(eq[-1]/START-1)*100:>10,.0f}%")
print(f"  CAGR                   {((eq[-1]/START)**(1/years)-1)*100:>10,.1f}%")
print(f"  Annualised volatility  {r.std()*np.sqrt(PPY)*100:>10,.0f}%")
print(f"  Sharpe                 {sharpe_ratio(r, PPY):>10.2f}")
print("-" * 60)
print(f"  Best day               ${eq.max()*0+START*r.max():>+10,.2f}  ({r.max()*100:+.1f}%)")
print(f"  Worst day              ${START*r.min():>+10,.2f}  ({r.min()*100:+.1f}%)")
print(f"  Worst drawdown         {mdd*100:>10,.0f}%  (peak ${peak_before_trough:,.0f} -> ${peak_before_trough*(1+mdd):,.0f})")
print(f"  % of days positive     {np.mean(r>0)*100:>10,.0f}%")

# Longest losing streak (consecutive down days).
streak = mx = 0
for x in r:
    streak = streak + 1 if x < 0 else 0
    mx = max(mx, streak)
print(f"  Longest losing streak  {mx:>10} days")

# Rough monthly picture (21 trading days).
monthly = [np.prod(1 + r[i:i+21]) - 1 for i in range(0, days - 21, 21)]
monthly = np.array(monthly)
print(f"  Typical month          {np.median(monthly)*100:>+10,.1f}%  "
      f"(best {monthly.max()*100:+.0f}%, worst {monthly.min()*100:+.0f}%)")

print("\n" + "=" * 60)
print("  REALITY CHECK: can $100 actually run this?")
print("=" * 60)
print("""  Short answer: no - and this is the important lesson.

  The strategy holds ~10-20 coins LONG and SHORT at once, and
  rebalances weekly. On a $100 account that means:

  * Position sizes of ~$3-8 each. Binance's minimum order is
    ~$5-10, so half your trades are impossible or all-or-nothing.
  * SHORTING requires perp futures or margin - not a $100 spot
    account. Shorts also pay borrow/funding the backtest ignores.
  * Fees: real all-in cost on tiny orders is far above the 10bp
    modelled here. Weekly rebalancing of 20 dust positions can
    lose several % a month to fees + spread alone.

  Realistic minimum to run a 20-name long/short book with meaning-
  ful sizing is ~$10k-25k (and a futures account for the shorts).

  So treat the numbers above as the STRATEGY'S profile, not what
  $100 would make. At $100 the right move is to PAPER-trade it
  (zero fees, learn the mechanics) - which is what vfund paper does.""")
