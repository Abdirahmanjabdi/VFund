"""Search for a HIGH-CAPACITY large-cap edge (complements the small-cap book).

Large caps (BTC, ETH, top ~12) absorb serious money but are efficient, so edge
is scarce. We test the economically-motivated candidates on this universe,
selecting in-sample (2021-2024) and judging out-of-sample (2025-2026):

  * time-series trend  — directional; judged by ALPHA vs equal-weight buy&hold
  * cross-sec momentum — long strongest majors, short weakest (market-neutral)
  * cross-sec reversal — short recent winners, long losers (market-neutral)
  * cross-sec low-vol  — long the calm majors, short the wild

Requires data/uni_broad.parquet.   python examples/largecap_search.py
"""

import numpy as np

from vfund.analytics.performance import alpha_beta, sharpe_ratio
from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.panel import load_panel
from vfund.strategy import (
    CrossSectionalLowVol, CrossSectionalMomentum, CrossSectionalReversal,
    TimeSeriesTrendEnsemble,
)
from vfund.strategy.cross_sectional import CrossSectionalStrategy

PPY = 365
LARGE_CAP = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
             "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "TRXUSDT",
             "BCHUSDT", "XLMUSDT"]
panel = load_panel("data/uni_broad.parquet").filter(
    __import__("polars").col("symbol").is_in(LARGE_CAP))
SHORT = dict(cost_bps=10, short_cost_bps_annual=1000, interval="1d")


class EqualWeightLong(CrossSectionalStrategy):
    def scores(self, ctx):
        return np.ones(ctx.closes.shape[1])


def rets(strat, **kw):
    res = CrossSectionalBacktester(panel, strat, rebalance_every=7, **SHORT, **kw).run()
    eq = res.equity_curve["equity"].to_numpy()
    ts = res.equity_curve["timestamp"].to_numpy()[1:]
    return eq[1:] / eq[:-1] - 1.0, np.array([int(str(t)[:4]) for t in ts])


bench_r, bench_yr = rets(EqualWeightLong(), neutralize=False)


def evaluate(name, make, grid, **kw):
    best = None
    for p in grid:
        r, yr = rets(make(p), **kw)
        ins, oos = yr < 2025, yr >= 2025
        if ins.sum() < 30 or oos.sum() < 30:
            continue
        row = dict(p=p, is_sh=sharpe_ratio(r[ins], PPY),
                   oos_sh=sharpe_ratio(r[oos], PPY), r=r, oos=oos)
        if best is None or row["is_sh"] > best["is_sh"]:
            best = row
    extra = ""
    if kw.get("neutralize") is False:  # directional -> show alpha vs buy&hold
        ab = alpha_beta(best["r"][best["oos"]], bench_r[-best["r"].size:][best["oos"]], PPY)
        extra = f"  (OOS beta {ab['beta']:.2f}, alpha_t {ab['alpha_t']:.1f})"
    v = ("CANDIDATE" if best["is_sh"] > 0.5 and best["oos_sh"] > 0.5 else
         "overfit" if best["is_sh"] > 0.5 and best["oos_sh"] <= 0 else
         "faded" if best["is_sh"] > 0.5 else
         "noise?" if best["oos_sh"] > 0.5 else "dead")
    print(f"{name:>16} | {best['p']:>4} | {best['is_sh']:>7.2f} | {best['oos_sh']:>8.2f}  {v}{extra}")


print(f"Large-cap edge search ({panel['symbol'].n_unique()} coins), select IS / judge OOS\n")
print(f"{'strategy':>16} | {'sel':>4} | {'IS Shrp':>7} | {'OOS Shrp':>8}")
print("-" * 70)
evaluate("trend* (dir)", lambda p: TimeSeriesTrendEnsemble(), [0],
         neutralize=False, vol_target=0.30, vol_lookback=30, max_leverage=3.0)
evaluate("momentum", lambda p: CrossSectionalMomentum(p), [30, 60, 90, 180], top_k=3)
evaluate("reversal", lambda p: CrossSectionalReversal(p), [1, 2, 3, 7], top_k=3)
evaluate("lowvol", lambda p: CrossSectionalLowVol(p), [30, 60, 90], top_k=3)
print("-" * 70)
print(f"buy&hold benchmark: OOS Sharpe {sharpe_ratio(bench_r[bench_yr >= 2025], PPY):.2f}")
print("* trend is directional: only real if alpha_t > ~2 (beats buy&hold, not just beta).")
