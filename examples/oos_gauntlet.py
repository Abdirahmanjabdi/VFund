"""Honest out-of-sample hunt.

For each hypothesis: pick the lookback that did best IN-SAMPLE (2021-2024), then
report how that exact config did OUT-OF-SAMPLE (2025-2026) - data never used for
selection. The OOS column is the only one that counts. ALL hypotheses are shown
(no cherry-picking), and even so, repeatedly consulting the OOS set slowly burns
its credibility - so treat a lone OOS winner as a lead to re-test, not a verdict.

Requires data/uni_2026.parquet.  python examples/oos_gauntlet.py
"""

import numpy as np
import polars as pl

from vfund.analytics.performance import sharpe_ratio
from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.panel import load_panel
from vfund.strategy import (
    CrossSectionalIlliquidity,
    CrossSectionalLowVol,
    CrossSectionalMaxReturn,
    CrossSectionalResidualMomentum,
    CrossSectionalSize,
    CrossSectionalValue,
    TimeSeriesTrendEnsemble,
)

PPY = 365
panel = load_panel("data/uni_2026.parquet")
gmax = panel["timestamp"].max()
keep = (panel.group_by("symbol").agg(pl.col("timestamp").max().alias("m"))
        .filter(pl.col("m") == gmax)["symbol"].to_list())
panel = panel.filter(pl.col("symbol").is_in(keep))


def run(strat, **kw):
    res = CrossSectionalBacktester(panel, strat, interval="1d", cost_bps=10, **kw).run()
    eq = res.equity_curve["equity"].to_numpy()
    ts = res.equity_curve["timestamp"].to_numpy()[1:]
    return eq[1:] / eq[:-1] - 1.0, np.array([int(str(t)[:4]) for t in ts])


# name -> (factory, param grid, engine kwargs)
NEUTRAL = dict(rebalance_every=7, top_k=5, neutralize=True)
HYPS = {
    "max/lottery":   (CrossSectionalMaxReturn, [14, 30, 60], NEUTRAL),
    "resid-mom":     (CrossSectionalResidualMomentum, [30, 60, 90, 180], NEUTRAL),
    "illiquidity":   (CrossSectionalIlliquidity, [14, 30, 60], NEUTRAL),
    "size":          (CrossSectionalSize, [20, 30, 50], NEUTRAL),
    "value":         (CrossSectionalValue, [30, 60, 90], NEUTRAL),
    "lowvol":        (CrossSectionalLowVol, [30, 60, 90], NEUTRAL),
    "trend* (dir)":  (lambda p: TimeSeriesTrendEnsemble(), [0],
                      dict(rebalance_every=7, neutralize=False, vol_target=0.30,
                           vol_lookback=30, max_leverage=3.0)),
}

print("Select config on 2021-2024, judge on 2025-2026 (all shown, no cherry-pick)\n")
print(f"{'hypothesis':>14} | {'sel':>5} | {'IS Shrp':>7} | {'OOS Shrp':>8} {'OOS ret':>8}  verdict")
print("-" * 66)
for name, (make, grid, kw) in HYPS.items():
    best = None
    for p in grid:
        strat = make(p) if not isinstance(make, type) else make(p)
        r, yr = run(strat, **kw)
        is_m, oos_m = yr < 2025, yr >= 2025
        if is_m.sum() < 30 or oos_m.sum() < 30:
            continue
        is_sh = sharpe_ratio(r[is_m], PPY)
        row = dict(p=p, is_sh=is_sh, oos_sh=sharpe_ratio(r[oos_m], PPY),
                   oos_ret=float(np.prod(1 + r[oos_m]) - 1))
        if best is None or is_sh > best["is_sh"]:
            best = row
    is_sh, oos_sh = best["is_sh"], best["oos_sh"]
    # A credible edge must work in BOTH periods. Good-IS/bad-OOS = overfit;
    # bad-IS/good-OOS = you'd never have traded it, its OOS win is likely luck.
    if is_sh > 0.5 and oos_sh > 0.5:
        v = "CANDIDATE"
    elif is_sh > 0.5 and oos_sh <= 0:
        v = "overfit (died OOS)"
    elif is_sh > 0.5:
        v = "faded"
    elif oos_sh > 0.5:
        v = "noise? (bad IS)"
    else:
        v = "dead"
    print(f"{name:>14} | {best['p']:>5} | {is_sh:>7.2f} | "
          f"{oos_sh:>8.2f} {best['oos_ret']*100:>7.0f}%  {v}")

print("\nA real edge is positive in BOTH periods. Good-IS/bad-OOS = overfit;")
print("bad-IS/good-OOS = regime luck you couldn't have selected on. Only")
print("'CANDIDATE' (consistent both) is worth a further look.")
