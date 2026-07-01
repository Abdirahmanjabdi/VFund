"""How much were survivorship bias and free-shorting inflating the edge?

Re-runs the combined trend+size book under progressively more honest conditions:
  * survivor universe (22 hand-picked coins) vs a broad ~60-coin universe
    (ragged: coins enter/exit as listed, including late-listers), and
  * free shorting vs a 10%/yr short-financing charge.

Each row splits in-sample (2021-2024) from out-of-sample (2025-2026).

Requires data/uni_2026.parquet (survivor) and data/uni_broad.parquet (broad).
    python examples/survivorship_check.py
"""

import numpy as np
import polars as pl

from vfund.analytics.performance import sharpe_ratio
from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.panel import load_panel
from vfund.strategy import CrossSectionalSize, TimeSeriesTrendEnsemble

PPY = 365

# Not tradable-alpha assets: stablecoins and metal-pegged tokens. "Top by volume"
# drags these in, so a clean universe must exclude them.
PEGGED = {"USDCUSDT", "USD1USDT", "FDUSDUSDT", "EURUSDT", "RLUSDUSDT", "TUSDUSDT",
          "DAIUSDT", "PYUSDUSDT", "USDPUSDT", "BUSDUSDT", "PAXGUSDT", "XAUTUSDT"}


def clean_universe(panel, min_bars=365):
    """Drop pegged tokens, non-standard symbols, and coins with < min_bars history."""
    counts = panel.group_by("symbol").agg(pl.len().alias("n"))
    enough = counts.filter(pl.col("n") >= min_bars)["symbol"].to_list()
    return panel.filter(
        pl.col("symbol").is_in(enough)
        & ~pl.col("symbol").is_in(list(PEGGED))
        & pl.col("symbol").str.contains(r"^[A-Z0-9]+USDT$")
    )


def _r(res):
    eq = res.equity_curve["equity"].to_numpy()
    ts = res.equity_curve["timestamp"].to_numpy()[1:]
    return eq[1:] / eq[:-1] - 1.0, np.array([int(str(t)[:4]) for t in ts])


def combo(panel, short_cost):
    common = dict(interval="1d", cost_bps=10, short_cost_bps_annual=short_cost)
    t, yr = _r(CrossSectionalBacktester(
        panel, TimeSeriesTrendEnsemble(), rebalance_every=7, neutralize=False,
        vol_target=0.30, vol_lookback=30, max_leverage=3.0, **common).run())
    s, _ = _r(CrossSectionalBacktester(
        panel, CrossSectionalSize(20), rebalance_every=7, top_k=5, **common).run())
    n = min(t.size, s.size)
    t, s, yr = t[-n:], s[-n:], yr[-n:]
    wa, wb = 1 / t.std(), 1 / s.std()
    wa, wb = wa / (wa + wb), wb / (wa + wb)
    return wa * t + wb * s, yr


def row(label, panel, short_cost):
    r, yr = combo(panel, short_cost)
    ins, oos = yr < 2025, yr >= 2025
    print(f"{label:>34} | {sharpe_ratio(r,PPY):>6.2f} {sharpe_ratio(r[ins],PPY):>7.2f} "
          f"{sharpe_ratio(r[oos],PPY):>8.2f} {(np.prod(1+r[oos])-1)*100:>7.0f}%")


print(f"{'universe / shorting':>34} | {'full':>6} {'IS':>7} {'OOS':>8} {'OOSret':>8}")
print("-" * 70)
surv = load_panel("data/uni_2026.parquet")
row("survivor(22), free shorts", surv, 0)
row("survivor(22), 10%/yr shorts", surv, 1000)
try:
    broad = clean_universe(load_panel("data/uni_broad.parquet"))
    n_syms = broad["symbol"].n_unique()
    row(f"broad({n_syms}), free shorts", broad, 0)
    row(f"broad({n_syms}), 10%/yr shorts", broad, 1000)
except FileNotFoundError:
    print("  (data/uni_broad.parquet not found - fetch it to compare)")

print("\nEach honesty adjustment (broader universe, real short costs) should pull")
print("the edge DOWN. What's left after all of them - especially OOS - is the")
print("only part worth believing.")
