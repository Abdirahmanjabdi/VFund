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


def combo(panel, short_cost, min_short_dv=None):
    common = dict(interval="1d", cost_bps=10, short_cost_bps_annual=short_cost,
                  min_short_dollar_volume=min_short_dv)
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


def row(label, panel, short_cost, min_short_dv=None):
    r, yr = combo(panel, short_cost, min_short_dv)
    ins, oos = yr < 2025, yr >= 2025
    print(f"{label:>34} | {sharpe_ratio(r,PPY):>6.2f} {sharpe_ratio(r[ins],PPY):>7.2f} "
          f"{sharpe_ratio(r[oos],PPY):>8.2f} {(np.prod(1+r[oos])-1)*100:>7.0f}%")


print(f"{'universe / shorting':>36} | {'full':>6} {'IS':>7} {'OOS':>8} {'OOSret':>8}")
print("-" * 72)
surv = load_panel("data/uni_2026.parquet")
row("survivor(22), 10%/yr shorts", surv, 1000)

# Survivorship-corrected: add the coins that actually DIED (data ends at delist).
try:
    dead = load_panel("data/delisted.parquet")
    pit_surv = pl.concat([surv, dead])
    row(f"survivor+dead({pit_surv['symbol'].n_unique()}), 10%/yr", pit_surv, 1000)
except FileNotFoundError:
    print("  (data/delisted.parquet not found)")

try:
    broad = clean_universe(load_panel("data/uni_broad.parquet"))
    dead = load_panel("data/delisted.parquet")
    pit_broad = pl.concat([broad, dead])
    n = pit_broad["symbol"].n_unique()
    row(f"broad+dead({n}), free shorts", pit_broad, 1000)
    # Hard-to-short: a coin is only shortable with enough recent liquidity.
    row(f"broad+dead, no-short <$1M/day", pit_broad, 1000, 1_000_000)
    row(f"broad+dead, no-short <$5M/day", pit_broad, 1000, 5_000_000)
    row(f"broad+dead, no-short <$20M/day", pit_broad, 1000, 20_000_000)
except FileNotFoundError:
    print("  (broad/delisted parquet not found - fetch to compare)")

print("\nThe last rows forbid shorting illiquid names (you can't borrow them).")
print("If OOS survives even the $20M/day gate, the edge isn't just shorting dust.")
