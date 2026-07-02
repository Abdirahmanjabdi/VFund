"""Stablecoin supply as a macro risk-on/off timing signal.

Thesis: when aggregate stablecoin supply is growing, fresh capital is entering
crypto (risk-on) and the market tends to rise; when it contracts, capital is
leaving (risk-off). Unlike the cross-sectional sleeves, this is a *market-timing*
overlay: hold the market when supply is expanding, step aside when it isn't.

We test it against buy-and-hold on an equal-weight majors basket, selecting the
lookback in-sample (pre-2025) and judging out-of-sample (2025-26). The bar isn't
"more return" — a timing signal earns its keep by improving *risk-adjusted*
return and cutting drawdown.

Standalone — no changes to the live signal/paper. Prereqs: data/uni_broad.parquet.
    python examples/stablecoin_macro.py
"""

import numpy as np
import polars as pl

from vfund.analytics.performance import max_drawdown, sharpe_ratio
from vfund.data.onchain import fetch_stablecoin_supply
from vfund.data.panel import load_panel

PPY = 365
MAJORS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
          "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]

# Equal-weight majors market return (daily).
px = (load_panel("data/uni_broad.parquet")
      .filter(pl.col("symbol").is_in(MAJORS))
      .pivot(values="close", index="timestamp", on="symbol").sort("timestamp").drop_nulls())
mkt = px.drop("timestamp").to_numpy()
mret = np.zeros(mkt.shape[0]); mret[1:] = (mkt[1:] / mkt[:-1] - 1.0).mean(axis=1)
mdays = px["timestamp"].to_numpy()

# Stablecoin supply, aligned to the market days (forward-filled).
sup = fetch_stablecoin_supply(start="2020-06-01", end="2026-07-01")
sup_j = px.select("timestamp").join(sup, on="timestamp", how="left").sort("timestamp")
supply = sup_j["supply"].fill_null(strategy="forward").to_numpy()
yr = np.array([int(str(t)[:4]) for t in mdays])


def timed(lookback):
    growth = np.zeros_like(supply)
    growth[lookback:] = supply[lookback:] / supply[:-lookback] - 1.0
    pos = (growth > 0).astype(float)      # risk-on when supply expanding
    r = np.zeros_like(mret)
    r[1:] = pos[:-1] * mret[1:]           # yesterday's signal, today's return (no look-ahead)
    return r


def stat(label, r):
    eq = np.cumprod(1 + r); yrs = r.size / PPY
    mdd, _, _ = max_drawdown(np.concatenate([[1.0], eq]))
    ins, oos = yr < 2025, yr >= 2025
    print(f"{label:>22} | {sharpe_ratio(r,PPY):>6.2f} {(eq[-1]**(1/yrs)-1)*100:>6.0f}% "
          f"{mdd*100:>6.0f}% | {sharpe_ratio(r[ins],PPY):>6.2f} {sharpe_ratio(r[oos],PPY):>7.2f}")


# Select the lookback in-sample by Sharpe.
best_lb = max([30, 60, 90, 120], key=lambda lb: sharpe_ratio(timed(lb)[yr < 2025], PPY))
print(f"Stablecoin-supply timing on equal-weight majors "
      f"({str(mdays[0])[:10]}..{str(mdays[-1])[:10]})\n")
print(f"best lookback (in-sample): {best_lb} days\n")
print(f"{'strategy':>22} | {'Sharpe':>6} {'CAGR':>6} {'MaxDD':>6} | {'IS':>6} {'OOS':>7}")
print("-" * 66)
stat("buy & hold", mret)
stat(f"stablecoin-timed", timed(best_lb))
print("-" * 66)
print("A timing overlay wins by cutting drawdown / raising Sharpe, not by out-")
print("returning a bull market. Judge it there, and on the out-of-sample column.")
