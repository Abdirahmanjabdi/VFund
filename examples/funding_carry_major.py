"""Funding-basis carry on the majors — a structural, high-capacity edge.

Perpetual futures usually trade at a premium to spot: leveraged longs pay
"funding" to shorts. A delta-neutral book (long spot, short perp) collects that
funding with ~no price risk. It's structural (someone must pay the premium),
deep, and scales to serious size — the opposite of the small-cap edge.

Two variants, on 13 majors:
  * always-on  — always short the perp, harvest whatever funding is (incl. the
                 negative-funding pain in deleveraging crashes)
  * timed      — only harvest coins whose funding has been positive lately

Costs: a delta-neutral carry is low-turnover once on; we net out a modest daily
friction for hedge maintenance. Splits in-sample vs out-of-sample.

Requires data/funding_major.parquet.   python examples/funding_carry_major.py
"""

import numpy as np
import polars as pl

from vfund.analytics.performance import max_drawdown, sharpe_ratio

PPY = 365
f = pl.read_parquet("data/funding_major.parquet")
# Daily funding per coin (sum the three 8h events), then a wide day x coin matrix.
daily = (f.with_columns(pl.col("timestamp").dt.truncate("1d").alias("day"))
         .group_by(["day", "symbol"]).agg(pl.col("funding_rate").sum())
         .sort("day"))
wide = daily.pivot(values="funding_rate", index="day", on="symbol").sort("day").fill_null(0.0)
days = wide["day"].to_numpy()
F = wide.drop("day").to_numpy()            # (T, N) daily funding per coin
yr = np.array([int(str(d)[:4]) for d in days])

FRICTION = 0.00005  # ~0.5 bp/day hedge-maintenance friction


def report(label, r):
    r = r - FRICTION
    eq = np.cumprod(1 + r)
    years = r.size / PPY
    mdd, _, _ = max_drawdown(np.concatenate([[1.0], eq]))
    ins, oos = yr < 2025, yr >= 2025
    print(f"{label:>14} | {sharpe_ratio(r, PPY):>6.2f} {(eq[-1]**(1/years)-1)*100:>6.1f}% "
          f"{mdd*100:>6.0f}% | {sharpe_ratio(r[ins], PPY):>6.2f} {sharpe_ratio(r[oos], PPY):>7.2f}")


# always-on: short every perp -> receive funding (equal weight across coins).
always = F.mean(axis=1)

# timed: only harvest a coin if its trailing 30d mean funding is positive.
timed = np.zeros(F.shape[0])
for t in range(30, F.shape[0]):
    mask = F[t - 30:t].mean(axis=0) > 0
    if mask.any():
        timed[t] = F[t, mask].mean()

print(f"Funding-basis carry, {F.shape[1]} majors, {str(days[0])[:10]}..{str(days[-1])[:10]}\n")
print(f"{'strategy':>14} | {'Sharpe':>6} {'CAGR':>6} {'MaxDD':>6} | {'IS':>6} {'OOS':>7}")
print("-" * 58)
report("always-on", always)
report("timed (pos)", timed)
print("-" * 58)
print("High capacity (BTC/ETH perps are the deepest crypto markets). Watch the")
print("MaxDD - funding turns sharply negative in deleveraging crashes.")
