"""Does an intraday squeeze liquidate the funding carry's short-perp leg?

The daily-close basis model said the delta-neutral hedge contains drawdowns. But
margin is checked *intraday*: a short-perp leg can be liquidated when the perp
spikes up, before the daily close shows the hedge worked. The daily perp HIGH is
the intraday max, so we can measure the worst adverse move against a short and
ask whether it would have blown through the margin at a given leverage.

Two regimes:
  * SILOED margin — the perp leg has its own margin; a big enough spike liquidates
    it (you lose that margin, must re-enter). Risky at high leverage.
  * CROSS margin — spot collateralises the perp; the delta-neutral book barely
    moves, so liquidation is rare. (How real basis desks run it.)

Requires data/perp_major.parquet, data/uni_broad.parquet, data/funding_major.parquet.
    python examples/carry_liquidation.py
"""

import numpy as np
import polars as pl

from vfund.analytics.performance import max_drawdown, sharpe_ratio

PPY = 365
MAJORS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
          "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "TRXUSDT", "BCHUSDT"]


def wide(path, value):
    return pl.read_parquet(path).pivot(values=value, index="timestamp", on="symbol").sort("timestamp")


spot = wide("data/uni_broad.parquet", "close")
perp_c = wide("data/perp_major.parquet", "close")
perp_h = wide("data/perp_major.parquet", "high")
fund = (pl.read_parquet("data/funding_major.parquet")
        .with_columns(pl.col("timestamp").dt.truncate("1d"))
        .group_by(["timestamp", "symbol"]).agg(pl.col("funding_rate").sum())
        .pivot(values="funding_rate", index="timestamp", on="symbol").sort("timestamp"))

syms = sorted(set(MAJORS) & set(spot.columns) & set(perp_c.columns) & set(fund.columns))


def mat(w, pfx):
    return w.select(["timestamp", *syms]).rename({s: f"{pfx}{s}" for s in syms})


j = (mat(spot, "s_").join(mat(perp_c, "c_"), on="timestamp")
     .join(mat(perp_h, "h_"), on="timestamp").join(mat(fund, "f_"), on="timestamp")
     .sort("timestamp").drop_nulls())
S = j.select([f"s_{s}" for s in syms]).to_numpy()
C = j.select([f"c_{s}" for s in syms]).to_numpy()
H = j.select([f"h_{s}" for s in syms]).to_numpy()
F = j.select([f"f_{s}" for s in syms]).to_numpy()
days = j["timestamp"].to_numpy()
yr = np.array([int(str(t)[:4]) for t in days])

basis = (C - S) / S
dbasis = np.zeros_like(basis); dbasis[1:] = basis[1:] - basis[:-1]
coin_ret = F - dbasis

# Worst intraday adverse move against a short = (high - prior close) / prior close.
adverse = np.zeros_like(H)
adverse[1:] = (H[1:] - C[:-1]) / C[:-1]

print(f"Intraday squeeze risk for the short-perp leg, {len(syms)} majors\n")
print(f"{'leverage':>10} | {'liq threshold':>13} | {'liq-days (siloed)':>17}")
print("-" * 48)
for L in [3, 5, 10, 20]:
    liq = adverse > (1.0 / L)
    print(f"{L:>9}x | {'>'+format(100/L,'.0f')+'% intraday':>13} | {int(liq.sum()):>17}")

worst = np.argsort(adverse.ravel())[-6:][::-1]
print("\nWorst intraday spikes against a short (coin, date, move):")
for idx in worst:
    t, i = divmod(idx, len(syms))
    print(f"  {syms[i]:>9}  {str(days[t])[:10]}  +{adverse[t, i]*100:.0f}%")


def stats(label, r):
    eq = np.cumprod(1 + r); years = r.size / PPY
    mdd, _, _ = max_drawdown(np.concatenate([[1.0], eq]))
    ins, oos = yr < 2025, yr >= 2025
    print(f"{label:>26} | {sharpe_ratio(r,PPY):>6.2f} {(eq[-1]**(1/years)-1)*100:>6.1f}% "
          f"{mdd*100:>6.0f}% | {sharpe_ratio(r[ins],PPY):>6.2f} {sharpe_ratio(r[oos],PPY):>7.2f}")


# Timed carry (positive-funding coins), cross vs siloed margin at L=5.
def timed_book(liq_L=None):
    out = np.zeros(coin_ret.shape[0])
    for t in range(30, coin_ret.shape[0]):
        m = F[t - 30:t].mean(axis=0) > 0
        if not m.any():
            continue
        r = coin_ret[t, m].copy()
        if liq_L is not None:  # siloed: liquidated legs lose their margin (1/L)
            hit = adverse[t, m] > (1.0 / liq_L)
            r[hit] = -1.0 / liq_L
        out[t] = r.mean()
    return out


print("\nTIMED carry P&L under each margin regime:\n")
print(f"{'regime':>26} | {'Sharpe':>6} {'CAGR':>6} {'MaxDD':>6} | {'IS':>6} {'OOS':>7}")
print("-" * 70)
stats("cross-margin (no liq)", timed_book(None))
stats("siloed 10x (liq)", timed_book(10))
stats("siloed 5x (liq)", timed_book(5))
print("-" * 70)
print("Read: cross-margin (spot collateralises perp) barely liquidates - that's")
print("how it's run in practice. Siloed high leverage takes rare but brutal hits")
print("in squeezes. The carry is safe run cross-margined at modest leverage.")
