"""The full composed book: three different *types* of edge in one portfolio.

Standalone measurement — no changes to the live signal or paper tracker.

  A) ALPHA  — small-cap 4-sleeve (trend + size + TVL-value + fees-value),
              near market-neutral cross-sectional alpha.
  B) CARRY  — timed majors funding-basis carry, delta-neutral yield.
  C) MACRO  — stablecoin-timed majors: a risk-on/off *directional* overlay.

A and B are (near) market-neutral; C deliberately adds cushioned beta. We show
the engine correlations, then a few blends, in-sample and out-of-sample.

Prereqs: uni_broad, delisted, tvl_prices, tvl, fees, perp_major, funding_major.
    python examples/full_book.py
"""

import numpy as np
import polars as pl

from vfund.analytics.performance import max_drawdown, sharpe_ratio
from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.onchain import fetch_stablecoin_supply, load_tvl
from vfund.data.panel import load_panel
from vfund.data.universe import clean_universe
from vfund.strategy import CrossSectionalSize, TimeSeriesTrendEnsemble, TVLDivergence

PPY = 365
MAJORS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
          "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
SHORT = dict(short_cost_bps_annual=1000, min_short_dollar_volume=5_000_000)
TREND = dict(neutralize=False, vol_target=0.30, vol_lookback=30, max_leverage=3.0)


def ser(name, panel, strat, **kw):
    ec = CrossSectionalBacktester(panel, strat, rebalance_every=7, interval="1d",
                                  cost_bps=10, **kw).run().equity_curve
    eq = ec["equity"].to_numpy()
    return pl.DataFrame({"timestamp": ec["timestamp"][1:], name: eq[1:] / eq[:-1] - 1.0})


# ---- A) 4-sleeve alpha --------------------------------------------------
broad_dead = pl.concat([clean_universe(load_panel("data/uni_broad.parquet")),
                        load_panel("data/delisted.parquet")])
defi, tvl, fees = load_panel("data/tvl_prices.parquet"), load_tvl("data/tvl.parquet"), load_tvl("data/fees.parquet")
sl = [ser("trend", broad_dead, TimeSeriesTrendEnsemble(), **TREND, **SHORT),
      ser("size", broad_dead, CrossSectionalSize(20), top_k=5, **SHORT),
      ser("tvl", defi, TVLDivergence(60), top_k=5, tvl=tvl),
      ser("fees", defi, TVLDivergence(30), top_k=5, tvl=fees)]
al = sl[0]
for d in sl[1:]:
    al = al.join(d, on="timestamp", how="inner")
Ra = np.column_stack([al[c].to_numpy() for c in al.columns if c != "timestamp"])
wa = 1 / Ra.std(axis=0); wa /= wa.sum()
alpha = pl.DataFrame({"timestamp": al["timestamp"], "alpha": Ra @ wa})

# ---- B) majors funding carry (timed, net) -------------------------------
def wide(path, v):
    return pl.read_parquet(path).pivot(values=v, index="timestamp", on="symbol").sort("timestamp")

spot, perp = wide("data/uni_broad.parquet", "close"), wide("data/perp_major.parquet", "close")
fund = (pl.read_parquet("data/funding_major.parquet").with_columns(pl.col("timestamp").dt.truncate("1d"))
        .group_by(["timestamp", "symbol"]).agg(pl.col("funding_rate").sum())
        .pivot(values="funding_rate", index="timestamp", on="symbol").sort("timestamp"))
cs = sorted(set(MAJORS) & set(spot.columns) & set(perp.columns) & set(fund.columns))
jm = (spot.select(["timestamp", *cs]).rename({s: f"s_{s}" for s in cs})
      .join(perp.select(["timestamp", *cs]).rename({s: f"p_{s}" for s in cs}), on="timestamp")
      .join(fund.select(["timestamp", *cs]).rename({s: f"f_{s}" for s in cs}), on="timestamp").drop_nulls())
Sm, Pm, Fm = (jm.select([f"{p}_{s}" for s in cs]).to_numpy() for p in "spf")
bas = (Pm - Sm) / Sm; dbas = np.zeros_like(bas); dbas[1:] = bas[1:] - bas[:-1]
cr = Fm - dbas; tc = np.zeros(cr.shape[0])
for t in range(30, cr.shape[0]):
    m = Fm[t - 30:t].mean(axis=0) > 0
    if m.any():
        tc[t] = cr[t, m].mean()
carry = pl.DataFrame({"timestamp": jm["timestamp"], "carry": tc - 1e-4})

# ---- C) stablecoin-timed majors (macro beta) ----------------------------
px = spot.select(["timestamp", *cs]).drop_nulls()
mkt = px.drop("timestamp").to_numpy(); mret = np.zeros(mkt.shape[0]); mret[1:] = (mkt[1:] / mkt[:-1] - 1).mean(axis=1)
sup = px.select("timestamp").join(fetch_stablecoin_supply(start="2020-06-01", end="2026-07-01"),
                                  on="timestamp", how="left").sort("timestamp")["supply"].fill_null(strategy="forward").to_numpy()
g = np.zeros_like(sup); g[30:] = sup[30:] / sup[:-30] - 1
pos = (g > 0).astype(float); mtimed = np.zeros_like(mret); mtimed[1:] = pos[:-1] * mret[1:]
macro = pl.DataFrame({"timestamp": px["timestamp"], "macro": mtimed})

# ---- combine ------------------------------------------------------------
df = alpha.join(carry, on="timestamp").join(macro, on="timestamp").sort("timestamp")
A, C, M = df["alpha"].to_numpy(), df["carry"].to_numpy(), df["macro"].to_numpy()
yr = np.array([int(str(t)[:4]) for t in df["timestamp"].to_numpy()])

print("ENGINE CORRELATIONS\n        alpha  carry  macro")
for nm, r in [("alpha", A), ("carry", C), ("macro", M)]:
    print(f"{nm:>7} " + "".join(f"{np.corrcoef(r, x)[0,1]:>7.2f}" for x in (A, C, M)))


def stat(label, r):
    eq = np.cumprod(1 + r); yrs = r.size / PPY
    mdd, _, _ = max_drawdown(np.concatenate([[1.0], eq]))
    ins, oos = yr < 2025, yr >= 2025
    print(f"{label:>26} | {sharpe_ratio(r,PPY):>6.2f} {(eq[-1]**(1/yrs)-1)*100:>6.0f}% "
          f"{mdd*100:>6.0f}% | {sharpe_ratio(r[ins],PPY):>6.2f} {sharpe_ratio(r[oos],PPY):>7.2f}")


print("\n" + "=" * 66)
print(f"{'book':>26} | {'Sharpe':>6} {'CAGR':>6} {'MaxDD':>6} | {'IS':>6} {'OOS':>7}")
print("-" * 66)
stat("alpha only", A)
stat("alpha + carry (50/50)", 0.5 * A + 0.5 * C)
stat("alpha+carry+macro (1/3)", (A + C + M) / 3)
print("=" * 66)
print("\nA & B are ~market-neutral; adding C trades some neutrality for a")
print("cushioned-beta return stream. Blends smooth the ride; judge on OOS.")
