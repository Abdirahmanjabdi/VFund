"""The two-engine book: small-cap alpha + funding-carry yield, combined.

Standalone analysis — it reuses the library but adds no coupling to the live
signal or paper tracker. Two weakly-related return streams:

  * ALPHA  — the small-cap 3-sleeve (trend + size + on-chain), honest costs.
  * CARRY  — timed funding-basis carry on majors, cross-margined (the safe
             regime from carry_liquidation.py), net of a modest cost.

We report their correlation and the blended profile under a few allocations.

Honest caveat: risk-parity over-weights CARRY because its high Sharpe comes from
low volatility that *hides* thin-margin and squeeze tail-risk — so treat the
risk-parity row as an upper bound, not a recommendation.

Prereqs: uni_broad, delisted, tvl_prices, tvl, uni_broad (spot majors),
perp_major, funding_major parquet files.   python examples/two_engine.py
"""

import numpy as np
import polars as pl

from vfund.analytics.performance import max_drawdown, sharpe_ratio
from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.onchain import load_tvl
from vfund.data.panel import load_panel
from vfund.data.universe import clean_universe
from vfund.strategy import CrossSectionalSize, TimeSeriesTrendEnsemble, TVLDivergence

PPY = 365
MAJORS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
          "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "TRXUSDT", "BCHUSDT"]
SHORT = dict(short_cost_bps_annual=1000, min_short_dollar_volume=5_000_000)
TREND = dict(neutralize=False, vol_target=0.30, vol_lookback=30, max_leverage=3.0)


# ---- ENGINE 1: small-cap 3-sleeve alpha ---------------------------------
def sleeve(panel, strat, **kw):
    res = CrossSectionalBacktester(panel, strat, rebalance_every=7, interval="1d",
                                   cost_bps=10, **kw).run()
    ec = res.equity_curve
    eq = ec["equity"].to_numpy()
    return pl.DataFrame({"timestamp": ec["timestamp"][1:], "r": eq[1:] / eq[:-1] - 1.0})


broad_dead = pl.concat([clean_universe(load_panel("data/uni_broad.parquet")),
                        load_panel("data/delisted.parquet")])
defi = load_panel("data/tvl_prices.parquet")
tvl = load_tvl("data/tvl.parquet")
sleeves = [sleeve(broad_dead, TimeSeriesTrendEnsemble(), **TREND, **SHORT),
           sleeve(broad_dead, CrossSectionalSize(20), top_k=5, **SHORT),
           sleeve(defi, TVLDivergence(60), top_k=5, tvl=tvl, **SHORT)]
a = sleeves[0].rename({"r": "r0"})
for i, d in enumerate(sleeves[1:], 1):
    a = a.join(d.rename({"r": f"r{i}"}), on="timestamp", how="inner")
Ra = np.column_stack([a[c].to_numpy() for c in a.columns if c.startswith("r")])
wa = 1 / Ra.std(axis=0); wa /= wa.sum()
alpha = pl.DataFrame({"timestamp": a["timestamp"], "alpha": Ra @ wa})


# ---- ENGINE 2: funding carry (cross-margin, net) ------------------------
def wide(path, value):
    return pl.read_parquet(path).pivot(values=value, index="timestamp", on="symbol").sort("timestamp")


spot = wide("data/uni_broad.parquet", "close")
perp = wide("data/perp_major.parquet", "close")
fund = (pl.read_parquet("data/funding_major.parquet")
        .with_columns(pl.col("timestamp").dt.truncate("1d"))
        .group_by(["timestamp", "symbol"]).agg(pl.col("funding_rate").sum())
        .pivot(values="funding_rate", index="timestamp", on="symbol").sort("timestamp"))
syms = sorted(set(MAJORS) & set(spot.columns) & set(perp.columns) & set(fund.columns))
jm = (spot.select(["timestamp", *syms]).rename({s: f"s_{s}" for s in syms})
      .join(perp.select(["timestamp", *syms]).rename({s: f"p_{s}" for s in syms}), on="timestamp")
      .join(fund.select(["timestamp", *syms]).rename({s: f"f_{s}" for s in syms}), on="timestamp")
      .sort("timestamp").drop_nulls())
S = jm.select([f"s_{s}" for s in syms]).to_numpy()
P = jm.select([f"p_{s}" for s in syms]).to_numpy()
Fm = jm.select([f"f_{s}" for s in syms]).to_numpy()
basis = (P - S) / S
db = np.zeros_like(basis); db[1:] = basis[1:] - basis[:-1]
cret = Fm - db
timed = np.zeros(cret.shape[0])
for t in range(30, cret.shape[0]):
    m = Fm[t - 30:t].mean(axis=0) > 0
    if m.any():
        timed[t] = cret[t, m].mean()
timed -= 1.0 / 10_000.0  # ~1bp/day realistic cost (low-turnover majors carry)
carry = pl.DataFrame({"timestamp": jm["timestamp"], "carry": timed})


# ---- COMBINE ------------------------------------------------------------
df = alpha.join(carry, on="timestamp", how="inner").sort("timestamp")
A = df["alpha"].to_numpy(); Cc = df["carry"].to_numpy()
yr = np.array([int(str(t)[:4]) for t in df["timestamp"].to_numpy()])
print(f"Two-engine book, {str(df['timestamp'][0])[:10]}..{str(df['timestamp'][-1])[:10]}\n")
print(f"alpha/carry correlation: {np.corrcoef(A, Cc)[0,1]:+.2f}  (low = good diversifier)\n")


def stat(label, r):
    eq = np.cumprod(1 + r); years = r.size / PPY
    mdd, _, _ = max_drawdown(np.concatenate([[1.0], eq]))
    ins, oos = yr < 2025, yr >= 2025
    print(f"{label:>22} | {sharpe_ratio(r,PPY):>6.2f} {(eq[-1]**(1/years)-1)*100:>6.1f}% "
          f"{mdd*100:>6.0f}% | {sharpe_ratio(r[ins],PPY):>6.2f} {sharpe_ratio(r[oos],PPY):>7.2f}")


rp = np.array([1 / A.std(), 1 / Cc.std()]); rp /= rp.sum()
print(f"{'allocation':>22} | {'Sharpe':>6} {'CAGR':>6} {'MaxDD':>6} | {'IS':>6} {'OOS':>7}")
print("-" * 66)
stat("alpha only", A)
stat("carry only", Cc)
stat("50/50 capital", 0.5 * A + 0.5 * Cc)
stat(f"risk-parity {rp[0]:.0%}/{rp[1]:.0%}", rp[0] * A + rp[1] * Cc)
print("-" * 66)
print("Two weakly-correlated engines: the small-cap alpha and the majors carry.")
print("Blending smooths the ride; risk-parity leans hard on carry (caveat above).")
