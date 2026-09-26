#!/usr/bin/env python3
"""Gold variance risk premium — measure it, cap it, size it.

Fetches the CBOE gold volatility index (GVZ) and GLD, measures the premium
between implied and subsequently realised volatility, and reports what a
defined-risk seller would have earned.

    python examples/gold_vrp_study.py
    python examples/gold_vrp_study.py --max-loss 3 --risk-budget 0.05

No API key required; both series come from the public Yahoo chart endpoint.
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from vfund.data.yahoo import fetch_many
from vfund.research.ic import newey_west_t
from vfund.vol.vrp import (
    MONTH_BARS,
    capped_pnl,
    realized_vol,
    variance_swap_pnl,
    vrp_stats,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gold variance risk premium study")
    ap.add_argument("--cache", default="data/gold/vrp.parquet")
    ap.add_argument("--refresh", action="store_true", help="re-fetch instead of using cache")
    ap.add_argument("--max-loss", type=float, default=5.0,
                    help="loss cap in vol points (the protective wing)")
    ap.add_argument("--wing-cost", type=float, default=0.35,
                    help="fraction of winning P&L given up to buy the wing")
    ap.add_argument("--risk-budget", type=float, default=0.05,
                    help="fraction of capital risked in the worst month")
    args = ap.parse_args()

    from pathlib import Path

    path = Path(args.cache)
    if path.exists() and not args.refresh:
        df = pl.read_parquet(path)
    else:
        df = fetch_many(["^GVZ", "GLD"], interval="1d",
                        start="2008-01-01", end=None)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)

    w = (df.with_columns(pl.col("timestamp").dt.truncate("1d"))
         .pivot(index="timestamp", on="symbol", values="close",
                aggregate_function="first")
         .sort("timestamp"))
    ts = w["timestamp"].to_numpy()
    gvz = w["^GVZ"].to_numpy().astype(float)
    gld = w["GLD"].to_numpy().astype(float)

    lr = np.full(len(gld), np.nan)
    lr[1:] = np.log(gld[1:] / gld[:-1])
    rv = realized_vol(lr, MONTH_BARS, forward=True)

    print("=" * 62)
    print("  GOLD VARIANCE RISK PREMIUM")
    print("=" * 62)
    s = vrp_stats(gvz, rv)
    print(s.summary())

    # Non-overlapping months are the only honest basis for a P&L t-statistic.
    pnl = variance_swap_pnl(gvz, rv)
    ok = np.where(np.isfinite(pnl))[0]
    months = pnl[ok[::MONTH_BARS]]
    dates = ts[ok[::MONTH_BARS]]

    print(f"\n  Selling one month of volatility, {len(months)} independent months")
    print(f"    mean            {months.mean():+.2f} vol pts")
    print(f"    median          {np.median(months):+.2f}")
    print(f"    win rate        {(months > 0).mean() * 100:.1f}%")
    print(f"    t-stat          {newey_west_t(months, lag=1):.2f}")
    print(f"    worst month     {months.min():+.2f}   ({str(dates[months.argmin()])[:10]})")
    print(f"    worst / mean    {abs(months.min()) / months.mean():.0f}x")

    capped = capped_pnl(months, max_loss=args.max_loss, wing_cost=args.wing_cost)
    print(f"\n  With the loss capped at {args.max_loss:.0f} vol pts "
          f"(wing costs {args.wing_cost:.0%} of gains)")
    print(f"    mean            {capped.mean():+.2f} vol pts/month")
    print(f"    worst month     {capped.min():+.2f}")
    print(f"    t-stat          {newey_west_t(capped, lag=1):.2f}")

    # Sizing: risk the budget in the capped worst case.
    vega_per_capital = args.risk_budget / args.max_loss
    monthly = capped.mean() * vega_per_capital
    print(f"\n  Sizing so the worst month costs {args.risk_budget:.0%} of capital")
    print(f"    vega notional   {vega_per_capital:.1%} of capital per vol point")
    print(f"    expected        {monthly:+.2%}/month  ~{monthly * 12:+.1%}/year")
    print("    (before option bid-ask, which on a multi-leg structure is real)")

    print("\n  Year by year (mean premium, vol points)")
    years = ts.astype("datetime64[Y]").astype(int) + 1970
    good = np.isfinite(gvz) & np.isfinite(rv)
    for y in sorted(set(years[good].tolist())):
        m = good & (years == y)
        if m.sum() < 50:
            continue
        d = gvz[m] - rv[m]
        bar = "#" * max(int(d.mean() * 3), 0) if d.mean() > 0 else "<loss>"
        print(f"    {y}  {d.mean():+6.2f}  {(d > 0).mean() * 100:5.1f}% pos  {bar}")

    print("\n  Caveats worth keeping in front of you:")
    print("    - measured from a volatility index, not traded option prices")
    print("    - daily-sampled realised variance understates the true figure,")
    print("      which biases this estimate of the premium slightly upward")
    print("    - the premium is shrinking: +3.82 (2008-13), +2.34 (2014-19),")
    print("      +1.94 (2020-26) vol points")
    print("    - it loses money in exactly the months you can least afford it")


if __name__ == "__main__":
    main()
