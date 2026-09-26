#!/usr/bin/env python3
"""DMA Strategy Study — Derivative Microstructure Alpha backtest.

Fetches positioning data (open interest) and carry-cost data (funding rates)
from Binance public APIs, aligns it to the existing price panel, runs the
DMA strategy through VFund's cross-sectional backtester, and prints results.

The same architecture works for any market — swap CryptoPerpsAdapter for
USEquityAdapter or FuturesAdapter when those are implemented.

Usage:
    # Fetch derivative data and run backtest
    python examples/deriv_alpha_study.py --data data/live.parquet

    # Use previously cached derivative data
    python examples/deriv_alpha_study.py --data data/live.parquet --skip-fetch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.panel import pivot_to_wide, validate_panel
from vfund.deriv_alpha.composer import DerivAlphaComposer, format_dashboard
from vfund.deriv_alpha.markets import CryptoPerpsAdapter
from vfund.deriv_alpha.regime import FundingDispersionRegime, Regime
from vfund.strategy.cross_sectional import PanelContext


def main() -> None:
    ap = argparse.ArgumentParser(description="DMA strategy backtest")
    ap.add_argument("--data", required=True, help="OHLCV panel parquet")
    ap.add_argument("--funding-cache", default="data/deriv/funding.parquet")
    ap.add_argument("--oi-cache", default="data/deriv/oi.parquet")
    ap.add_argument("--initial-cash", type=float, default=10_000)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--skip-fetch", action="store_true")
    args = ap.parse_args()

    # --- Load price panel ---------------------------------------------------
    print("Loading price panel...")
    panel = validate_panel(pl.read_parquet(args.data))
    symbols = sorted(panel["symbol"].unique().to_list())
    ts_min = panel["timestamp"].min()
    ts_max = panel["timestamp"].max()
    print(f"  {len(symbols)} symbols, {ts_min} to {ts_max}")

    # --- Fetch or load derivative data via the market adapter ---------------
    adapter = CryptoPerpsAdapter()

    funding_path = Path(args.funding_cache)
    oi_path = Path(args.oi_cache)

    if args.skip_fetch and funding_path.exists() and oi_path.exists():
        print("Loading cached derivative data...")
        adapter.load(str(funding_path), str(oi_path))
    else:
        perp_syms = [s for s in symbols if s.endswith("USDT")]
        if not perp_syms:
            perp_syms = [f"{s}USDT" for s in symbols]
        print(f"Fetching derivative data for {len(perp_syms)} symbols...")
        adapter.fetch(perp_syms, str(ts_min), str(ts_max))
        from vfund.deriv_alpha.collector import save

        save(adapter._funding, funding_path)
        save(adapter._oi, oi_path)
        print(f"  cached to {funding_path.parent}/")

    # --- Align to price timeline --------------------------------------------
    wide = pivot_to_wide(panel, "close", drop_incomplete=False)
    bt_symbols = [c for c in wide.columns if c != "timestamp"]
    timestamps = wide["timestamp"]

    print("Aligning positioning data...")
    oi_matrix = adapter.positioning(timestamps, bt_symbols)
    funding_df = adapter.carry_cost(timestamps, bt_symbols)
    print(f"  positioning matrix: {oi_matrix.shape}")

    # --- Set up and run the DMA strategy ------------------------------------
    print("Setting up DMA strategy...")
    composer = DerivAlphaComposer(oi_matrix)

    bt = CrossSectionalBacktester(
        panel,
        composer,
        rebalance_every=7,
        top_k=args.top_k,
        interval="1d",
        cost_bps=10,
        short_cost_bps_annual=1000,
        min_short_dollar_volume=5_000_000,
        initial_cash=args.initial_cash,
        funding=funding_df,
    )

    print("Running backtest...")
    result = bt.run()

    print("\n" + "=" * 56)
    print("  DMA Strategy — Derivative Microstructure Alpha")
    print("  Market: Crypto Perpetual Futures (Binance USD-M)")
    print("=" * 56)
    print(result.summary())

    # --- Live dashboard (latest bar) ----------------------------------------
    T = bt.closes.shape[0]
    ctx = PanelContext(
        T - 1,
        bt.closes,
        bt.symbols,
        funding=bt.funding_prevailing,
        volumes=bt.volumes,
    )
    live_scores = composer.scores(ctx)

    regime_det = FundingDispersionRegime()
    regime = (
        regime_det.detect(bt.funding_prevailing[:T])
        if bt.funding_prevailing is not None
        else Regime.NEUTRAL
    )

    print("\n" + format_dashboard(live_scores, bt.symbols, regime, top_k=args.top_k))

    # --- Year-by-year breakdown ---------------------------------------------
    eq = result.equity_curve
    years = sorted(
        eq.with_columns(pl.col("timestamp").dt.year().alias("year"))["year"]
        .unique()
        .to_list()
    )
    print("\nYear-by-year:")
    for y in years:
        yr = eq.filter(pl.col("timestamp").dt.year() == y)
        start_eq = yr["equity"][0]
        end_eq = yr["equity"][-1]
        ret = end_eq / start_eq - 1.0
        print(f"  {y}: {ret:+.1%}  (${start_eq:,.0f} -> ${end_eq:,.0f})")


if __name__ == "__main__":
    main()
