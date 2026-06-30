"""VFund command-line interface.

    vfund demo                          # offline: synthetic data -> backtest -> report
    vfund fetch  --symbol BTCUSDT ...   # pull real crypto bars from Binance
    vfund backtest --data btc.parquet   # backtest a strategy on stored data

Run ``vfund <command> -h`` for per-command options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vfund import __version__


def _add_strategy_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--strategy", choices=["ma", "buyhold"], default="ma")
    p.add_argument("--fast", type=int, default=20, help="fast MA window")
    p.add_argument("--slow", type=int, default=50, help="slow MA window")
    p.add_argument("--allow-short", action="store_true", help="let MA go short")
    p.add_argument("--cash", type=float, default=10_000.0, help="initial equity")
    p.add_argument("--commission-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--plot", metavar="PNG", help="save an equity/drawdown chart")


def _build_strategy(args):
    from vfund.strategy import BuyAndHold, MACrossover

    if args.strategy == "buyhold":
        return BuyAndHold()
    return MACrossover(fast=args.fast, slow=args.slow, allow_short=args.allow_short)


def _run_backtest(data, args, interval: str):
    from vfund.backtest import Backtester

    bt = Backtester(
        data,
        _build_strategy(args),
        initial_cash=args.cash,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        interval=interval,
    )
    result = bt.run()
    print(result.summary())
    if getattr(args, "plot", None):
        from vfund.analytics.plot import plot_equity

        out = plot_equity(result, args.plot)
        print(f"\nchart saved -> {out}")
    return result


def cmd_demo(args) -> int:
    from vfund.data.synthetic import generate_gbm_bars

    print(f"VFund {__version__} - offline demo (synthetic GBM data)\n")
    data = generate_gbm_bars(args.bars, interval=args.interval, seed=args.seed)
    _run_backtest(data, args, args.interval)
    return 0


def cmd_fetch(args) -> int:
    from vfund.data.ingest import fetch_klines
    from vfund.data.storage import save_parquet

    print(f"fetching {args.symbol} {args.interval} from Binance ...")
    df = fetch_klines(args.symbol, args.interval, start=args.start, end=args.end)
    out = save_parquet(df, args.out)
    print(f"saved {df.height:,} bars -> {out}")
    return 0


def cmd_backtest(args) -> int:
    from vfund.data.storage import load_parquet

    data = load_parquet(args.data)
    _run_backtest(data, args, args.interval)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vfund", description=__doc__)
    parser.add_argument("--version", action="version", version=f"vfund {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # demo
    d = sub.add_parser("demo", help="run an offline synthetic-data backtest")
    d.add_argument("--bars", type=int, default=2000)
    d.add_argument("--interval", default="1h")
    d.add_argument("--seed", type=int, default=42)
    _add_strategy_args(d)
    d.set_defaults(func=cmd_demo)

    # fetch
    f = sub.add_parser("fetch", help="download crypto bars from Binance")
    f.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    f.add_argument("--interval", default="1h")
    f.add_argument("--start", required=True, help="ISO date, e.g. 2023-01-01")
    f.add_argument("--end", default=None, help="ISO date (default: now)")
    f.add_argument("--out", required=True, help="output .parquet path")
    f.set_defaults(func=cmd_fetch)

    # backtest
    b = sub.add_parser("backtest", help="backtest a strategy on stored data")
    b.add_argument("--data", required=True, help="input .parquet path")
    b.add_argument("--interval", default="1h", help="bar interval (for annualising)")
    _add_strategy_args(b)
    b.set_defaults(func=cmd_backtest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
