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


def _build_xs_strategy(hypothesis: str, lookback: int):
    from vfund.strategy import CrossSectionalMomentum, CrossSectionalReversal

    if hypothesis == "momentum":
        return CrossSectionalMomentum(lookback)
    return CrossSectionalReversal(lookback)


def cmd_fetch_universe(args) -> int:
    from vfund.data.panel import save_panel
    from vfund.data.universe import DEFAULT_UNIVERSE, fetch_universe, top_symbols_by_volume

    if args.top:
        symbols = top_symbols_by_volume(args.top)
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = DEFAULT_UNIVERSE
    print(f"fetching {len(symbols)} symbols {args.interval} from Binance ...")
    panel = fetch_universe(symbols, args.interval, start=args.start, end=args.end)
    out = save_panel(panel, args.out)
    print(f"saved panel ({panel.height:,} rows, {panel['symbol'].n_unique()} symbols) -> {out}")
    return 0


def cmd_research(args) -> int:
    bt_kwargs = dict(
        rebalance_every=args.rebalance_every,
        leverage=args.leverage,
        top_k=args.top_k,
        cost_bps=args.cost_bps,
        interval=args.interval,
    )

    if args.demo:
        from vfund.data.synthetic import generate_gbm_panel

        panel = generate_gbm_panel(
            args.assets, args.bars, interval=args.interval,
            reversion=args.reversion, seed=args.seed,
        )
        print(
            f"synthetic panel: {args.assets} assets x {args.bars} bars, "
            f"reversion={args.reversion}\n"
        )
    else:
        from vfund.data.panel import load_panel

        panel = load_panel(args.data)

    if args.walkforward:
        from vfund.research import walk_forward

        grid = [{"lookback": lb} for lb in args.grid]
        res = walk_forward(
            panel,
            lambda lookback: _build_xs_strategy(args.hypothesis, lookback),
            grid,
            train_size=args.train_size,
            test_size=args.test_size,
            interval=args.interval,
            backtest_kwargs=bt_kwargs,
        )
        import polars as pl

        print(res.summary())
        print("\nper-window detail:")
        with pl.Config(tbl_rows=50, ascii_tables=True, tbl_width_chars=100):
            print(res.windows)
    else:
        from vfund.backtest.cross_sectional import CrossSectionalBacktester

        strat = _build_xs_strategy(args.hypothesis, args.lookback)
        res = CrossSectionalBacktester(panel, strat, **bt_kwargs).run()
        print(res.summary())
        if getattr(args, "plot", None):
            from vfund.analytics.plot import plot_equity

            print(f"\nchart saved -> {plot_equity(res, args.plot)}")
    return 0


def _add_xs_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hypothesis", choices=["reversal", "momentum"], default="reversal")
    p.add_argument("--interval", default="1h")
    p.add_argument("--lookback", type=int, default=1, help="signal lookback (single run)")
    p.add_argument("--rebalance-every", type=int, default=1)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None, help="trade only k names per side")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--plot", metavar="PNG", help="save an equity/drawdown chart")


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

    # fetch-universe
    u = sub.add_parser("fetch-universe", help="download a multi-asset panel from Binance")
    u.add_argument("--symbols", nargs="+", help="explicit symbol list (e.g. BTCUSDT ETHUSDT)")
    u.add_argument("--top", type=int, default=0, help="instead: top N by 24h volume")
    u.add_argument("--interval", default="1h")
    u.add_argument("--start", required=True, help="ISO date, e.g. 2023-01-01")
    u.add_argument("--end", default=None)
    u.add_argument("--out", required=True, help="output panel .parquet path")
    u.set_defaults(func=cmd_fetch_universe)

    # research — cross-sectional long/short, with optional walk-forward
    r = sub.add_parser("research", help="cross-sectional long/short research")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="input panel .parquet path")
    src.add_argument("--demo", action="store_true", help="use an offline synthetic panel")
    _add_xs_args(r)
    # walk-forward
    r.add_argument("--walkforward", action="store_true", help="run out-of-sample validation")
    r.add_argument("--grid", type=int, nargs="+", default=[1, 2, 3, 6, 12, 24],
                   help="candidate lookbacks to select from (walk-forward)")
    r.add_argument("--train-size", type=int, default=1500)
    r.add_argument("--test-size", type=int, default=500)
    # demo panel params
    r.add_argument("--assets", type=int, default=20)
    r.add_argument("--bars", type=int, default=6000)
    r.add_argument("--reversion", type=float, default=0.15)
    r.add_argument("--seed", type=int, default=42)
    r.set_defaults(func=cmd_research)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy codepage; UTF-8 keeps report and
    # table output from choking on non-ASCII characters.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # pragma: no cover - stream may not support it
        pass

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
