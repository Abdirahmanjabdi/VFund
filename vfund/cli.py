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


def _build_xs_strategy(hypothesis: str, param: int):
    from vfund.strategy import (
        CrossSectionalLowVol,
        CrossSectionalMomentum,
        CrossSectionalReversal,
        CrossSectionalValue,
        FundingCarry,
        TimeSeriesTrend,
    )

    return {
        "reversal": lambda p: CrossSectionalReversal(p),
        "momentum": lambda p: CrossSectionalMomentum(p),
        "lowvol": lambda p: CrossSectionalLowVol(p),
        "value": lambda p: CrossSectionalValue(p),
        "trend": lambda p: TimeSeriesTrend(p),
        "carry": lambda p: FundingCarry(smooth=p),
    }[hypothesis](param)


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


def cmd_fetch_tvl(args) -> int:
    from vfund.data.onchain import fetch_tvl_panel, save_tvl
    from vfund.data.universe import DEFI_UNIVERSE

    symbols = args.symbols or DEFI_UNIVERSE
    print(f"fetching TVL for {len(symbols)} DeFi tokens from DefiLlama ...")
    tvl = fetch_tvl_panel(symbols, start=args.start, end=args.end)
    out = save_tvl(tvl, args.out)
    print(f"saved TVL ({tvl.height:,} rows, {tvl['symbol'].n_unique()} tokens) -> {out}")
    return 0


def cmd_fetch_funding(args) -> int:
    from vfund.data.panel import save_funding
    from vfund.data.universe import DEFAULT_UNIVERSE, fetch_funding_universe, top_symbols_by_volume

    if args.top:
        symbols = top_symbols_by_volume(args.top)
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = DEFAULT_UNIVERSE
    print(f"fetching funding for {len(symbols)} symbols ...")
    funding = fetch_funding_universe(symbols, start=args.start, end=args.end)
    out = save_funding(funding, args.out)
    print(f"saved funding ({funding.height:,} rows, {funding['symbol'].n_unique()} symbols) -> {out}")
    return 0


def _load_clean(path):
    from vfund.data.panel import load_panel
    from vfund.data.universe import clean_universe

    return clean_universe(load_panel(path))


def cmd_signal(args) -> int:
    from vfund.live.signal import combined_book, format_book

    panel = _load_clean(args.data)
    book = combined_book(
        panel, size_lookback=args.size_lookback, vol_target=args.vol_target,
        top_k=args.top_k, interval=args.interval,
        min_short_dollar_volume=args.min_short_dv,
    )
    print(format_book(book))
    return 0


def cmd_paper(args) -> int:
    from vfund.live.paper import PaperTracker
    from vfund.live.signal import format_book

    tracker = PaperTracker(args.state, start_equity=args.start_equity)
    if args.three_sleeve:
        from vfund.data.onchain import load_tvl
        from vfund.data.panel import load_panel

        res = tracker.update_three_sleeve(
            _load_clean(args.data), load_panel(args.defi_data), load_tvl(args.tvl_data),
            cost_bps=args.cost_bps, min_short_dollar_volume=args.min_short_dv,
        )
    else:
        res = tracker.update(
            _load_clean(args.data), cost_bps=args.cost_bps,
            size_lookback=args.size_lookback, vol_target=args.vol_target,
            top_k=args.top_k, interval=args.interval,
            min_short_dollar_volume=args.min_short_dv,
        )
    st = tracker.load()
    ret = st["equity"] / st["start_equity"] - 1.0
    print(f"[{res['status']}] as of {res['asof']}")
    print(f"  last-period return   {res['port_ret']*100:+.2f}%")
    print(f"  paper equity         ${st['equity']:,.2f}  ({ret*100:+.1f}% since {st['created'][:10]})")
    print(f"  updates recorded     {len(st['history'])}\n")
    print(format_book(res["book"]))
    return 0


def cmd_research(args) -> int:
    bt_kwargs = dict(
        rebalance_every=args.rebalance_every,
        leverage=args.leverage,
        top_k=args.top_k,
        neutralize=not (args.directional or args.hypothesis == "trend"),
        cost_bps=args.cost_bps,
        short_cost_bps_annual=args.short_cost_bps_annual,
        interval=args.interval,
    )

    if args.demo:
        if args.hypothesis == "carry":
            raise ValueError("carry needs real funding data; use --data + --funding")
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

    funding = None
    if args.funding:
        from vfund.data.panel import load_funding

        funding = load_funding(args.funding)
    elif args.hypothesis == "carry":
        raise ValueError("carry hypothesis requires --funding <panel.parquet>")

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
            funding=funding,
        )
        import polars as pl

        print(res.summary())
        print("\nper-window detail:")
        with pl.Config(tbl_rows=50, ascii_tables=True, tbl_width_chars=100):
            print(res.windows)
    else:
        from vfund.backtest.cross_sectional import CrossSectionalBacktester

        strat = _build_xs_strategy(args.hypothesis, args.lookback)
        res = CrossSectionalBacktester(panel, strat, funding=funding, **bt_kwargs).run()
        print(res.summary())
        if getattr(args, "plot", None):
            from vfund.analytics.plot import plot_equity

            print(f"\nchart saved -> {plot_equity(res, args.plot)}")
    return 0


def _add_xs_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--hypothesis",
        choices=["reversal", "momentum", "lowvol", "value", "trend", "carry"],
        default="reversal",
    )
    p.add_argument("--interval", default="1h")
    p.add_argument("--lookback", type=int, default=1, help="signal lookback/smooth (single run)")
    p.add_argument("--rebalance-every", type=int, default=1)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None, help="trade only k names per side")
    p.add_argument("--directional", action="store_true",
                   help="net long/short book (auto for trend); else dollar-neutral")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--short-cost-bps-annual", type=float, default=0.0,
                   help="annualised financing charged on short notional")
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

    # fetch-funding
    fu = sub.add_parser("fetch-funding", help="download perp funding-rate history")
    fu.add_argument("--symbols", nargs="+", help="explicit symbol list")
    fu.add_argument("--top", type=int, default=0, help="instead: top N by 24h volume")
    fu.add_argument("--start", required=True, help="ISO date")
    fu.add_argument("--end", default=None)
    fu.add_argument("--out", required=True, help="output funding .parquet path")
    fu.set_defaults(func=cmd_fetch_funding)

    # fetch-tvl
    tv = sub.add_parser("fetch-tvl", help="download DeFi TVL history from DefiLlama")
    tv.add_argument("--symbols", nargs="+", help="DeFi token symbols (default: DEFI_UNIVERSE)")
    tv.add_argument("--start", required=True, help="ISO date")
    tv.add_argument("--end", default=None)
    tv.add_argument("--out", required=True, help="output TVL .parquet path")
    tv.set_defaults(func=cmd_fetch_tvl)

    # research — cross-sectional long/short, with optional walk-forward
    r = sub.add_parser("research", help="cross-sectional long/short research")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="input panel .parquet path")
    src.add_argument("--demo", action="store_true", help="use an offline synthetic panel")
    r.add_argument("--funding", help="funding panel .parquet (required for --hypothesis carry)")
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

    # signal — today's target book from the combined edge
    sig = sub.add_parser("signal", help="print today's combined trend+size target book")
    sig.add_argument("--data", required=True, help="panel .parquet path")
    sig.add_argument("--interval", default="1d")
    sig.add_argument("--size-lookback", type=int, default=20)
    sig.add_argument("--vol-target", type=float, default=0.30)
    sig.add_argument("--top-k", type=int, default=5)
    sig.add_argument("--min-short-dv", type=float, default=5_000_000,
                     help="min trailing $/day to short a name (hard-to-short gate)")
    sig.set_defaults(func=cmd_signal)

    # paper — forward-track a hypothetical account
    pap = sub.add_parser("paper", help="update a persistent paper-trading account")
    pap.add_argument("--data", required=True, help="panel .parquet path (refetch to advance)")
    pap.add_argument("--state", required=True, help="paper account .json path")
    pap.add_argument("--start-equity", type=float, default=10_000.0)
    pap.add_argument("--cost-bps", type=float, default=10.0)
    pap.add_argument("--interval", default="1d")
    pap.add_argument("--size-lookback", type=int, default=20)
    pap.add_argument("--vol-target", type=float, default=0.30)
    pap.add_argument("--top-k", type=int, default=5)
    pap.add_argument("--min-short-dv", type=float, default=5_000_000,
                     help="min trailing $/day to short a name (hard-to-short gate)")
    pap.add_argument("--three-sleeve", action="store_true",
                     help="track the diversified trend+size+on-chain book")
    pap.add_argument("--defi-data", help="DeFi price panel .parquet (for --three-sleeve)")
    pap.add_argument("--tvl-data", help="TVL .parquet (for --three-sleeve)")
    pap.set_defaults(func=cmd_paper)

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
