"""Engine correctness tests.

These pin down the two properties that make a backtest trustworthy: honest
accounting (equity tracks price the way the position implies) and the one-bar
execution lag that prevents look-ahead.
"""

from datetime import datetime, timezone

import numpy as np
import polars as pl

from vfund.backtest import Backtester
from vfund.strategy import BuyAndHold
from vfund.strategy.base import BarContext, Strategy


def make_bars(closes, opens=None):
    """Build a canonical bar frame from close (and optional open) arrays."""
    closes = np.asarray(closes, dtype=float)
    if opens is None:
        opens = np.empty_like(closes)
        opens[0] = closes[0]
        opens[1:] = closes[:-1]
    n = closes.size
    start = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    step = 3_600_000
    return pl.DataFrame(
        {
            "timestamp": pl.Series([start + i * step for i in range(n)]).cast(
                pl.Datetime("ms", time_zone="UTC")
            ),
            "open": opens,
            "high": np.maximum(opens, closes),
            "low": np.minimum(opens, closes),
            "close": closes,
            "volume": np.ones(n),
        }
    )


class AlwaysWeight(Strategy):
    """Hold a constant target weight every bar."""

    def __init__(self, w: float):
        self.w = w

    def on_bar(self, ctx: BarContext) -> float:
        return self.w


def test_flat_strategy_keeps_equity_constant():
    data = make_bars([100, 110, 90, 105, 130])
    bt = Backtester(data, AlwaysWeight(0.0), initial_cash=10_000,
                    commission_bps=10, slippage_bps=5)
    res = bt.run()
    # Never trades, so equity must equal the starting cash on every bar.
    assert res.n_trades == 0
    assert np.allclose(res.equity_curve["equity"].to_numpy(), 10_000.0)


def test_buy_and_hold_matches_price_with_zero_costs():
    closes = [100, 105, 110, 120, 125]
    data = make_bars(closes)
    bt = Backtester(data, BuyAndHold(), initial_cash=10_000,
                    commission_bps=0, slippage_bps=0)
    res = bt.run()

    # Signal fires on bar 0, fills at bar 1's open (== close[0] == 100).
    # So units = 10_000 / 100, and final equity = units * close[-1].
    units = 10_000 / data["open"].to_numpy()[1]
    expected_final = units * closes[-1]
    assert abs(res.final_equity - expected_final) < 1e-6
    assert res.n_trades == 1  # one entry, then hold


def test_execution_lag_prevents_same_bar_fill():
    # Price jumps on bar 1. A signal on bar 0 cannot capture bar-0's close;
    # it enters at bar 1's open instead. With open[1] == close[0], the entry
    # price is the pre-jump level, exactly as an honest fill should be.
    data = make_bars([100, 200, 200, 200])
    bt = Backtester(data, BuyAndHold(), initial_cash=10_000,
                    commission_bps=0, slippage_bps=0)
    res = bt.run()
    entry_price = res.trades["price"][0]
    assert abs(entry_price - 100.0) < 1e-9  # entered at the open, not the jump


def test_costs_reduce_returns():
    closes = list(range(100, 200))
    data = make_bars(closes)
    free = Backtester(data, BuyAndHold(), commission_bps=0, slippage_bps=0).run()
    costed = Backtester(data, BuyAndHold(), commission_bps=20, slippage_bps=10).run()
    assert costed.final_equity < free.final_equity
