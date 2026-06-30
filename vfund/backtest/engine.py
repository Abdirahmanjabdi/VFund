"""The event-driven backtester.

The one rule this engine exists to enforce: **a decision made on bar ``t`` may
only use information available at bar ``t``, and is executed at bar ``t+1``.**

Concretely, each bar we:

1. Execute any order the strategy requested on the *previous* bar, filling it at
   *this* bar's open (you can't act on a close you haven't seen yet).
2. Mark the portfolio to market at this bar's close.
3. Record the equity point.
4. Ask the strategy for a new target weight, given history through this close.
   That request becomes the order executed at the next open.

This single-bar execution lag is what separates an honest backtest from a
fantasy that "buys the dip" at a price it could never have gotten.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from vfund.backtest.broker import SimulatedBroker
from vfund.backtest.portfolio import Portfolio
from vfund.backtest.result import BacktestResult
from vfund.data.intervals import bars_per_year as _bars_per_year
from vfund.data.models import Bar, validate_bars
from vfund.strategy.base import BarContext, Strategy


class Backtester:
    def __init__(
        self,
        data: pl.DataFrame,
        strategy: Strategy,
        *,
        initial_cash: float = 10_000.0,
        commission_bps: float = 10.0,
        slippage_bps: float = 5.0,
        interval: str = "1h",
    ):
        self.data = validate_bars(data)
        self.strategy = strategy
        self.portfolio = Portfolio(initial_cash)
        self.broker = SimulatedBroker(commission_bps, slippage_bps)
        self.interval = interval
        self.bars_per_year = _bars_per_year(interval)

    def run(self) -> BacktestResult:
        df = self.data
        n = df.height

        ts = df["timestamp"].to_numpy()
        opens = df["open"].to_numpy()
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()
        volumes = df["volume"].to_numpy()

        self.strategy.on_start()

        # Records for the equity curve and trade log. The equity curve has one
        # row per bar (so its timestamps are just ``df["timestamp"]``); the
        # trade log stores the *index* of each fill and gathers timestamps at
        # the end, sidestepping fragile per-scalar datetime construction.
        eq_close, eq_pos, eq_val = [], [], []
        tr_idx, tr_side, tr_units, tr_price, tr_comm, tr_eq = [], [], [], [], [], []

        pending_target: float | None = None  # weight requested on the prior bar
        current_target: float = 0.0          # last weight the engine is holding to

        for i in range(n):
            open_px = float(opens[i])

            # (1) Execute the order requested on the previous bar at this open.
            if pending_target is not None and pending_target != current_target:
                self._rebalance_to(
                    target_weight=pending_target,
                    price=open_px,
                    idx=i,
                    log=(tr_idx, tr_side, tr_units, tr_price, tr_comm, tr_eq),
                )
                current_target = pending_target
            pending_target = None

            # (2) Mark to market at the close.
            close_px = float(closes[i])
            equity = self.portfolio.equity(close_px)

            # (3) Record the equity point.
            eq_close.append(close_px)
            eq_pos.append(self.portfolio.position)
            eq_val.append(equity)

            # (4) Ask the strategy for the next target (no future data in scope).
            bar = Bar(ts[i], open_px, float(highs[i]), float(lows[i]), close_px, float(volumes[i]))
            ctx = BarContext(i, bar, closes, highs, lows, volumes, ts)
            target = self.strategy.on_bar(ctx)
            if target is not None:
                target = float(np.clip(target, -1.0, 1.0))
                pending_target = target

        equity_curve = pl.DataFrame(
            {
                "timestamp": df["timestamp"],  # one row per bar, in order
                "close": eq_close,
                "position": eq_pos,
                "equity": eq_val,
            }
        )
        trades = pl.DataFrame(
            {
                "timestamp": df["timestamp"].gather(tr_idx)
                if tr_idx
                else pl.Series("timestamp", [], dtype=pl.Datetime("ms", time_zone="UTC")),
                "side": tr_side,
                "units": tr_units,
                "price": tr_price,
                "commission": tr_comm,
                "equity": tr_eq,
            }
        )

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            initial_cash=self.portfolio.initial_cash,
            bars_per_year=self.bars_per_year,
            meta={
                "strategy": type(self.strategy).__name__,
                "interval": self.interval,
                "n_bars": n,
            },
        )

    def _rebalance_to(self, target_weight, price, idx, log) -> None:
        """Trade toward ``target_weight`` of current equity at ``price``."""
        tr_idx, tr_side, tr_units, tr_price, tr_comm, tr_eq = log

        equity = self.portfolio.equity(price)
        target_units = (target_weight * equity) / price
        delta = target_units - self.portfolio.position
        if abs(delta) < 1e-12:
            return

        side = 1 if delta > 0 else -1
        fill = self.broker.execute(side, abs(delta), price)
        self.portfolio.apply_fill(fill)

        tr_idx.append(idx)
        tr_side.append(side)
        tr_units.append(fill.units)
        tr_price.append(fill.price)
        tr_comm.append(fill.commission)
        tr_eq.append(self.portfolio.equity(price))
