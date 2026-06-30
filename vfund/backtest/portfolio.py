"""Portfolio accounting: cash, position, and mark-to-market equity.

A deliberately tiny class. v0 trades a single asset, so the book is just cash
plus units held. Keeping it isolated means the multi-asset version later is a
drop-in replacement, not a rewrite.
"""

from __future__ import annotations

from vfund.backtest.broker import Fill


class Portfolio:
    def __init__(self, initial_cash: float = 10_000.0):
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.position = 0.0  # units of the asset held (can be negative if short)

    def equity(self, price: float) -> float:
        """Mark-to-market account value at ``price``."""
        return self.cash + self.position * price

    def apply_fill(self, fill: Fill) -> None:
        """Update cash and position from an executed fill."""
        if fill.units == 0:
            return
        signed_units = fill.side * fill.units
        # Buying spends cash (notional + fee); selling raises cash (notional - fee).
        self.cash -= fill.side * fill.notional
        self.cash -= fill.commission
        self.position += signed_units
