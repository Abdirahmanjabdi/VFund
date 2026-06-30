"""The output of a backtest: the equity curve, the trade log, and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl


@dataclass
class BacktestResult:
    """Everything a backtest produces, ready for analysis.

    ``equity_curve`` has columns: timestamp, close, position, equity.
    ``trades`` has columns: timestamp, side, units, price, commission, equity.
    """

    equity_curve: pl.DataFrame
    trades: pl.DataFrame
    initial_cash: float
    bars_per_year: float
    meta: dict = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve["equity"][-1])

    @property
    def n_trades(self) -> int:
        return self.trades.height

    def metrics(self) -> dict:
        """Compute performance metrics (lazy import to avoid a cycle)."""
        from vfund.analytics.performance import compute_metrics

        return compute_metrics(self)

    def summary(self) -> str:
        from vfund.analytics.performance import format_report

        return format_report(self.metrics())
