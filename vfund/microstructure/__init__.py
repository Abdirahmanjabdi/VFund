"""Market microstructure: a limit order book and a simulator for honestly
evaluating maker (market-making) execution — including adverse selection, the
effect that a naive maker backtest silently ignores.

This is the foundation for validating short-horizon / market-making edges (like
cross-sectional reversal) that die as a taker but might live as a maker — *if*
they earn more spread than adverse selection costs them.
"""

from vfund.microstructure.orderbook import LimitOrderBook, Fill
from vfund.microstructure.simulator import MarketMakingSim, MMResult

__all__ = ["LimitOrderBook", "Fill", "MarketMakingSim", "MMResult"]
