"""VFund — an open-source quant research & trading platform.

VFund is a learning-first, local-first toolkit for systematic trading research:
ingest market data, backtest strategies without fooling yourself, and measure
performance the way a real fund would.

The public tools are open. The edge you find with them is yours.
"""

__version__ = "0.0.1"

from vfund.data.models import Bar, BAR_SCHEMA
from vfund.backtest.engine import Backtester
from vfund.backtest.result import BacktestResult
from vfund.strategy.base import Strategy, BarContext

__all__ = [
    "__version__",
    "Bar",
    "BAR_SCHEMA",
    "Backtester",
    "BacktestResult",
    "Strategy",
    "BarContext",
]
