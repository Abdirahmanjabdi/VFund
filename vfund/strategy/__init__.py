"""Strategies: the research surface where edge is discovered.

A strategy sees only data up to and including the current bar and returns a
*target weight* — the fraction of equity it wants in the asset. The engine does
the rest (rebalancing, costs, accounting). This keeps strategy code about the
*idea*, not the plumbing.
"""

from vfund.strategy.base import Strategy, BarContext
from vfund.strategy.ma_crossover import MACrossover
from vfund.strategy.buy_and_hold import BuyAndHold
from vfund.strategy.cross_sectional import (
    CrossSectionalStrategy,
    CrossSectionalReversal,
    CrossSectionalMomentum,
    CrossSectionalLowVol,
    CrossSectionalValue,
    CrossSectionalSize,
    CrossSectionalMaxReturn,
    CrossSectionalResidualMomentum,
    CrossSectionalIlliquidity,
    TVLMomentum,
    TVLDivergence,
    TimeSeriesTrend,
    TimeSeriesTrendEnsemble,
    FundingCarry,
    PanelContext,
)

__all__ = [
    "Strategy",
    "BarContext",
    "MACrossover",
    "BuyAndHold",
    "CrossSectionalStrategy",
    "CrossSectionalReversal",
    "CrossSectionalMomentum",
    "CrossSectionalLowVol",
    "CrossSectionalValue",
    "CrossSectionalSize",
    "CrossSectionalMaxReturn",
    "CrossSectionalResidualMomentum",
    "CrossSectionalIlliquidity",
    "TVLMomentum",
    "TVLDivergence",
    "TimeSeriesTrend",
    "TimeSeriesTrendEnsemble",
    "FundingCarry",
    "PanelContext",
]
