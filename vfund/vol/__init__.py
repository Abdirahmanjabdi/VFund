"""Volatility risk premium — being paid to sell insurance, not to forecast.

Every strategy elsewhere in this package tries to predict a direction. This one
does not. It collects the gap between what options imply and what the
underlying subsequently delivers, which exists because someone on the other
side wants protection and is willing to overpay for it.

That difference matters for why it should keep working. A predictive edge stops
paying once enough people find it. An insurance premium stops paying only if
buyers stop wanting insurance, and the reason it is priced above fair value is
that the seller bears a real, ugly, correlated risk. It is compensation, not a
mistake — which is also the reason it can hurt badly and must be sized as if it
will.
"""

from vfund.vol.vrp import (
    VRPStats,
    capped_pnl,
    realized_vol,
    variance_swap_pnl,
    vrp_stats,
)

__all__ = [
    "VRPStats",
    "capped_pnl",
    "realized_vol",
    "variance_swap_pnl",
    "vrp_stats",
]
