"""Simulated execution: slippage and commission.

The fastest way to fool yourself is to assume you trade at the printed price for
free. You don't. ``SimulatedBroker`` pushes the fill price *against* you by a
slippage margin and skims a commission off every notional — the two costs that
quietly kill most paper edges.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Fill:
    """The result of executing one order."""

    side: int          # +1 buy, -1 sell
    units: float       # absolute units transacted (>= 0)
    price: float       # slippage-adjusted fill price
    commission: float  # cash paid in fees
    notional: float    # units * price (pre-commission)


class SimulatedBroker:
    def __init__(self, commission_bps: float = 10.0, slippage_bps: float = 5.0):
        """``*_bps`` are in basis points (1 bp = 0.01%)."""
        self.commission = commission_bps / 10_000.0
        self.slippage = slippage_bps / 10_000.0

    def execute(self, side: int, units: float, ref_price: float) -> Fill:
        """Fill ``units`` at ``ref_price`` adjusted for slippage + commission.

        ``side`` is +1 to buy, -1 to sell. Buys fill a touch above the reference
        price, sells a touch below — slippage always works against you.
        """
        if units <= 0:
            return Fill(side=side, units=0.0, price=ref_price, commission=0.0, notional=0.0)

        fill_price = ref_price * (1 + self.slippage) if side > 0 else ref_price * (1 - self.slippage)
        notional = units * fill_price
        commission = notional * self.commission
        return Fill(
            side=side,
            units=units,
            price=fill_price,
            commission=commission,
            notional=notional,
        )
