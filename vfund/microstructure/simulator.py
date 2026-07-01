"""A market-making simulation that exposes adverse selection.

The setup, following the classic informed/uninformed (Glosten-Milgrom) story:

* A fundamental value random-walks.
* Market orders arrive each step. A fraction are *informed* — they buy just
  before the value rises and sell just before it falls. The rest are noise.
* A market maker (the only liquidity here) posts a bid and an ask around the
  value, each ``half_spread`` ticks wide, and fills whatever hits it.

A naive backtest assumes the MM simply *earns the spread* on every fill. But
informed orders lift the ask right before the value rises (the MM sold too low)
and hit the bid right before it falls (bought too high). That's **adverse
selection**, and it emerges here for free. The result reports the naive spread
capture, the realised P&L, and the gap between them — the cost a maker backtest
must not ignore.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vfund.microstructure.orderbook import LimitOrderBook


@dataclass
class MMResult:
    total_pnl: float          # realised, marked at the final value
    n_fills: int
    spread_captured: float    # what a naive "earn the spread" backtest assumes
    adverse_selection: float  # spread_captured - total_pnl (the hidden cost)
    half_spread: int
    informed_frac: float

    @property
    def pnl_per_fill(self) -> float:
        return self.total_pnl / self.n_fills if self.n_fills else 0.0

    def summary(self) -> str:
        return (
            f"half_spread={self.half_spread}t informed={self.informed_frac:.0%} | "
            f"fills={self.n_fills:5d}  spread_capt={self.spread_captured:8.0f}  "
            f"adverse={self.adverse_selection:8.0f}  net={self.total_pnl:8.0f}  "
            f"({'PROFIT' if self.total_pnl > 0 else 'LOSS'})"
        )


class MarketMakingSim:
    def __init__(
        self,
        n_steps: int = 50_000,
        *,
        half_spread: int = 2,
        informed_frac: float = 0.3,
        order_rate: float = 1.0,
        sigma: float = 1.0,
        mm_size: float = 1.0,
        seed: int | None = 0,
    ):
        self.n_steps = n_steps
        self.half_spread = half_spread
        self.informed_frac = informed_frac
        self.order_rate = order_rate
        self.sigma = sigma
        self.mm_size = mm_size
        self.rng = np.random.default_rng(seed)

    def run(self) -> MMResult:
        rng = self.rng
        h = self.half_spread
        # Fundamental value path (integer ticks).
        steps = np.round(self.sigma * rng.standard_normal(self.n_steps + 1)).astype(int)
        value = np.cumsum(steps) + 10_000

        book = LimitOrderBook()
        total_pnl = 0.0
        n_fills = 0

        for t in range(self.n_steps):
            v = int(value[t])
            future = float(value[t + 1])          # value right after this step
            future_up = future > value[t]         # informed know the next move

            # MM re-quotes around fair value.
            book.bids.clear()
            book.asks.clear()
            book.add(+1, v - h, self.mm_size)     # bid
            book.add(-1, v + h, self.mm_size)     # ask

            n_orders = rng.poisson(self.order_rate)
            for _ in range(n_orders):
                if rng.random() < self.informed_frac:
                    side = 1 if future_up else -1          # informed
                else:
                    side = 1 if rng.random() < 0.5 else -1  # noise
                for fill in book.match_market(side, self.mm_size):
                    # Realised-spread P&L: mark the passive fill against the value
                    # right after. MM sold (side +1) at price p -> p - future; MM
                    # bought (side -1) -> future - p. Adverse selection appears when
                    # the post-fill move exceeds the half-spread.
                    total_pnl += fill.side * (fill.price - future) * fill.qty
                    n_fills += 1

        spread_captured = h * n_fills  # naive: half-spread earned per fill
        return MMResult(
            total_pnl=total_pnl, n_fills=n_fills, spread_captured=spread_captured,
            adverse_selection=spread_captured - total_pnl,
            half_spread=h, informed_frac=self.informed_frac,
        )
