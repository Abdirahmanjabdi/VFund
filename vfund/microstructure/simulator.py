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

    def _flow(self):
        """Pre-generate the random flow (vectorised): value path + order stream.

        Separating flow generation (fast, vectorised NumPy) from the sequential
        matching loop lets the loop run in Rust while staying deterministic and
        parity-testable given the same flow.
        """
        rng = self.rng
        steps = np.round(self.sigma * rng.standard_normal(self.n_steps + 1)).astype(np.int64)
        value = (np.cumsum(steps) + 10_000).astype(np.int64)
        n_orders = rng.poisson(self.order_rate, self.n_steps).astype(np.int64)
        total = int(n_orders.sum())
        informed = (rng.random(total) < self.informed_frac).astype(np.int64)
        noise = np.where(rng.random(total) < 0.5, 1, -1).astype(np.int64)
        return value, n_orders, informed, noise

    def run(self) -> MMResult:
        value, n_orders, informed, noise = self._flow()
        total_pnl, n_fills = _mm_loop(value, n_orders, informed, noise, self.half_spread)
        spread_captured = self.half_spread * n_fills
        return MMResult(
            total_pnl=total_pnl, n_fills=n_fills, spread_captured=spread_captured,
            adverse_selection=spread_captured - total_pnl,
            half_spread=self.half_spread, informed_frac=self.informed_frac,
        )


def mm_loop_py(value, n_orders, informed, noise, half_spread):
    """Reference market-making matching loop (the spec for the Rust core).

    One fill per side per step (the MM quotes one unit each side and re-quotes).
    Realised-spread P&L: a fill is marked against the value right after — which is
    where adverse selection shows up.
    """
    h = half_spread
    total_pnl = 0.0
    n_fills = 0
    ptr = 0
    for t in range(len(n_orders)):
        v = int(value[t])
        future = float(value[t + 1])
        up = value[t + 1] > value[t]
        ask_avail = bid_avail = True
        for _ in range(int(n_orders[t])):
            side = (1 if up else -1) if informed[ptr] else int(noise[ptr])
            ptr += 1
            if side > 0 and ask_avail:               # lifts the ask at v+h
                total_pnl += (v + h) - future
                n_fills += 1
                ask_avail = False
            elif side < 0 and bid_avail:             # hits the bid at v-h
                total_pnl += future - (v - h)
                n_fills += 1
                bid_avail = False
    return total_pnl, n_fills


def _mm_loop(value, np_orders, informed, noise, half_spread):
    """Run the matching loop natively if the Rust core is built, else in Python."""
    try:
        import vfund_core

        if hasattr(vfund_core, "mm_loop"):
            return vfund_core.mm_loop(value, np_orders, informed, noise, int(half_spread))
    except ImportError:
        pass
    return mm_loop_py(value, np_orders, informed, noise, half_spread)
