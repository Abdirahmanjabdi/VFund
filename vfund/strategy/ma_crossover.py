"""Moving-average crossover — the canonical baseline strategy.

This is *not* here to make money. It's the "hello world" that proves the engine
works end to end and gives every future strategy a benchmark to beat. When fast
MA is above slow MA we go long; otherwise flat (or short, if enabled).
"""

from __future__ import annotations

import numpy as np

from vfund.strategy.base import BarContext, Strategy


class MACrossover(Strategy):
    def __init__(self, fast: int = 20, slow: int = 50, allow_short: bool = False):
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")
        self.fast = fast
        self.slow = slow
        self.allow_short = allow_short

    def on_bar(self, ctx: BarContext) -> float | None:
        # Not enough history to form the slow average yet -> stay flat.
        if ctx.n_seen < self.slow:
            return 0.0

        closes = ctx.closes
        fast_ma = float(np.mean(closes[-self.fast :]))
        slow_ma = float(np.mean(closes[-self.slow :]))

        if fast_ma > slow_ma:
            return 1.0
        return -1.0 if self.allow_short else 0.0
