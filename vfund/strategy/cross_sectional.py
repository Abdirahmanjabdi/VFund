"""Cross-sectional strategies: rank a universe, not a single asset.

A cross-sectional strategy looks at all coins at once and emits a *score* per
coin. The engine turns scores into a dollar-neutral long/short book (long the
high scores, short the low ones), so the shared market move cancels out and what
remains is *relative* skill.

The economic stories:

* **Reversal** — over short horizons retail overreacts; the coin that just
  spiked tends to give some back. Score = minus the recent return (long the
  losers, short the winners).
* **Momentum** — over longer horizons trends persist. Score = the recent return.

Neither is guaranteed edge. They are *hypotheses with a reason*, which is the
only kind worth testing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PanelContext:
    """Point-in-time view of the whole universe handed to ``scores``.

    ``closes`` is a ``(n_seen, n_assets)`` matrix truncated at the current bar —
    so, as in the single-asset engine, the future is simply not in scope.
    """

    __slots__ = ("i", "_closes", "symbols")

    def __init__(self, i: int, closes: np.ndarray, symbols: list[str]):
        self.i = i
        self._closes = closes
        self.symbols = symbols

    @property
    def closes(self) -> np.ndarray:
        return self._closes[: self.i + 1]

    @property
    def n_seen(self) -> int:
        return self.i + 1


class CrossSectionalStrategy(ABC):
    @abstractmethod
    def scores(self, ctx: PanelContext) -> np.ndarray:
        """Return one score per asset (NaN to exclude an asset this bar)."""
        raise NotImplementedError


class CrossSectionalReversal(CrossSectionalStrategy):
    def __init__(self, lookback: int = 1):
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        self.lookback = lookback

    def scores(self, ctx: PanelContext) -> np.ndarray:
        closes = ctx.closes
        if closes.shape[0] <= self.lookback:
            return np.full(closes.shape[1], np.nan)
        past_return = closes[-1] / closes[-1 - self.lookback] - 1.0
        return -past_return  # long the losers, short the winners


class CrossSectionalMomentum(CrossSectionalStrategy):
    def __init__(self, lookback: int = 168):  # ~1 week of hourly bars
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        self.lookback = lookback

    def scores(self, ctx: PanelContext) -> np.ndarray:
        closes = ctx.closes
        if closes.shape[0] <= self.lookback:
            return np.full(closes.shape[1], np.nan)
        past_return = closes[-1] / closes[-1 - self.lookback] - 1.0
        return past_return  # long the winners, short the losers
