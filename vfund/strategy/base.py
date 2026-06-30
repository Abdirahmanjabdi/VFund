"""The strategy interface.

A ``Strategy`` is a function of history -> target weight. The single most
important property of this design is that ``on_bar`` is handed a ``BarContext``
that *only* exposes data up to and including the current bar. You cannot
accidentally peek at the future, because the future isn't in scope.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from vfund.data.models import Bar


class BarContext:
    """A point-in-time view of the market handed to ``Strategy.on_bar``.

    ``i`` is the index of the current bar. ``closes``/``highs``/``lows`` are
    NumPy views truncated at ``i`` (inclusive), so any indicator you compute is
    automatically free of look-ahead bias.
    """

    __slots__ = ("i", "bar", "_closes", "_highs", "_lows", "_volumes", "_ts")

    def __init__(
        self,
        i: int,
        bar: Bar,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        timestamps: np.ndarray,
    ) -> None:
        self.i = i
        self.bar = bar
        self._closes = closes
        self._highs = highs
        self._lows = lows
        self._volumes = volumes
        self._ts = timestamps

    @property
    def closes(self) -> np.ndarray:
        """All closes up to and including the current bar."""
        return self._closes[: self.i + 1]

    @property
    def highs(self) -> np.ndarray:
        return self._highs[: self.i + 1]

    @property
    def lows(self) -> np.ndarray:
        return self._lows[: self.i + 1]

    @property
    def volumes(self) -> np.ndarray:
        return self._volumes[: self.i + 1]

    @property
    def n_seen(self) -> int:
        """How many bars the strategy has observed so far."""
        return self.i + 1


class Strategy(ABC):
    """Base class for all VFund strategies."""

    def on_start(self) -> None:
        """Called once before the first bar. Override for setup."""

    @abstractmethod
    def on_bar(self, ctx: BarContext) -> float | None:
        """Return the desired target weight in ``[-1, 1]``.

        * ``1.0``  -> put 100% of equity long the asset
        * ``0.0``  -> hold no position (all cash)
        * ``-1.0`` -> 100% short
        * ``None`` -> leave the target unchanged from the previous bar

        The engine rebalances toward this weight at the *next* bar's open.
        """
        raise NotImplementedError
