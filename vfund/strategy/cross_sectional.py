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

    __slots__ = ("i", "_closes", "symbols", "_funding", "_volumes")

    def __init__(
        self,
        i: int,
        closes: np.ndarray,
        symbols: list[str],
        funding: np.ndarray | None = None,
        volumes: np.ndarray | None = None,
    ):
        self.i = i
        self._closes = closes
        self.symbols = symbols
        self._funding = funding
        self._volumes = volumes

    @property
    def closes(self) -> np.ndarray:
        return self._closes[: self.i + 1]

    @property
    def funding(self) -> np.ndarray | None:
        """Prevailing (forward-filled) funding rates up to now, or None."""
        if self._funding is None:
            return None
        return self._funding[: self.i + 1]

    @property
    def volumes(self) -> np.ndarray | None:
        """Base-asset volumes up to now, or None."""
        if self._volumes is None:
            return None
        return self._volumes[: self.i + 1]

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


class FundingCarry(CrossSectionalStrategy):
    """Harvest the perpetual funding spread.

    Score = minus the prevailing funding rate (optionally smoothed over the last
    ``smooth`` bars). High positive funding -> low score -> short it (you collect
    the funding the crowded longs pay); negative funding -> long it. The book is
    dollar-neutral, so it's the *funding differential* you earn, largely
    independent of where the market goes.

    Needs a funding-aware backtest (``CrossSectionalBacktester(..., funding=...)``)
    so the funding cashflow actually shows up in P&L.
    """

    def __init__(self, smooth: int = 1):
        if smooth < 1:
            raise ValueError("smooth must be >= 1")
        self.smooth = smooth

    def scores(self, ctx: PanelContext) -> np.ndarray:
        funding = ctx.funding
        if funding is None:
            raise ValueError(
                "FundingCarry needs a funding-aware backtest — pass funding=... "
                "to CrossSectionalBacktester"
            )
        window = funding[-self.smooth :]
        return -window.mean(axis=0)  # short high funding, long low/negative


class CrossSectionalLowVol(CrossSectionalStrategy):
    """Betting-against-beta: long the calm coins, short the wild ones.

    Across many markets, low-volatility assets have delivered better
    *risk-adjusted* returns than high-vol ones (investors overpay for lottery-like
    upside). Score = minus trailing volatility.
    """

    def __init__(self, lookback: int = 168):
        if lookback < 5:
            raise ValueError("lookback too small to estimate volatility")
        self.lookback = lookback

    def scores(self, ctx: PanelContext) -> np.ndarray:
        closes = ctx.closes
        if closes.shape[0] <= self.lookback + 1:
            return np.full(closes.shape[1], np.nan)
        window = closes[-self.lookback - 1 :]
        rets = window[1:] / window[:-1] - 1.0
        return -rets.std(axis=0)  # long low vol, short high vol


class CrossSectionalValue(CrossSectionalStrategy):
    """Medium-horizon mean reversion — a crude 'value' proxy.

    Price stretched far above its own long moving average is 'rich'; far below is
    'cheap'. Long the cheap, short the rich, and bet on convergence. Score =
    minus the distance from the moving average.
    """

    def __init__(self, lookback: int = 720):  # ~30 days of hourly bars
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        self.lookback = lookback

    def scores(self, ctx: PanelContext) -> np.ndarray:
        closes = ctx.closes
        if closes.shape[0] <= self.lookback:
            return np.full(closes.shape[1], np.nan)
        ma = closes[-self.lookback :].mean(axis=0)
        distance = closes[-1] / ma - 1.0
        return -distance  # long below-MA (cheap), short above-MA (rich)


class CrossSectionalSize(CrossSectionalStrategy):
    """The size effect: long small coins, short large ones.

    In Liu, Tsyvinski & Wu (J. Finance 2022), size is one of just three factors
    that price the crypto cross-section — small coins earn higher returns. We
    lack circulating supply, so we proxy 'size' by trailing dollar volume
    (price x volume), a liquidity measure that tracks it closely. Score = minus
    log dollar volume, so the smallest/least-liquid names are longed.

    This edge lives in names too small for large funds to touch — precisely the
    capacity-constrained niche a nimble book can exploit.
    """

    def __init__(self, lookback: int = 30):
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        self.lookback = lookback

    def scores(self, ctx: PanelContext) -> np.ndarray:
        vols = ctx.volumes
        closes = ctx.closes
        if vols is None:
            raise ValueError("CrossSectionalSize needs volume data in the panel")
        if closes.shape[0] < self.lookback:
            return np.full(closes.shape[1], np.nan)
        dollar_vol = (closes[-self.lookback :] * vols[-self.lookback :]).mean(axis=0)
        return -np.log(np.maximum(dollar_vol, 1e-9))  # long small, short large


class TimeSeriesTrend(CrossSectionalStrategy):
    """Per-asset trend following (time-series momentum).

    For each coin independently: long if its trailing return is positive, short
    if negative. Unlike the cross-sectional strategies this is *directional* —
    long everything in a broad uptrend — so run it with ``neutralize=False`` and
    judge it against buy-and-hold, since it carries market exposure. Trend is one
    of the most robust anomalies across asset classes and time.
    """

    def __init__(self, lookback: int = 168):
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        self.lookback = lookback

    def scores(self, ctx: PanelContext) -> np.ndarray:
        closes = ctx.closes
        if closes.shape[0] <= self.lookback:
            return np.full(closes.shape[1], np.nan)
        trailing = closes[-1] / closes[-1 - self.lookback] - 1.0
        return np.sign(trailing)  # +1 uptrend (long), -1 downtrend (short)


class TimeSeriesTrendEnsemble(CrossSectionalStrategy):
    """Multi-horizon trend — the managed-futures answer to lookback fragility.

    A single trend lookback is a bet on one speed; the best one in-sample is
    rarely the best out-of-sample. Real CTAs average signals across many
    horizons instead. Score = the mean of the per-horizon trend signs, so a coin
    trending up on every timescale gets +1, one with mixed signals gets a small
    number, and the book leans toward the clearest trends. Directional — run with
    ``neutralize=False``.

    Reference: Hurst, Ooi & Pedersen, "A Century of Evidence on Trend-Following".
    """

    def __init__(self, lookbacks=(20, 30, 50, 100)):
        self.lookbacks = tuple(int(x) for x in lookbacks)
        if min(self.lookbacks) < 1:
            raise ValueError("lookbacks must be >= 1")

    def scores(self, ctx: PanelContext) -> np.ndarray:
        closes = ctx.closes
        n = closes.shape[0]
        signals = [
            np.sign(closes[-1] / closes[-1 - lb] - 1.0)
            for lb in self.lookbacks
            if n > lb
        ]
        if not signals:
            return np.full(closes.shape[1], np.nan)
        return np.mean(signals, axis=0)
