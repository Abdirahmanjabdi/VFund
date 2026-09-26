"""Liquidity-provision reversal — paid for standing where size cannot follow.

The finding
-----------
Short-horizon reversal in perpetual futures is two completely different things
depending on where you look:

* In **liquid** names it is bid-ask bounce. The recorded close alternates
  between bid and ask, manufacturing negative autocorrelation that looks like a
  strong signal and cannot be captured. Wait one day before entering and it is
  gone (IC 0.018 -> 0.003, t 3.08 -> 0.49).
* In **thin** names it is genuine mean reversion. It survives the same one-day
  skip at IC 0.040, t = 7.22, and decays smoothly over the following days
  rather than vanishing at once.

Measured on 749 Binance USD-M perpetuals, 2021-12 to 2026-08, with a
survivorship-free universe drawn from the Vision dump rather than from
currently-listed symbols. IC against the next day's return, by trailing
30-day median dollar volume:

======================  ==========  ==========  ==========
Liquidity quintile      skip 0      skip 1      skip 2
======================  ==========  ==========  ==========
q1  (~$5.7M ADV)        +0.066      **+0.040**  +0.026
q2  (~$9.6M ADV)        +0.069      **+0.042**  +0.034
q5  (~$172M ADV)        +0.018      +0.003      +0.002
======================  ==========  ==========  ==========

Why it should persist
---------------------
This is a capacity-constrained edge, and that is precisely why it survives. A
fund running $500M cannot express a view in a perp trading $5M a day: a 1%
position would be days of volume to enter and days more to exit. So the
mispricing is not arbitraged away — not because nobody has noticed it, but
because almost nobody who notices it can act on it at their size. The return is
compensation for supplying liquidity in venues that institutional capital is
structurally excluded from.

That also bounds it honestly: an edge that exists *because* it does not scale
will stop working for you at some size too.

The skip is not optional
------------------------
``skip=1`` is what separates the real signal from the artefact, so it is the
default and lowering it to 0 is a decision to trade microstructure noise. It
also makes the strategy humane: the signal is formed on one close and acted on
the next, so nothing depends on reacting instantly.
"""

from __future__ import annotations

import numpy as np

from vfund.strategy.cross_sectional import CrossSectionalStrategy, PanelContext


def _rank_cs(x: np.ndarray) -> np.ndarray:
    """Cross-sectional midrank percentile, centred on zero, NaN-preserving."""
    valid = np.isfinite(x)
    if valid.sum() < 2:
        return np.full(x.shape, np.nan)
    out = np.full(x.shape, np.nan)
    v = x[valid]
    n = len(v)
    less = (v[:, None] > v[None, :]).sum(axis=1)
    equal = (v[:, None] == v[None, :]).sum(axis=1)
    out[valid] = (less + equal / 2.0) / n - 0.5
    return out


def trailing_adv(
    closes: np.ndarray, volumes: np.ndarray, window: int = 30
) -> np.ndarray:
    """(T, N) trailing median dollar volume, strictly backward-looking.

    Row ``t`` uses bars ``[t-window, t-1]`` — it excludes bar ``t`` itself, so a
    name cannot be classified using volume it has not yet traded. The median
    rather than the mean, because a single listing-day volume spike would
    otherwise promote a thin name into the liquid bucket for a month.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    T, N = closes.shape
    dv = volumes * closes
    adv = np.full((T, N), np.nan)
    for t in range(window, T):
        w = dv[t - window : t]
        ok = np.isfinite(w) & (w > 0)
        if not ok.any():
            continue
        col = np.where(ok, w, np.nan)
        adv[t] = np.where(ok.any(axis=0), np.nanmedian(col, axis=0), np.nan)
    return adv


class IlliquidReversal(CrossSectionalStrategy):
    """Reversal, traded only in the thin tail of the cross-section.

    Args:
        adv: (T, N) trailing dollar volume from :func:`trailing_adv`.
        lookback: bars of past return to reverse. 5 was the strongest tested.
        skip: bars between forming the signal and acting on it. **Keep at 1** —
            at 0 the measured edge is largely bid-ask bounce and is not
            capturable.
        max_quantile: trade only names below this cross-sectional liquidity
            quantile. 0.4 keeps the two thinnest quintiles, which is where the
            skip-surviving signal lives.
        min_adv: hard dollar-volume floor. The quantile is relative, so in a
            thin cross-section it would otherwise select names too small to
            trade at any size at all.
        min_names: require at least this many eligible names before taking any
            position, so the book is never concentrated into a handful.
    """

    name = "illiquid_reversal"

    def __init__(
        self,
        adv: np.ndarray,
        *,
        lookback: int = 5,
        skip: int = 1,
        max_quantile: float = 0.4,
        min_adv: float = 1_000_000.0,
        min_names: int = 10,
    ):
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if skip < 0:
            raise ValueError("skip must be >= 0")
        if not 0.0 < max_quantile <= 1.0:
            raise ValueError("max_quantile must be in (0, 1]")
        self._adv = np.asarray(adv, dtype=float)
        self.lookback = lookback
        self.skip = skip
        self.max_quantile = max_quantile
        self.min_adv = min_adv
        self.min_names = min_names

    def eligible(self, i: int) -> np.ndarray:
        """Boolean mask of names thin enough to trade at bar ``i``."""
        a = self._adv[i]
        ok = np.isfinite(a) & (a >= self.min_adv)
        if ok.sum() < self.min_names:
            return np.zeros(a.shape, dtype=bool)
        cut = np.quantile(a[ok], self.max_quantile)
        return ok & (a <= cut)

    def scores(self, ctx: PanelContext) -> np.ndarray:
        i, lb, sk = ctx.i, self.lookback, self.skip
        N = len(ctx.symbols)
        # The signal is formed at bar i - skip, so the return window ends there.
        end = i - sk
        if end - lb < 0:
            return np.full(N, np.nan)

        closes = ctx._closes
        with np.errstate(divide="ignore", invalid="ignore"):
            past = closes[end] / closes[end - lb] - 1.0
        past = np.where(np.isfinite(past), past, np.nan)

        mask = self.eligible(i)
        if not mask.any():
            return np.full(N, np.nan)

        out = np.full(N, np.nan)
        out[mask] = past[mask]
        # Reversal: long what fell, short what rose.
        return _rank_cs(-out)
