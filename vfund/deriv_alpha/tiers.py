"""Positioning-tier divergence — the informed/uninformed spread as an alpha.

The construct
-------------
Most positioning research asks *how* crowded an asset is. This module asks a
different question: **who** is crowded. Markets that publish positioning almost
always publish it split by participant tier, and the tiers disagree:

=============  ===========================  ==============================
Market         Informed tier                Uninformed tier
=============  ===========================  ==============================
Crypto perps   top accounts by margin,      all accounts, equally weighted
               position-weighted L/S        (numerically retail-dominated)
Futures        CFTC Commercial hedgers      Non-reportable small traders
Equities       13F institutional flow       retail / odd-lot flow
FX             IMM large speculators        retail broker client books
=============  ===========================  ==============================

The signal is the signed disagreement between them. When large, well-capitalised
accounts lean long while the account-count crowd leans short, that is a
different state from both leaning long — even though aggregate positioning may
look identical in the two cases.

Why it should persist
---------------------
This is not an arbitrage that closes when discovered. The uninformed side is
structurally replenished: new retail entrants arrive continuously, cannot
coordinate, and are subject to well-documented behavioural biases (chasing
recent returns, asymmetric loss realisation). The informed side cannot simply
absorb the whole flow because doing so means taking unbounded inventory in a
levered market. The disagreement is a standing feature of market structure, not
a mispricing with a closing date.

Design notes that matter
------------------------
Every factor here is **symbol-demeaned before cross-sectional ranking**. A raw
long/short ratio is not comparable across assets: some assets are structurally
retail-heavy and would otherwise dominate one end of the ranking permanently,
turning the factor into a static sector bet. Taking each symbol's trailing
z-score first strips that persistent bias out and leaves the deviation, which is
what the thesis is actually about.

Ratios are handled in logs throughout, so that "twice as many longs" and "twice
as many shorts" are equal and opposite rather than 2.0 and 0.5.
"""

from __future__ import annotations

import numpy as np

from vfund.strategy.cross_sectional import CrossSectionalStrategy, PanelContext


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def _rank_cs(x: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank, NaN-preserving, centred on 0."""
    valid = np.isfinite(x)
    if valid.sum() < 2:
        return np.full(x.shape, np.nan)
    out = np.full(x.shape, np.nan)
    v = x[valid]
    n = len(v)
    less = (v[:, None] > v[None, :]).sum(axis=1)
    equal = (v[:, None] == v[None, :]).sum(axis=1)
    # Midrank: half of each tie group, self included. (equal + 1) / 2 would
    # double-count the self-comparison and leave a +0.5/n bias.
    out[valid] = (less + equal / 2.0) / n - 0.5
    return out


def _safe_log(x: np.ndarray) -> np.ndarray:
    """Log of a positive ratio; non-positive and non-finite entries become NaN."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x) & (x > 0)
    out[ok] = np.log(x[ok])
    return out


def _ts_z(mat: np.ndarray, i: int, window: int, min_obs: int | None = None) -> np.ndarray:
    """Per-symbol trailing z-score of ``mat`` at bar ``i``.

    Uses only rows ``[i-window+1, i]`` — strictly backward-looking. Returns NaN
    for symbols with too few observations or zero trailing variance.
    """
    lo = max(0, i - window + 1)
    win = mat[lo : i + 1]
    if win.shape[0] < 2:
        return np.full(mat.shape[1], np.nan)
    min_obs = min_obs if min_obs is not None else max(5, window // 3)

    n = np.isfinite(win).sum(axis=0)
    usable = n >= max(min_obs, 2)
    z = np.full(mat.shape[1], np.nan)
    if not usable.any():
        return z
    # Restrict to columns with data before averaging: nanmean/nanstd warn and
    # return NaN on an all-NaN slice, which floods the log during a backtest.
    w = win[:, usable]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(w, axis=0)
        sd = np.nanstd(w, axis=0, ddof=1)
        zz = (mat[i][usable] - mu) / sd
    # A near-constant series has a standard deviation of floating-point noise;
    # dividing by it manufactures huge z-scores out of rounding error.
    floor = np.maximum(1e-12, 1e-8 * np.abs(mu))
    z[usable] = np.where(np.isfinite(sd) & (sd > floor) & np.isfinite(zz), zz, np.nan)
    return z


# --------------------------------------------------------------------------- #
# Factors
# --------------------------------------------------------------------------- #
class TierDivergence(CrossSectionalStrategy):
    """Long where the informed tier leans more long than the crowd.

    ``D = log(informed L/S) - log(crowd L/S)``, symbol-demeaned over a trailing
    window, then ranked cross-sectionally. Positive D means large accounts are
    positioned more bullishly than the account-count crowd.
    """

    name = "tier_divergence"

    def __init__(self, informed: np.ndarray, crowd: np.ndarray, *, window: int = 60):
        if window < 10:
            raise ValueError("window must be >= 10")
        self._d = _safe_log(informed) - _safe_log(crowd)
        self.window = window

    def scores(self, ctx: PanelContext) -> np.ndarray:
        return _rank_cs(_ts_z(self._d, ctx.i, self.window))


class TierDivergenceDelta(CrossSectionalStrategy):
    """Trade the *change* in tier disagreement, not its level.

    A widening spread means the informed tier is actively moving away from the
    crowd — a rotation in progress rather than a standing difference of opinion.
    """

    name = "tier_divergence_delta"

    def __init__(
        self,
        informed: np.ndarray,
        crowd: np.ndarray,
        *,
        lookback: int = 5,
        window: int = 60,
    ):
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        self._d = _safe_log(informed) - _safe_log(crowd)
        self.lookback = lookback
        self.window = window

    def scores(self, ctx: PanelContext) -> np.ndarray:
        i, lb = ctx.i, self.lookback
        if i < lb:
            return np.full(self._d.shape[1], np.nan)
        delta = self._d[i] - self._d[i - lb]
        # Demean the *change* series so structurally noisier symbols do not
        # dominate purely through higher variance.
        dmat = np.full_like(self._d, np.nan)
        dmat[lb:] = self._d[lb:] - self._d[:-lb]
        z = _ts_z(dmat, i, self.window)
        return _rank_cs(np.where(np.isfinite(delta), z, np.nan))


class CrowdExtremity(CrossSectionalStrategy):
    """Fade the crowd when it is one-sided.

    Score = minus the symbol-demeaned crowd long/short tilt. When the
    account-count crowd is unusually long an asset relative to its own history,
    lean against it.
    """

    name = "crowd_extremity"

    def __init__(self, crowd: np.ndarray, *, window: int = 60):
        self._c = _safe_log(crowd)
        self.window = window

    def scores(self, ctx: PanelContext) -> np.ndarray:
        return _rank_cs(-_ts_z(self._c, ctx.i, self.window))


class SmartMoneyFlow(CrossSectionalStrategy):
    """Follow the informed tier's own repositioning.

    Score = symbol-demeaned change in informed-tier long/short over ``lookback``
    bars. Where TierDivergence asks whether big money disagrees with the crowd,
    this asks only which way big money is moving.
    """

    name = "smart_money_flow"

    def __init__(self, informed: np.ndarray, *, lookback: int = 5, window: int = 60):
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        self._s = _safe_log(informed)
        self.lookback = lookback
        self.window = window

    def scores(self, ctx: PanelContext) -> np.ndarray:
        i, lb = ctx.i, self.lookback
        if i < lb:
            return np.full(self._s.shape[1], np.nan)
        dmat = np.full_like(self._s, np.nan)
        dmat[lb:] = self._s[lb:] - self._s[:-lb]
        return _rank_cs(_ts_z(dmat, i, self.window))


class TakerImbalance(CrossSectionalStrategy):
    """Fade aggressive taker flow.

    ``sum_taker_long_short_vol_ratio`` measures who is crossing the spread.
    Aggressive buying pays the spread and is disproportionately impatient,
    uninformed flow; sustained one-sided aggression tends to exhaust.
    """

    name = "taker_imbalance"

    def __init__(self, taker: np.ndarray, *, window: int = 60):
        self._t = _safe_log(taker)
        self.window = window

    def scores(self, ctx: PanelContext) -> np.ndarray:
        return _rank_cs(-_ts_z(self._t, ctx.i, self.window))


class OIWeightedDivergence(CrossSectionalStrategy):
    """Tier divergence, amplified where open interest is unusually elevated.

    The same disagreement carries more information when there is more capital
    behind it. Scales the divergence z-score by a squashed OI z-score so that a
    stale, illiquid disagreement does not rank alongside a well-funded one.
    """

    name = "oi_weighted_divergence"

    def __init__(
        self,
        informed: np.ndarray,
        crowd: np.ndarray,
        oi_value: np.ndarray,
        *,
        window: int = 60,
    ):
        self._d = _safe_log(informed) - _safe_log(crowd)
        self._oi = _safe_log(oi_value)
        self.window = window

    def scores(self, ctx: PanelContext) -> np.ndarray:
        i = ctx.i
        d = _ts_z(self._d, i, self.window)
        o = _ts_z(self._oi, i, self.window)
        # tanh keeps an OI outlier from dominating the whole cross-section.
        amp = 1.0 + np.tanh(np.where(np.isfinite(o), o, 0.0))
        return _rank_cs(d * amp)


ALL_FACTORS = (
    TierDivergence,
    TierDivergenceDelta,
    CrowdExtremity,
    SmartMoneyFlow,
    TakerImbalance,
    OIWeightedDivergence,
)
