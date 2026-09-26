"""Flow decomposition — separating forced trading from voluntary trading.

The idea
--------
Volume tells you that trading happened. It does not tell you whether that
trading *opened* positions or *closed* them, and those are different events with
different consequences. Open interest supplies the missing bit: it rises when
positions are opened and falls when they are closed. Combined with the sign of
the price move, it partitions every asset-day into four states:

======================  =========================================
Price up, OI up         new longs opening — voluntary, informed
Price up, OI down       **shorts covering — forced**
Price down, OI up       new shorts opening — voluntary, informed
Price down, OI down     **longs liquidating — forced**
======================  =========================================

Why the distinction should pay
------------------------------
Forced flow is not an opinion. A trader meeting a margin call sells at whatever
price clears, and keeps selling until the requirement is met — then stops. This
has two consequences that voluntary flow does not have: it pushes price away
from fair value while it runs, and it is *self-terminating*, because the
positions being closed are a finite stock. Voluntary flow has neither property;
someone opening a new position has chosen the price and can keep going.

So the two states call for opposite trades. Fade a move driven by position
closing; follow a move confirmed by position opening.

The construction
----------------
All of that collapses into one product:

    score = normalised return x normalised change in open interest

Price up with OI up is positive (follow). Price up with OI down is negative
(fade). Price down with OI down is positive (fade the fall). Price down with OI
up is negative (follow the fall). The four quadrants fall out of the sign
algebra without needing to be enumerated, and the magnitude scales with the
conviction of both legs.

A human can read the same thing off two lines on a chart, which is the point:
the signal is not an opaque fit, it is a statement about who is trading and why.

Nothing here is validated. These are hypotheses with a mechanism, which is the
only kind worth testing — see :mod:`vfund.research.ic` for the gate they have to
pass.
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


def _ts_z(mat: np.ndarray, i: int, window: int, min_obs: int = 10) -> np.ndarray:
    """Per-symbol trailing z-score at bar ``i``, using only rows up to ``i``.

    Normalising each symbol against its own history is what makes the product
    below comparable across assets: a 5% move means something different in a
    major than in a freshly listed altcoin, and the raw product would otherwise
    rank purely by volatility.
    """
    lo = max(0, i - window + 1)
    win = mat[lo : i + 1]
    n = np.isfinite(win).sum(axis=0)
    usable = n >= max(min_obs, 2)
    z = np.full(mat.shape[1], np.nan)
    if not usable.any():
        return z
    w = win[:, usable]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(w, axis=0)
        sd = np.nanstd(w, axis=0, ddof=1)
        zz = (mat[i][usable] - mu) / sd
    # A near-constant series has a standard deviation of pure floating-point
    # noise (~1e-18), and dividing by it manufactures enormous z-scores out of
    # rounding error. Require the spread to be meaningful relative to the
    # level before trusting it; otherwise the symbol simply has no view.
    floor = np.maximum(1e-12, 1e-8 * np.abs(mu))
    good = np.isfinite(sd) & (sd > floor) & np.isfinite(zz)
    z[usable] = np.where(good, zz, np.nan)
    return z


def _clip(x: np.ndarray, lim: float) -> np.ndarray:
    return np.clip(x, -lim, lim)


class FlowConfirmation(CrossSectionalStrategy):
    """The four-quadrant product: follow opening flow, fade closing flow.

    Args:
        oi: (T, N) open-interest notional aligned to the price timeline.
        lookback: bars over which the return and the OI change are measured.
        window: trailing window for the per-symbol z-scores.
        clip: z-scores are clipped to +/- this before multiplying. A product of
            two heavy-tailed terms is *very* heavy-tailed, and without clipping
            one asset-day dominates the whole cross-section.
        smooth: average the raw score over this many trailing bars. Turnover
            control — the same lesson crowding acceleration taught.
    """

    name = "flow_confirmation"

    def __init__(
        self,
        oi: np.ndarray,
        *,
        lookback: int = 1,
        window: int = 60,
        clip: float = 3.0,
        smooth: int = 1,
    ):
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if window < 10:
            raise ValueError("window must be >= 10")
        if smooth < 1:
            raise ValueError("smooth must be >= 1")
        self._oi = np.asarray(oi, dtype=float)
        self.lookback = lookback
        self.window = window
        self.clip = clip
        self.smooth = smooth
        self._ret_cache: np.ndarray | None = None
        self._doi_cache: np.ndarray | None = None

    def _prepare(self, closes: np.ndarray) -> None:
        """Build the return and OI-change panels once per backtest."""
        if self._ret_cache is not None:
            return
        lb = self.lookback
        T = closes.shape[0]
        ret = np.full(closes.shape, np.nan)
        doi = np.full(self._oi.shape, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            if lb < T:
                ret[lb:] = closes[lb:] / closes[:-lb] - 1.0
            if lb < self._oi.shape[0]:
                prev = self._oi[:-lb]
                doi[lb:] = np.where(prev > 0, self._oi[lb:] / prev - 1.0, np.nan)
        self._ret_cache = np.where(np.isfinite(ret), ret, np.nan)
        self._doi_cache = np.where(np.isfinite(doi), doi, np.nan)

    def raw(self, i: int) -> np.ndarray:
        # Signs come from the RAW changes, magnitudes from the normalised ones.
        # Open interest trends upward, so its trailing mean is positive and a
        # negative z-score means "grew less than usual" rather than "positions
        # were closed" — using the z sign would silently corrupt the quadrants.
        r_raw, d_raw = self._ret_cache[i], self._doi_cache[i]
        r_mag = np.abs(_clip(_ts_z(self._ret_cache, i, self.window), self.clip))
        d_mag = np.abs(_clip(_ts_z(self._doi_cache, i, self.window), self.clip))
        return np.sign(r_raw) * np.sign(d_raw) * r_mag * d_mag

    def scores(self, ctx: PanelContext) -> np.ndarray:
        self._prepare(ctx._closes)
        if self.smooth == 1:
            a = self.raw(ctx.i)
        else:
            lo = max(0, ctx.i - self.smooth + 1)
            stack = np.vstack([self.raw(j) for j in range(lo, ctx.i + 1)])
            n_obs = np.isfinite(stack).sum(axis=0)
            a = np.full(stack.shape[1], np.nan)
            if (n_obs > 0).any():
                cols = n_obs > 0
                a[cols] = np.nanmean(stack[:, cols], axis=0)
        return _rank_cs(a)


class ForcedFlowReversal(CrossSectionalStrategy):
    """Fade price moves that happen on *closing* open interest only.

    The pure form of the thesis: ignore the two voluntary quadrants entirely and
    trade only where positions are being closed. Long assets that fell while OI
    fell (liquidation), short assets that rose while OI fell (short covering).
    Assets whose OI rose score exactly zero — no view.
    """

    name = "forced_flow_reversal"

    def __init__(
        self,
        oi: np.ndarray,
        *,
        lookback: int = 1,
        window: int = 60,
        clip: float = 3.0,
        smooth: int = 1,
    ):
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if smooth < 1:
            raise ValueError("smooth must be >= 1")
        self._inner = FlowConfirmation(
            oi, lookback=lookback, window=window, clip=clip, smooth=1
        )
        self.smooth = smooth

    def raw(self, i: int) -> np.ndarray:
        w, c = self._inner.window, self._inner.clip
        r_raw, d_raw = self._inner._ret_cache[i], self._inner._doi_cache[i]
        r_mag = np.abs(_clip(_ts_z(self._inner._ret_cache, i, w), c))
        d_mag = np.abs(_clip(_ts_z(self._inner._doi_cache, i, w), c))
        # Gate on the RAW change: positions must actually have been closed.
        # Gating on the z-score would fire whenever OI merely grew more slowly
        # than its trailing average, which is not the same event at all.
        closing = np.where(d_raw < 0, d_mag, 0.0)
        return -np.sign(r_raw) * r_mag * closing

    def scores(self, ctx: PanelContext) -> np.ndarray:
        self._inner._prepare(ctx._closes)
        if self.smooth == 1:
            a = self.raw(ctx.i)
        else:
            lo = max(0, ctx.i - self.smooth + 1)
            stack = np.vstack([self.raw(j) for j in range(lo, ctx.i + 1)])
            n_obs = np.isfinite(stack).sum(axis=0)
            a = np.full(stack.shape[1], np.nan)
            if (n_obs > 0).any():
                cols = n_obs > 0
                a[cols] = np.nanmean(stack[:, cols], axis=0)
        return _rank_cs(a)


class PositionChurn(CrossSectionalStrategy):
    """Rank by instability of intraday positioning.

    ``oi_churn`` is the dispersion of within-day log changes in open interest.
    A day where OI grinds steadily to its close and a day where it thrashes can
    end at the same level while describing very different books: the second is
    positions being opened and closed against each other, which is the signature
    of leveraged participants being shaken out rather than accumulating.

    Sign is deliberately configurable — this is an exploratory factor and the
    direction is an empirical question, not something to assert up front.
    """

    name = "position_churn"

    def __init__(self, churn: np.ndarray, *, window: int = 60, sign: float = -1.0):
        if sign not in (1.0, -1.0):
            raise ValueError("sign must be +1.0 or -1.0")
        self._c = np.asarray(churn, dtype=float)
        self.window = window
        self.sign = sign

    def scores(self, ctx: PanelContext) -> np.ndarray:
        return _rank_cs(self.sign * _ts_z(self._c, ctx.i, self.window))
