"""Crowding acceleration — the one factor that survived validation.

What this is
------------
The second derivative of open interest, ranked cross-sectionally, traded
**short**: sell the assets where leveraged position build-up is accelerating,
buy the ones where it is decelerating.

The sign is the point, and it was not the hypothesis
-----------------------------------------------------
The intuitive story for accelerating open interest is bullish — fresh
conviction arriving, the trend has fuel. That is what an earlier version of this
work assumed. It is wrong, and the data says so consistently.

Accelerating position growth marks *late-stage* crowding. When new leveraged
positions arrive at an increasing rate, the marginal entrant is progressively
less informed, and the stock of positions becomes increasingly fragile: it is
levered, it is one-sided, and it must be unwound through the same narrow exit.
The acceleration is the tell that the build-up is closer to exhaustion than to
its beginning.

Evidence
--------
Binance USD-M perpetuals, 2021-12 to 2026-08, ~30 names in the daily
cross-section (1,704 bars):

* Raw IC at a 1-day horizon: **-0.029**, Newey-West t = **-5.06**
* Orthogonalised against 1/5/14-day past returns: IC -0.018, t = **-3.58**
* Same sign in all three sub-periods, at every horizon tested
* Survives Bonferroni correction for the full 52-combination search (|t| > 3.30)
* Under a paranoid one-bar data lag: t = **-3.55** — essentially unchanged

It is *not* short-term reversal. At the 5-day horizon reversal is dead
(t = 0.31 once crowding acceleration is controlled for) while crowding
acceleration holds. They are complementary signals, not the same one.

What this is not
----------------
An IC is not a P&L. This factor's horizon is short, which means turnover, and
turnover is where cross-sectional edges die. Nothing here is a profitability
claim until it has cleared a cost-charged backtest.
"""

from __future__ import annotations

import numpy as np

from vfund.strategy.cross_sectional import CrossSectionalStrategy, PanelContext


def _rank_cs(x: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank, NaN-preserving, centred on zero."""
    valid = np.isfinite(x)
    if valid.sum() < 2:
        return np.full(x.shape, np.nan)
    out = np.full(x.shape, np.nan)
    v = x[valid]
    n = len(v)
    less = (v[:, None] > v[None, :]).sum(axis=1)
    equal = (v[:, None] == v[None, :]).sum(axis=1)
    # Midrank: half of each tie group, self included. Using (equal + 1) / 2
    # here would count the self-comparison twice and leave a +0.5/n bias, so
    # the "centred" scores would not actually be centred.
    out[valid] = (less + equal / 2.0) / n - 0.5
    return out


class CrowdingAcceleration(CrossSectionalStrategy):
    """Short accelerating position build-up, long decelerating.

    Args:
        oi: (T, N) open-interest notional aligned to the price timeline.
        lookback: bars per growth leg. The score compares growth over the most
            recent ``lookback`` bars against growth over the ``lookback`` bars
            before that.
        winsorize: clip the raw acceleration at this many median-absolute
            deviations before ranking. A newly listed perp can print an
            arbitrarily large OI growth ratio from a near-zero base; without
            this, one such name would occupy an extreme rank every day.
        smooth: average the raw acceleration over this many trailing bars
            before ranking. This is a turnover control, and it is the single
            most important parameter here. The un-smoothed signal is noisy
            enough that the book churns every rebalance, and at realistic costs
            that churn removes more than half the gross return. Averaging is a
            low-pass filter on the estimate, not a change to the thesis.
    """

    name = "crowding_acceleration"

    def __init__(
        self,
        oi: np.ndarray,
        *,
        lookback: int = 7,
        winsorize: float = 5.0,
        smooth: int = 1,
    ):
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        if winsorize <= 0:
            raise ValueError("winsorize must be > 0")
        if smooth < 1:
            raise ValueError("smooth must be >= 1")
        self._oi = np.asarray(oi, dtype=float)
        self.lookback = lookback
        self.winsorize = winsorize
        self.smooth = smooth

    def raw(self, i: int) -> np.ndarray:
        """Un-ranked acceleration at bar ``i`` (positive = accelerating)."""
        lb = self.lookback
        N = self._oi.shape[1]
        if i < 2 * lb:
            return np.full(N, np.nan)
        now, mid, far = self._oi[i], self._oi[i - lb], self._oi[i - 2 * lb]
        with np.errstate(divide="ignore", invalid="ignore"):
            g_recent = np.where(mid > 0, now / mid - 1.0, np.nan)
            g_past = np.where(far > 0, mid / far - 1.0, np.nan)
        a = g_recent - g_past
        return np.where(np.isfinite(a), a, np.nan)

    def scores(self, ctx: PanelContext) -> np.ndarray:
        if self.smooth == 1:
            a = self.raw(ctx.i)
        else:
            lo = max(0, ctx.i - self.smooth + 1)
            stack = np.vstack([self.raw(j) for j in range(lo, ctx.i + 1)])
            # A column that is NaN across every smoothing bar has no estimate;
            # averaging it would warn and yield NaN anyway, so mask it out first.
            n_obs = np.isfinite(stack).sum(axis=0)
            a = np.full(stack.shape[1], np.nan)
            if (n_obs > 0).any():
                cols = n_obs > 0
                a[cols] = np.nanmean(stack[:, cols], axis=0)
        ok = np.isfinite(a)
        if ok.sum() >= 3:
            med = np.median(a[ok])
            mad = np.median(np.abs(a[ok] - med))
            if mad > 0:
                lim = self.winsorize * 1.4826 * mad
                a = np.clip(a, med - lim, med + lim)
        # Negative: short the accelerating names, long the decelerating ones.
        return -_rank_cs(a)
