"""Measuring and harvesting the variance risk premium.

The measured edge (gold, GVZ vs GLD, 2008-06 to 2026-08, 4,568 daily
observations and 217 non-overlapping months):

===============================  ==========================================
mean implied volatility          18.94
mean subsequent realised          16.29
**mean premium**                 **+2.65 vol points (+14.0% of implied)**
days implied exceeded realised    79.0%
Newey-West t (lag 21)             8.23
===============================  ==========================================

By sub-period the premium is +3.82 (2008-13), +2.34 (2014-19), +1.94
(2020-26): shrinking, but positive and significant throughout — unlike the
crypto basis trade, which went outright negative once institutional capital
arrived, or the month-end FX effect, which regulation removed.

The tail is the whole problem
-----------------------------
Selling volatility wins about four months in five and occasionally loses
enormously. Over 217 months the worst single month cost 55 vol points against a
mean gain of 1.77 — a ratio of 31 to 1 — and the worst ten months put 55% of
cumulative profit at risk. A naked short volatility position is therefore not a
strategy, it is a fuse.

Capping the loss is what makes it one. Truncating the monthly loss at 5 vol
points raises the t-statistic from 3.45 to 6.91 while retaining 81% of the mean,
because in this distribution the tail is almost the entire source of variance.
That cap is not a modelling convenience: it is what buying the protective wing
of a spread actually does, and it is the structural advantage options hold over
a short futures position, which cannot be capped at all.

Timing does not help
--------------------
Conditioning entry on elevated implied volatility, on the implied-to-trailing-
realised spread, or on volatility percentiles all *reduced* performance
(selling only above the 70th percentile of implied vol: t = 0.94, against 3.45
for selling unconditionally). The honest conclusion is to sell every month and
add no discretion.

Two honest caveats
------------------
Realised variance sampled at daily frequency slightly understates true
integrated variance, which biases any VRP estimate of this construction
modestly upward.

And these figures come from a volatility *index*, not from traded option
prices. They establish that the premium exists and how it behaves; they are not
a backtest of an executable structure, and real bid-ask on a multi-leg position
will take a further bite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from vfund.research.ic import newey_west_t

#: Trading days used to proxy the 30-calendar-day horizon these indices quote.
MONTH_BARS = 21


def realized_vol(
    log_returns: np.ndarray,
    horizon: int = MONTH_BARS,
    *,
    forward: bool = True,
    min_frac: float = 0.5,
) -> np.ndarray:
    """Annualised realised volatility in vol points (percent).

    Args:
        log_returns: 1-D log returns of the underlying.
        horizon: window in bars.
        forward: when True, row ``t`` looks at bars ``t+1 .. t+horizon`` — the
            volatility a seller at ``t`` is exposed to. When False it looks
            backward over ``t-horizon .. t-1``, which is the only version
            knowable at ``t``.
        min_frac: fraction of the window that must be finite to report a value.
    """
    if horizon < 2:
        raise ValueError("horizon must be >= 2")
    r = np.asarray(log_returns, dtype=float)
    n = len(r)
    out = np.full(n, np.nan)
    need = max(int(horizon * min_frac), 2)
    for t in range(n):
        if forward:
            if t + 1 + horizon > n:
                continue
            w = r[t + 1: t + 1 + horizon]
        else:
            if t - horizon < 0:
                continue
            w = r[t - horizon: t]
        w = w[np.isfinite(w)]
        if len(w) >= need:
            out[t] = w.std(ddof=1) * math.sqrt(252) * 100.0
    return out


def variance_swap_pnl(implied: np.ndarray, realized_fwd: np.ndarray) -> np.ndarray:
    """P&L of selling variance, expressed in vol points.

    ``(IV^2 - RV^2) / (2 * IV)`` converts the variance-swap payoff into
    vega-equivalent vol points, so a result of +1.5 means the position earned
    one and a half volatility points on its vega notional. Quoting it this way
    keeps the number comparable across volatility levels, where a raw variance
    difference would not be.
    """
    iv = np.asarray(implied, dtype=float)
    rv = np.asarray(realized_fwd, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (iv**2 - rv**2) / (2.0 * iv)
    return np.where(np.isfinite(out) & (iv > 0), out, np.nan)


def capped_pnl(
    pnl: np.ndarray,
    *,
    max_loss: float = 5.0,
    wing_cost: float = 0.35,
) -> np.ndarray:
    """Convert naked P&L into the defined-risk equivalent.

    Buying a protective wing truncates the loss and gives up part of the
    premium when the trade wins. ``wing_cost`` is the fraction of the gain
    surrendered; 0.35 is a deliberately conservative placeholder and the real
    figure depends on the strikes actually traded.

    Args:
        pnl: naked per-period P&L in vol points.
        max_loss: the floor, in vol points. Must be positive.
        wing_cost: fraction of winning-period P&L given up, in [0, 1).
    """
    if max_loss <= 0:
        raise ValueError("max_loss must be > 0")
    if not 0.0 <= wing_cost < 1.0:
        raise ValueError("wing_cost must be in [0, 1)")
    p = np.asarray(pnl, dtype=float)
    capped = np.maximum(p, -max_loss)
    return np.where(capped > 0, capped * (1.0 - wing_cost), capped)


@dataclass(frozen=True)
class VRPStats:
    """Summary of a variance risk premium measurement."""

    n: int
    mean_implied: float
    mean_realized: float
    premium: float           # mean(implied - realised), vol points
    pct_positive: float
    t_stat: float
    worst: float
    best: float

    @property
    def premium_pct_of_implied(self) -> float:
        return self.premium / self.mean_implied * 100.0 if self.mean_implied else float("nan")

    def summary(self) -> str:  # pragma: no cover - formatting only
        return (
            f"  observations          {self.n}\n"
            f"  mean implied          {self.mean_implied:.2f}\n"
            f"  mean realised         {self.mean_realized:.2f}\n"
            f"  premium               {self.premium:+.2f} vol pts "
            f"({self.premium_pct_of_implied:+.1f}% of implied)\n"
            f"  implied > realised    {self.pct_positive * 100:.1f}% of the time\n"
            f"  t-stat (Newey-West)   {self.t_stat:.2f}\n"
            f"  worst / best          {self.worst:+.2f} / {self.best:+.2f}"
        )


def vrp_stats(
    implied: np.ndarray,
    realized_fwd: np.ndarray,
    *,
    nw_lag: int = MONTH_BARS,
) -> VRPStats:
    """Measure the premium between implied and subsequent realised volatility.

    ``nw_lag`` defaults to the horizon because overlapping forward windows are
    mechanically autocorrelated; a naive t-statistic on daily observations of a
    monthly forward quantity is inflated by roughly the square root of the
    horizon.
    """
    iv = np.asarray(implied, dtype=float)
    rv = np.asarray(realized_fwd, dtype=float)
    ok = np.isfinite(iv) & np.isfinite(rv)
    if ok.sum() < 30:
        nan = float("nan")
        return VRPStats(int(ok.sum()), nan, nan, nan, nan, nan, nan, nan)
    d = iv[ok] - rv[ok]
    return VRPStats(
        n=int(ok.sum()),
        mean_implied=float(iv[ok].mean()),
        mean_realized=float(rv[ok].mean()),
        premium=float(d.mean()),
        pct_positive=float((d > 0).mean()),
        t_stat=newey_west_t(d, lag=nw_lag),
        worst=float(d.min()),
        best=float(d.max()),
    )


def non_overlapping(values: np.ndarray, horizon: int = MONTH_BARS) -> np.ndarray:
    """Take every ``horizon``-th finite observation.

    Overlapping windows make a series look far more significant than it is.
    Sampling one observation per horizon gives genuinely independent periods at
    the cost of most of the data, which is the right trade when the alternative
    is a t-statistic that cannot be believed.
    """
    v = np.asarray(values, dtype=float)
    idx = np.where(np.isfinite(v))[0]
    return v[idx[::horizon]]
