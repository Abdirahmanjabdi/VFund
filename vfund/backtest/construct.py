"""Portfolio construction: turn a vector of scores into position weights.

This is the portfolio-manager's job, kept deliberately separate from the
signal. The same reversal score can be sized many ways; isolating that choice
here means you can improve *sizing* without touching *research*.

The default is a dollar-neutral book: demean the scores so longs and shorts
cancel, then scale so gross exposure (sum of absolute weights) equals
``leverage``. With ``top_k`` set, only the strongest names on each side trade —
usually cleaner and cheaper than spreading thin across the whole universe.
"""

from __future__ import annotations

import math

import numpy as np


def scores_to_weights(
    scores: np.ndarray,
    *,
    leverage: float = 1.0,
    top_k: int | None = None,
    neutralize: bool = True,
) -> np.ndarray:
    """Convert per-asset scores to target weights.

    ``neutralize=True`` (default) builds a dollar-neutral long/short book by
    demeaning the scores — the market move cancels and you're left with the
    *relative* bet. That's right for cross-sectional signals.

    ``neutralize=False`` keeps the raw scores, so the book can be net long or
    short. That's right for *time-series* signals (e.g. per-asset trend), where
    being long everything in an uptrend is the whole point — but note it then
    carries market exposure, so judge it against buy-and-hold, not zero.
    """
    s = np.asarray(scores, dtype=float)
    w = np.zeros_like(s)
    valid = ~np.isnan(s)
    n_valid = int(valid.sum())
    if n_valid < 2:
        return w  # not enough names to form a cross-section -> stay flat

    sv = s[valid]

    if neutralize and top_k is not None and top_k > 0 and 2 * top_k <= n_valid:
        # Long the top_k highest scores, short the top_k lowest — equal weight.
        order = np.argsort(sv)
        book = np.zeros_like(sv)
        book[order[-top_k:]] = 1.0
        book[order[:top_k]] = -1.0
    elif neutralize:
        book = sv - sv.mean()  # demean -> dollar neutral
    else:
        book = sv.copy()  # directional: keep net exposure

    gross = np.abs(book).sum()
    if gross > 0:
        book = book / gross * leverage
    w[valid] = book
    return w


def vol_scale_weights(
    weights: np.ndarray,
    recent_returns: np.ndarray,
    *,
    vol_target: float,
    bars_per_year: float,
    max_leverage: float = 3.0,
) -> np.ndarray:
    """Scale a weight vector so its predicted volatility hits ``vol_target``.

    ``recent_returns`` is an ``(L, N)`` window of asset returns. Predicted book
    vol is ``sqrt(w' Σ w)`` annualised; weights are scaled by
    ``vol_target / pred_vol`` and capped at ``max_leverage`` gross. Shared by the
    backtester and the live signal so they size positions identically.
    """
    w = np.asarray(weights, dtype=float)
    if recent_returns.shape[0] < 5:
        return w
    cov = np.atleast_2d(np.cov(recent_returns, rowvar=False))
    var = float(w @ cov @ w)
    pred_vol = math.sqrt(max(var, 0.0) * bars_per_year)
    if pred_vol <= 1e-9:
        return w
    w = w * (vol_target / pred_vol)
    gross = float(np.abs(w).sum())
    if gross > max_leverage:
        w = w * (max_leverage / gross)
    return w
