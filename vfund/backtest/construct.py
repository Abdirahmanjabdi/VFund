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

import numpy as np


def scores_to_weights(
    scores: np.ndarray,
    *,
    leverage: float = 1.0,
    top_k: int | None = None,
) -> np.ndarray:
    """Convert per-asset scores to dollar-neutral target weights."""
    s = np.asarray(scores, dtype=float)
    w = np.zeros_like(s)
    valid = ~np.isnan(s)
    n_valid = int(valid.sum())
    if n_valid < 2:
        return w  # not enough names to form a cross-section -> stay flat

    sv = s[valid]

    if top_k is not None and top_k > 0 and 2 * top_k <= n_valid:
        # Long the top_k highest scores, short the top_k lowest — equal weight.
        order = np.argsort(sv)
        book = np.zeros_like(sv)
        book[order[-top_k:]] = 1.0
        book[order[:top_k]] = -1.0
    else:
        book = sv - sv.mean()  # demean -> dollar neutral

    gross = np.abs(book).sum()
    if gross > 0:
        book = book / gross * leverage
    w[valid] = book
    return w
