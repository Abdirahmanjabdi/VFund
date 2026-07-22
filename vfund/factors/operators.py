"""Cross-sectional alpha operators — the vocabulary formulaic alphas are written in.

Why this exists
---------------
Every VFund strategy so far is a hand-written class: to test an idea you write
code. That bounds how many hypotheses you can afford to try, which is the wrong
thing to be bounded by — the whole platform is built on the premise that *most
ideas die*, so the throughput of killing them matters.

These operators let an alpha be written as an expression instead::

    rank(ts_corr(close, volume, 10)) * -1

which is how the published formulaic-alpha literature (Kakushadze 2015, GTJA
2014, Qlib's Alpha158) states its factors. With the vocabulary in place, porting
a paper is transcription rather than programming.

Conventions
-----------
All operators take and return **(T, N) float arrays**: ``T`` bars by ``N``
symbols, time increasing down axis 0 — the same orientation as
``PanelContext.closes``. Cross-sectional operators act along axis 1 (across
symbols at one instant); ``ts_*`` operators act along axis 0 (through time for
one symbol).

NaN policy
----------
NaN propagates. Nothing is silently filled with zero: a constant window returns
NaN from ``ts_corr``, an all-NaN cross-section returns all-NaN from ``rank``, and
a zero-sum row returns NaN from ``scale``. This matters because VFund's panels
are *ragged* — a coin that had not listed yet is NaN, and quietly turning that
into 0.0 would fabricate a tradable value for an asset that did not exist.

Look-ahead ban
--------------
This is the load-bearing property. Every ``ts_*`` operator reads backwards only,
``delta`` refuses a lag below 1, and there is deliberately no negative-shift
operator. A formula written in this vocabulary *cannot* express look-ahead —
enforced structurally here, and checked again by :mod:`vfund.factors.purity`.

Attribution
-----------
The operator set and its NaN/look-ahead semantics follow the Alpha Zoo base
layer of HKUDS/Vibe-Trading (MIT). The implementations here are independent —
that project operates on wide pandas DataFrames, VFund on numpy panels.
"""

from __future__ import annotations

import contextlib
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "rank", "zscore", "scale",
    "ts_rank", "ts_mean", "ts_std", "ts_sum", "ts_max", "ts_min",
    "ts_argmax", "ts_argmin", "ts_corr", "ts_cov",
    "delta", "delay", "decay_linear", "signed_power", "safe_div", "quiet_nan",
    "returns", "vwap", "adv",
]

#: Above this many symbols the O(N^2) pairwise rank would get memory-hungry, so
#: we fall back to a per-row sort. Crypto universes sit far below it.
_RANK_PAIRWISE_MAX_N = 512


def _f(x: np.ndarray) -> np.ndarray:
    """Coerce to float64 without copying when already correct."""
    a = np.asarray(x)
    return a if a.dtype == np.float64 else a.astype(np.float64)


# --- cross-sectional (axis 1) ------------------------------------------------


def rank(x: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank per row, ties averaged, NaN preserved.

    Returns values in (0, 1]. An all-NaN row stays all-NaN rather than becoming
    a uniform 0.5, so a bar with no live symbols cannot masquerade as a neutral
    signal.
    """
    a = _f(x)
    if a.ndim != 2:
        raise ValueError(f"rank expects a (T, N) panel, got shape {a.shape}")
    T, N = a.shape
    out = np.full((T, N), np.nan)
    n_valid = np.sum(~np.isnan(a), axis=1)

    if N <= _RANK_PAIRWISE_MAX_N:
        # Average rank = (#strictly less) + (#equal + 1)/2. NaN compares False
        # on both counts, so invalid entries drop out for free.
        less = (a[:, None, :] < a[:, :, None]).sum(axis=2)
        equal = (a[:, None, :] == a[:, :, None]).sum(axis=2)
        raw = less + (equal + 1.0) / 2.0
    else:  # pragma: no cover - large universes are not VFund's target
        raw = np.full((T, N), np.nan)
        for t in range(T):
            row = a[t]
            m = ~np.isnan(row)
            if not m.any():
                continue
            v = row[m]
            raw[t, m] = [(v < s).sum() + ((v == s).sum() + 1) / 2.0 for s in v]

    with np.errstate(invalid="ignore", divide="ignore"):
        pct = raw / n_valid[:, None]
    valid = ~np.isnan(a) & (n_valid[:, None] > 0)
    out[valid] = pct[valid]
    return out


def zscore(x: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score per row (sample std). Zero-std rows -> NaN."""
    a = _f(x)
    with quiet_nan():
        mean = np.nanmean(a, axis=1, keepdims=True)
        std = np.nanstd(a, axis=1, ddof=1, keepdims=True)
    std = np.where(std > 0, std, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (a - mean) / std
    return _kill_inf(z)


def scale(x: np.ndarray, a_: float = 1.0) -> np.ndarray:
    """L1-normalise each row so the absolute values sum to ``a_``.

    Zero-sum rows become NaN — a book with no gross exposure is not a book.
    """
    a = _f(x)
    s = np.nansum(np.abs(a), axis=1, keepdims=True)
    s = np.where(s > 0, s, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _kill_inf(a * a_ / s)


# --- time-series (axis 0) ----------------------------------------------------


def _windows(a: np.ndarray, n: int) -> np.ndarray:
    """Rolling windows along time: (T-n+1, N, n) view, no copy."""
    return sliding_window_view(a, n, axis=0)


def _rolling(a: np.ndarray, n: int, fn) -> np.ndarray:
    """Apply ``fn(windows) -> (T-n+1, N)`` and left-pad warmup with NaN."""
    if n < 1:
        raise ValueError(f"window must be >= 1, got {n}")
    a = _f(a)
    T = a.shape[0]
    out = np.full(a.shape, np.nan)
    if T < n:
        return out
    with quiet_nan():
        out[n - 1:] = fn(_windows(a, n))
    return out


def ts_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Rolling mean over the trailing ``n`` bars (inclusive of the current one)."""
    return _rolling(x, n, lambda w: np.nanmean((w), axis=2))


def ts_sum(x: np.ndarray, n: int) -> np.ndarray:
    """Rolling sum over the trailing ``n`` bars."""
    return _rolling(x, n, lambda w: np.nansum(w, axis=2))


def ts_std(x: np.ndarray, n: int) -> np.ndarray:
    """Rolling sample standard deviation over the trailing ``n`` bars."""
    if n < 2:
        raise ValueError(f"ts_std window must be >= 2, got {n}")
    return _rolling(x, n, lambda w: np.nanstd((w), axis=2, ddof=1))


def ts_max(x: np.ndarray, n: int) -> np.ndarray:
    """Rolling maximum over the trailing ``n`` bars."""
    return _rolling(x, n, lambda w: np.nanmax((w), axis=2))


def ts_min(x: np.ndarray, n: int) -> np.ndarray:
    """Rolling minimum over the trailing ``n`` bars."""
    return _rolling(x, n, lambda w: np.nanmin((w), axis=2))


def ts_argmax(x: np.ndarray, n: int) -> np.ndarray:
    """Bars since the rolling maximum: 0 = the max is the current bar."""
    return _rolling(x, n, lambda w: (n - 1) - _nanarg(w, np.nanargmax))


def ts_argmin(x: np.ndarray, n: int) -> np.ndarray:
    """Bars since the rolling minimum: 0 = the min is the current bar."""
    return _rolling(x, n, lambda w: (n - 1) - _nanarg(w, np.nanargmin))


def ts_rank(x: np.ndarray, n: int) -> np.ndarray:
    """Percentile rank of the current value within its trailing ``n``-window.

    In (0, 1]; compositionally compatible with cross-sectional :func:`rank`.
    """
    def _fn(w):
        cur = w[:, :, -1][:, :, None]
        valid = ~np.isnan(w)
        less = np.sum((w < cur) & valid, axis=2)
        equal = np.sum((w == cur) & valid, axis=2)
        cnt = valid.sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (less + (equal + 1.0) / 2.0) / cnt
        return np.where((cnt > 0) & ~np.isnan(w[:, :, -1]), r, np.nan)

    return _rolling(x, n, _fn)


def ts_corr(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Rolling Pearson correlation of two panels. Constant window -> NaN."""
    if n < 2:
        raise ValueError(f"ts_corr window must be >= 2, got {n}")
    a, b = _f(x), _f(y)
    if a.shape != b.shape:
        raise ValueError(f"ts_corr needs matching shapes, got {a.shape} vs {b.shape}")
    out = np.full(a.shape, np.nan)
    if a.shape[0] < n:
        return out
    wa, wb = _windows(a, n), _windows(b, n)
    ok = ~np.isnan(wa) & ~np.isnan(wb)
    wa_ = np.where(ok, wa, np.nan)
    wb_ = np.where(ok, wb, np.nan)
    with quiet_nan():
        ma = np.nanmean(wa_, axis=2, keepdims=True)
        mb = np.nanmean(wb_, axis=2, keepdims=True)
    da, db = wa_ - ma, wb_ - mb
    cov = np.nansum(da * db, axis=2)
    sa = np.sqrt(np.nansum(da * da, axis=2))
    sb = np.sqrt(np.nansum(db * db, axis=2))
    denom = sa * sb
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0, cov / denom, np.nan)
    # Require a full window of paired observations, like min_periods=n.
    corr = np.where(ok.sum(axis=2) == n, corr, np.nan)
    out[n - 1:] = corr
    return _kill_inf(out)


def ts_cov(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Rolling sample covariance of two panels."""
    if n < 2:
        raise ValueError(f"ts_cov window must be >= 2, got {n}")
    a, b = _f(x), _f(y)
    if a.shape != b.shape:
        raise ValueError(f"ts_cov needs matching shapes, got {a.shape} vs {b.shape}")
    out = np.full(a.shape, np.nan)
    if a.shape[0] < n:
        return out
    wa, wb = _windows(a, n), _windows(b, n)
    ok = ~np.isnan(wa) & ~np.isnan(wb)
    wa_ = np.where(ok, wa, np.nan)
    wb_ = np.where(ok, wb, np.nan)
    with quiet_nan():
        da = wa_ - np.nanmean(wa_, axis=2, keepdims=True)
        db = wb_ - np.nanmean(wb_, axis=2, keepdims=True)
    cov = np.nansum(da * db, axis=2) / (n - 1)
    out[n - 1:] = np.where(ok.sum(axis=2) == n, cov, np.nan)
    return _kill_inf(out)


def delay(x: np.ndarray, d: int) -> np.ndarray:
    """Value ``d`` bars ago.

    Raises:
        ValueError: if ``d < 1``. A negative delay would read the future; the
            operator does not exist so a formula cannot express it.
    """
    if d < 1:
        raise ValueError(f"delay must be >= 1 (look-ahead ban), got {d}")
    a = _f(x)
    out = np.full(a.shape, np.nan)
    if d < a.shape[0]:
        out[d:] = a[:-d]
    return out


def delta(x: np.ndarray, d: int) -> np.ndarray:
    """Change over ``d`` bars: ``x - delay(x, d)``. ``d >= 1`` strictly."""
    if d < 1:
        raise ValueError(f"delta lag must be >= 1 (look-ahead ban), got {d}")
    return _f(x) - delay(x, d)


def decay_linear(x: np.ndarray, n: int) -> np.ndarray:
    """Linearly decayed weighted mean, weights ``n, n-1, ..., 1`` normalised.

    Heaviest weight on the most recent bar. Any NaN in the window -> NaN.
    """
    if n < 1:
        raise ValueError(f"decay_linear window must be >= 1, got {n}")
    w = np.arange(n, 0, -1, dtype=np.float64)[::-1]  # oldest..newest
    w /= w.sum()
    return _rolling(x, n, lambda win: np.einsum("tnk,k->tn", win, w))


def signed_power(x: np.ndarray, e: float) -> np.ndarray:
    """``sign(x) * |x| ** e`` — magnitude shaping that preserves direction."""
    a = _f(x)
    with np.errstate(invalid="ignore"):
        return _kill_inf(np.sign(a) * np.abs(a) ** e)


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise division where a zero (or NaN) denominator yields NaN."""
    x, y = _f(a), _f(b)
    y = np.where(y == 0, np.nan, y)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _kill_inf(x / y)


# --- derived panels ----------------------------------------------------------


def returns(close: np.ndarray, d: int = 1) -> np.ndarray:
    """Simple return over ``d`` bars."""
    return safe_div(delta(close, d), delay(close, d))


def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Typical price ``(H + L + C) / 3`` as the VWAP proxy.

    Real VWAP needs intra-bar trade data. Binance daily klines do not carry it,
    so the standard OHLC proxy is used — and named honestly, because a formula
    citing "vwap" on daily bars is always some proxy.
    """
    return (_f(high) + _f(low) + _f(close)) / 3.0


def adv(close: np.ndarray, volume: np.ndarray, n: int) -> np.ndarray:
    """Average daily dollar volume over ``n`` bars."""
    return ts_mean(_f(close) * _f(volume), n)


# --- internals ---------------------------------------------------------------


@contextlib.contextmanager
def quiet_nan():
    """Silence numpy's all-NaN / empty-slice reduction warnings.

    Those warnings fire exactly when the NaN policy is working as designed: a
    window containing only NaN *should* reduce to NaN. Ragged crypto panels hit
    this on every bar before a coin listed, so the warning is pure noise that
    would train the reader to ignore warnings - the opposite of useful.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="(All-NaN|Mean of empty slice|Degrees of freedom)",
            category=RuntimeWarning,
        )
        yield


def _nanarg(w: np.ndarray, fn) -> np.ndarray:
    """Apply nanargmax/nanargmin over the window axis, all-NaN slices -> NaN."""
    allnan = np.isnan(w).all(axis=2)
    safe = np.where(np.isnan(w), -np.inf if fn is np.nanargmax else np.inf, w)
    idx = np.argmax(safe, axis=2) if fn is np.nanargmax else np.argmin(safe, axis=2)
    return np.where(allnan, np.nan, idx.astype(np.float64))


def _kill_inf(a: np.ndarray) -> np.ndarray:
    """Map +/-inf to NaN. Infinities are never a valid factor value."""
    return np.where(np.isfinite(a), a, np.nan)
