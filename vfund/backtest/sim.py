"""The core portfolio-simulation loop, isolated as a hot-path primitive.

This is the innermost loop of the cross-sectional engine: given per-bar returns
and a schedule of target weights, walk forward bar by bar — earn the return,
drift the weights, pay turnover at each rebalance. It runs T times per backtest
and is re-run thousands of times in bootstraps and walk-forwards, so it's the
natural candidate for a native (Rust) implementation.

``simulate`` dispatches to the compiled Rust version if available (see
``vfund/backtest/_accel.py`` and ``rust/``), else this pure-Python reference.
Both must produce identical results; ``simulate_py`` is the specification.
"""

from __future__ import annotations

import numpy as np


def simulate_py(
    rets: np.ndarray,
    reb_indices: np.ndarray,
    reb_weights: np.ndarray,
    *,
    initial: float = 10_000.0,
    cost_rate: float = 0.0,
    short_cost_per_bar: float = 0.0,
    funding: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference simulation.

    Parameters
    ----------
    rets : (T, N) float array
        Per-bar simple returns; ``rets[t]`` is the return from ``t-1`` to ``t``.
        Must be finite (caller sanitises NaNs to 0).
    reb_indices : (R,) int array
        Sorted bar indices at which the book is rebalanced.
    reb_weights : (R, N) float array
        Target weights applied at each rebalance index.
    initial, cost_rate, short_cost_per_bar : float
        Starting equity, turnover cost rate, per-bar financing on short notional.
    funding : (T, N) float array, optional
        Funding paid on each bar (a short in positive funding receives it).

    Returns
    -------
    equity : (T,) float array
    turnovers : (R,) float array
    """
    rets = np.ascontiguousarray(rets, dtype=float)
    T, N = rets.shape
    reb_indices = np.asarray(reb_indices, dtype=np.int64)
    reb_weights = np.ascontiguousarray(reb_weights, dtype=float)

    equity = np.empty(T)
    equity[0] = initial
    turnovers = np.empty(reb_indices.size)

    w_active = np.zeros(N)
    eq = initial
    ptr = 0

    for t in range(1, T):
        r = rets[t]
        price_ret = float(w_active @ r)
        fund_ret = -float(w_active @ funding[t]) if funding is not None else 0.0
        short_ret = -float(np.abs(np.minimum(w_active, 0.0)).sum()) * short_cost_per_bar
        eq *= 1.0 + price_ret + fund_ret + short_ret
        growth = 1.0 + price_ret
        w_drifted = w_active * (1.0 + r) / growth if growth != 0.0 else w_active.copy()

        if ptr < reb_indices.size and reb_indices[ptr] == t:
            w_target = reb_weights[ptr]
            turnover = float(np.abs(w_target - w_drifted).sum())
            eq *= 1.0 - turnover * cost_rate
            turnovers[ptr] = turnover
            w_active = w_target
            ptr += 1
        else:
            w_active = w_drifted
        equity[t] = eq

    return equity, turnovers


def simulate(*args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Run the simulation via the Rust core if built, else the Python reference."""
    from vfund.backtest._accel import rust_simulate

    if rust_simulate is not None:
        return rust_simulate(*args, **kwargs)
    return simulate_py(*args, **kwargs)
