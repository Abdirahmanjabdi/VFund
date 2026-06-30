"""Equity-curve and drawdown plotting.

matplotlib is an optional dependency (``pip install vfund[viz]``). Importing this
module without it raises a clear message rather than a cryptic ImportError deep
in a backtest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vfund.analytics.performance import max_drawdown


def plot_equity(result, path: str | Path, *, title: str | None = None) -> Path:
    """Save a two-panel equity + drawdown chart for a ``BacktestResult``."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "plotting needs matplotlib — install with `pip install vfund[viz]`"
        ) from exc

    ec = result.equity_curve
    ts = ec["timestamp"].to_numpy()
    equity = ec["equity"].to_numpy()

    running_peak = np.maximum.accumulate(equity)
    dd = (equity / running_peak - 1.0) * 100.0
    mdd, _, trough_i = max_drawdown(equity)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, height_ratios=[3, 1]
    )

    ax1.plot(ts, equity, color="#2563eb", lw=1.4)
    ax1.set_ylabel("Equity ($)")
    ax1.set_title(title or f"VFund — {result.meta.get('strategy', 'strategy')}")
    ax1.grid(alpha=0.25)

    ax2.fill_between(ts, dd, 0, color="#dc2626", alpha=0.35)
    ax2.scatter([ts[trough_i]], [mdd * 100.0], color="#dc2626", zorder=5, s=18)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
