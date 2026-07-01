"""Performance metrics computed from an equity curve.

These are the numbers an allocator asks for before wiring you a dollar: how much
did you make, how bumpy was the ride, and how deep was the worst hole. Get
comfortable with them now — your future track record is just this report, run on
real money instead of a backtest.
"""

from __future__ import annotations

import math

import numpy as np

# Lazily annotated to avoid importing BacktestResult at module load (cycle-free).


def equity_returns(equity: np.ndarray) -> np.ndarray:
    """Simple per-bar returns from an equity series."""
    equity = np.asarray(equity, dtype=float)
    if equity.size < 2:
        return np.zeros(0)
    return equity[1:] / equity[:-1] - 1.0


def sharpe_ratio(returns: np.ndarray, periods_per_year: float, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio. ``rf`` is the per-year risk-free rate."""
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return 0.0
    rf_per_bar = rf / periods_per_year
    excess = returns - rf_per_bar
    sd = excess.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * math.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, periods_per_year: float, rf: float = 0.0) -> float:
    """Annualised Sortino ratio — Sharpe but penalising only downside vol."""
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return 0.0
    rf_per_bar = rf / periods_per_year
    excess = returns - rf_per_bar
    downside = excess[excess < 0]
    if downside.size == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    dd = math.sqrt(np.mean(downside**2))
    if dd == 0:
        return 0.0
    return float(excess.mean() / dd * math.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> tuple[float, int, int]:
    """Maximum peak-to-trough drawdown.

    Returns ``(max_dd, peak_idx, trough_idx)`` where ``max_dd`` is a negative
    fraction (e.g. ``-0.32`` for a 32% drawdown).
    """
    equity = np.asarray(equity, dtype=float)
    if equity.size == 0:
        return 0.0, 0, 0
    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / running_peak - 1.0
    trough_idx = int(np.argmin(drawdowns))
    peak_idx = int(np.argmax(equity[: trough_idx + 1])) if trough_idx > 0 else 0
    return float(drawdowns[trough_idx]), peak_idx, trough_idx


def cagr(equity: np.ndarray, periods_per_year: float) -> float:
    """Compound annual growth rate implied by the equity curve."""
    equity = np.asarray(equity, dtype=float)
    if equity.size < 2 or equity[0] <= 0:
        return 0.0
    total_return = equity[-1] / equity[0]
    if total_return <= 0:
        return -1.0
    years = (equity.size - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float(total_return ** (1 / years) - 1.0)


def alpha_beta(returns: np.ndarray, benchmark: np.ndarray, periods_per_year: float) -> dict:
    """Regress strategy returns on a benchmark to separate alpha from beta.

    The single most important adjustment for a *directional* strategy: a book
    that's long in a bull market looks brilliant, but most of that is just market
    exposure (beta). Real skill is the intercept (alpha) that survives *after*
    subtracting beta times the market. A high alpha t-stat (> ~2) is the bar.

    Returns annualised alpha, beta, and the alpha t-statistic.
    """
    r = np.asarray(returns, float)
    b = np.asarray(benchmark, float)
    n = min(r.size, b.size)
    if n < 3:
        return {"alpha_ann": 0.0, "beta": 0.0, "alpha_t": 0.0}
    r, b = r[-n:], b[-n:]
    X = np.column_stack([np.ones(n), b])
    coef, _, _, _ = np.linalg.lstsq(X, r, rcond=None)
    alpha_per, beta = float(coef[0]), float(coef[1])
    resid = r - X @ coef
    dof = max(n - 2, 1)
    s2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se_alpha = math.sqrt(s2 * xtx_inv[0, 0]) if s2 > 0 else 0.0
    t_alpha = alpha_per / se_alpha if se_alpha > 0 else 0.0
    return {"alpha_ann": alpha_per * periods_per_year, "beta": beta, "alpha_t": t_alpha}


def compute_metrics(result) -> dict:
    """Compute the full metric set for a ``BacktestResult``."""
    equity = result.equity_curve["equity"].to_numpy()
    ppy = result.bars_per_year
    rets = equity_returns(equity)
    mdd, peak_i, trough_i = max_drawdown(equity)

    final = float(equity[-1])
    total_ret = final / result.initial_cash - 1.0
    ann_vol = float(rets.std(ddof=1) * math.sqrt(ppy)) if rets.size > 1 else 0.0

    return {
        "strategy": result.meta.get("strategy", "?"),
        "n_bars": int(equity.size),
        "n_trades": result.n_trades,
        "initial_equity": result.initial_cash,
        "final_equity": final,
        "total_return": total_ret,
        "cagr": cagr(equity, ppy),
        "ann_volatility": ann_vol,
        "sharpe": sharpe_ratio(rets, ppy),
        "sortino": sortino_ratio(rets, ppy),
        "max_drawdown": mdd,
        "max_dd_peak_idx": peak_i,
        "max_dd_trough_idx": trough_i,
    }


def format_report(m: dict) -> str:
    """Render a metrics dict as a human-readable text report."""

    def pct(x: float) -> str:
        return f"{x * 100:,.2f}%"

    def money(x: float) -> str:
        return f"${x:,.2f}"

    lines = [
        "=" * 46,
        f"  VFund backtest report - {m['strategy']}",
        "=" * 46,
        f"  Bars                {m['n_bars']:>20,}",
        f"  Trades              {m['n_trades']:>20,}",
        f"  Initial equity      {money(m['initial_equity']):>20}",
        f"  Final equity        {money(m['final_equity']):>20}",
        "-" * 46,
        f"  Total return        {pct(m['total_return']):>20}",
        f"  CAGR                {pct(m['cagr']):>20}",
        f"  Ann. volatility     {pct(m['ann_volatility']):>20}",
        f"  Sharpe (ann.)       {m['sharpe']:>20.2f}",
        f"  Sortino (ann.)      {m['sortino']:>20.2f}",
        f"  Max drawdown        {pct(m['max_drawdown']):>20}",
        "=" * 46,
    ]
    return "\n".join(lines)
