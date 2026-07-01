"""Robustness tests — the machine that separates a real edge from a lucky fit.

A single positive walk-forward means little: it could be one favourable period,
one lucky set of coins, or the best of many configs you happened to try. This
module attacks a result from every angle:

* **sub-period stability** — is the edge positive across *most* time windows, or
  driven by one?
* **universe bootstrap** — does it survive when you randomly drop coins, or does
  it hinge on a couple of names?
* **Probabilistic / Deflated Sharpe** — given the sample length, the return
  distribution's fat tails, and *how many configs you tried*, what's the
  probability the true Sharpe is actually above zero?

The last one is the antidote to the single most common way quant researchers
fool themselves: testing 50 ideas and getting excited about the best one.
References: Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.panel import pivot_to_wide, validate_panel

_GAMMA = 0.5772156649015329  # Euler–Mascheroni


# --------------------------------------------------------------------------- #
# Normal distribution helpers (no scipy dependency)
# --------------------------------------------------------------------------- #
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse normal CDF via Acklam's rational approximation."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# --------------------------------------------------------------------------- #
# Probabilistic & Deflated Sharpe
# --------------------------------------------------------------------------- #
def _raw_sharpe(returns: np.ndarray) -> float:
    """Per-observation (non-annualised) Sharpe."""
    r = np.asarray(returns, float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def probabilistic_sharpe_ratio(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true per-obs Sharpe > ``sr_benchmark``), adjusting for skew, kurtosis, n.

    ``sr_benchmark`` is in per-observation units (0 = "better than nothing").
    """
    r = np.asarray(returns, float)
    n = r.size
    if n < 3:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    mu = r.mean()
    sr = mu / sd
    skew = float(((r - mu) ** 3).mean() / sd**3)
    kurt = float(((r - mu) ** 4).mean() / sd**4)  # non-excess (normal = 3)
    denom = math.sqrt(1 - skew * sr + ((kurt - 1) / 4) * sr**2)
    if denom == 0:
        return float("nan")
    return norm_cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom)


def expected_max_sharpe(trial_sharpes: np.ndarray) -> float:
    """Expected maximum per-obs Sharpe from N independent trials of zero edge."""
    s = np.asarray(trial_sharpes, float)
    n = s.size
    v = s.var(ddof=1)
    if n < 2 or v <= 0:
        return 0.0
    z1 = norm_ppf(1 - 1.0 / n)
    z2 = norm_ppf(1 - 1.0 / (n * math.e))
    return math.sqrt(v) * ((1 - _GAMMA) * z1 + _GAMMA * z2)


def deflated_sharpe_ratio(best_returns: np.ndarray, trial_sharpes: np.ndarray) -> float:
    """P(the *selected* config's true Sharpe > 0) after multiple-testing.

    ``trial_sharpes`` are the per-obs Sharpes of *every* config you tried;
    ``best_returns`` is the chosen config's return series.
    """
    sr0 = expected_max_sharpe(trial_sharpes)
    return probabilistic_sharpe_ratio(best_returns, sr_benchmark=sr0)


# --------------------------------------------------------------------------- #
# Resampling-based robustness
# --------------------------------------------------------------------------- #
@dataclass
class RobustnessResult:
    subperiod: pl.DataFrame
    universe_sharpes: np.ndarray
    psr: float
    dsr: float
    n_trials: int

    def summary(self) -> str:
        sp = self.subperiod
        frac_pos = float((sp["sharpe"] > 0).mean())
        us = self.universe_sharpes
        lines = [
            "=" * 60,
            "  Robustness report",
            "=" * 60,
            f"  Sub-periods positive       {int((sp['sharpe'] > 0).sum())}/{sp.height}"
            f"  ({frac_pos * 100:.0f}%)",
            f"  Sub-period Sharpe range     [{sp['sharpe'].min():.2f}, {sp['sharpe'].max():.2f}]",
            "-" * 60,
            f"  Universe bootstrap draws    {us.size}",
            f"  Median Sharpe               {np.median(us):>8.2f}",
            f"  5th percentile Sharpe       {np.percentile(us, 5):>8.2f}",
            f"  Fraction positive           {float(np.mean(us > 0)) * 100:>7.0f}%",
            "-" * 60,
            f"  Probabilistic Sharpe (>0)   {self.psr * 100:>7.1f}%",
            f"  Deflated Sharpe ({self.n_trials} trials) {self.dsr * 100:>6.1f}%",
            "=" * 60,
        ]
        return "\n".join(lines)


def _slice(panel, funding, ts):
    p = panel.filter(pl.col("timestamp").is_in(ts.implode()))
    f = funding.filter(pl.col("timestamp").is_in(ts.implode())) if funding is not None else None
    return p, f


def subperiod_stability(
    panel: pl.DataFrame,
    make_strategy,
    *,
    n_periods: int,
    backtest_kwargs: dict,
    funding: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Run the strategy on ``n_periods`` contiguous chunks; report each."""
    panel = validate_panel(panel)
    ts = pivot_to_wide(panel, "close")["timestamp"]
    n = ts.len()
    size = n // n_periods
    rows = []
    for k in range(n_periods):
        chunk = ts.slice(k * size, size if k < n_periods - 1 else n - k * size)
        p, f = _slice(panel, funding, chunk)
        res = CrossSectionalBacktester(p, make_strategy(), funding=f, **backtest_kwargs).run()
        m = res.metrics()
        rows.append({"period": k, "sharpe": m["sharpe"], "total_return": m["total_return"]})
    return pl.DataFrame(rows)


def universe_bootstrap(
    panel: pl.DataFrame,
    make_strategy,
    *,
    n_draws: int,
    subset_size: int,
    backtest_kwargs: dict,
    funding: pl.DataFrame | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Rerun on ``n_draws`` random coin subsets; return the Sharpe distribution."""
    panel = validate_panel(panel)
    symbols = panel["symbol"].unique().to_list()
    if subset_size >= len(symbols):
        raise ValueError("subset_size must be smaller than the universe")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_draws):
        pick = list(rng.choice(symbols, size=subset_size, replace=False))
        p = panel.filter(pl.col("symbol").is_in(pick))
        f = funding.filter(pl.col("symbol").is_in(pick)) if funding is not None else None
        res = CrossSectionalBacktester(p, make_strategy(), funding=f, **backtest_kwargs).run()
        out.append(res.metrics()["sharpe"])
    return np.array(out)
