"""Score alphas by information coefficient, and categorise alive / reversed / dead.

What an IC study is, and is not
-------------------------------
The information coefficient is the cross-sectional rank correlation between an
alpha's scores at bar ``t`` and the realised forward return from ``t`` to
``t+1``. It answers one narrow question: *does this alpha rank assets in an
order that matches what happens next?*

It does **not** answer whether the alpha makes money. A statistically strong IC
on a high-turnover alpha routinely loses after costs, and this module charges no
costs at all. Treat its output as a **signal-quality scan** that decides what is
worth backtesting properly — never as a profitability claim.

The categorisation thresholds follow the convention used in published alpha
survival studies so results are comparable:

* **alive** — mean IC > 0.02, t-stat > 2, and >= 55% of bars with positive IC;
* **reversed** — mean IC < -0.02 and t-stat < -2 (the alpha predicts, backwards);
* **dead** — everything else.

"Reversed" is a real category, not a bookkeeping quirk: an alpha whose sign has
flipped with significance is evidence the anomaly it captured has been crowded
out and the opposite trade is now the one being paid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vfund.factors.alpha import Alpha, Panel
from vfund.factors.operators import quiet_nan, rank

#: Categorisation thresholds — see the module docstring.
ALIVE_IC = 0.02
ALIVE_T = 2.0
ALIVE_HIT = 0.55


@dataclass(frozen=True)
class ICResult:
    """One alpha's IC profile over a window."""

    name: str
    mean_ic: float
    t_stat: float
    hit_rate: float          # fraction of bars with IC > 0
    ir: float                # mean IC / std IC (information ratio)
    n_bars: int
    verdict: str             # "alive" | "reversed" | "dead"
    theme: tuple[str, ...] = ()

    def row(self) -> str:  # pragma: no cover - formatting only
        return (f"{self.name:<24} {self.mean_ic:>+8.4f} {self.t_stat:>7.2f} "
                f"{self.ir:>7.3f} {self.hit_rate*100:>6.1f}% {self.n_bars:>6} "
                f"  {self.verdict}")


def _spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row rank correlation of two (T, N) panels, NaN where undefined.

    Both sides are rank-transformed cross-sectionally first, so this is Spearman
    rather than Pearson — robust to the fat tails and outliers that dominate
    crypto returns, where a single 400% day would otherwise drive the estimate.
    """
    ra, rb = rank(a), rank(b)
    ok = ~np.isnan(ra) & ~np.isnan(rb)
    n = ok.sum(axis=1)
    ra = np.where(ok, ra, np.nan)
    rb = np.where(ok, rb, np.nan)
    with quiet_nan():
        ma = np.nanmean(ra, axis=1, keepdims=True)
        mb = np.nanmean(rb, axis=1, keepdims=True)
    da, db = ra - ma, rb - mb
    cov = np.nansum(da * db, axis=1)
    sa = np.sqrt(np.nansum(da * da, axis=1))
    sb = np.sqrt(np.nansum(db * db, axis=1))
    denom = sa * sb
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = np.where(denom > 0, cov / denom, np.nan)
    # Need a real cross-section to correlate; 2 names is not a ranking.
    return np.where(n >= 5, ic, np.nan)


def forward_returns(close: np.ndarray, horizon: int = 1) -> np.ndarray:
    """Return from bar ``t`` to ``t + horizon``, NaN on the trailing edge.

    This is the only place the future is used, and it is used correctly: row
    ``t`` holds a return that is *unknown* at ``t``, which is exactly what an
    alpha at ``t`` is being scored against. It is never fed back into a signal.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    out = np.full(close.shape, np.nan)
    if close.shape[0] > horizon:
        fwd = close[horizon:] / close[:-horizon] - 1.0
        out[:-horizon] = np.where(np.isfinite(fwd), fwd, np.nan)
    return out


def information_coefficient(
    a: Alpha, panel: Panel, *, horizon: int = 1, skip_warmup: bool = True
) -> ICResult:
    """Score one alpha's IC over the whole panel.

    Args:
        a: the registered alpha.
        panel: OHLCV panel to evaluate on.
        horizon: forward-return horizon in bars.
        skip_warmup: drop the alpha's declared warmup bars before scoring.

    Returns:
        An :class:`ICResult` with the mean IC, its t-statistic, hit rate, IR and
        the alive/reversed/dead verdict.
    """
    scores = a.compute(panel)
    fwd = forward_returns(panel.close, horizon)
    start = a.warmup if skip_warmup else 0
    scores, fwd = scores[start:], fwd[start:]

    ic = _spearman_rows(scores, fwd)
    ic = ic[~np.isnan(ic)]
    n = int(ic.size)
    if n < 30:  # too few valid cross-sections to say anything
        return ICResult(a.name, float("nan"), float("nan"), float("nan"),
                        float("nan"), n, "dead", a.theme)

    mean = float(ic.mean())
    sd = float(ic.std(ddof=1))
    t = mean / (sd / np.sqrt(n)) if sd > 0 else 0.0
    ir = mean / sd if sd > 0 else 0.0
    hit = float((ic > 0).mean())

    if mean > ALIVE_IC and t > ALIVE_T and hit >= ALIVE_HIT:
        verdict = "alive"
    elif mean < -ALIVE_IC and t < -ALIVE_T:
        verdict = "reversed"
    else:
        verdict = "dead"
    return ICResult(a.name, mean, float(t), hit, float(ir), n, verdict, a.theme)


def bench(alphas: list[Alpha], panel: Panel, *, horizon: int = 1) -> list[ICResult]:
    """Score every alpha, strongest absolute IC first."""
    out = [information_coefficient(a, panel, horizon=horizon) for a in alphas]
    return sorted(out, key=lambda r: -abs(r.mean_ic if np.isfinite(r.mean_ic) else 0))


def summarise(results: list[ICResult]) -> str:  # pragma: no cover - formatting
    """Render a bench table plus survival counts."""
    head = (f"{'alpha':<24} {'mean IC':>8} {'t-stat':>7} {'IR':>7} "
            f"{'hit':>7} {'bars':>6}   verdict")
    lines = [head, "-" * len(head)]
    lines += [r.row() for r in results]
    counts = {v: sum(1 for r in results if r.verdict == v)
              for v in ("alive", "reversed", "dead")}
    total = len(results) or 1
    lines += [
        "-" * len(head),
        f"alive {counts['alive']} ({counts['alive']/total:.0%})  "
        f"reversed {counts['reversed']} ({counts['reversed']/total:.0%})  "
        f"dead {counts['dead']} ({counts['dead']/total:.0%})",
    ]
    return "\n".join(lines)
