"""Information-coefficient testing for cross-sectional strategies.

This is the gate a factor passes *before* anyone builds a portfolio out of it.
It answers one question — does the factor rank assets in the order the next
period actually delivers? — and answers it with the corrections that make the
answer trustworthy.

Two corrections do most of the work:

**Newey-West standard errors.** Testing a 5-day forward return every day means
consecutive observations share four days of outcome. They are mechanically
autocorrelated, and a naive t-stat on overlapping windows is inflated by roughly
sqrt(horizon) — enough to turn noise into a "significant" result. The
Newey-West estimator with lag = horizon absorbs that.

**Sub-period stability.** A factor that earns its whole t-stat in one regime is
a regime bet wearing a factor's clothes. Splitting the sample and requiring
consistency of sign catches that, and no amount of full-sample significance
substitutes for it.

Neither correction charges transaction costs. A strong IC on a fast-turnover
factor still routinely loses money after costs. Treat everything here as
signal-quality evidence that decides what is worth backtesting, never as a
profitability claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vfund.strategy.cross_sectional import CrossSectionalStrategy, PanelContext


def _rank_rows(x: np.ndarray) -> np.ndarray:
    """Rank-transform each row, NaN-preserving, mean-centred per row."""
    out = np.full(x.shape, np.nan)
    for t in range(x.shape[0]):
        row = x[t]
        ok = np.isfinite(row)
        if ok.sum() < 2:
            continue
        v = row[ok]
        order = v.argsort().argsort().astype(float)
        out[t, ok] = order - order.mean()
    return out


def spearman(a: np.ndarray, b: np.ndarray, min_names: int = 5) -> np.ndarray:
    """Per-row Spearman correlation of two (T, N) panels.

    Rank-transforming both sides first makes this robust to the fat tails that
    dominate crypto returns, where one 400% day would otherwise drive a Pearson
    estimate on its own.
    """
    ra, rb = _rank_rows(a), _rank_rows(b)
    ok = np.isfinite(ra) & np.isfinite(rb)
    n = ok.sum(axis=1)
    out = np.full(a.shape[0], np.nan)
    for t in range(a.shape[0]):
        if n[t] < min_names:
            continue
        u, v = ra[t, ok[t]], rb[t, ok[t]]
        u = u - u.mean()
        v = v - v.mean()
        den = math.sqrt(float((u * u).sum()) * float((v * v).sum()))
        if den > 0:
            out[t] = float((u * v).sum()) / den
    return out


def newey_west_t(x: np.ndarray, lag: int) -> float:
    """t-statistic for mean(x) != 0 with Newey-West HAC standard errors.

    ``lag`` should be at least the forward-return horizon; with overlapping
    windows the raw t-stat is inflated and this is the standard fix.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float("nan")
    mu = x.mean()
    e = x - mu
    gamma0 = float((e * e).sum()) / n
    s = gamma0
    for L in range(1, min(lag, n - 1) + 1):
        w = 1.0 - L / (lag + 1.0)  # Bartlett kernel
        cov = float((e[L:] * e[:-L]).sum()) / n
        s += 2.0 * w * cov
    if s <= 0:
        return float("nan")
    se = math.sqrt(s / n)
    return float(mu / se) if se > 0 else float("nan")


@dataclass
class ICReport:
    """One factor's IC profile at one forward horizon."""

    name: str
    horizon: int
    mean_ic: float
    t_naive: float
    t_nw: float
    ir: float
    hit_rate: float
    n_bars: int
    sub_ics: tuple[float, ...] = field(default_factory=tuple)

    @property
    def stable(self) -> bool:
        """True when every sub-period agrees with the full-sample sign."""
        if not self.sub_ics or not math.isfinite(self.mean_ic):
            return False
        sign = math.copysign(1.0, self.mean_ic)
        return all(
            math.isfinite(s) and math.copysign(1.0, s) == sign for s in self.sub_ics
        )

    @property
    def verdict(self) -> str:
        if not math.isfinite(self.t_nw):
            return "insufficient"
        if abs(self.mean_ic) < 0.01 or abs(self.t_nw) < 2.0:
            return "dead"
        if not self.stable:
            return "unstable"
        return "alive" if self.mean_ic > 0 else "reversed"

    def row(self) -> str:  # pragma: no cover - formatting only
        subs = " ".join(f"{s:+.3f}" if math.isfinite(s) else "  n/a" for s in self.sub_ics)
        return (
            f"{self.name:<26} {self.horizon:>2}d {self.mean_ic:>+8.4f} "
            f"{self.t_naive:>7.2f} {self.t_nw:>7.2f} {self.hit_rate * 100:>5.1f}% "
            f"{self.n_bars:>5}  [{subs}]  {self.verdict}"
        )


def factor_panel(
    strategy: CrossSectionalStrategy,
    closes: np.ndarray,
    symbols: list[str],
    *,
    funding: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
    start: int = 0,
    step: int = 1,
) -> np.ndarray:
    """Evaluate ``strategy`` at every bar, returning a (T, N) score panel.

    Bars are evaluated through :class:`PanelContext`, so each score sees only
    data up to and including its own bar.
    """
    T, N = closes.shape
    out = np.full((T, N), np.nan)
    for t in range(start, T, step):
        ctx = PanelContext(t, closes, symbols, funding=funding, volumes=volumes)
        try:
            s = strategy.scores(ctx)
        except (ValueError, IndexError):
            continue
        if s is not None and len(s) == N:
            out[t] = s
    return out


def forward_returns(closes: np.ndarray, horizon: int) -> np.ndarray:
    """(T, N) simple return from bar ``t`` to ``t + horizon``; NaN at the tail."""
    T = closes.shape[0]
    out = np.full(closes.shape, np.nan)
    if horizon < T:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[: T - horizon] = closes[horizon:] / closes[: T - horizon] - 1.0
    return np.where(np.isfinite(out), out, np.nan)


def ic_report(
    name: str,
    scores: np.ndarray,
    closes: np.ndarray,
    horizon: int = 1,
    *,
    n_subperiods: int = 3,
    min_names: int = 5,
) -> ICReport:
    """Full IC profile for one factor at one horizon."""
    fwd = forward_returns(closes, horizon)
    ics = spearman(scores, fwd, min_names=min_names)
    valid = ics[np.isfinite(ics)]
    n = len(valid)
    if n < 10:
        return ICReport(name, horizon, float("nan"), float("nan"), float("nan"),
                        float("nan"), float("nan"), n)

    mu = float(valid.mean())
    sd = float(valid.std(ddof=1))
    t_naive = mu / (sd / math.sqrt(n)) if sd > 0 else float("nan")
    t_nw = newey_west_t(ics, lag=max(horizon, 1))
    ir = mu / sd if sd > 0 else float("nan")
    hit = float((valid > 0).mean())

    # Sub-periods are cut on the *observed* IC series so each holds a comparable
    # number of observations even when the panel starts ragged.
    subs: list[float] = []
    if n_subperiods > 1:
        chunks = np.array_split(valid, n_subperiods)
        subs = [float(c.mean()) if len(c) else float("nan") for c in chunks]

    return ICReport(name, horizon, mu, t_naive, t_nw, ir, hit, n, tuple(subs))


HEADER = (
    f"{'factor':<26} {'h':>3} {'mean_IC':>8} {'t_raw':>7} {'t_NW':>7} "
    f"{'hit':>6} {'bars':>5}  {'sub-periods':<24} verdict"
)


@dataclass
class SpreadReport:
    """Mean-based decile spread — what a portfolio would actually earn.

    Rank IC is blind to magnitude, and in a fat-tailed market magnitude *is*
    the P&L. A factor can order the cross-section correctly, post a large IC
    t-statistic, and still have no spread between the deciles you would trade,
    because the extreme bucket's mean is set by a handful of enormous moves
    against you. This measures the thing rank correlation cannot see.
    """

    name: str
    top_mean: float
    bottom_mean: float
    spread: float
    spread_t: float
    top_median: float
    bottom_median: float
    n_obs: int

    @property
    def median_spread(self) -> float:
        return self.top_median - self.bottom_median

    @property
    def tail_driven(self) -> bool:
        """True when the median and mean spreads disagree in sign.

        The signature of an unmonetisable factor: it ranks the typical case
        correctly while the extremes pay the other way.
        """
        if not (math.isfinite(self.spread) and math.isfinite(self.median_spread)):
            return False
        return math.copysign(1.0, self.spread) != math.copysign(1.0, self.median_spread)

    def row(self) -> str:  # pragma: no cover - formatting only
        flag = "  TAIL-DRIVEN" if self.tail_driven else ""
        return (
            f"{self.name:<26} spread {self.spread:>+8.4%}/bar  t {self.spread_t:>6.2f}  "
            f"(median {self.median_spread:>+8.4%}){flag}"
        )


def decile_spread(
    name: str,
    scores: np.ndarray,
    closes: np.ndarray,
    horizon: int = 1,
    *,
    n_buckets: int = 10,
    min_names: int = 20,
) -> SpreadReport:
    """Mean forward return of the top bucket minus the bottom bucket.

    This is the gate that matters: it is what an equal-weight long/short book
    over those buckets would earn before costs, and unlike :func:`ic_report` it
    cannot be fooled by a correct ordering whose extremes are dominated by
    outliers.
    """
    fwd = forward_returns(closes, horizon)
    top: list[float] = []
    bot: list[float] = []
    per_bar: list[float] = []
    for t in range(scores.shape[0]):
        ok = np.isfinite(scores[t]) & np.isfinite(fwd[t])
        if ok.sum() < min_names:
            continue
        s, r = scores[t, ok], fwd[t, ok]
        order = s.argsort()
        chunks = np.array_split(order, n_buckets)
        b, u = r[chunks[0]], r[chunks[-1]]
        bot.extend(b.tolist())
        top.extend(u.tolist())
        per_bar.append(float(u.mean() - b.mean()))

    if len(per_bar) < 10:
        nan = float("nan")
        return SpreadReport(name, nan, nan, nan, nan, nan, nan, 0)

    t_arr = np.asarray(top)
    b_arr = np.asarray(bot)
    pb = np.asarray(per_bar)
    return SpreadReport(
        name=name,
        top_mean=float(t_arr.mean()),
        bottom_mean=float(b_arr.mean()),
        spread=float(pb.mean()),
        spread_t=newey_west_t(pb, lag=max(horizon, 1)),
        top_median=float(np.median(t_arr)),
        bottom_median=float(np.median(b_arr)),
        n_obs=len(pb),
    )
