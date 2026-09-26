"""Calendar and session effects, tested honestly.

Seasonality studies are the easiest place in quantitative finance to fool
yourself. There are 24 hours, 5 weekdays, 12 months, and ~21 trading days in a
month; slice a return series enough ways and something will look significant at
the 5% level purely because you looked many times. Two disciplines make the
difference, and both are enforced here rather than left to the caller:

**Family-wise correction.** Every bucket in a family is one test. A family of 24
hourly buckets needs |t| > 3.2 for a genuine 5% result, not 2.0. The
:class:`BucketFamily` reports the corrected threshold alongside the raw one and
will not label anything significant without it.

**A cost hurdle.** Statistical significance is not tradeability. An effect of
half a basis point per day is real and worthless if crossing the spread costs
two. Every result carries the round-trip cost it would have to clear, and the
verdict accounts for it.

What survives both is worth a backtest. Nothing else is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from vfund.research.ic import newey_west_t


@dataclass(frozen=True)
class Bucket:
    """One calendar bucket's return profile."""

    label: str
    mean: float          # mean return per occurrence
    t_stat: float        # Newey-West t
    n: int
    hit_rate: float
    median: float
    std: float

    @property
    def ann_bps(self) -> float:
        """Mean effect in basis points per occurrence."""
        return self.mean * 10_000.0


@dataclass
class BucketFamily:
    """A family of buckets tested together, with the correction applied."""

    name: str
    buckets: tuple[Bucket, ...]
    cost_bps_round_trip: float
    alpha: float = 0.05

    @property
    def n_tests(self) -> int:
        return len(self.buckets)

    @property
    def t_threshold(self) -> float:
        """Two-sided Bonferroni-corrected |t| threshold for this family.

        Uses a normal approximation, which is accurate at the sample sizes
        seasonality studies produce and avoids a scipy dependency.
        """
        if self.n_tests < 1:
            return float("inf")
        p = self.alpha / (2.0 * self.n_tests)
        # Inverse normal survival function via the same rational approximation
        # used elsewhere in the research package.
        from vfund.research.robustness import norm_ppf

        return abs(norm_ppf(p))

    def survivors(self) -> list[Bucket]:
        """Buckets that clear both the corrected t and the cost hurdle."""
        thr = self.t_threshold
        return [
            b for b in self.buckets
            if math.isfinite(b.t_stat)
            and abs(b.t_stat) > thr
            and abs(b.ann_bps) > self.cost_bps_round_trip
        ]

    def report(self) -> str:  # pragma: no cover - formatting only
        thr = self.t_threshold
        lines = [
            f"{self.name}  ({self.n_tests} buckets, "
            f"Bonferroni |t| > {thr:.2f}, cost hurdle {self.cost_bps_round_trip:.1f} bps)",
            f"  {'bucket':<14} {'mean bps':>9} {'t':>7} {'hit':>7} {'n':>7}  flag",
        ]
        for b in sorted(self.buckets, key=lambda x: -abs(x.t_stat)):
            flag = ""
            if math.isfinite(b.t_stat) and abs(b.t_stat) > thr:
                flag = "SIGNIF" if abs(b.ann_bps) > self.cost_bps_round_trip else "signif, below cost"
            elif math.isfinite(b.t_stat) and abs(b.t_stat) > 2.0:
                flag = "(raw p<.05 only)"
            lines.append(
                f"  {b.label:<14} {b.ann_bps:>+9.2f} {b.t_stat:>7.2f} "
                f"{b.hit_rate * 100:>6.1f}% {b.n:>7}  {flag}"
            )
        return "\n".join(lines)


def bucket_returns(
    returns: np.ndarray,
    labels: np.ndarray,
    *,
    name: str,
    cost_bps_round_trip: float = 0.0,
    nw_lag: int = 1,
    min_obs: int = 30,
    alpha: float = 0.05,
) -> BucketFamily:
    """Group ``returns`` by ``labels`` and test each group's mean against zero.

    Args:
        returns: 1-D array of per-period simple returns.
        labels: same length; any hashable bucket label per observation.
        name: family name, used in the report.
        cost_bps_round_trip: what it costs to enter and exit once. A bucket
            whose effect is smaller than this is not tradeable however
            significant it is.
        nw_lag: Newey-West lag. Same-bucket observations a day apart can be
            autocorrelated, so this defaults to 1 rather than 0.
        min_obs: buckets with fewer observations are reported with a NaN t
            rather than a t computed from too little data.
    """
    returns = np.asarray(returns, dtype=float)
    labels = np.asarray(labels)
    if returns.shape != labels.shape:
        raise ValueError("returns and labels must have the same shape")

    ok = np.isfinite(returns)
    returns, labels = returns[ok], labels[ok]

    out: list[Bucket] = []
    for lab in sorted(set(labels.tolist()), key=str):
        r = returns[labels == lab]
        if len(r) < min_obs:
            out.append(Bucket(str(lab), float(r.mean()) if len(r) else float("nan"),
                              float("nan"), len(r), float("nan"), float("nan"),
                              float("nan")))
            continue
        out.append(
            Bucket(
                label=str(lab),
                mean=float(r.mean()),
                t_stat=newey_west_t(r, lag=nw_lag),
                n=len(r),
                hit_rate=float((r > 0).mean()),
                median=float(np.median(r)),
                std=float(r.std(ddof=1)),
            )
        )
    return BucketFamily(name, tuple(out), cost_bps_round_trip, alpha)


def turn_of_month_labels(days: np.ndarray, window: int = 5) -> np.ndarray:
    """Label each bar by its trading-day offset from the month boundary.

    ``days`` must be an array of numpy datetime64 values in bar order. Offsets
    run ``-window .. -1`` for the final trading days of a month and ``+1 ..
    +window`` for the first of the next; everything else is ``"mid"``.

    Trading-day offsets, not calendar days: month-end rebalancing flow is keyed
    to sessions, and a calendar-day rule would smear the effect across
    weekends differently in different months.
    """
    months = days.astype("datetime64[M]")
    labels = np.full(len(days), "mid", dtype=object)
    boundaries = np.where(months[1:] != months[:-1])[0]  # last index of each month
    for b in boundaries:
        for k in range(1, window + 1):
            if b - k + 1 >= 0:
                labels[b - k + 1] = f"-{k}"
            if b + k < len(days):
                labels[b + k] = f"+{k}"
    return labels
