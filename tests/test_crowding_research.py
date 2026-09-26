"""Tests for the crowding-acceleration research layer.

All offline and synthetic — no network, no API keys.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from vfund.deriv_alpha.crowding import CrowdingAcceleration, _rank_cs
from vfund.deriv_alpha.tiers import TierDivergence, _safe_log, _ts_z
from vfund.deriv_alpha.vision import align_metrics
from vfund.research.ic import (
    decile_spread,
    forward_returns,
    ic_report,
    newey_west_t,
    spearman,
)
from vfund.strategy.cross_sectional import PanelContext


def _ctx(i, closes, symbols):
    return PanelContext(i, closes, symbols)


# --------------------------------------------------------------------------- #
# Ranking and normalisation primitives
# --------------------------------------------------------------------------- #
class TestPrimitives:
    def test_rank_is_centred_on_zero(self):
        r = _rank_cs(np.array([1.0, 2.0, 3.0, 4.0]))
        assert r.mean() == pytest.approx(0.0, abs=1e-12)

    def test_rank_orders_ascending(self):
        r = _rank_cs(np.array([3.0, 1.0, 2.0]))
        assert r[1] < r[2] < r[0]

    def test_rank_needs_two_valid_points(self):
        assert np.all(np.isnan(_rank_cs(np.array([5.0, np.nan]))))

    def test_safe_log_rejects_non_positive(self):
        out = _safe_log(np.array([1.0, 0.0, -2.0, np.nan]))
        assert out[0] == pytest.approx(0.0)
        assert np.isnan(out[1:]).all()

    def test_ts_z_is_backward_looking(self):
        # A spike at the final bar must not contaminate the trailing mean that
        # the same bar is scored against.
        mat = np.ones((40, 2))
        mat[39, 0] = 100.0
        z = _ts_z(mat, 39, window=30)
        assert z[0] > 3.0  # the spike reads as a large deviation
        assert np.isnan(z[1])  # a flat series has zero variance -> undefined

    def test_ts_z_requires_min_obs(self):
        mat = np.full((40, 2), np.nan)
        mat[38:, 0] = [1.0, 2.0]
        assert np.isnan(_ts_z(mat, 39, window=30)[0])


# --------------------------------------------------------------------------- #
# CrowdingAcceleration
# --------------------------------------------------------------------------- #
class TestCrowdingAcceleration:
    def test_validates_arguments(self):
        oi = np.ones((50, 3))
        with pytest.raises(ValueError, match="lookback"):
            CrowdingAcceleration(oi, lookback=1)
        with pytest.raises(ValueError, match="winsorize"):
            CrowdingAcceleration(oi, winsorize=0)
        with pytest.raises(ValueError, match="smooth"):
            CrowdingAcceleration(oi, smooth=0)

    def test_nan_before_enough_history(self):
        oi = np.ones((50, 3))
        f = CrowdingAcceleration(oi, lookback=7)
        assert np.all(np.isnan(f.scores(_ctx(10, np.ones((50, 3)), list("abc")))))

    def test_accelerating_oi_scores_negative(self):
        """The validated sign: accelerating build-up is the SHORT leg."""
        T, N, lb = 60, 3, 5
        oi = np.ones((T, N)) * 1e9
        # symbol 0 accelerates (quadratic), symbol 2 decelerates (sqrt)
        for t in range(T):
            oi[t, 0] = 1e9 * (1.0 + 0.0008 * t**2)
            oi[t, 2] = 1e9 * (1.0 + 0.05 * np.sqrt(max(t, 1)))
        f = CrowdingAcceleration(oi, lookback=lb)
        s = f.scores(_ctx(40, np.ones((T, N)) * 100, ["ACC", "FLAT", "DECEL"]))
        assert s[0] < s[2], "accelerating name must rank below decelerating"

    def test_winsorize_tames_a_new_listing(self):
        """A near-zero OI base can print an unbounded growth ratio."""
        T, N, lb = 60, 4, 5
        oi = np.ones((T, N)) * 1e9
        oi[:, 0] = 1.0            # near-zero base
        oi[30:, 0] = 1e9          # then explodes
        f = CrowdingAcceleration(oi, lookback=lb, winsorize=5.0)
        s = f.scores(_ctx(40, np.ones((T, N)) * 100, list("abcd")))
        assert np.isfinite(s).all()
        assert np.abs(s).max() <= 0.5 + 1e-9

    def test_smoothing_reduces_score_churn(self):
        """Smoothing is a turnover control — it must actually damp changes."""
        rng = np.random.default_rng(0)
        T, N = 120, 8
        oi = 1e9 * np.exp(np.cumsum(rng.normal(0, 0.05, (T, N)), axis=0))
        closes = np.ones((T, N)) * 100
        syms = [f"S{i}" for i in range(N)]

        def churn(smooth):
            f = CrowdingAcceleration(oi, lookback=5, smooth=smooth)
            prev, tot = None, 0.0
            for t in range(40, T):
                s = f.scores(_ctx(t, closes, syms))
                if prev is not None and np.isfinite(s).all():
                    tot += float(np.abs(s - prev).sum())
                prev = s
            return tot

        assert churn(10) < churn(1)

    def test_smooth_handles_all_nan_window(self):
        oi = np.full((40, 3), np.nan)
        f = CrowdingAcceleration(oi, lookback=5, smooth=5)
        s = f.scores(_ctx(30, np.ones((40, 3)), list("abc")))
        assert np.all(np.isnan(s))


# --------------------------------------------------------------------------- #
# Tier factors
# --------------------------------------------------------------------------- #
class TestTierFactors:
    def test_divergence_ranks_informed_tilt(self):
        T, N = 100, 3
        informed = np.ones((T, N))
        crowd = np.ones((T, N))
        informed[:, 0] = np.linspace(1.0, 3.0, T)  # big money turning long
        crowd[:, 2] = np.linspace(1.0, 3.0, T)     # crowd turning long
        f = TierDivergence(informed, crowd, window=60)
        s = f.scores(_ctx(90, np.ones((T, N)) * 100, list("abc")))
        assert s[0] > s[2]

    def test_window_validation(self):
        with pytest.raises(ValueError, match="window"):
            TierDivergence(np.ones((50, 2)), np.ones((50, 2)), window=5)


# --------------------------------------------------------------------------- #
# IC harness
# --------------------------------------------------------------------------- #
class TestICHarness:
    def test_newey_west_penalises_autocorrelation(self):
        rng = np.random.default_rng(3)
        # Strongly autocorrelated series with a positive mean: the naive t-stat
        # overstates significance, NW must pull it down.
        e = rng.normal(0, 1, 800)
        x = np.empty(800)
        x[0] = e[0]
        for i in range(1, 800):
            x[i] = 0.9 * x[i - 1] + e[i]
        x = x + 0.5
        n = len(x)
        t_naive = x.mean() / (x.std(ddof=1) / np.sqrt(n))
        assert abs(newey_west_t(x, lag=20)) < abs(t_naive)

    def test_newey_west_needs_data(self):
        assert np.isnan(newey_west_t(np.array([1.0, 2.0]), lag=1))

    def test_spearman_detects_perfect_ordering(self):
        a = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        b = np.array([[10.0, 20.0, 30.0, 40.0, 50.0]])
        assert spearman(a, b)[0] == pytest.approx(1.0)

    def test_spearman_nan_when_too_few_names(self):
        a = np.array([[1.0, 2.0, np.nan, np.nan, np.nan]])
        assert np.isnan(spearman(a, a, min_names=5)[0])

    def test_forward_returns_have_no_lookahead_at_tail(self):
        closes = np.arange(1, 11, dtype=float).reshape(10, 1)
        fwd = forward_returns(closes, 3)
        assert np.isnan(fwd[-3:]).all()
        assert fwd[0, 0] == pytest.approx(4 / 1 - 1)

    def test_ic_report_flags_perfect_predictor(self):
        rng = np.random.default_rng(7)
        T, N = 300, 10
        closes = np.ones((T, N))
        rets = rng.normal(0, 0.02, (T, N))
        for t in range(1, T):
            closes[t] = closes[t - 1] * (1 + rets[t])
        # Score = next bar's return plus noise: a strong but imperfect oracle.
        # A noiseless one has zero IC variance, so its t-stat is undefined.
        scores = np.full((T, N), np.nan)
        scores[:-1] = rets[1:] + rng.normal(0, 0.02, (T - 1, N))
        r = ic_report("oracle", scores, closes, horizon=1, n_subperiods=3)
        assert r.mean_ic > 0.3
        assert r.verdict == "alive"

    def test_ic_report_calls_noise_dead(self):
        rng = np.random.default_rng(11)
        T, N = 300, 10
        closes = np.cumprod(1 + rng.normal(0, 0.02, (T, N)), axis=0)
        scores = rng.normal(0, 1, (T, N))
        assert ic_report("noise", scores, closes, horizon=1).verdict == "dead"

    def test_ic_report_insufficient_data(self):
        closes = np.ones((5, 3))
        assert ic_report("tiny", np.ones((5, 3)), closes).verdict == "insufficient"


class TestDecileSpread:
    """The gate that rank IC cannot provide."""

    def _build(self, T, N, rets):
        closes = np.ones((T, N))
        for t in range(1, T):
            closes[t] = closes[t - 1] * (1 + rets[t])
        return closes

    def test_detects_a_real_spread(self):
        rng = np.random.default_rng(31)
        T, N = 400, 40
        scores = rng.normal(0, 1, (T, N))
        # Forward return genuinely increasing in the score.
        rets = np.zeros((T, N))
        rets[1:] = 0.01 * scores[:-1] + rng.normal(0, 0.005, (T - 1, N))
        closes = self._build(T, N, rets)
        d = decile_spread("real", scores, closes, horizon=1, n_buckets=5)
        assert d.spread > 0
        assert d.spread_t > 3
        assert not d.tail_driven

    def test_flags_a_tail_driven_factor(self):
        """Correct ordering of the median, destroyed by the extremes.

        This is the failure mode that a rank IC cannot see: the factor sorts the
        typical case correctly while the bottom bucket carries rare enormous
        gains that flip the mean.
        """
        rng = np.random.default_rng(32)
        T, N = 600, 40
        scores = rng.normal(0, 1, (T, N))
        rets = np.zeros((T, N))
        # Median effect: high score -> mildly positive next return.
        rets[1:] = 0.002 * scores[:-1]
        # Tail: the lowest-scoring names occasionally explode upward.
        low = scores[:-1] < np.quantile(scores[:-1], 0.1)
        shock = (rng.random((T - 1, N)) < 0.02) & low
        rets[1:][shock] += 1.5
        closes = self._build(T, N, rets)
        d = decile_spread("tail", scores, closes, horizon=1, n_buckets=10)
        assert d.median_spread > 0, "the median ordering should look right"
        assert d.spread < 0, "the mean should be flipped by the tail"
        assert d.tail_driven

    def test_insufficient_data(self):
        d = decile_spread("tiny", np.ones((5, 3)), np.ones((5, 3)))
        assert d.n_obs == 0
        assert np.isnan(d.spread)

    def test_respects_min_names(self):
        rng = np.random.default_rng(33)
        T, N = 200, 8  # fewer names than min_names
        scores = rng.normal(0, 1, (T, N))
        closes = self._build(T, N, rng.normal(0, 0.01, (T, N)))
        assert decile_spread("thin", scores, closes, min_names=20).n_obs == 0


# --------------------------------------------------------------------------- #
# Metric alignment
# --------------------------------------------------------------------------- #
class TestAlignMetrics:
    def _metrics(self):
        return pl.DataFrame(
            {
                "timestamp": [
                    pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 2),
                ],
                "symbol": ["BTCUSDT", "BTCUSDT"],
                "oi_value": [100.0, 200.0],
            }
        ).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us"))
        ) if False else pl.DataFrame(
            {
                "timestamp": np.array(
                    ["2024-01-01", "2024-01-02"], dtype="datetime64[us]"
                ),
                "symbol": ["BTCUSDT", "BTCUSDT"],
                "oi_value": [100.0, 200.0],
            }
        )

    def _ts(self, n=4):
        return pl.Series(
            "timestamp",
            np.array(
                [f"2024-01-{d:02d}" for d in range(1, n + 1)], dtype="datetime64[us]"
            ),
        )

    def test_lag_shifts_series_back(self):
        m, ts = self._metrics(), self._ts()
        a0 = align_metrics(m, ts, ["BTCUSDT"], ("oi_value",), lag=0)["oi_value"]
        a1 = align_metrics(m, ts, ["BTCUSDT"], ("oi_value",), lag=1)["oi_value"]
        assert a0[0, 0] == pytest.approx(100.0)
        assert np.isnan(a1[0, 0])
        assert a1[1, 0] == pytest.approx(100.0)

    def test_ffill_limit_stops_carrying_stale_data(self):
        m, ts = self._metrics(), self._ts(6)
        a = align_metrics(m, ts, ["BTCUSDT"], ("oi_value",), ffill_limit=1)["oi_value"]
        assert a[2, 0] == pytest.approx(200.0)  # carried one bar
        assert np.isnan(a[3, 0])                # then stops

    def test_bare_symbol_maps_to_usdt_pair(self):
        m, ts = self._metrics(), self._ts()
        a = align_metrics(m, ts, ["BTC"], ("oi_value",))["oi_value"]
        assert a[0, 0] == pytest.approx(100.0)

    def test_unknown_symbol_is_all_nan(self):
        m, ts = self._metrics(), self._ts()
        a = align_metrics(m, ts, ["NOPE"], ("oi_value",))["oi_value"]
        assert np.isnan(a).all()

    def test_negative_lag_rejected(self):
        with pytest.raises(ValueError, match="lag"):
            align_metrics(self._metrics(), self._ts(), ["BTCUSDT"], ("oi_value",), lag=-1)
