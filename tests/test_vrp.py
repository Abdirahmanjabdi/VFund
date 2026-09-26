"""Tests for the variance risk premium layer."""

from __future__ import annotations

import numpy as np
import pytest

from vfund.vol.vrp import (
    MONTH_BARS,
    capped_pnl,
    non_overlapping,
    realized_vol,
    variance_swap_pnl,
    vrp_stats,
)


class TestRealizedVol:
    def test_recovers_a_known_volatility(self):
        rng = np.random.default_rng(1)
        target = 0.20  # 20% annualised
        r = rng.normal(0, target / np.sqrt(252), 5000)
        rv = realized_vol(r, horizon=252, forward=True)
        got = np.nanmean(rv)
        assert got == pytest.approx(target * 100, rel=0.1)

    def test_forward_window_excludes_the_current_bar(self):
        r = np.zeros(100)
        r[50] = 10.0  # one enormous move at bar 50
        rv = realized_vol(r, horizon=10, forward=True)
        # bar 50 looks at 51..60, which is flat, so it must not see its own shock
        assert rv[50] == pytest.approx(0.0, abs=1e-9)
        # bar 45 looks at 46..55, which contains it
        assert rv[45] > 0

    def test_backward_window_excludes_the_current_bar(self):
        r = np.zeros(100)
        r[50] = 10.0
        rv = realized_vol(r, horizon=10, forward=False)
        assert rv[50] == pytest.approx(0.0, abs=1e-9)  # looks at 40..49
        assert rv[55] > 0                              # looks at 45..54

    def test_tail_is_nan_when_window_runs_off_the_end(self):
        rv = realized_vol(np.zeros(30), horizon=21, forward=True)
        assert np.isnan(rv[-1])

    def test_horizon_validated(self):
        with pytest.raises(ValueError, match="horizon"):
            realized_vol(np.zeros(50), horizon=1)


class TestVarianceSwapPnl:
    def test_seller_profits_when_implied_exceeds_realized(self):
        p = variance_swap_pnl(np.array([20.0]), np.array([15.0]))
        assert p[0] > 0

    def test_seller_loses_when_realized_exceeds_implied(self):
        p = variance_swap_pnl(np.array([20.0]), np.array([30.0]))
        assert p[0] < 0

    def test_zero_when_they_match(self):
        assert variance_swap_pnl(np.array([20.0]), np.array([20.0]))[0] == pytest.approx(0.0)

    def test_expressed_in_vega_equivalent_points(self):
        # implied 20, realised 10 -> (400-100)/40 = 7.5 vol points
        assert variance_swap_pnl(np.array([20.0]), np.array([10.0]))[0] == pytest.approx(7.5)

    def test_non_positive_implied_is_nan(self):
        assert np.isnan(variance_swap_pnl(np.array([0.0]), np.array([10.0]))[0])


class TestCappedPnl:
    def test_floors_the_loss(self):
        out = capped_pnl(np.array([-50.0]), max_loss=5.0, wing_cost=0.0)
        assert out[0] == pytest.approx(-5.0)

    def test_gives_up_part_of_the_gain(self):
        out = capped_pnl(np.array([10.0]), max_loss=5.0, wing_cost=0.35)
        assert out[0] == pytest.approx(6.5)

    def test_small_losses_pass_through_untouched(self):
        out = capped_pnl(np.array([-2.0]), max_loss=5.0, wing_cost=0.35)
        assert out[0] == pytest.approx(-2.0)

    def test_capping_raises_risk_adjusted_return(self):
        """The point of the wing: the tail is nearly all of the variance."""
        rng = np.random.default_rng(4)
        p = rng.normal(1.5, 3.0, 400)
        p[::40] = -50.0  # rare catastrophes
        raw_ir = p.mean() / p.std()
        cap = capped_pnl(p, max_loss=5.0, wing_cost=0.35)
        assert cap.mean() / cap.std() > raw_ir

    def test_validates_arguments(self):
        with pytest.raises(ValueError, match="max_loss"):
            capped_pnl(np.array([1.0]), max_loss=0.0)
        with pytest.raises(ValueError, match="wing_cost"):
            capped_pnl(np.array([1.0]), wing_cost=1.0)


class TestVrpStats:
    def test_measures_a_planted_premium(self):
        rng = np.random.default_rng(6)
        n = 2000
        iv = rng.normal(20, 3, n)
        rv = iv - 2.5 + rng.normal(0, 2, n)   # implied exceeds realised by 2.5
        s = vrp_stats(iv, rv, nw_lag=21)
        assert s.premium == pytest.approx(2.5, abs=0.3)
        assert s.pct_positive > 0.8
        assert s.t_stat > 5

    def test_reports_no_premium_when_there_is_none(self):
        rng = np.random.default_rng(7)
        n = 2000
        iv = rng.normal(20, 3, n)
        rv = iv + rng.normal(0, 2, n)
        s = vrp_stats(iv, rv, nw_lag=21)
        assert abs(s.t_stat) < 2.5

    def test_premium_as_percent_of_implied(self):
        s = vrp_stats(np.full(100, 20.0), np.full(100, 18.0), nw_lag=1)
        assert s.premium_pct_of_implied == pytest.approx(10.0)

    def test_insufficient_data(self):
        s = vrp_stats(np.ones(5), np.ones(5))
        assert s.n < 30 and np.isnan(s.premium)

    def test_newey_west_lag_matters_for_overlapping_windows(self):
        """A monthly quantity sampled daily is autocorrelated by construction."""
        rng = np.random.default_rng(8)
        base = rng.normal(2.0, 1.0, 200)
        overlapping = np.repeat(base, 21)[:2000]  # crude overlap
        iv = np.full(2000, 20.0)
        rv = iv - overlapping
        naive = vrp_stats(iv, rv, nw_lag=1).t_stat
        corrected = vrp_stats(iv, rv, nw_lag=21).t_stat
        assert abs(corrected) < abs(naive)


class TestNonOverlapping:
    def test_takes_one_observation_per_horizon(self):
        v = np.arange(100, dtype=float)
        out = non_overlapping(v, horizon=21)
        assert len(out) == len(range(0, 100, 21))
        assert out[0] == 0.0 and out[1] == 21.0

    def test_skips_nans_before_sampling(self):
        v = np.full(100, np.nan)
        v[50:60] = 1.0
        out = non_overlapping(v, horizon=5)
        assert np.isfinite(out).all()

    def test_month_bars_constant(self):
        assert MONTH_BARS == 21
