"""Tests for the Derivative Microstructure Alpha sub-project.

All tests use synthetic data — no network calls, no API keys needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from vfund.deriv_alpha.factors import (
    CrowdingPressure,
    FundingMomentum,
    LiquidationBounce,
    OIAcceleration,
    OIPriceDivergence,
    _rank_cs,
)
from vfund.deriv_alpha.regime import FundingDispersionRegime, Regime
from vfund.deriv_alpha.composer import DerivAlphaComposer
from vfund.strategy.cross_sectional import PanelContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synth(T: int = 100, N: int = 10, seed: int = 42):
    """Build synthetic closes, funding, volumes, and OI for testing."""
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    funding = rng.normal(0.0001, 0.0005, (T, N))
    volumes = rng.uniform(1e6, 1e8, (T, N))
    oi = 1e9 * np.exp(np.cumsum(rng.normal(0, 0.03, (T, N)), axis=0))
    symbols = [f"SYM{i}" for i in range(N)]
    return closes, funding, volumes, oi, symbols


def _ctx(i, closes, symbols, funding=None, volumes=None):
    return PanelContext(i, closes, symbols, funding=funding, volumes=volumes)


# ---------------------------------------------------------------------------
# _rank_cs
# ---------------------------------------------------------------------------


class TestRankCS:
    def test_ordering(self):
        x = np.array([3.0, 1.0, 2.0])
        r = _rank_cs(x)
        assert r[1] < r[2] < r[0]

    def test_in_unit_interval(self):
        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        r = _rank_cs(x)
        assert np.all(r > 0) and np.all(r <= 1)

    def test_nan_preserved(self):
        x = np.array([1.0, np.nan, 3.0])
        r = _rank_cs(x)
        assert np.isnan(r[1])
        assert np.isfinite(r[0]) and np.isfinite(r[2])

    def test_all_nan(self):
        r = _rank_cs(np.array([np.nan, np.nan]))
        assert np.all(np.isnan(r))

    def test_single_value_sits_at_its_own_median(self):
        # A lone observation is the midpoint of its own distribution, not the
        # top of it — the midrank convention puts it at 0.5.
        r = _rank_cs(np.array([5.0]))
        assert r[0] == 0.5

    def test_ranks_are_unbiased(self):
        # Distinct values must average to 0.5, so that centring by subtracting
        # 0.5 actually produces a zero-mean score.
        r = _rank_cs(np.array([1.0, 2.0, 3.0, 4.0]))
        assert r.mean() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# OIPriceDivergence
# ---------------------------------------------------------------------------


class TestOIPriceDivergence:
    def test_returns_finite_scores(self):
        closes, funding, volumes, oi, symbols = _synth()
        factor = OIPriceDivergence(oi, lookback=10)
        ctx = _ctx(50, closes, symbols)
        scores = factor.scores(ctx)
        assert scores.shape == (10,)
        assert np.any(np.isfinite(scores))

    def test_nan_for_short_history(self):
        _, _, _, oi, symbols = _synth()
        closes = np.ones((100, 10))
        factor = OIPriceDivergence(oi, lookback=10)
        ctx = _ctx(5, closes, symbols)
        assert np.all(np.isnan(factor.scores(ctx)))

    def test_lookback_validation(self):
        with pytest.raises(ValueError, match="lookback"):
            OIPriceDivergence(np.zeros((10, 5)), lookback=1)

    def test_divergence_direction(self):
        T, N = 50, 3
        closes = np.ones((T, N)) * 100.0
        oi = np.ones((T, N)) * 1e9
        # Step must be INSIDE the lookback window (i=45, lb=10 → past=35)
        # Symbol 0: OI doubles but price flat -> should score high
        oi[38:, 0] = 2e9
        # Symbol 2: price doubles but OI flat -> should score low
        closes[38:, 2] = 200.0
        factor = OIPriceDivergence(oi, lookback=10)
        ctx = _ctx(45, closes, ["A", "B", "C"])
        scores = factor.scores(ctx)
        assert scores[0] > scores[2]


# ---------------------------------------------------------------------------
# CrowdingPressure
# ---------------------------------------------------------------------------


class TestCrowdingPressure:
    def test_needs_funding(self):
        _, _, _, oi, symbols = _synth()
        closes = np.ones((100, 10))
        factor = CrowdingPressure(oi)
        ctx = _ctx(50, closes, symbols, funding=None)
        with pytest.raises(ValueError, match="funding"):
            factor.scores(ctx)

    def test_returns_scores(self):
        closes, funding, _, oi, symbols = _synth()
        factor = CrowdingPressure(oi, lookback=14)
        ctx = _ctx(50, closes, symbols, funding=funding)
        scores = factor.scores(ctx)
        assert scores.shape == (10,)
        assert np.any(np.isfinite(scores))

    def test_crowded_coins_score_low(self):
        T, N = 50, 3
        closes = np.ones((T, N)) * 100
        oi = np.ones((T, N)) * 1e9
        funding = np.zeros((T, N))
        # Symbol 0: high funding + growing OI = crowded
        funding[:, 0] = 0.01
        oi[35:, 0] = 2e9  # inside lookback (i=45, lb=14 → past=31)
        # Symbol 2: low funding + flat OI = uncrowded
        funding[:, 2] = -0.001
        factor = CrowdingPressure(oi, lookback=14)
        ctx = _ctx(45, closes, ["CROWD", "MID", "EMPTY"], funding=funding)
        scores = factor.scores(ctx)
        assert scores[0] < scores[2]


# ---------------------------------------------------------------------------
# LiquidationBounce
# ---------------------------------------------------------------------------


class TestLiquidationBounce:
    def test_zero_when_no_liquidation(self):
        T, N = 30, 5
        closes = np.ones((T, N)) * 100
        oi = np.ones((T, N)) * 1e9
        factor = LiquidationBounce(oi, lookback=5)
        ctx = _ctx(20, closes, [f"S{i}" for i in range(N)])
        scores = factor.scores(ctx)
        np.testing.assert_array_almost_equal(scores, 0.0)

    def test_positive_after_crash(self):
        T, N = 30, 3
        closes = np.ones((T, N)) * 100
        oi = np.ones((T, N)) * 1e9
        # Step inside lookback window (i=25, lb=5 → past=20)
        closes[22:, 0] = 80.0
        oi[22:, 0] = 7e8
        factor = LiquidationBounce(oi, lookback=5)
        ctx = _ctx(25, closes, ["CRASH", "FLAT1", "FLAT2"])
        scores = factor.scores(ctx)
        assert scores[0] > scores[1]
        assert scores[1] == 0.0

    def test_price_drop_without_oi_drop_is_zero(self):
        T, N = 30, 2
        closes = np.ones((T, N)) * 100
        oi = np.ones((T, N)) * 1e9
        # Price drops but OI stays (fundamental, not liquidation)
        closes[22:, 0] = 70.0
        factor = LiquidationBounce(oi, lookback=5)
        ctx = _ctx(25, closes, ["NEWS", "FLAT"])
        scores = factor.scores(ctx)
        assert scores[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# FundingMomentum
# ---------------------------------------------------------------------------


class TestFundingMomentum:
    def test_needs_funding(self):
        closes = np.ones((100, 5))
        factor = FundingMomentum(lookback=7)
        ctx = _ctx(50, closes, [f"S{i}" for i in range(5)], funding=None)
        with pytest.raises(ValueError, match="funding"):
            factor.scores(ctx)

    def test_returns_scores(self):
        closes, funding, _, _, symbols = _synth()
        factor = FundingMomentum(lookback=7)
        ctx = _ctx(50, closes, symbols, funding=funding)
        scores = factor.scores(ctx)
        assert scores.shape == (10,)

    def test_rising_funding_scores_low(self):
        T, N = 50, 3
        closes = np.ones((T, N)) * 100
        funding = np.zeros((T, N))
        # Symbol 0: funding rising from 0 to 0.01
        for t in range(T):
            funding[t, 0] = 0.001 * t / T
        # Symbol 2: funding falling
        for t in range(T):
            funding[t, 2] = 0.01 * (1 - t / T)
        factor = FundingMomentum(lookback=7)
        ctx = _ctx(40, closes, ["RISING", "FLAT", "FALLING"], funding=funding)
        scores = factor.scores(ctx)
        assert scores[0] < scores[2]


# ---------------------------------------------------------------------------
# OIAcceleration
# ---------------------------------------------------------------------------


class TestOIAcceleration:
    def test_detects_acceleration(self):
        T, N = 50, 3
        closes = np.ones((T, N)) * 100
        oi = np.ones((T, N)) * 1e9
        # Symbol 0: accelerating OI (quadratic growth)
        for t in range(T):
            oi[t, 0] = 1e9 * (1.0 + 0.0005 * t**2)
        # Symbol 2: decelerating OI (sqrt growth)
        for t in range(T):
            oi[t, 2] = 1e9 * (1.0 + 0.05 * np.sqrt(max(t, 1)))
        factor = OIAcceleration(oi, lookback=7)
        ctx = _ctx(40, closes, ["ACC", "FLAT", "DECEL"])
        scores = factor.scores(ctx)
        assert scores[0] > scores[2]

    def test_nan_for_short_history(self):
        _, _, _, oi, symbols = _synth()
        closes = np.ones((100, 10))
        factor = OIAcceleration(oi, lookback=7)
        ctx = _ctx(10, closes, symbols)
        assert np.all(np.isnan(factor.scores(ctx)))


# ---------------------------------------------------------------------------
# Regime detector
# ---------------------------------------------------------------------------


class TestRegimeDetector:
    def test_neutral_for_short_history(self):
        det = FundingDispersionRegime(window=30)
        funding = np.zeros((10, 5))
        assert det.detect(funding) == Regime.NEUTRAL

    def test_neutral_for_none(self):
        det = FundingDispersionRegime(window=30)
        assert det.detect(None) == Regime.NEUTRAL

    def test_weights_sum_differs_by_regime(self):
        det = FundingDispersionRegime()
        tw = det.weights(Regime.TRENDING)
        rw = det.weights(Regime.REVERTING)
        assert tw["funding_momentum"] > rw["funding_momentum"]
        assert rw["crowding_pressure"] > tw["crowding_pressure"]

    def test_window_validation(self):
        with pytest.raises(ValueError, match="window"):
            FundingDispersionRegime(window=2)


# ---------------------------------------------------------------------------
# Composer (end to end)
# ---------------------------------------------------------------------------


class TestComposer:
    def test_end_to_end(self):
        closes, funding, volumes, oi, symbols = _synth()
        composer = DerivAlphaComposer(oi)
        ctx = _ctx(80, closes, symbols, funding=funding, volumes=volumes)
        scores = composer.scores(ctx)
        assert scores.shape == (10,)
        assert np.any(np.isfinite(scores))

    def test_scores_centered_around_zero(self):
        closes, funding, volumes, oi, symbols = _synth()
        composer = DerivAlphaComposer(oi)
        ctx = _ctx(80, closes, symbols, funding=funding, volumes=volumes)
        scores = composer.scores(ctx)
        valid = scores[np.isfinite(scores)]
        assert valid.mean() == pytest.approx(0.0, abs=0.15)

    def test_without_funding_still_works(self):
        closes, _, volumes, oi, symbols = _synth()
        composer = DerivAlphaComposer(oi)
        ctx = _ctx(80, closes, symbols, funding=None, volumes=volumes)
        scores = composer.scores(ctx)
        assert scores.shape == (10,)

    def test_factor_names(self):
        oi = np.ones((50, 5))
        composer = DerivAlphaComposer(oi)
        assert len(composer.factor_names) == 5
        assert "oi_price_divergence" in composer.factor_names


# ---------------------------------------------------------------------------
# Market adapter interface
# ---------------------------------------------------------------------------


class TestCryptoPerpsAdapter:
    def test_requires_fetch_before_positioning(self):
        from vfund.deriv_alpha.markets import CryptoPerpsAdapter
        import polars as pl

        adapter = CryptoPerpsAdapter()
        with pytest.raises(RuntimeError, match="fetch"):
            adapter.positioning(pl.Series("ts", []), [])

    def test_name(self):
        from vfund.deriv_alpha.markets import CryptoPerpsAdapter

        assert CryptoPerpsAdapter().name == "crypto_perps"


class TestUSEquityAdapter:
    def test_not_implemented(self):
        from vfund.deriv_alpha.markets import USEquityAdapter

        adapter = USEquityAdapter()
        with pytest.raises(NotImplementedError):
            adapter.fetch([], "2024-01-01", "2024-12-31")


class TestFuturesAdapter:
    def test_not_implemented(self):
        from vfund.deriv_alpha.markets import FuturesAdapter

        adapter = FuturesAdapter()
        with pytest.raises(NotImplementedError):
            adapter.fetch([], "2024-01-01", "2024-12-31")
