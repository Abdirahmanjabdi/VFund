"""Tests for the flow-decomposition factors and the Vision listing parser."""

from __future__ import annotations

import numpy as np
import pytest

from vfund.deriv_alpha.flow import (
    FlowConfirmation,
    ForcedFlowReversal,
    PositionChurn,
    _rank_cs,
    _ts_z,
)
from vfund.deriv_alpha.vision import _DATED
from vfund.strategy.cross_sectional import PanelContext


def _ctx(i, closes, symbols):
    return PanelContext(i, closes, symbols)


def _panel(T=120, N=4, seed=0):
    rng = np.random.default_rng(seed)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    oi = 1e9 * np.exp(np.cumsum(rng.normal(0, 0.03, (T, N)), axis=0))
    return closes, oi, [f"S{i}" for i in range(N)]


class TestVisionListingParser:
    def test_accepts_a_real_key(self):
        m = _DATED.search(
            "data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2023-06-15.zip"
        )
        assert m and m.group(1) == "2023-06-15"

    def test_rejects_the_mislabelled_1993_object(self):
        # These exist in the dump for MATICUSDT and TRBUSDT and sort first.
        assert _DATED.search(
            "data/futures/um/daily/metrics/MATICUSDT/MATICUSDT-metrics-1993.zip"
        ) is None

    def test_rejects_checksum_sidecar(self):
        assert _DATED.search(
            "data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2023-06-15.zip.CHECKSUM"
        ) is None


class TestFlowQuadrants:
    """The sign algebra is the whole thesis, so pin all four quadrants."""

    def _score_for(self, price_dir, oi_dir):
        """Build a panel where symbol 0 moves as specified and others are flat."""
        T, N, lb = 80, 5, 1
        rng = np.random.default_rng(1)
        # Give every symbol a little noise so trailing z-scores are defined.
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, (T, N)), axis=0))
        oi = 1e9 * np.exp(np.cumsum(rng.normal(0, 0.005, (T, N)), axis=0))
        i = 70
        closes[i, 0] = closes[i - 1, 0] * (1 + 0.15 * price_dir)
        oi[i, 0] = oi[i - 1, 0] * (1 + 0.30 * oi_dir)
        f = FlowConfirmation(oi, lookback=lb, window=60)
        return f.scores(_ctx(i, closes, [f"S{j}" for j in range(N)]))[0]

    def test_price_up_oi_up_is_positive(self):
        assert self._score_for(+1, +1) > 0  # new longs — follow

    def test_price_up_oi_down_is_negative(self):
        assert self._score_for(+1, -1) < 0  # short covering — fade

    def test_price_down_oi_down_is_positive(self):
        assert self._score_for(-1, -1) > 0  # capitulation — fade the fall

    def test_price_down_oi_up_is_negative(self):
        assert self._score_for(-1, +1) < 0  # new shorts — follow the fall


class TestForcedFlowReversal:
    def test_ignores_rising_open_interest(self):
        """Assets whose OI is rising must contribute no view."""
        T, N = 80, 4
        rng = np.random.default_rng(2)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, N)), axis=0))
        # Rising, but with genuine variation so the z-score is well defined.
        oi = 1e9 * np.exp(
            np.cumsum(np.abs(rng.normal(0.02, 0.005, (T, N))), axis=0)
        )
        f = ForcedFlowReversal(oi, lookback=1, window=60)
        f._inner._prepare(closes)
        raw = f.raw(70)
        assert np.allclose(raw[np.isfinite(raw)], 0.0, atol=1e-12)

    def test_steadily_growing_oi_yields_no_view(self):
        """Perfectly steady growth is never position closing — score exactly 0."""
        T, N = 80, 4
        rng = np.random.default_rng(9)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, N)), axis=0))
        oi = 1e9 * np.exp(np.cumsum(np.full((T, N), 0.01), axis=0))
        f = ForcedFlowReversal(oi, lookback=1, window=60)
        f._inner._prepare(closes)
        assert np.allclose(f.raw(70), 0.0, atol=1e-12)

    def test_fades_a_move_on_falling_oi(self):
        T, N, i = 80, 5, 70
        rng = np.random.default_rng(3)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, (T, N)), axis=0))
        oi = 1e9 * np.exp(np.cumsum(rng.normal(0, 0.005, (T, N)), axis=0))
        closes[i, 0] = closes[i - 1, 0] * 1.15   # rose
        oi[i, 0] = oi[i - 1, 0] * 0.70           # on closing positions
        f = ForcedFlowReversal(oi, lookback=1, window=60)
        s = f.scores(_ctx(i, closes, [f"S{j}" for j in range(N)]))
        assert s[0] < 0  # short-covering rally — fade it


class TestFlowMechanics:
    def test_validates_arguments(self):
        _, oi, _ = _panel()
        with pytest.raises(ValueError, match="lookback"):
            FlowConfirmation(oi, lookback=0)
        with pytest.raises(ValueError, match="window"):
            FlowConfirmation(oi, window=5)
        with pytest.raises(ValueError, match="smooth"):
            FlowConfirmation(oi, smooth=0)
        with pytest.raises(ValueError, match="sign"):
            PositionChurn(oi, sign=0.0)

    def test_clip_bounds_the_product(self):
        """Without clipping, one outlier day dominates the cross-section."""
        T, N, i = 80, 5, 70
        rng = np.random.default_rng(4)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, (T, N)), axis=0))
        oi = 1e9 * np.exp(np.cumsum(rng.normal(0, 0.005, (T, N)), axis=0))
        closes[i, 0] *= 20.0   # absurd outlier
        oi[i, 0] *= 20.0
        f = FlowConfirmation(oi, lookback=1, window=60, clip=3.0)
        f._prepare(closes)
        assert np.nanmax(np.abs(f.raw(i))) <= 9.0 + 1e-9  # clip * clip

    def test_smoothing_damps_score_churn(self):
        closes, oi, syms = _panel(T=150, N=8, seed=5)

        def churn(sm):
            f = FlowConfirmation(oi, lookback=1, window=60, smooth=sm)
            prev, tot = None, 0.0
            for t in range(80, 150):
                s = f.scores(_ctx(t, closes, syms))
                if prev is not None and np.isfinite(s).all():
                    tot += float(np.abs(s - prev).sum())
                prev = s
            return tot

        assert churn(10) < churn(1)

    def test_scores_are_centred(self):
        closes, oi, syms = _panel()
        f = FlowConfirmation(oi, lookback=1, window=60)
        s = f.scores(_ctx(100, closes, syms))
        assert np.nanmean(s) == pytest.approx(0.0, abs=1e-9)

    def test_all_nan_oi_yields_no_scores(self):
        closes, _, syms = _panel()
        f = FlowConfirmation(np.full(closes.shape, np.nan), lookback=1, window=60)
        assert np.all(np.isnan(f.scores(_ctx(100, closes, syms))))

    def test_ts_z_rejects_degenerate_variance(self):
        """A near-constant series must not produce a huge z from rounding error."""
        rng = np.random.default_rng(21)
        mat = np.empty((60, 3))
        mat[:, 0] = 1e9                                  # exactly constant
        # Large level, spread ~1e-12 of it: below the relative floor.
        mat[:, 1] = 1e9 + rng.normal(0, 1e-3, 60)
        mat[:, 2] = 1e9 + rng.normal(0, 1e7, 60)         # genuine spread
        z = _ts_z(mat, 59, window=30)
        assert np.isnan(z[0]), "constant series must give no view"
        assert np.isnan(z[1]), "spread below the relative floor is rounding error"
        assert np.isfinite(z[2])

    def test_ts_z_ignores_all_nan_columns(self):
        mat = np.full((60, 3), np.nan)
        mat[:, 1] = np.arange(60, dtype=float)
        z = _ts_z(mat, 59, window=30)
        assert np.isnan(z[0]) and np.isnan(z[2])
        assert np.isfinite(z[1])

    def test_rank_is_zero_mean(self):
        r = _rank_cs(np.array([5.0, 1.0, 3.0, 9.0]))
        assert r.mean() == pytest.approx(0.0, abs=1e-12)


class TestPositionChurn:
    def test_sign_flips_the_ranking(self):
        churn = np.tile(np.linspace(0.1, 0.5, 4), (80, 1))
        churn += np.random.default_rng(6).normal(0, 0.01, (80, 4))
        closes = np.ones((80, 4)) * 100
        syms = [f"S{i}" for i in range(4)]
        neg = PositionChurn(churn, sign=-1.0).scores(_ctx(70, closes, syms))
        pos = PositionChurn(churn, sign=+1.0).scores(_ctx(70, closes, syms))
        assert np.allclose(neg, -pos, equal_nan=True)
