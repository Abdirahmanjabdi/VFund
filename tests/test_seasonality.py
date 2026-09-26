"""Tests for the seasonality harness and the Yahoo data layer."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from vfund.data.yahoo import INTERVALS, _epoch
from vfund.research.seasonality import (
    Bucket,
    BucketFamily,
    bucket_returns,
    turn_of_month_labels,
)


class TestBucketFamily:
    def _fam(self, buckets, n=None, cost=0.0):
        return BucketFamily("test", tuple(buckets), cost)

    def test_threshold_tightens_with_more_tests(self):
        """More buckets means a stricter bar — that is the whole point."""
        few = self._fam([Bucket(str(i), 0.0, 1.0, 100, 0.5, 0.0, 0.01) for i in range(3)])
        many = self._fam([Bucket(str(i), 0.0, 1.0, 100, 0.5, 0.0, 0.01) for i in range(24)])
        assert many.t_threshold > few.t_threshold
        assert few.t_threshold > 2.0  # even 3 tests demand more than the naive 1.96

    def test_survivor_needs_both_significance_and_size(self):
        big_t_small_effect = Bucket("a", 0.00001, 8.0, 500, 0.6, 0.0, 0.01)   # 0.1 bps
        big_t_big_effect = Bucket("b", 0.0010, 8.0, 500, 0.6, 0.0, 0.01)      # 10 bps
        small_t_big_effect = Bucket("c", 0.0010, 1.0, 500, 0.6, 0.0, 0.01)
        fam = self._fam([big_t_small_effect, big_t_big_effect, small_t_big_effect],
                        cost=2.0)
        labels = [b.label for b in fam.survivors()]
        assert labels == ["b"], "only the significant AND economically large one"

    def test_nan_t_never_survives(self):
        fam = self._fam([Bucket("a", 0.01, float("nan"), 5, 0.5, 0.0, 0.01)], cost=0.0)
        assert fam.survivors() == []

    def test_ann_bps_conversion(self):
        assert Bucket("a", 0.0015, 3.0, 100, 0.5, 0.0, 0.01).ann_bps == pytest.approx(15.0)


class TestBucketReturns:
    def test_finds_a_planted_effect(self):
        rng = np.random.default_rng(5)
        n = 4000
        labels = np.array(["A", "B", "C", "D"] * (n // 4))
        r = rng.normal(0, 0.005, n)
        r[labels == "B"] += 0.004  # 40 bps, unmissable
        fam = bucket_returns(r, labels, name="planted", cost_bps_round_trip=2.0)
        surv = fam.survivors()
        assert len(surv) == 1 and surv[0].label == "B"

    def test_rejects_pure_noise(self):
        rng = np.random.default_rng(6)
        n = 4000
        labels = np.array([f"h{i%24:02d}" for i in range(n)])
        fam = bucket_returns(rng.normal(0, 0.005, n), labels, name="noise",
                            cost_bps_round_trip=1.0)
        assert fam.survivors() == []

    def test_effect_below_cost_is_not_a_survivor(self):
        rng = np.random.default_rng(7)
        n = 20000
        labels = np.array(["A", "B"] * (n // 2))
        r = rng.normal(0, 0.002, n)
        r[labels == "B"] += 0.00005  # 0.5 bps: statistically findable, not tradeable
        fam = bucket_returns(r, labels, name="tiny", cost_bps_round_trip=2.0)
        assert fam.survivors() == []

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same shape"):
            bucket_returns(np.zeros(10), np.array(["a"] * 9), name="x")

    def test_small_bucket_gets_nan_t(self):
        labels = np.array(["A"] * 100 + ["B"] * 5)
        r = np.random.default_rng(8).normal(0, 0.01, 105)
        fam = bucket_returns(r, labels, name="x", min_obs=30)
        b = {x.label: x for x in fam.buckets}
        assert np.isnan(b["B"].t_stat)
        assert np.isfinite(b["A"].t_stat)

    def test_nan_returns_dropped(self):
        labels = np.array(["A"] * 100)
        r = np.full(100, np.nan)
        r[:60] = 0.001
        fam = bucket_returns(r, labels, name="x", min_obs=30)
        assert fam.buckets[0].n == 60


class TestTurnOfMonth:
    def test_labels_straddle_the_boundary(self):
        days = np.arange(np.datetime64("2024-01-25"), np.datetime64("2024-02-08"))
        lab = turn_of_month_labels(days, window=3)
        # 2024-01-31 is the last January day in this contiguous range
        i_last = int(np.where(days == np.datetime64("2024-01-31"))[0][0])
        assert lab[i_last] == "-1"
        assert lab[i_last - 1] == "-2"
        assert lab[i_last + 1] == "+1"
        assert lab[i_last + 3] == "+3"

    def test_middle_days_are_mid(self):
        days = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-03-01"))
        lab = turn_of_month_labels(days, window=3)
        i = int(np.where(days == np.datetime64("2024-01-15"))[0][0])
        assert lab[i] == "mid"

    def test_window_size_respected(self):
        days = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-03-01"))
        lab = turn_of_month_labels(days, window=2)
        assert "-3" not in set(lab.tolist())
        assert "-2" in set(lab.tolist())


class TestYahooLayer:
    def test_known_intervals_carry_limits(self):
        assert INTERVALS["1h"] == 730
        assert INTERVALS["1d"] is None  # unbounded, needs explicit period bounds

    def test_epoch_assumes_utc_for_naive(self):
        from datetime import datetime, timezone

        naive = _epoch(datetime(2024, 1, 1))
        aware = _epoch(datetime(2024, 1, 1, tzinfo=timezone.utc))
        assert naive == aware

    def test_epoch_parses_iso_string(self):
        from datetime import datetime, timezone

        assert _epoch("2024-01-01") == _epoch(
            datetime(2024, 1, 1, tzinfo=timezone.utc)
        )

    def test_unknown_interval_rejected(self):
        from vfund.data.yahoo import fetch

        with pytest.raises(ValueError, match="unknown interval"):
            fetch("GBPUSD=X", interval="3h")
