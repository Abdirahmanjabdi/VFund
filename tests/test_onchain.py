"""Tests for on-chain (TVL) signal mechanics."""

import numpy as np
import pytest

from vfund.strategy import TVLDivergence, TVLMomentum
from vfund.strategy.cross_sectional import PanelContext


def _ctx(closes, tvl, n):
    return PanelContext(n - 1, closes, ["A", "B"], tvl=tvl)


def test_tvl_momentum_prefers_growing_tvl():
    closes = np.column_stack([np.full(120, 100.0), np.full(120, 100.0)])
    tvl = np.column_stack([np.linspace(1e6, 3e6, 120), np.linspace(3e6, 1e6, 120)])
    s = TVLMomentum(lookback=60).scores(_ctx(closes, tvl, 120))
    assert s[0] > s[1]  # growing TVL scores higher (long it)


def test_tvl_divergence_longs_cheap_fundamentals():
    # A: TVL doubled, price flat (cheap). B: price doubled, TVL flat (rich).
    tvl = np.column_stack([np.linspace(1e6, 2e6, 120), np.full(120, 1e6)])
    closes = np.column_stack([np.full(120, 100.0), np.linspace(100, 200, 120)])
    s = TVLDivergence(lookback=60).scores(_ctx(closes, tvl, 120))
    assert s[0] > s[1]  # usage grew but price didn't -> long


def test_tvl_needs_data():
    closes = np.column_stack([np.full(120, 100.0), np.full(120, 100.0)])
    ctx = PanelContext(119, closes, ["A", "B"])  # no tvl
    with pytest.raises(ValueError):
        TVLMomentum(lookback=30).scores(ctx)
