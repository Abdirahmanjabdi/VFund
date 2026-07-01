"""Tests for vol-targeting and the size factor."""

import numpy as np

from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.synthetic import generate_gbm_panel
from vfund.strategy import CrossSectionalSize, TimeSeriesTrend
from vfund.strategy.cross_sectional import PanelContext


def test_size_scores_prefer_low_dollar_volume():
    closes = np.column_stack([np.full(50, 100.0), np.full(50, 100.0)])
    volumes = np.column_stack([np.full(50, 10.0), np.full(50, 1_000.0)])  # A small, B large
    ctx = PanelContext(49, closes, ["A", "B"], volumes=volumes)
    s = CrossSectionalSize(lookback=20).scores(ctx)
    assert s[0] > s[1]  # smaller (lower dollar volume) scores higher -> long it


def test_vol_targeting_controls_realised_vol():
    panel = generate_gbm_panel(10, 2000, market_vol=0.8, idio_vol=0.9, seed=4)
    plain = CrossSectionalBacktester(
        panel, TimeSeriesTrend(48), rebalance_every=5, neutralize=False, cost_bps=0
    ).run()
    targeted = CrossSectionalBacktester(
        panel, TimeSeriesTrend(48), rebalance_every=5, neutralize=False, cost_bps=0,
        vol_target=0.20, vol_lookback=30, interval="1h",
    ).run()

    def ann_vol(res):
        eq = res.equity_curve["equity"].to_numpy()
        r = eq[1:] / eq[:-1] - 1.0
        return r.std() * np.sqrt(res.bars_per_year)

    # Targeting to 20% should land far closer to 20% than the un-targeted book.
    assert abs(ann_vol(targeted) - 0.20) < abs(ann_vol(plain) - 0.20)


def test_size_needs_volume_context():
    import pytest

    closes = np.column_stack([np.full(50, 100.0), np.full(50, 100.0)])
    ctx = PanelContext(49, closes, ["A", "B"])  # no volumes
    with pytest.raises(ValueError):
        CrossSectionalSize(lookback=20).scores(ctx)
