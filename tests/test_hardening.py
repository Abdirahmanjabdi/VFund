"""Tests for capacity limits and the drawdown circuit-breaker."""

import numpy as np

from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.synthetic import generate_gbm_panel
from vfund.strategy import CrossSectionalReversal
from vfund.strategy.cross_sectional import CrossSectionalStrategy


class AllShort(CrossSectionalStrategy):
    def scores(self, ctx):
        return -np.ones(ctx.closes.shape[1])


def _avg_gross(res):
    return res.equity_curve["gross_exposure"].to_numpy().mean()


def test_capacity_shrinks_book_at_large_aum():
    panel = generate_gbm_panel(8, 400, interval="1d", seed=1)  # ~$40k/day dollar vol
    small = CrossSectionalBacktester(
        panel, CrossSectionalReversal(2), rebalance_every=5, cost_bps=0, interval="1d",
        capacity_aum=1_000, max_participation=0.02,
    ).run()
    huge = CrossSectionalBacktester(
        panel, CrossSectionalReversal(2), rebalance_every=5, cost_bps=0, interval="1d",
        capacity_aum=1_000_000_000, max_participation=0.02,
    ).run()
    # At tiny AUM there's no real cap; at huge AUM positions are clipped hard.
    assert _avg_gross(huge) < _avg_gross(small)
    assert _avg_gross(huge) < 0.05


def test_dd_scale_curve():
    bt = CrossSectionalBacktester(
        generate_gbm_panel(5, 100, interval="1d", seed=1), CrossSectionalReversal(1),
        interval="1d", dd_derisk_start=0.15, dd_derisk_full=0.30, dd_derisk_floor=0.3,
    )
    assert bt._dd_scale(0.0) == 1.0            # no drawdown -> full exposure
    assert bt._dd_scale(-0.10) == 1.0          # shallow -> unaffected
    assert bt._dd_scale(-0.30) == 0.3          # deep -> floored
    mid = bt._dd_scale(-0.225)                 # halfway -> between floor and 1
    assert 0.3 < mid < 1.0


def test_circuit_breaker_cuts_losses_in_drawdown():
    # Short a rising market -> sustained drawdown; de-risking should lose less.
    panel = generate_gbm_panel(6, 400, interval="1d", market_vol=0.1, seed=3)
    off = CrossSectionalBacktester(
        panel, AllShort(), neutralize=False, rebalance_every=5, cost_bps=0, interval="1d"
    ).run()
    on = CrossSectionalBacktester(
        panel, AllShort(), neutralize=False, rebalance_every=5, cost_bps=0, interval="1d",
        dd_derisk_start=0.10, dd_derisk_full=0.30, dd_derisk_floor=0.2,
    ).run()
    assert on.final_equity >= off.final_equity
