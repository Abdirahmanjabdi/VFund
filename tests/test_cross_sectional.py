"""The decisive test: the engine profits on edge that is really there, and does
NOT manufacture edge on data that has none. That is what makes it a microscope
and not a mirror.
"""

from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.synthetic import generate_gbm_panel
from vfund.strategy import CrossSectionalMomentum, CrossSectionalReversal


def _run(panel, strat, cost_bps=0.0):
    return CrossSectionalBacktester(
        panel, strat, rebalance_every=1, leverage=1.0, cost_bps=cost_bps
    ).run()


def test_reversal_beats_random_when_edge_exists():
    edge = generate_gbm_panel(15, 3000, reversion=0.25, seed=1)
    none = generate_gbm_panel(15, 3000, reversion=0.0, seed=1)

    on_edge = _run(edge, CrossSectionalReversal(lookback=1))
    on_noise = _run(none, CrossSectionalReversal(lookback=1))

    # With a real reversal effect the strategy should make money...
    assert on_edge.final_equity > on_edge.initial_cash
    # ...and clearly more than on structureless data.
    assert on_edge.final_equity > on_noise.final_equity


def test_momentum_loses_on_a_reversal_market():
    edge = generate_gbm_panel(15, 3000, reversion=0.25, seed=2)
    # Momentum is the wrong hypothesis here; it should bleed.
    res = _run(edge, CrossSectionalMomentum(lookback=1))
    assert res.final_equity < res.initial_cash


def test_costs_erode_the_edge():
    edge = generate_gbm_panel(15, 3000, reversion=0.25, seed=3)
    free = _run(edge, CrossSectionalReversal(lookback=1), cost_bps=0.0)
    costed = _run(edge, CrossSectionalReversal(lookback=1), cost_bps=20.0)
    assert costed.final_equity < free.final_equity


def test_result_shapes():
    panel = generate_gbm_panel(10, 500, reversion=0.1, seed=4)
    res = _run(panel, CrossSectionalReversal(lookback=2))
    assert res.equity_curve.height == 500
    assert "gross_exposure" in res.equity_curve.columns
    assert res.n_trades > 0
