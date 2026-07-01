"""Correctness tests for the simulation primitive.

`simulate_py` is the specification the Rust core must match, so these also pin
down the native implementation. The key test reproduces the full engine's equity
curve from the primitive.
"""

import numpy as np

from vfund.backtest.construct import scores_to_weights
from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.backtest.sim import simulate, simulate_py
from vfund.data.synthetic import generate_gbm_panel
from vfund.strategy import CrossSectionalReversal
from vfund.strategy.cross_sectional import PanelContext


def test_buy_and_hold_hand_computed():
    # Asset 0 returns +10%/bar from t=1; go fully long it at t=1.
    rets = np.array([[0.0, 0.0], [0.0, 0.0], [0.1, 0.0], [0.1, 0.0]])
    eq, turn = simulate_py(rets, np.array([1]), np.array([[1.0, 0.0]]),
                           initial=100.0, cost_rate=0.0)
    assert np.allclose(eq, [100.0, 100.0, 110.0, 121.0])
    assert np.allclose(turn, [1.0])


def test_flat_book_is_constant():
    rng = np.random.default_rng(0)
    rets = rng.standard_normal((50, 4)) * 0.01
    rets[0] = 0
    eq, _ = simulate_py(rets, np.array([1]), np.zeros((1, 4)), initial=1000.0)
    assert np.allclose(eq, 1000.0)


def test_matches_full_engine():
    panel = generate_gbm_panel(6, 200, interval="1d", reversion=0.2, seed=3)
    strat = CrossSectionalReversal(2)
    bt = CrossSectionalBacktester(panel, strat, rebalance_every=5, cost_bps=10,
                                  interval="1d")
    res = bt.run()

    C, V, symbols = bt.closes, bt.volumes, bt.symbols
    rets = np.zeros_like(C)
    rets[1:] = C[1:] / C[:-1] - 1.0
    rets = np.where(np.isfinite(rets), rets, 0.0)

    reb_idx, reb_w = [], []
    for t in range(1, C.shape[0]):
        if t % 5 == 0:
            ctx = PanelContext(t, C, symbols, volumes=V)
            reb_idx.append(t)
            reb_w.append(scores_to_weights(strat.scores(ctx), leverage=1.0, neutralize=True))

    eq, _ = simulate_py(rets, np.array(reb_idx), np.array(reb_w),
                        initial=bt.initial_cash, cost_rate=10 / 10_000.0)
    assert np.allclose(eq, res.equity_curve["equity"].to_numpy())


def test_dispatch_falls_back_to_python():
    rets = np.zeros((10, 3))
    rets[1:] = 0.01
    a = simulate(rets, np.array([1]), np.ones((1, 3)) / 3, initial=100.0)
    b = simulate_py(rets, np.array([1]), np.ones((1, 3)) / 3, initial=100.0)
    assert np.allclose(a[0], b[0])
