import numpy as np

from vfund.data.synthetic import generate_gbm_panel
from vfund.research.robustness import (
    norm_cdf,
    norm_ppf,
    probabilistic_sharpe_ratio,
    expected_max_sharpe,
    deflated_sharpe_ratio,
    subperiod_stability,
    universe_bootstrap,
)
from vfund.strategy import CrossSectionalReversal


def test_norm_ppf_inverts_cdf():
    for p in (0.01, 0.25, 0.5, 0.84, 0.99):
        assert abs(norm_cdf(norm_ppf(p)) - p) < 1e-4
    assert abs(norm_ppf(0.5)) < 1e-6


def test_psr_zero_edge_is_half():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(5000) * 0.01
    r = r - r.mean()  # force realised Sharpe to exactly 0
    assert abs(probabilistic_sharpe_ratio(r) - 0.5) < 1e-6


def test_psr_rises_with_sample_size():
    rng = np.random.default_rng(1)
    small = 0.0005 + 0.01 * rng.standard_normal(200)
    big = 0.0005 + 0.01 * rng.standard_normal(20000)
    assert probabilistic_sharpe_ratio(big) > probabilistic_sharpe_ratio(small)


def test_deflated_is_stricter_than_probabilistic():
    rng = np.random.default_rng(2)
    best = 0.0008 + 0.01 * rng.standard_normal(5000)
    trials = np.array([_raw for _raw in rng.normal(0, 0.03, 40)])  # many noisy trial Sharpes
    psr = probabilistic_sharpe_ratio(best, 0.0)
    dsr = deflated_sharpe_ratio(best, trials)
    assert dsr <= psr  # accounting for many trials can only lower confidence


def test_expected_max_sharpe_grows_with_trials():
    rng = np.random.default_rng(3)
    few = rng.normal(0, 0.05, 5)
    many = rng.normal(0, 0.05, 200)
    assert expected_max_sharpe(many) > expected_max_sharpe(few)


def test_subperiod_and_universe_shapes():
    panel = generate_gbm_panel(12, 2400, reversion=0.2, seed=5)
    bt = dict(rebalance_every=1, cost_bps=0.0, interval="1h")
    sp = subperiod_stability(
        panel, lambda: CrossSectionalReversal(1), n_periods=4, backtest_kwargs=bt
    )
    assert sp.height == 4
    draws = universe_bootstrap(
        panel, lambda: CrossSectionalReversal(1),
        n_draws=6, subset_size=8, backtest_kwargs=bt, seed=1,
    )
    assert draws.shape == (6,)
    # A real injected edge should be positive in most random sub-universes.
    assert float(np.mean(draws > 0)) > 0.5
