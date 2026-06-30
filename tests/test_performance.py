import numpy as np

from vfund.analytics.performance import (
    max_drawdown,
    sharpe_ratio,
    cagr,
    equity_returns,
)


def test_max_drawdown_known_series():
    # Peak 100 -> trough 50 == -50% drawdown.
    equity = np.array([100.0, 120.0, 60.0, 90.0, 50.0, 80.0])
    mdd, peak_i, trough_i = max_drawdown(equity)
    assert peak_i == 1  # the 120 peak
    assert trough_i == 4  # the 50 trough
    assert abs(mdd - (50.0 / 120.0 - 1.0)) < 1e-12


def test_max_drawdown_monotonic_up_is_zero():
    equity = np.array([10.0, 11.0, 12.0, 13.0])
    mdd, _, _ = max_drawdown(equity)
    assert mdd == 0.0


def test_sharpe_sign_follows_drift():
    rng = np.random.default_rng(0)
    up = 0.001 + 0.005 * rng.standard_normal(5000)
    down = -0.001 + 0.005 * rng.standard_normal(5000)
    assert sharpe_ratio(up, periods_per_year=8760) > 0
    assert sharpe_ratio(down, periods_per_year=8760) < 0


def test_cagr_doubles_in_one_year():
    # 365 daily steps doubling -> ~100% CAGR.
    equity = np.linspace(100.0, 200.0, 366)
    assert abs(cagr(equity, periods_per_year=365) - 1.0) < 0.02


def test_equity_returns_length():
    assert equity_returns(np.array([1.0, 2.0, 3.0])).shape == (2,)
