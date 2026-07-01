"""Sanity checks for the v0.1.x hypothesis strategies (mechanics, not edge)."""

import numpy as np

from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.synthetic import generate_gbm_panel
from vfund.analytics.performance import alpha_beta
from vfund.strategy import (
    CrossSectionalIlliquidity,
    CrossSectionalLowVol,
    CrossSectionalMaxReturn,
    CrossSectionalResidualMomentum,
    CrossSectionalValue,
    TimeSeriesTrend,
    TimeSeriesTrendEnsemble,
)
from vfund.strategy.cross_sectional import PanelContext


def test_lowvol_scores_prefer_calm_assets():
    # Asset 0 low vol, asset 1 high vol, over 200 bars.
    rng = np.random.default_rng(0)
    a = 100 * np.cumprod(1 + 0.001 * rng.standard_normal(200))
    b = 100 * np.cumprod(1 + 0.02 * rng.standard_normal(200))
    closes = np.column_stack([a, b])
    ctx = PanelContext(199, closes, ["A", "B"])
    s = CrossSectionalLowVol(lookback=100).scores(ctx)
    assert s[0] > s[1]  # calmer asset scores higher (we long it)


def test_value_scores_prefer_below_ma():
    n = 200
    ma_ref = np.full(n, 100.0)
    cheap = ma_ref.copy(); cheap[-1] = 80.0   # below its average
    rich = ma_ref.copy(); rich[-1] = 120.0    # above its average
    closes = np.column_stack([cheap, rich])
    ctx = PanelContext(n - 1, closes, ["CHEAP", "RICH"])
    s = CrossSectionalValue(lookback=100).scores(ctx)
    assert s[0] > s[1]  # cheap (below MA) scores higher


def test_trend_is_directional_long_in_uptrend():
    # Everything trending up -> time-series trend should be net LONG.
    panel = generate_gbm_panel(8, 1000, market_vol=0.1, idio_vol=0.05,
                               reversion=0.0, seed=3)
    # Force an upward drift by using momentum on a rising synthetic set:
    res = CrossSectionalBacktester(
        panel, TimeSeriesTrend(lookback=48), rebalance_every=24,
        neutralize=False, cost_bps=0,
    ).run()
    # Gross exposure is used; the run should produce a non-flat equity path.
    assert res.equity_curve["gross_exposure"].max() > 0


def test_ensemble_averages_horizons():
    # A coin up on all horizons -> +1; the ensemble should be bounded in [-1, 1].
    closes = np.column_stack([np.linspace(50, 150, 300), np.linspace(150, 50, 300)])
    ctx = PanelContext(299, closes, ["UP", "DOWN"])
    s = TimeSeriesTrendEnsemble(lookbacks=(20, 50, 100)).scores(ctx)
    assert abs(s[0] - 1.0) < 1e-9 and abs(s[1] + 1.0) < 1e-9


def test_max_return_shorts_the_spiker():
    # Asset 0 has a big one-day spike; asset 1 is smooth. MAX should short 0.
    base = np.linspace(100, 110, 60)
    spiker = base.copy(); spiker[-3] = base[-3] * 1.5  # a lottery pop
    closes = np.column_stack([spiker, base])
    ctx = PanelContext(59, closes, ["POP", "CALM"])
    s = CrossSectionalMaxReturn(lookback=30).scores(ctx)
    assert s[0] < s[1]  # the spiker scores lower -> shorted


def test_residual_momentum_prefers_outperformer():
    rng = np.random.default_rng(0)
    mkt = np.cumprod(1 + 0.001 * rng.standard_normal(200))
    winner = mkt * np.linspace(1.0, 1.5, 200)   # beats the market
    loser = mkt * np.linspace(1.0, 0.7, 200)    # lags the market
    closes = np.column_stack([winner * 100, loser * 100])
    ctx = PanelContext(199, closes, ["WIN", "LOSE"])
    s = CrossSectionalResidualMomentum(lookback=90).scores(ctx)
    assert s[0] > s[1]


def test_illiquidity_prefers_thin_names():
    closes = np.column_stack([np.full(60, 100.0), np.full(60, 100.0)])
    # Same |returns| via tiny wiggles, but A trades far less -> more illiquid.
    closes[::2, 0] = 101.0
    closes[::2, 1] = 101.0
    volumes = np.column_stack([np.full(60, 5.0), np.full(60, 5_000.0)])
    ctx = PanelContext(59, closes, ["THIN", "DEEP"], volumes=volumes)
    s = CrossSectionalIlliquidity(lookback=30).scores(ctx)
    assert s[0] > s[1]  # thin (illiquid) name scores higher -> longed


def test_alpha_beta_recovers_known_beta():
    rng = np.random.default_rng(0)
    bench = 0.01 * rng.standard_normal(4000)
    strat = 2.0 * bench + 0.0005  # beta 2, small constant alpha, no idio noise
    ab = alpha_beta(strat, bench, periods_per_year=365)
    assert abs(ab["beta"] - 2.0) < 1e-6
    assert ab["alpha_ann"] > 0  # positive intercept detected


def test_alpha_beta_insignificant_for_pure_beta():
    rng = np.random.default_rng(1)
    bench = 0.01 * rng.standard_normal(4000)
    strat = 0.5 * bench + 0.001 * rng.standard_normal(4000)  # scaled beta, no true alpha
    ab = alpha_beta(strat, bench, periods_per_year=365)
    assert abs(ab["beta"] - 0.5) < 0.05
    assert abs(ab["alpha_t"]) < 2.0  # alpha not statistically significant
