"""Tests for the formulaic-alpha layer: operators, purity gate, zoo, bench.

The load-bearing test here is ``test_every_alpha_is_causal``: it corrupts the
future and asserts every registered alpha's past output is unchanged. The
operator vocabulary is *designed* so look-ahead cannot be expressed, and this
checks that design holds for all 41 alphas at once rather than trusting it.
"""

import numpy as np
import pytest

import vfund.factors.zoo  # noqa: F401  - registers the zoo
from vfund.factors import Panel, all_alphas, bench, information_coefficient
from vfund.factors.alpha import FormulaicStrategy, get
from vfund.factors.operators import (
    decay_linear,
    delay,
    delta,
    rank,
    safe_div,
    scale,
    ts_corr,
    ts_max,
    ts_mean,
    ts_rank,
    ts_std,
    zscore,
)
from vfund.factors.purity import PurityError, assert_pure, check_source


def _panel(T=400, N=12, seed=0, drift=None):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.03, (T, N))
    if drift is not None:
        steps += drift
    close = 100.0 * np.exp(np.cumsum(steps, axis=0))
    high = close * (1 + np.abs(rng.normal(0, 0.01, (T, N))))
    low = close * (1 - np.abs(rng.normal(0, 0.01, (T, N))))
    open_ = close * (1 + rng.normal(0, 0.005, (T, N)))
    volume = np.abs(rng.normal(1e6, 2e5, (T, N)))
    return Panel(close=close, open=open_, high=high, low=low, volume=volume,
                 symbols=[f"S{i}" for i in range(N)])


# --- operators ---------------------------------------------------------------


def test_rank_is_percentile_with_ties_averaged():
    x = np.array([[1.0, 2.0, 2.0, 4.0]])
    r = rank(x)
    # ranks 1, 2.5, 2.5, 4 over 4 valid -> /4
    np.testing.assert_allclose(r, [[0.25, 0.625, 0.625, 1.0]])


def test_rank_preserves_nan_and_never_invents_a_neutral_value():
    x = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan]])
    r = rank(x)
    assert np.isnan(r[0, 1])
    assert np.isnan(r[1]).all()          # all-NaN row stays all-NaN, not 0.5


def test_scale_l1_normalises_and_nans_a_zero_row():
    x = np.array([[1.0, -3.0], [0.0, 0.0]])
    s = scale(x)
    assert np.abs(s[0]).sum() == pytest.approx(1.0)
    assert np.isnan(s[1]).all()


def test_zscore_nans_a_constant_row():
    assert np.isnan(zscore(np.array([[2.0, 2.0, 2.0]]))).all()


def test_safe_div_nans_zero_denominator():
    out = safe_div(np.array([[1.0, 1.0]]), np.array([[0.0, 2.0]]))
    assert np.isnan(out[0, 0]) and out[0, 1] == pytest.approx(0.5)


def test_ts_mean_warmup_is_nan_then_correct():
    x = np.arange(5, dtype=float).reshape(5, 1)
    m = ts_mean(x, 3)
    assert np.isnan(m[:2, 0]).all()
    np.testing.assert_allclose(m[2:, 0], [1.0, 2.0, 3.0])


def test_ts_corr_nans_a_constant_window():
    x = np.arange(10, dtype=float).reshape(10, 1)
    const = np.ones((10, 1))
    assert np.isnan(ts_corr(x, const, 5)[-1, 0])


def test_ts_rank_is_a_percentile_of_the_window():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(5, 1)
    # last value is the largest in every full window -> 1.0
    assert ts_rank(x, 3)[-1, 0] == pytest.approx(1.0)


def test_decay_linear_weights_the_recent_bar_hardest():
    x = np.array([0.0, 0.0, 1.0]).reshape(3, 1)
    y = np.array([1.0, 0.0, 0.0]).reshape(3, 1)
    assert decay_linear(x, 3)[-1, 0] > decay_linear(y, 3)[-1, 0]


def test_delta_and_delay_refuse_non_positive_lag():
    x = _panel(50, 3).close
    for bad in (0, -1):
        with pytest.raises(ValueError, match="look-ahead ban"):
            delta(x, bad)
        with pytest.raises(ValueError, match="look-ahead ban"):
            delay(x, bad)


def test_ts_operators_are_causal():
    """Corrupting the tail must not change any earlier operator output."""
    x = _panel(300, 5, seed=3).close
    cut = 200
    y = x.copy()
    y[cut:] *= 7.5           # violent corruption of the future

    for fn in (lambda a: ts_mean(a, 20), lambda a: ts_std(a, 20),
               lambda a: ts_max(a, 20), lambda a: ts_rank(a, 20),
               lambda a: decay_linear(a, 20), lambda a: delta(a, 5)):
        base, corrupt = fn(x), fn(y)
        np.testing.assert_array_equal(base[:cut], corrupt[:cut])


# --- purity gate -------------------------------------------------------------


@pytest.mark.parametrize("src,rule", [
    ("def f(p):\n    import os\n    return p.close", "import"),
    ("def f(p):\n    return eval('1')", "forbidden"),
    ("def f(p):\n    return p.__class__", "dunder"),
    ("def f(p):\n    return p.close[::-1]", "time-reversal"),
    ("def f(p):\n    return np.roll(p.close, -1)", "numpy-call"),
    ("def f(p):\n    return p.close.shift(-1)", "method-call"),
    ("def f(p):\n    return mystery(p.close)", "unknown-call"),
])
def test_purity_gate_rejects(src, rule):
    violations = check_source(src)
    assert any(v.rule == rule for v in violations), (
        f"expected rule {rule!r}, got {[v.rule for v in violations]}"
    )


def test_purity_gate_allows_elementwise_numpy_and_operators():
    src = ("def f(p):\n"
           "    return np.where(p.close > 0, rank(delta(p.close, 5)), np.nan)")
    assert check_source(src) == []


def test_assert_pure_rejects_a_peeking_alpha():
    def peeker(p):
        return np.roll(p.close, -1)     # tomorrow's price, today

    with pytest.raises(PurityError, match="numpy-call"):
        assert_pure(peeker)


def test_every_registered_alpha_passes_the_gate():
    for a in all_alphas():
        assert_pure(a.fn, name=a.name)


# --- the zoo -----------------------------------------------------------------


def test_zoo_is_registered():
    alphas = all_alphas()
    assert len(alphas) >= 40
    assert len({a.name for a in alphas}) == len(alphas)      # unique names
    assert all(a.formula and a.source for a in alphas)       # provenance kept


def test_alphas_return_the_panel_shape_and_no_infinities():
    p = _panel(400, 10, seed=1)
    for a in all_alphas():
        out = a.compute(p)
        assert out.shape == p.shape, a.name
        assert not np.isinf(out).any(), a.name


def test_every_alpha_is_causal():
    """The sentinel: corrupt the future, assert every past score is identical.

    If any alpha reached forward - via a bad operator, a stray slice, or a
    formula transcription error - its pre-cut scores would move when the
    post-cut data changed. Nothing may move.
    """
    p = _panel(400, 10, seed=2)
    cut = 260
    corrupted = Panel(
        close=p.close.copy(), open=p.open.copy(), high=p.high.copy(),
        low=p.low.copy(), volume=p.volume.copy(), symbols=p.symbols,
    )
    for arr in (corrupted.close, corrupted.open, corrupted.high, corrupted.low):
        arr[cut:] *= 9.1
    corrupted.volume[cut:] *= 41.0

    for a in all_alphas():
        base = a.compute(p)[:cut]
        after = a.compute(corrupted)[:cut]
        both_nan = np.isnan(base) & np.isnan(after)
        assert np.array_equal(np.where(both_nan, 0.0, base),
                              np.where(both_nan, 0.0, after)), (
            f"{a.name} changed its past when the future changed - look-ahead"
        )


# --- bench -------------------------------------------------------------------


def test_bench_finds_a_planted_edge():
    """A synthetic panel with real mean reversion must score the reversal alpha alive.

    The mirror of the causality test: prove the bench *can* detect signal, so a
    'dead' verdict elsewhere means absence of edge rather than a broken metric.
    """
    rng = np.random.default_rng(11)
    T, N = 900, 20
    close = np.zeros((T, N))
    close[0] = 100.0
    prev = np.zeros(N)
    for t in range(1, T):
        shock = rng.normal(0, 0.03, N)
        step = shock - 0.45 * prev          # strong next-bar reversal
        close[t] = close[t - 1] * np.exp(step)
        prev = shock
    p = Panel(close=close, open=close, high=close, low=close,
              volume=np.full((T, N), 1e6), symbols=[f"S{i}" for i in range(N)])

    res = information_coefficient(get("academic_strev"), p)
    assert res.verdict == "alive", res
    assert res.mean_ic > 0.02 and res.t_stat > 2


def test_bench_reports_dead_on_pure_noise():
    """Random walks contain no cross-sectional edge; the bench must say so."""
    p = _panel(900, 20, seed=17)
    results = bench(all_alphas(), p)
    alive = [r for r in results if r.verdict == "alive"]
    # A handful of false positives is expected from multiple testing over 41
    # alphas; a large number would mean the metric is broken.
    assert len(alive) <= 4, [r.name for r in alive]


def test_forward_returns_do_not_leak_into_scores():
    from vfund.factors.bench import forward_returns

    close = np.array([[1.0], [2.0], [4.0], [8.0]])
    fwd = forward_returns(close, 1)
    np.testing.assert_allclose(fwd[:3, 0], [1.0, 1.0, 1.0])
    assert np.isnan(fwd[-1, 0])          # last bar has no known future


def test_formulaic_strategy_adapts_to_the_engine():
    from vfund.strategy.cross_sectional import PanelContext

    p = _panel(300, 8, seed=5)
    strat = FormulaicStrategy("academic_strev")
    ctx = PanelContext(299, p.close, p.symbols, volumes=p.volume)
    s = strat.scores(ctx)
    assert s.shape == (8,)
    assert np.isfinite(s).any()


def test_formulaic_strategy_returns_nan_during_warmup():
    from vfund.strategy.cross_sectional import PanelContext

    p = _panel(300, 8, seed=5)
    strat = FormulaicStrategy("academic_high52w")     # 370-bar warmup
    ctx = PanelContext(299, p.close, p.symbols, volumes=p.volume)
    assert np.isnan(strat.scores(ctx)).all()


def test_null_control_produces_no_false_positives():
    """The study's control: no alpha may score on data with no edge in it.

    This is what licenses the crypto study's headline. If the bench flagged
    alphas on independent zero-drift GBM with heterogeneous volatility, every
    "alive" verdict on real data would be suspect - the metric would be reading
    an artifact of ranking skewed returns rather than a signal.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from crypto_alpha_study import null_control

    results = bench(all_alphas(), null_control(60, 2000, seed=42))
    flagged = [r.name for r in results if r.verdict in ("alive", "reversed")]
    assert not flagged, f"bench found signal in pure noise: {flagged}"
