"""Research/live parity: the live book must BE the engine's book.

Why this file exists
--------------------
The single most expensive class of bug in a trading system is not a wrong
strategy — it is a *right* strategy whose live implementation quietly differs
from the one that was validated. The backtest stays honest, the live signal
disagrees with it, and nothing in the output looks wrong.

VFund had exactly that. ``combined_book`` re-implemented the weight pipeline
instead of asking the engine, and drifted in two ways:

* it never applied the capacity cap, so with ``capacity_aum`` configured the
  live book was up to **1.75x too large** (gross 1.000 vs the engine's 0.573);
* it vol-targeted *before* masking un-shortable names instead of after, so on
  the vol-targeted trend sleeve **all 30 names were wrong** and gross came out
  at 0.910x the engine's.

Both are invisible to a reader and to every other test in the suite. So parity
is asserted directly, across the overlay combinations that actually differ, and
a regression becomes a CI failure rather than a wrong position.

The rule being enforced: **live code chooses parameters, never steps.** The
engine owns the overlay chain and its order.
"""

import numpy as np
import pytest

from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.synthetic import generate_gbm_panel
from vfund.live.signal import _engine_weights, combined_book
from vfund.strategy import CrossSectionalSize, TimeSeriesTrendEnsemble


def _panel(n_assets=14, n_bars=420, seed=7):
    return generate_gbm_panel(n_assets, n_bars, interval="1d",
                              reversion=0.1, seed=seed)


def _engine_book(panel, strategy, **kw):
    """Weights the engine would target on the final bar."""
    bt = CrossSectionalBacktester(panel, strategy, rebalance_every=7,
                                  interval="1d", cost_bps=10, **kw)
    return _engine_weights(bt)


def _assert_same(engine_w, live_w, label):
    names = sorted(set(engine_w) | set(live_w))
    bad = [(n, engine_w.get(n, 0.0), live_w.get(n, 0.0)) for n in names
           if abs(engine_w.get(n, 0.0) - live_w.get(n, 0.0)) > 1e-9]
    assert not bad, (
        f"{label}: live book diverged from the engine on {len(bad)}/{len(names)} "
        f"names. First few: {bad[:3]}"
    )


# Overlay combinations. Each row turns on a different part of the chain
# (shortability -> vol-target -> capacity); the capacity rows are the ones the
# old hand-rolled path got wrong, and the vol-target rows caught the ordering bug.
SHORT_DV = 5_000_000.0
# combined_book always vol-targets the trend sleeve, so every case keeps
# vol_target on and varies the overlays around it. Vol-targeting is the overlay
# whose *ordering* caused the original bug, so it belongs in all of them anyway.
TREND_CASES = [
    pytest.param({"vol_target": 0.30}, id="trend-voltarget"),
    pytest.param({"vol_target": 0.15}, id="trend-voltarget-tight"),
    pytest.param({"vol_target": 0.30, "min_short_dollar_volume": SHORT_DV},
                 id="trend-voltarget+shortability"),
    pytest.param({"vol_target": 0.30, "capacity_aum": 1_000_000.0},
                 id="trend-voltarget+capacity"),
    pytest.param({"vol_target": 0.30, "min_short_dollar_volume": SHORT_DV,
                  "capacity_aum": 5_000_000.0}, id="trend-all-overlays"),
    # A vol target this high makes the scaler want more leverage than allowed,
    # so max_leverage clamps. Without a binding case, mutating max_leverage is
    # unobservable and that branch of the overlay chain goes untested.
    pytest.param({"vol_target": 5.0}, id="trend-maxleverage-binding"),
]
SIZE_CASES = [
    pytest.param({}, id="size-bare"),
    pytest.param({"min_short_dollar_volume": SHORT_DV}, id="size-shortability"),
    pytest.param({"capacity_aum": 5_000_000.0}, id="size-capacity"),
    pytest.param({"min_short_dollar_volume": SHORT_DV,
                  "capacity_aum": 2_000_000.0}, id="size-shortability+capacity"),
]


@pytest.mark.parametrize("overlays", TREND_CASES)
def test_trend_sleeve_matches_engine(overlays):
    panel = _panel()
    engine_kw = dict(neutralize=False, vol_lookback=30, max_leverage=3.0, **overlays)
    engine_w = _engine_book(panel, TimeSeriesTrendEnsemble(), **engine_kw)

    live = combined_book(
        panel, interval="1d", trend_weight=1.0,
        vol_target=overlays["vol_target"],
        vol_lookback=30, max_leverage=3.0,
        min_short_dollar_volume=overlays.get("min_short_dollar_volume"),
        capacity_aum=overlays.get("capacity_aum"),
    )
    _assert_same(engine_w, live.weights, f"trend {overlays}")


@pytest.mark.parametrize("overlays", SIZE_CASES)
def test_size_sleeve_matches_engine(overlays):
    panel = _panel()
    engine_w = _engine_book(panel, CrossSectionalSize(20), top_k=5, **overlays)
    live = combined_book(
        panel, interval="1d", trend_weight=0.0, size_lookback=20, top_k=5,
        min_short_dollar_volume=overlays.get("min_short_dollar_volume"),
        capacity_aum=overlays.get("capacity_aum"),
    )
    _assert_same(engine_w, live.weights, f"size {overlays}")


def test_blended_book_is_the_weighted_sum_of_engine_sleeves():
    """The 50/50 blend must equal 0.5 x each engine sleeve, name by name."""
    panel = _panel()
    kw = dict(min_short_dollar_volume=SHORT_DV, capacity_aum=5_000_000.0)
    trend_w = _engine_book(panel, TimeSeriesTrendEnsemble(), neutralize=False,
                           vol_target=0.30, vol_lookback=30, max_leverage=3.0, **kw)
    size_w = _engine_book(panel, CrossSectionalSize(20), top_k=5, **kw)

    expected = {}
    for alloc, sleeve in ((0.5, trend_w), (0.5, size_w)):
        for s, x in sleeve.items():
            expected[s] = expected.get(s, 0.0) + alloc * x
    expected = {s: w for s, w in expected.items() if abs(w) > 1e-6}

    live = combined_book(panel, interval="1d", trend_weight=0.5, size_lookback=20,
                         top_k=5, vol_target=0.30, vol_lookback=30,
                         max_leverage=3.0, **kw)
    _assert_same(expected, live.weights, "blended 50/50")


def test_capacity_actually_binds():
    """Guard the guard: a tight capacity must shrink the book.

    If capacity silently did nothing, every parity test above would still pass
    while proving nothing about the overlay that caused the original bug.
    """
    panel = _panel()
    loose = combined_book(panel, interval="1d", trend_weight=0.0, top_k=5)
    tight = combined_book(panel, interval="1d", trend_weight=0.0, top_k=5,
                          capacity_aum=500_000_000.0)
    assert tight.gross < loose.gross, (
        "capacity_aum did not reduce gross exposure - the cap is not binding, "
        "so the parity tests that use it prove nothing"
    )


def test_max_leverage_actually_binds():
    """A binding leverage cap must be observable, or its parity case is vacuous.

    Found by mutation testing: perturbing ``max_leverage`` left every parity
    test green, because with a modest vol target the cap never engaged.
    """
    panel = _panel()
    common = dict(interval="1d", trend_weight=1.0, vol_target=5.0, vol_lookback=30)
    low = combined_book(panel, max_leverage=1.0, **common)
    high = combined_book(panel, max_leverage=3.0, **common)
    assert high.gross > low.gross + 1e-9, (
        "max_leverage did not change gross exposure - the cap is not binding, "
        "so the parity case using it proves nothing"
    )


def test_shortability_actually_binds():
    """Same guard for the hard-to-short gate."""
    panel = _panel()
    free = combined_book(panel, interval="1d", trend_weight=0.0, top_k=5)
    gated = combined_book(panel, interval="1d", trend_weight=0.0, top_k=5,
                          min_short_dollar_volume=1e15)   # nothing is shortable
    short_free = sum(1 for w in free.weights.values() if w < 0)
    short_gated = sum(1 for w in gated.weights.values() if w < 0)
    assert short_free > 0, "baseline had no shorts; test cannot discriminate"
    assert short_gated < short_free, "shortability gate did not remove any short"


def test_alpha_book_uses_the_engine_helper():
    """alpha_book and combined_book must share the one weight path.

    Cheap structural check: if someone re-introduces a hand-rolled pipeline,
    the imports come back and this fails loudly with a pointed message.
    """
    import vfund.live.signal as sig

    src = __import__("inspect").getsource(sig)
    for banned in ("scores_to_weights", "vol_scale_weights",
                   "short_liquidity_mask", "trailing_dollar_volume"):
        assert banned not in src, (
            f"vfund/live/signal.py calls {banned}() directly. Live code must ask "
            f"the engine for weights (_engine_weights), never rebuild the overlay "
            f"chain - that is exactly how research/live parity was lost before."
        )


def test_engine_weights_never_returns_nan_or_inf():
    panel = _panel()
    bt = CrossSectionalBacktester(panel, CrossSectionalSize(20), rebalance_every=7,
                                  interval="1d", cost_bps=10, top_k=5)
    w = np.array(list(_engine_weights(bt).values()))
    assert w.size and np.isfinite(w).all()
