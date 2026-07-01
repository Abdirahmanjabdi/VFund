import numpy as np

from vfund.backtest.construct import scores_to_weights


def test_weights_are_dollar_neutral_and_scaled():
    scores = np.array([3.0, 1.0, -1.0, -3.0])
    w = scores_to_weights(scores, leverage=1.0)
    assert abs(w.sum()) < 1e-12            # longs cancel shorts
    assert abs(np.abs(w).sum() - 1.0) < 1e-12  # gross exposure == leverage
    assert w[0] > 0 and w[-1] < 0          # highest score long, lowest short


def test_top_k_selects_extremes_only():
    scores = np.array([5.0, 4.0, 0.0, -4.0, -5.0])
    w = scores_to_weights(scores, leverage=1.0, top_k=1)
    # Only the single best and single worst name trade.
    assert np.count_nonzero(w) == 2
    assert w[0] > 0 and w[4] < 0 and w[2] == 0


def test_nan_assets_are_excluded():
    scores = np.array([np.nan, 1.0, -1.0])
    w = scores_to_weights(scores)
    assert w[0] == 0.0
    assert abs(w.sum()) < 1e-12


def test_insufficient_names_returns_flat():
    assert np.all(scores_to_weights(np.array([np.nan, 1.0])) == 0.0)
