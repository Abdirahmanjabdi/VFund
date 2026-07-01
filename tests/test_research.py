import polars as pl

from vfund.data.synthetic import generate_gbm_panel
from vfund.research import train_test_split, walk_forward_windows, walk_forward
from vfund.strategy import CrossSectionalReversal


def test_train_test_split_ranges():
    train, test = train_test_split(100, train_frac=0.6)
    assert train == range(0, 60)
    assert test == range(60, 100)


def test_walk_forward_windows_tile_the_timeline():
    wins = walk_forward_windows(100, train_size=50, test_size=20, step=20)
    assert len(wins) == 2
    assert wins[0] == (range(0, 50), range(50, 70))
    assert wins[1] == (range(20, 70), range(70, 90))


def test_walk_forward_finds_oos_edge_on_reversal_panel():
    panel = generate_gbm_panel(15, 4000, reversion=0.25, seed=7)
    res = walk_forward(
        panel,
        lambda lookback: CrossSectionalReversal(lookback),
        [{"lookback": lb} for lb in (1, 2, 3)],
        train_size=1500,
        test_size=500,
        backtest_kwargs=dict(rebalance_every=1, cost_bps=0.0),
    )
    assert res.windows.height >= 1
    assert f"test_sharpe" in res.windows.columns
    # The edge is real and persistent, so it should survive out-of-sample.
    assert res.oos_sharpe() > 0
    assert isinstance(res.oos_equity, pl.DataFrame)
