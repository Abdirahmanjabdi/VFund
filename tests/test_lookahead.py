"""The gold-standard causality test: the past cannot see the future.

If corrupting future data changes any *past* equity value, the engine has
look-ahead bias somewhere — and every backtest result is invalid. This test
exercises the full engine with every overlay active, so it guards the whole
stack. It is the single most important test in the suite.
"""

import numpy as np
import polars as pl

from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.synthetic import generate_gbm_panel
from vfund.strategy import CrossSectionalReversal


def _equity(panel):
    return CrossSectionalBacktester(
        panel, CrossSectionalReversal(3), rebalance_every=5, cost_bps=10, top_k=5,
        interval="1d", vol_target=0.3, short_cost_bps_annual=1000,
        min_short_dollar_volume=1000, capacity_aum=1e7,
    ).run().equity_curve["equity"].to_numpy()


def test_no_lookahead_future_cannot_change_past():
    panel = generate_gbm_panel(15, 600, reversion=0.2, seed=7)
    base = _equity(panel)

    # Violently corrupt every bar after the cut; the past must be untouched.
    ts = panel.select("timestamp").unique().sort("timestamp")["timestamp"]
    cut = ts[420]
    rng = np.random.default_rng(0)
    noise = pl.Series(rng.normal(0, 0.5, panel.height))
    corrupt = panel.with_columns([
        pl.when(pl.col("timestamp") > cut)
        .then(pl.col(c) * (1 + noise)).otherwise(pl.col(c)).alias(c)
        for c in ["open", "high", "low", "close", "volume"]
    ])
    corrupted = _equity(corrupt)

    assert np.array_equal(base[:421], corrupted[:421])   # past is byte-identical
    assert not np.allclose(base[421:], corrupted[421:])  # future did change (sanity)
