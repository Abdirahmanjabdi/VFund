"""Tests for ragged (enter/exit) universes and short-financing costs."""

import numpy as np
import polars as pl

from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.synthetic import generate_gbm_panel
from vfund.strategy import CrossSectionalReversal
from vfund.strategy.cross_sectional import CrossSectionalStrategy


class AllShort(CrossSectionalStrategy):
    def scores(self, ctx):
        return -np.ones(ctx.closes.shape[1])


def test_ragged_panel_runs_over_full_timeline():
    full = generate_gbm_panel(6, 400, interval="1d", seed=1)
    ts = full.select("timestamp").unique().sort("timestamp")["timestamp"]
    early, late = ts.head(100), ts.tail(100)
    # SYN00 lists late (drop its first 100 bars); SYN01 delists early.
    ragged = full.filter(
        ~((pl.col("symbol") == "SYN00") & pl.col("timestamp").is_in(early.implode()))
        & ~((pl.col("symbol") == "SYN01") & pl.col("timestamp").is_in(late.implode()))
    )
    res = CrossSectionalBacktester(
        ragged, CrossSectionalReversal(2), rebalance_every=1, cost_bps=0, interval="1d"
    ).run()
    # Union timeline is preserved and equity is finite throughout.
    assert res.equity_curve.height == ts.len()
    assert np.all(np.isfinite(res.equity_curve["equity"].to_numpy()))


def test_aligned_panel_unchanged_by_ragged_flag():
    panel = generate_gbm_panel(8, 300, reversion=0.2, seed=2)
    a = CrossSectionalBacktester(panel, CrossSectionalReversal(1), allow_ragged=True).run()
    b = CrossSectionalBacktester(panel, CrossSectionalReversal(1), allow_ragged=False).run()
    assert np.allclose(a.equity_curve["equity"].to_numpy(),
                       b.equity_curve["equity"].to_numpy())


def test_hard_to_short_blocks_illiquid_shorts():
    panel = generate_gbm_panel(6, 300, interval="1d", seed=7)
    free = CrossSectionalBacktester(
        panel, AllShort(), neutralize=False, rebalance_every=5, cost_bps=0, interval="1d"
    ).run()
    # Threshold above every coin's dollar volume -> no short is executable.
    blocked = CrossSectionalBacktester(
        panel, AllShort(), neutralize=False, rebalance_every=5, cost_bps=0, interval="1d",
        min_short_dollar_volume=1e15,
    ).run()
    assert free.equity_curve["gross_exposure"].max() > 0.5
    assert blocked.equity_curve["gross_exposure"].max() < 1e-9  # book forced flat


def test_short_financing_cost_reduces_equity():
    panel = generate_gbm_panel(6, 400, interval="1d", seed=3)
    free = CrossSectionalBacktester(
        panel, AllShort(), neutralize=False, rebalance_every=7, cost_bps=0, interval="1d"
    ).run()
    charged = CrossSectionalBacktester(
        panel, AllShort(), neutralize=False, rebalance_every=7, cost_bps=0, interval="1d",
        short_cost_bps_annual=1000,  # 10%/yr financing on shorts
    ).run()
    assert charged.final_equity < free.final_equity
