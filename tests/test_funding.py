"""Funding-carry accounting tests.

The engine must actually credit funding cashflows to P&L — otherwise the carry
edge is invisible. We hold prices flat so that *only* funding can move equity,
and check both signs: harvesting funding pays whether the rate is positive
(short the payer) or negative (long the payer).
"""

from datetime import datetime, timezone

import numpy as np
import polars as pl

from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.panel import align_funding
from vfund.strategy import FundingCarry

HOUR_MS = 3_600_000
START = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def _timestamps(n):
    return pl.Series("timestamp", [START + i * HOUR_MS for i in range(n)]).cast(
        pl.Datetime("ms", time_zone="UTC")
    )


def _flat_panel(symbols, n):
    frames = []
    for s in symbols:
        frames.append(
            pl.DataFrame(
                {
                    "timestamp": _timestamps(n),
                    "open": np.full(n, 100.0),
                    "high": np.full(n, 100.0),
                    "low": np.full(n, 100.0),
                    "close": np.full(n, 100.0),
                    "volume": np.ones(n),
                }
            ).with_columns(pl.lit(s).alias("symbol"))
        )
    return pl.concat(frames)


def _funding(rates: dict, n, every=8):
    rows = []
    for sym, rate in rates.items():
        for t in range(0, n, every):
            rows.append((START + t * HOUR_MS, sym, rate))
    return pl.DataFrame(
        {
            "timestamp": [r[0] for r in rows],
            "symbol": [r[1] for r in rows],
            "funding_rate": [r[2] for r in rows],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", time_zone="UTC")))


def test_align_funding_event_and_prevailing():
    ts = _timestamps(24)
    funding = pl.DataFrame(
        {
            "timestamp": pl.Series([START, START + 8 * HOUR_MS]).cast(
                pl.Datetime("ms", time_zone="UTC")
            ),
            "symbol": ["AAA", "AAA"],
            "funding_rate": [0.001, 0.002],
        }
    )
    event, prevailing = align_funding(funding, ts, ["AAA"])
    assert event[0, 0] == 0.001 and event[8, 0] == 0.002
    assert event[1, 0] == 0.0  # nothing paid between events
    assert prevailing[3, 0] == 0.001  # forward-filled
    assert prevailing[10, 0] == 0.002


def test_carry_harvests_positive_funding():
    panel = _flat_panel(["AAA", "BBB"], 48)
    funding = _funding({"AAA": 0.001, "BBB": 0.0}, 48)
    res = CrossSectionalBacktester(
        panel, FundingCarry(), funding=funding, rebalance_every=1, cost_bps=0
    ).run()
    # Flat prices: the only P&L is harvested funding, so equity must rise.
    assert res.final_equity > res.initial_cash


def test_carry_harvests_negative_funding():
    panel = _flat_panel(["AAA", "BBB"], 48)
    funding = _funding({"AAA": -0.001, "BBB": 0.0}, 48)
    res = CrossSectionalBacktester(
        panel, FundingCarry(), funding=funding, rebalance_every=1, cost_bps=0
    ).run()
    assert res.final_equity > res.initial_cash  # long the payer, still collect


def test_carry_needs_funding():
    panel = _flat_panel(["AAA", "BBB"], 20)
    import pytest

    with pytest.raises(ValueError):
        CrossSectionalBacktester(panel, FundingCarry(), rebalance_every=1).run()
