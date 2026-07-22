"""Tests for the funding-basis carry sleeve and the two-engine paper account.

The carry sleeve is the half of the two-engine book that cannot be expressed as
spot weights, so it needs its own accrual path — and its own causality proof.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from vfund.live.carry import (
    DEFAULT_COST_PER_DAY,
    carry_sleeve,
    format_carry,
    realised_carry,
)

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def _panels(n_days: int = 120, premium: float = 0.001, funding_rate: float = 0.0004):
    """Build spot/perp/funding panels with a known, constant carry.

    Perp trades at a constant ``premium`` over spot, so the basis never moves and
    the daily pair return collapses to exactly the funding rate. That makes the
    expected return analytically known, which is what lets the tests assert on a
    number instead of a vibe.
    """
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts = [t0 + timedelta(days=i) for i in range(n_days)]
    rng = np.random.default_rng(7)

    spot_rows, perp_rows, fund_rows = [], [], []
    for s in SYMS:
        # Prices wander; the carry must not depend on the price path at all.
        px = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
        for i, t in enumerate(ts):
            spot_rows.append({"timestamp": t, "symbol": s, "open": px[i], "high": px[i],
                              "low": px[i], "close": px[i], "volume": 1e7})
            p = px[i] * (1.0 + premium)
            perp_rows.append({"timestamp": t, "symbol": s, "open": p, "high": p,
                              "low": p, "close": p, "volume": 1e7})
            # One funding payment per day keeps the daily sum equal to the rate.
            fund_rows.append({"timestamp": t, "symbol": s, "funding_rate": funding_rate})

    schema = {"timestamp": pl.Datetime("ms", time_zone="UTC")}
    spot = pl.DataFrame(spot_rows).with_columns(pl.col("timestamp").cast(schema["timestamp"]))
    perp = pl.DataFrame(perp_rows).with_columns(pl.col("timestamp").cast(schema["timestamp"]))
    fund = pl.DataFrame(fund_rows).with_columns(pl.col("timestamp").cast(schema["timestamp"]))
    return spot, perp, fund


def test_constant_basis_yields_funding_minus_cost():
    """With a flat basis, the daily carry return is exactly funding - cost."""
    rate = 0.0004
    spot, perp, fund = _panels(funding_rate=rate)
    sleeve = carry_sleeve(spot, perp, fund, majors=SYMS, lookback=30)

    r = sleeve.returns["carry"].to_numpy()
    assert r.size == 120 - 30
    np.testing.assert_allclose(r, rate - DEFAULT_COST_PER_DAY, atol=1e-12)


def test_positive_funding_selects_the_basket():
    spot, perp, fund = _panels(funding_rate=0.0004)
    sleeve = carry_sleeve(spot, perp, fund, majors=SYMS, lookback=30)
    assert sorted(sleeve.basket) == sorted(SYMS)
    assert "LONG spot / SHORT perp" in format_carry(sleeve)


def test_negative_funding_goes_flat():
    """When longs are being paid, there is no carry to harvest - stand aside."""
    spot, perp, fund = _panels(funding_rate=-0.0004)
    sleeve = carry_sleeve(spot, perp, fund, majors=SYMS, lookback=30)
    assert sleeve.basket == []
    # Flat means you still pay the friction, but earn no funding.
    r = sleeve.returns["carry"].to_numpy()
    np.testing.assert_allclose(r, -DEFAULT_COST_PER_DAY, atol=1e-12)
    assert "flat" in format_carry(sleeve)


def test_carry_is_causal_future_funding_cannot_change_the_past():
    """Truncating the future must not alter any already-computed return.

    This is the look-ahead sentinel: compute on the full history, then on a
    prefix, and require the overlapping returns to match exactly. If the basket
    selection or the accrual peeked forward, the prefix would disagree.
    """
    spot, perp, fund = _panels(n_days=150)
    full = carry_sleeve(spot, perp, fund, majors=SYMS, lookback=30)

    cutoff = spot["timestamp"].unique().sort()[110]
    prefix = carry_sleeve(
        spot.filter(pl.col("timestamp") <= cutoff),
        perp.filter(pl.col("timestamp") <= cutoff),
        fund.filter(pl.col("timestamp") <= cutoff),
        majors=SYMS, lookback=30,
    )

    joined = full.returns.join(prefix.returns, on="timestamp", how="inner", suffix="_p")
    assert joined.height > 0
    np.testing.assert_allclose(
        joined["carry"].to_numpy(), joined["carry_p"].to_numpy(), atol=1e-12
    )


def test_realised_carry_compounds_only_the_window():
    spot, perp, fund = _panels(funding_rate=0.0004)
    sleeve = carry_sleeve(spot, perp, fund, majors=SYMS, lookback=30)

    whole = realised_carry(sleeve, None)
    assert whole > 0

    last_ts = str(sleeve.returns["timestamp"][-2])
    tail = realised_carry(sleeve, last_ts)
    per_day = 0.0004 - DEFAULT_COST_PER_DAY
    assert tail == pytest.approx(per_day, abs=1e-12)  # exactly one bar in window

    # Nothing after the final bar.
    assert realised_carry(sleeve, str(sleeve.returns["timestamp"][-1])) == 0.0


def test_rejects_symbols_missing_from_an_input():
    spot, perp, fund = _panels()
    with pytest.raises(ValueError, match="intersection"):
        carry_sleeve(spot, perp, fund, majors=["NOTAREALCOIN"], lookback=30)


def test_rejects_history_shorter_than_lookback():
    spot, perp, fund = _panels(n_days=20)
    with pytest.raises(ValueError, match="aligned bars"):
        carry_sleeve(spot, perp, fund, majors=SYMS, lookback=30)


# --- two-engine paper account ------------------------------------------------


class _Book:
    def __init__(self, weights):
        self.weights = weights


def test_two_engine_account_tracks_both_pots(tmp_path):
    from vfund.live.paper import PaperTracker

    spot, perp, fund = _panels(funding_rate=0.0004)
    sleeve = carry_sleeve(spot, perp, fund, majors=SYMS, lookback=30)

    tracker = PaperTracker(tmp_path / "two.json", start_equity=100_000)
    book = _Book({"AAA": 0.5, "BBB": -0.5})
    prices = {"AAA": 100.0, "BBB": 100.0}

    first = tracker.record_two_engine(
        book, sleeve, "2024-04-01 00:00:00+00:00", prices, realised_carry=realised_carry
    )
    assert first["status"] == "initialized"
    st = tracker.load()
    # 50/50 split, alpha half paying entry turnover, carry half untouched.
    assert st["carry_equity"] == pytest.approx(50_000.0)
    assert st["alpha_equity"] < 50_000.0

    # AAA doubles: the +50% long earns 50%, the -50% short on flat BBB earns 0.
    moved = tracker.record_two_engine(
        book, sleeve, "2024-04-08 00:00:00+00:00",
        {"AAA": 200.0, "BBB": 100.0}, realised_carry=realised_carry,
    )
    assert moved["status"] == "updated"
    assert moved["alpha_ret"] == pytest.approx(0.5)
    st = tracker.load()
    assert st["equity"] > 100_000
    # Pots are rebalanced back to the target split every update.
    assert st["alpha_equity"] == pytest.approx(st["carry_equity"])
    assert len(st["history"]) == 2


def test_two_engine_respects_carry_weight(tmp_path):
    from vfund.live.paper import PaperTracker

    spot, perp, fund = _panels()
    sleeve = carry_sleeve(spot, perp, fund, majors=SYMS, lookback=30)
    tracker = PaperTracker(tmp_path / "w.json", start_equity=100_000)
    tracker.record_two_engine(
        _Book({"AAA": 1.0}), sleeve, "2024-04-01 00:00:00+00:00", {"AAA": 10.0},
        carry_weight=0.25, realised_carry=realised_carry,
    )
    st = tracker.load()
    assert st["carry_equity"] == pytest.approx(25_000.0)


def test_two_engine_rejects_stale_state(tmp_path):
    from vfund.live.paper import PaperTracker

    spot, perp, fund = _panels()
    sleeve = carry_sleeve(spot, perp, fund, majors=SYMS, lookback=30)
    tracker = PaperTracker(tmp_path / "s.json", start_equity=100_000)
    book, prices = _Book({"AAA": 1.0}), {"AAA": 10.0}

    tracker.record_two_engine(book, sleeve, "2024-01-01 00:00:00+00:00", prices,
                              realised_carry=realised_carry)
    with pytest.raises(RuntimeError, match="stale"):
        tracker.record_two_engine(book, sleeve, "2024-06-01 00:00:00+00:00", prices,
                                  realised_carry=realised_carry)


def test_two_engine_rejects_bad_carry_weight(tmp_path):
    from vfund.live.paper import PaperTracker

    tracker = PaperTracker(tmp_path / "b.json")
    with pytest.raises(ValueError, match="carry_weight"):
        tracker.update_two_engine(
            None, None, None, None, None, None, carry_weight=1.5
        )


_LOCAL_DATA = [Path("data") / f for f in
               ("uni_broad.parquet", "tvl_prices.parquet", "tvl.parquet", "fees.parquet")]


@pytest.mark.skipif(
    not all(p.exists() for p in _LOCAL_DATA),
    reason="needs locally fetched panels (data/ is gitignored; CI runs network-free)",
)
def test_alpha_book_fees_sleeve_changes_the_book():
    """Supplying fees adds a 4th sleeve, so the blended book must differ.

    Guards the wiring: if `fees` were silently dropped, the 3- and 4-sleeve
    books would be identical and the live account would quietly track the wrong
    configuration - the exact failure this whole change exists to fix.
    """
    from vfund.data.onchain import load_tvl
    from vfund.data.panel import load_panel
    from vfund.data.universe import clean_universe
    from vfund.live.signal import alpha_book, three_sleeve_book

    broad = clean_universe(load_panel("data/uni_broad.parquet"))
    defi = load_panel("data/tvl_prices.parquet")
    tvl = load_tvl("data/tvl.parquet")
    fees = load_tvl("data/fees.parquet")

    three = three_sleeve_book(broad, defi, tvl)
    four = alpha_book(broad, defi, tvl, fees=fees)

    assert three.weights and four.weights
    assert three.weights != four.weights
    # Both remain sane books, not degenerate output.
    assert four.gross > 0 and abs(four.net) < four.gross
