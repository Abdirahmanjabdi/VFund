"""Tests for the live signal and paper tracker."""

import polars as pl

from vfund.data.synthetic import generate_gbm_panel
from vfund.live.signal import combined_book, format_book
from vfund.live.paper import PaperTracker


def _panel(n):
    return generate_gbm_panel(10, n, interval="1d", reversion=0.1, seed=5)


def test_combined_book_has_longs_and_shorts():
    book = combined_book(_panel(320), interval="1d")
    assert book.weights                     # non-empty
    assert book.gross > 0
    assert any(w > 0 for w in book.weights.values())
    assert any(w < 0 for w in book.weights.values())
    text = format_book(book)
    assert "LONG" in text and "SHORT" in text


def test_paper_tracker_rejects_stale_state(tmp_path):
    # A huge gap between updates means the state file is stale/wrong; marking
    # it forward in one step would corrupt the record. Must refuse.
    import pytest

    full = _panel(320)
    ts = full.select("timestamp").unique().sort("timestamp")["timestamp"]
    early = full.filter(pl.col("timestamp").is_in(ts.head(200).implode()))  # 120d gap

    tracker = PaperTracker(tmp_path / "acct.json", start_equity=10_000)
    tracker.update(early, interval="1d")
    with pytest.raises(RuntimeError, match="stale"):
        tracker.update(full, interval="1d")


def test_paper_tracker_initializes_and_advances(tmp_path):
    full = _panel(320)
    ts = full.select("timestamp").unique().sort("timestamp")["timestamp"]
    early = full.filter(pl.col("timestamp").is_in(ts.head(310).implode()))

    tracker = PaperTracker(tmp_path / "acct.json", start_equity=10_000)

    first = tracker.update(early, interval="1d")
    assert first["status"] == "initialized"
    assert len(tracker.load()["history"]) == 1

    # Same data again -> nothing new.
    again = tracker.update(early, interval="1d")
    assert again["status"] == "no new data"

    # More data -> marks forward and records.
    advanced = tracker.update(full, interval="1d")
    assert advanced["status"] == "updated"
    assert len(tracker.load()["history"]) == 2
    assert tracker.load()["equity"] > 0
