"""Tests for forward-account health checks.

These guard an ops failure that already happened: the 2026-07-20 weekly run was
killed mid-flight and nothing noticed. The value of this module is entirely in
its boundaries - warning while a gap is still recoverable - so the boundaries
are what get tested.
"""

import json
from datetime import date

import pytest

from vfund.live.health import (
    BROKEN,
    CRITICAL,
    CRITICAL_DAYS,
    FRESH_DAYS,
    MISSING,
    OK,
    STALE,
    check_account,
    check_accounts,
    exit_code,
    format_health,
)
from vfund.live.paper import PaperTracker


def _state(tmp_path, last_ts="2026-07-20", equity=95_000.0, updates=4, name="a.json"):
    p = tmp_path / name
    p.write_text(json.dumps({
        "created": "2026-07-01 00:00:00+00:00",
        "start_equity": 100_000.0,
        "equity": equity,
        "last_ts": f"{last_ts} 00:00:00+00:00",
        "last_prices": {}, "weights": {},
        "history": [{"ts": "x", "equity": equity}] * updates,
    }))
    return p


@pytest.mark.parametrize("age_days,expected", [
    (0, OK),
    (FRESH_DAYS - 1, OK),
    (FRESH_DAYS, STALE),                       # boundary: a run was missed
    (CRITICAL_DAYS - 1, STALE),
    (CRITICAL_DAYS, CRITICAL),                 # boundary: act now
    (PaperTracker.MAX_GAP_DAYS, CRITICAL),     # still just recoverable
    (PaperTracker.MAX_GAP_DAYS + 1, BROKEN),   # tracker will now refuse
])
def test_staleness_ladder_boundaries(tmp_path, age_days, expected):
    """Each rung must trip exactly at its threshold, not near it."""
    from datetime import timedelta
    last = date(2026, 7, 20)
    p = _state(tmp_path, last_ts=last.isoformat())
    h = check_account(p, now=last + timedelta(days=age_days))
    assert h.status == expected, f"{age_days}d stale -> {h.status}, wanted {expected}"


def test_broken_threshold_matches_the_trackers_own_refusal(tmp_path):
    """BROKEN must line up with MAX_GAP_DAYS, or the warning is a lie.

    If these drift apart, health would report an account fine that the tracker
    then refuses to advance - the exact silent failure this module exists to
    prevent.
    """
    from datetime import timedelta
    last = date(2026, 1, 1)
    p = _state(tmp_path, last_ts=last.isoformat())
    just_ok = check_account(p, now=last + timedelta(days=PaperTracker.MAX_GAP_DAYS))
    just_bad = check_account(p, now=last + timedelta(days=PaperTracker.MAX_GAP_DAYS + 1))
    assert just_ok.status != BROKEN and just_bad.status == BROKEN


def test_missing_file_is_reported_not_crashed(tmp_path):
    h = check_account(tmp_path / "nope.json")
    assert h.status == MISSING and "never initialised" in h.detail


def test_corrupt_and_incomplete_state_are_broken(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert check_account(bad).status == BROKEN

    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"equity": 1.0}))
    h = check_account(partial)
    assert h.status == BROKEN and "missing keys" in h.detail


def test_returns_and_counts_are_reported(tmp_path):
    p = _state(tmp_path, last_ts="2026-07-20", equity=95_000.0, updates=4)
    h = check_account(p, now=date(2026, 7, 21))
    assert h.equity == 95_000.0
    assert h.updates == 4
    assert h.total_return == pytest.approx(-0.05)
    assert h.ok


def test_results_sort_worst_first_and_exit_code_reflects_health(tmp_path):
    good = _state(tmp_path, last_ts="2026-07-20", name="good.json")
    old = _state(tmp_path, last_ts="2026-01-01", name="old.json")

    results = check_accounts({"good": good, "old": old}, now=date(2026, 7, 21))
    assert results[0].name == "old"          # worst first
    assert exit_code(results) == 1           # something needs attention

    healthy = check_accounts({"good": good}, now=date(2026, 7, 21))
    assert exit_code(healthy) == 0
    assert "all accounts healthy" in format_health(healthy)


def test_format_is_readable_and_flags_problems(tmp_path):
    old = _state(tmp_path, last_ts="2026-01-01", name="old.json")
    text = format_health(check_accounts({"old": old}, now=date(2026, 7, 21)))
    assert "BROKEN" in text and "need attention" in text
