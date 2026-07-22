"""Health of the forward paper accounts — the ops layer that failed first.

Why this exists
---------------
On 2026-07-20 the weekly scheduled task was killed mid-run (Windows terminated
it on battery, exit ``0xC000013A``). Nothing alerted. The failure was only
noticed later, and by then that Monday was gone: the paper tracker refuses to
mark a stale gap forward (:attr:`PaperTracker.MAX_GAP_DAYS`), precisely so a
missed week cannot be silently back-filled with fabricated history.

That is the right behaviour and it has a consequence: **a missed update is a
permanently lost data point in the only honest record this project has.** A
forward test is worth exactly the discipline of the process that feeds it, so
the process needs to be able to say when it is broken.

Staleness ladder
----------------
* ``OK``       — updated within :data:`FRESH_DAYS`; the weekly cadence is holding.
* ``STALE``    — a scheduled update was missed. Recoverable: run the update now
  and only the resolution of the record suffers.
* ``CRITICAL`` — approaching ``MAX_GAP_DAYS``. The next update is about to be
  refused; act before the account becomes unrecoverable.
* ``BROKEN``   — past ``MAX_GAP_DAYS``. The tracker will refuse to advance. The
  account must be archived and restarted, which destroys its forward record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from vfund.live.paper import PaperTracker

#: Updates land weekly, so ~8 days allows for a late run without crying wolf.
FRESH_DAYS = 8
#: Warn while there is still time to act before MAX_GAP_DAYS refuses the update.
CRITICAL_DAYS = 30

OK, STALE, CRITICAL, BROKEN, MISSING = "OK", "STALE", "CRITICAL", "BROKEN", "MISSING"

#: Non-zero exit for these, so a scheduler or CI surfaces the problem.
_FAILING = (STALE, CRITICAL, BROKEN, MISSING)


@dataclass(frozen=True)
class AccountHealth:
    """One paper account's state, judged against the staleness ladder."""

    name: str
    path: Path
    status: str
    days_stale: float | None = None
    last_ts: str | None = None
    equity: float | None = None
    start_equity: float | None = None
    updates: int = 0
    detail: str = ""

    @property
    def total_return(self) -> float | None:
        if self.equity is None or not self.start_equity:
            return None
        return self.equity / self.start_equity - 1.0

    @property
    def ok(self) -> bool:
        return self.status == OK


def _days_since(ts: str, now: date) -> float:
    return (now - date.fromisoformat(ts[:10])).days


def check_account(path: str | Path, *, name: str | None = None,
                  now: date | None = None) -> AccountHealth:
    """Judge a single paper-account state file.

    Args:
        path: the account's ``.json`` state file.
        name: label for display; defaults to the file stem.
        now: the date to measure staleness against (injectable for tests).
    """
    path = Path(path)
    label = name or path.stem
    now = now or datetime.now(timezone.utc).date()

    if not path.exists():
        return AccountHealth(label, path, MISSING,
                             detail="no state file - account never initialised")
    try:
        st = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return AccountHealth(label, path, BROKEN, detail=f"unreadable: {exc}")

    missing = [k for k in ("last_ts", "equity", "start_equity", "history") if k not in st]
    if missing:
        return AccountHealth(label, path, BROKEN,
                             detail=f"state file missing keys: {missing}")

    days = _days_since(st["last_ts"], now)
    if days > PaperTracker.MAX_GAP_DAYS:
        status, detail = BROKEN, (
            f"gap exceeds MAX_GAP_DAYS ({PaperTracker.MAX_GAP_DAYS}d) - the "
            f"tracker will refuse to advance this account"
        )
    elif days >= CRITICAL_DAYS:
        status, detail = CRITICAL, (
            f"{PaperTracker.MAX_GAP_DAYS - days:.0f}d until the tracker refuses "
            f"to advance - update now"
        )
    elif days >= FRESH_DAYS:
        status, detail = STALE, "a scheduled update was missed"
    else:
        status, detail = OK, ""

    return AccountHealth(
        label, path, status, days_stale=days, last_ts=st["last_ts"][:10],
        equity=float(st["equity"]), start_equity=float(st["start_equity"]),
        updates=len(st["history"]), detail=detail,
    )


def check_accounts(paths: dict[str, str | Path], *,
                   now: date | None = None) -> list[AccountHealth]:
    """Judge several accounts, worst status first."""
    order = {BROKEN: 0, MISSING: 1, CRITICAL: 2, STALE: 3, OK: 4}
    out = [check_account(p, name=n, now=now) for n, p in paths.items()]
    return sorted(out, key=lambda h: (order.get(h.status, 9), h.name))


def format_health(results: list[AccountHealth]) -> str:
    """Render a health report."""
    if not results:
        return "no accounts configured"
    head = (f"{'account':<22} {'status':<9} {'last':<12} {'age':>6} "
            f"{'equity':>12} {'return':>8} {'upd':>4}")
    lines = [head, "-" * len(head)]
    for h in results:
        age = f"{h.days_stale:.0f}d" if h.days_stale is not None else "-"
        eq = f"${h.equity:,.0f}" if h.equity is not None else "-"
        ret = f"{h.total_return*100:+.1f}%" if h.total_return is not None else "-"
        lines.append(f"{h.name:<22} {h.status:<9} {h.last_ts or '-':<12} {age:>6} "
                     f"{eq:>12} {ret:>8} {h.updates:>4}")
        if h.detail:
            lines.append(f"  -> {h.detail}")
    bad = [h for h in results if h.status in _FAILING]
    lines.append("-" * len(head))
    lines.append("all accounts healthy" if not bad
                 else f"{len(bad)} account(s) need attention")
    return "\n".join(lines)


def exit_code(results: list[AccountHealth]) -> int:
    """0 when every account is healthy, 1 otherwise.

    The scheduled task can gate on this, so a silent failure becomes a visible
    one instead of a gap discovered weeks later.
    """
    return 1 if any(h.status in _FAILING for h in results) else 0
