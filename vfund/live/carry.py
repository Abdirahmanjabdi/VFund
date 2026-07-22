"""The funding-basis carry sleeve, as a live-trackable return stream.

Why this module exists
----------------------
The alpha engine (trend + size + on-chain) is a book of *spot weights*: you hold
a fraction of equity in each coin and the return is the price move. The paper
tracker models exactly that.

Carry is not that shape. A carry position is a **pair** — long spot, short the
perpetual — and its return is

    funding received  -  change in the spot/perp basis

not a spot price move. So it cannot be expressed as a weight vector marked
forward by prices, and the two-engine book needs this second, separate accrual
path. This module provides it, factored out of ``examples/two_engine.py`` so the
live account runs the *same* arithmetic that was validated there.

Causality
---------
The basket at bar ``t`` is chosen from trailing funding over ``[t-lookback, t)``
— strictly before ``t`` — and the return credited at ``t`` uses funding paid at
``t`` and the basis move from ``t-1`` to ``t``. No look-ahead. ``tests/
test_lookahead.py`` covers the engine; ``test_carry.py`` covers this directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import polars as pl

#: Liquid USD-M perpetual majors — the capacity-scalable end of the market.
#: Carry lives here on purpose: small caps have fat funding but no borrow depth.
MAJORS: list[str] = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "TRXUSDT",
    "BCHUSDT",
]

#: ~1bp/day of round-trip friction. Majors carry is low-turnover, so this is a
#: realistic (not generous) drag. Matches examples/two_engine.py.
DEFAULT_COST_PER_DAY = 1.0 / 10_000.0

#: Trailing window used to decide whether a major is *paying* carry.
DEFAULT_LOOKBACK = 30


@dataclass
class CarrySleeve:
    """Daily carry returns plus the currently-selected basket."""

    returns: pl.DataFrame          # (timestamp, carry) — realised daily return
    basket: list[str] = field(default_factory=list)   # majors in the book now
    asof: object = None            # timestamp of the last bar


def _wide(panel: pl.DataFrame, value: str = "close") -> pl.DataFrame:
    """Long panel -> wide (timestamp x symbol) frame, sorted by time."""
    return (
        panel.pivot(values=value, index="timestamp", on="symbol").sort("timestamp")
    )


def _daily_funding(funding: pl.DataFrame) -> pl.DataFrame:
    """Sum 8-hourly funding into a daily wide frame.

    Binance pays funding every 8h; the carry backtest works on daily bars, so the
    three intraday payments are summed into the day they land on.
    """
    return (
        funding.with_columns(pl.col("timestamp").dt.truncate("1d"))
        .group_by(["timestamp", "symbol"])
        .agg(pl.col("funding_rate").sum())
        .pivot(values="funding_rate", index="timestamp", on="symbol")
        .sort("timestamp")
    )


def carry_sleeve(
    spot: pl.DataFrame,
    perp: pl.DataFrame,
    funding: pl.DataFrame,
    *,
    majors: list[str] | None = None,
    lookback: int = DEFAULT_LOOKBACK,
    cost_per_day: float = DEFAULT_COST_PER_DAY,
) -> CarrySleeve:
    """Compute the timed funding-basis carry sleeve.

    Args:
        spot: long-format spot price panel (needs the majors).
        perp: long-format USD-M perpetual price panel (same symbols).
        funding: long-format funding panel (timestamp, symbol, funding_rate).
        majors: symbols to consider; defaults to :data:`MAJORS`.
        lookback: trailing days used to decide if a major pays carry.
        cost_per_day: friction subtracted from every day's return.

    Returns:
        A :class:`CarrySleeve` with the realised daily return series and the
        basket selected as of the final bar.

    Raises:
        ValueError: if no symbol is present in all three inputs, or if the
            aligned history is shorter than ``lookback`` (nothing to decide on).
    """
    majors = list(majors or MAJORS)
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    s_wide = _wide(spot)
    p_wide = _wide(perp)
    f_wide = _daily_funding(funding)

    syms = sorted(
        set(majors) & set(s_wide.columns) & set(p_wide.columns) & set(f_wide.columns)
    )
    if not syms:
        raise ValueError(
            "carry needs symbols present in spot, perp and funding; the "
            f"intersection of {len(majors)} majors with the three inputs is empty"
        )

    joined = (
        s_wide.select(["timestamp", *syms]).rename({s: f"s_{s}" for s in syms})
        .join(
            p_wide.select(["timestamp", *syms]).rename({s: f"p_{s}" for s in syms}),
            on="timestamp", how="inner",
        )
        .join(
            f_wide.select(["timestamp", *syms]).rename({s: f"f_{s}" for s in syms}),
            on="timestamp", how="inner",
        )
        .sort("timestamp")
        .drop_nulls()
    )
    if joined.height <= lookback:
        raise ValueError(
            f"carry needs more than {lookback} aligned bars to form a trailing "
            f"funding view; got {joined.height}"
        )

    S = joined.select([f"s_{s}" for s in syms]).to_numpy()
    P = joined.select([f"p_{s}" for s in syms]).to_numpy()
    F = joined.select([f"f_{s}" for s in syms]).to_numpy()

    # Basis = perp premium over spot. Carrying the pair earns the funding and
    # loses whatever the basis widens by (mark-to-market on the short leg).
    basis = (P - S) / S
    dbasis = np.zeros_like(basis)
    dbasis[1:] = basis[1:] - basis[:-1]
    pair_ret = F - dbasis

    # Timed: hold only the majors whose trailing funding was positive, i.e. the
    # ones longs have been paying for. Decision window is strictly before t.
    timed = np.zeros(pair_ret.shape[0])
    basket_mask = np.zeros(len(syms), dtype=bool)
    for t in range(lookback, pair_ret.shape[0]):
        mask = F[t - lookback:t].mean(axis=0) > 0
        if mask.any():
            timed[t] = pair_ret[t, mask].mean()
        if t == pair_ret.shape[0] - 1:
            basket_mask = mask
    timed -= cost_per_day

    # The first `lookback` bars have no decision basis; drop rather than credit 0.
    ts = joined["timestamp"]
    returns = pl.DataFrame(
        {"timestamp": ts[lookback:], "carry": timed[lookback:]}
    )
    return CarrySleeve(
        returns=returns,
        basket=[s for s, m in zip(syms, basket_mask) if m],
        asof=ts[-1],
    )


def realised_carry(sleeve: CarrySleeve, since_ts: str | None) -> float:
    """Compound the carry sleeve's return over ``(since_ts, end]``.

    Args:
        sleeve: output of :func:`carry_sleeve`.
        since_ts: ISO timestamp string of the last account update, or ``None``
            to take the whole series.

    Returns:
        The compounded simple return over the window (0.0 if no bars fall in it).

    Raises:
        ValueError: if ``since_ts`` is not a parseable ISO timestamp. We refuse
            rather than silently fall back to the whole series — crediting
            months of carry to a single update would corrupt the record exactly
            like the stale-state gap this guards against.
    """
    df = sleeve.returns
    if since_ts is not None:
        # Compare as datetimes, not strings: the account stores ``str(datetime)``
        # while polars' own Utf8 cast uses a different layout, and a lexical
        # compare between the two silently off-by-ones the window.
        try:
            cutoff = datetime.fromisoformat(str(since_ts))
        except ValueError as exc:
            raise ValueError(f"since_ts is not an ISO timestamp: {since_ts!r}") from exc
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        df = df.filter(pl.col("timestamp") > cutoff)
    if df.height == 0:
        return 0.0
    return float(np.prod(1.0 + df["carry"].to_numpy()) - 1.0)


def format_carry(sleeve: CarrySleeve) -> str:
    """Human-readable summary of the carry sleeve's current stance."""
    n = len(sleeve.basket)
    head = f"Carry sleeve as of {sleeve.asof}  ({n} major{'' if n == 1 else 's'} paying)"
    if not sleeve.basket:
        return f"{head}\n{'-' * 44}\n  (flat - no major is paying positive carry)"
    names = ", ".join(s[:-4] if s.endswith("USDT") else s for s in sleeve.basket)
    return f"{head}\n{'-' * 44}\n  LONG spot / SHORT perp:  {names}"
