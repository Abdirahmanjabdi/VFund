"""Yahoo Finance chart API — FX, commodities, equities, indices.

Where :mod:`vfund.data.ingest` covers Binance, this covers everything else.
It is the data layer that makes the cross-market ambition testable: the same
endpoint serves ``GBPUSD=X``, ``GC=F`` (COMEX gold), ``^GSPC``, and ordinary
equity tickers, all in one schema.

Limits worth knowing before you plan a study
--------------------------------------------
* **Daily** history is deep — GBPUSD from 2003, gold futures from 2000 — but
  only if you pass explicit ``period1``/``period2``. Asking for ``range=max``
  with ``interval=1d`` silently returns *monthly* bars instead, which is a
  quiet way to run a study on the wrong data.
* **Hourly** is capped at roughly 730 days.
* **Finer than hourly** (15m, 5m) is capped at about 60 days.
* FX pairs report ``volume`` as zero. It is not missing data; there is no
  consolidated volume for OTC spot FX. Never build a signal on it.

No API key is used or required.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import polars as pl
import requests

_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_UA = "Mozilla/5.0 (compatible; VFund research)"

#: Intervals Yahoo accepts, and roughly how far back each reaches.
INTERVALS = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
    "60m": 730, "1h": 730, "1d": None, "1wk": None, "1mo": None,
}


def _epoch(dt: datetime | str) -> int:
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def fetch(
    symbol: str,
    *,
    interval: str = "1d",
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    session: requests.Session | None = None,
    retries: int = 3,
) -> pl.DataFrame:
    """Fetch OHLCV bars for one Yahoo symbol.

    Returns VFund's canonical schema — timestamp (UTC), symbol, open, high,
    low, close, volume — with bars whose close is null dropped. Yahoo pads its
    arrays with nulls for non-trading periods, and a null close is an absent
    bar rather than a zero.
    """
    if interval not in INTERVALS:
        raise ValueError(f"unknown interval {interval!r}; known: {sorted(INTERVALS)}")

    s = session or requests.Session()
    s.headers.setdefault("User-Agent", _UA)
    params: dict[str, object] = {"interval": interval}
    if start is not None or end is not None:
        params["period1"] = _epoch(start or datetime(2000, 1, 1, tzinfo=timezone.utc))
        params["period2"] = _epoch(end or datetime.now(timezone.utc))
    else:
        # Without explicit bounds Yahoo picks a range for the interval, and for
        # "max" it downgrades daily to monthly — so bounds are the safe default.
        params["range"] = f"{INTERVALS[interval]}d" if INTERVALS[interval] else "10y"

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = s.get(_CHART.format(symbol=symbol), params=params, timeout=40)
            r.raise_for_status()
            payload = r.json()["chart"]
            if payload.get("error"):
                raise ValueError(f"{symbol}: {payload['error']}")
            res = payload["result"][0]
            break
        except (requests.RequestException, KeyError, TypeError, ValueError) as e:
            last_err = e
            if attempt == retries - 1:
                raise RuntimeError(f"failed to fetch {symbol}: {e}") from e
            time.sleep(1.0 + attempt)
    else:  # pragma: no cover - loop always breaks or raises
        raise RuntimeError(str(last_err))

    stamps = res.get("timestamp") or []
    if not stamps:
        return pl.DataFrame(
            schema={"timestamp": pl.Datetime("us", "UTC"), "symbol": pl.Utf8,
                    "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
                    "close": pl.Float64, "volume": pl.Float64}
        )
    q = res["indicators"]["quote"][0]
    df = pl.DataFrame(
        {
            "timestamp": stamps,
            "symbol": symbol,
            "open": [None if v is None else float(v) for v in q.get("open", [])],
            "high": [None if v is None else float(v) for v in q.get("high", [])],
            "low": [None if v is None else float(v) for v in q.get("low", [])],
            "close": [None if v is None else float(v) for v in q.get("close", [])],
            "volume": [0.0 if v is None else float(v) for v in q.get("volume", [])],
        },
        strict=False,
    )
    return (
        df.with_columns(
            pl.from_epoch("timestamp", time_unit="s").dt.replace_time_zone("UTC")
        )
        .drop_nulls("close")
        .unique(subset="timestamp", keep="last")
        .sort("timestamp")
    )


def fetch_many(
    symbols: list[str],
    *,
    interval: str = "1d",
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    pause_s: float = 0.3,
) -> pl.DataFrame:
    """Fetch several symbols into one long panel, skipping those that fail."""
    s = requests.Session()
    s.headers["User-Agent"] = _UA
    frames: list[pl.DataFrame] = []
    for sym in symbols:
        try:
            df = fetch(sym, interval=interval, start=start, end=end, session=s)
        except RuntimeError:
            continue
        if df.height:
            frames.append(df)
        time.sleep(pause_s)
    if not frames:
        raise RuntimeError("no symbols returned data")
    return pl.concat(frames, how="diagonal_relaxed").sort(["symbol", "timestamp"])
