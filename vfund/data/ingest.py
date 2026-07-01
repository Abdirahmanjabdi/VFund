"""Ingest historical crypto bars from Binance's public REST API.

No API key required — Binance serves historical klines anonymously. We chose
crypto for VFund's v0 precisely because the data is free, unlicensed, and the
market never closes, so you can learn the whole pipeline without paperwork.

Reference: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import polars as pl
import requests

from vfund.data.intervals import INTERVAL_MS
from vfund.data.models import validate_bars

_BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
_MAX_LIMIT = 1000  # Binance hard cap per request.


def _to_ms(dt: datetime | str) -> int:
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_klines(
    symbol: str,
    interval: str = "1h",
    *,
    start: datetime | str,
    end: datetime | str | None = None,
    session: requests.Session | None = None,
    pause_s: float = 0.25,
) -> pl.DataFrame:
    """Fetch OHLCV bars for ``symbol`` between ``start`` and ``end``.

    Paginates transparently across Binance's 1000-bar limit. Returns a frame in
    VFund's canonical schema. ``symbol`` is a Binance pair like ``"BTCUSDT"``.
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"unknown interval {interval!r}; known: {sorted(INTERVAL_MS)}")

    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end is not None else int(time.time() * 1000)
    step_ms = INTERVAL_MS[interval]

    sess = session or requests.Session()
    rows: list[list] = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": _MAX_LIMIT,
        }
        resp = sess.get(_BINANCE_KLINES_URL, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break

        rows.extend(batch)
        # Advance the cursor just past the last bar's open time.
        last_open = batch[-1][0]
        cursor = last_open + step_ms
        if len(batch) < _MAX_LIMIT:
            break
        time.sleep(pause_s)  # be polite to the public endpoint

    if not rows:
        raise RuntimeError(
            f"Binance returned no data for {symbol} {interval} "
            f"in the requested range"
        )

    # Kline columns: [openTime, open, high, low, close, volume, closeTime, ...].
    df = pl.DataFrame(
        {
            "timestamp": [r[0] for r in rows],
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
            "volume": [float(r[5]) for r in rows],
        }
    ).with_columns(
        pl.col("timestamp").cast(pl.Datetime("ms", time_zone="UTC"))
    )

    # Binance can return a partial trailing bar twice across pages; dedupe.
    df = df.unique(subset="timestamp", keep="first")
    return validate_bars(df)
