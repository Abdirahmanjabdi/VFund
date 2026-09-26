"""Derivative data collector — funding rates and open interest from Binance.

.. warning::

   **Do not use this module for open-interest history.** The
   ``/futures/data/openInterestHist`` endpoint is hard-capped at **30 days**
   and rejects any older ``startTime`` with error -1130. Any backtest built on
   it is impossible, not merely short. Use :mod:`vfund.deriv_alpha.vision`
   instead, which reads the Vision data archive and reaches back to 2020-09.

   The funding-rate fetcher here is fine — that endpoint paginates freely.

All endpoints are public (no API key required). Data is fetched at daily
granularity and stored as long-format polars DataFrames matching VFund's
panel convention, so existing alignment and backtest tools work directly.

Endpoints used:
    /fapi/v1/fundingRate           — 8-hourly funding rates (paginated, deep)
    /futures/data/openInterestHist — open interest (30-day cap; see warning)
    /fapi/v1/exchangeInfo          — active USD-M perpetual symbols
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import requests

_FAPI_BASE = "https://fapi.binance.com"
_FUNDING_URL = f"{_FAPI_BASE}/fapi/v1/fundingRate"
_OI_URL = f"{_FAPI_BASE}/futures/data/openInterestHist"
_EXCHANGE_INFO_URL = f"{_FAPI_BASE}/fapi/v1/exchangeInfo"

_FUNDING_LIMIT = 1000
_OI_LIMIT = 500
_PAUSE = 0.15


def _to_ms(dt: datetime | str) -> int:
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def perp_symbols(session: requests.Session | None = None) -> list[str]:
    """Return active USD-M perpetual contract symbols on Binance."""
    sess = session or requests.Session()
    resp = sess.get(_EXCHANGE_INFO_URL, timeout=30)
    resp.raise_for_status()
    info = resp.json()
    return sorted(
        s["symbol"]
        for s in info["symbols"]
        if s.get("contractType") == "PERPETUAL"
        and s.get("status") == "TRADING"
        and s["symbol"].endswith("USDT")
    )


def fetch_funding(
    symbols: list[str],
    start: datetime | str,
    end: datetime | str,
    *,
    session: requests.Session | None = None,
) -> pl.DataFrame:
    """Fetch 8-hourly funding rates for *symbols* over [start, end].

    Returns a long DataFrame (timestamp, symbol, funding_rate) with Datetime[ms, UTC].
    Paginates automatically past Binance's 1000-record limit.
    """
    sess = session or requests.Session()
    start_ms, end_ms = _to_ms(start), _to_ms(end)
    records: list[dict] = []

    for sym in symbols:
        cursor = start_ms
        n = 0
        while cursor < end_ms:
            resp = sess.get(
                _FUNDING_URL,
                params={
                    "symbol": sym,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": _FUNDING_LIMIT,
                },
                timeout=30,
            )
            data = resp.json()
            if not data or isinstance(data, dict):
                break
            for r in data:
                records.append(
                    {
                        "timestamp": int(r["fundingTime"]),
                        "symbol": str(r["symbol"]),
                        "funding_rate": float(r["fundingRate"]),
                    }
                )
                n += 1
            cursor = int(data[-1]["fundingTime"]) + 1
            if len(data) < _FUNDING_LIMIT:
                break
            time.sleep(_PAUSE)
        print(f"  funding {sym}: {n} records")

    if not records:
        raise RuntimeError("no funding data fetched")
    return (
        pl.DataFrame(records)
        .with_columns(pl.col("timestamp").cast(pl.Datetime("ms", time_zone="UTC")))
        .sort(["symbol", "timestamp"])
    )


def fetch_oi(
    symbols: list[str],
    start: datetime | str,
    end: datetime | str,
    *,
    period: str = "1d",
    session: requests.Session | None = None,
) -> pl.DataFrame:
    """Fetch open-interest history for *symbols* over [start, end].

    Returns a long DataFrame (timestamp, symbol, oi, oi_value) where *oi* is
    in base-asset contracts and *oi_value* is the USD notional.
    """
    sess = session or requests.Session()
    start_ms, end_ms = _to_ms(start), _to_ms(end)
    records: list[dict] = []

    for sym in symbols:
        cursor = start_ms
        n = 0
        while cursor < end_ms:
            resp = sess.get(
                _OI_URL,
                params={
                    "symbol": sym,
                    "period": period,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": _OI_LIMIT,
                },
                timeout=30,
            )
            data = resp.json()
            if not data or isinstance(data, dict):
                break
            for r in data:
                records.append(
                    {
                        "timestamp": int(r["timestamp"]),
                        "symbol": str(r["symbol"]),
                        "oi": float(r["sumOpenInterest"]),
                        "oi_value": float(r["sumOpenInterestValue"]),
                    }
                )
                n += 1
            cursor = int(data[-1]["timestamp"]) + 1
            if len(data) < _OI_LIMIT:
                break
            time.sleep(_PAUSE)
        print(f"  oi {sym}: {n} records")

    if not records:
        raise RuntimeError("no OI data fetched")
    return (
        pl.DataFrame(records)
        .with_columns(pl.col("timestamp").cast(pl.Datetime("ms", time_zone="UTC")))
        .sort(["symbol", "timestamp"])
    )


def align_oi(
    oi: pl.DataFrame,
    timestamps: pl.Series,
    symbols: list[str],
) -> np.ndarray:
    """Forward-fill OI onto a price timeline -> (T, N) matrix.

    *symbols* may be bare (``BTC``) or suffixed (``BTCUSDT``); both forms are
    tried against the OI data so the caller need not normalise.
    """
    base = pl.DataFrame({"timestamp": timestamps})
    cols: list[np.ndarray] = []
    for sym in symbols:
        candidates = (
            [sym, f"{sym}USDT"] if not sym.endswith("USDT") else [sym, sym[:-4]]
        )
        found = False
        for cand in candidates:
            sub = oi.filter(pl.col("symbol") == cand)
            if sub.height == 0:
                continue
            f = sub.select(["timestamp", "oi_value"]).sort("timestamp")
            f = f.with_columns(pl.col("timestamp").dt.truncate("1d"))
            f = f.group_by("timestamp").agg(pl.col("oi_value").last()).sort("timestamp")
            joined = base.join(f, on="timestamp", how="left").sort("timestamp")
            series = joined["oi_value"].fill_null(strategy="forward")
            cols.append(series.to_numpy())
            found = True
            break
        if not found:
            cols.append(np.full(timestamps.len(), np.nan))
    return np.column_stack(cols) if cols else np.zeros((timestamps.len(), 0))


def save(df: pl.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to parquet, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def load(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(path)
