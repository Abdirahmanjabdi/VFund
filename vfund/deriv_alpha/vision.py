"""Historical derivative metrics from the Binance Vision data dump.

Why this module exists
----------------------
The obvious source for open-interest history is the REST endpoint
``/futures/data/openInterestHist``. It is a trap: it is hard-capped at **30
days** and rejects any ``startTime`` older than that with error -1130. Any
backtest built on it is impossible, not merely short.

The Vision data dump (``data.binance.vision``) publishes one zip per symbol per
day under ``futures/um/daily/metrics/``, at 5-minute resolution, back to
2020-09. It also carries four fields the REST endpoint never exposes:

===============================  =========================================
``sum_open_interest``            OI, in contracts
``sum_open_interest_value``      OI, in USD notional
``count_long_short_ratio``       long/short ratio across **all** accounts
``count_toptrader_long_short_ratio``  same, **top accounts by margin balance**
``sum_toptrader_long_short_ratio``    top accounts, **position-weighted**
``sum_taker_long_short_vol_ratio``    aggressive taker buy/sell imbalance
===============================  =========================================

The two ``*_long_short_ratio`` families are the interesting part: one measures
where the crowd sits, the other where the largest accounts sit. Their spread is
a direct read on informed-versus-uninformed positioning, and it has structural
analogues in every other market (COT commercial-vs-noncommercial in futures,
13F-vs-retail in equities).

Everything here is public and unauthenticated. No API key is used or needed.
"""

from __future__ import annotations

import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import requests

VISION = "https://data.binance.vision"
_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

# Raw CSV column -> the name we carry forward.
_COLS = {
    "sum_open_interest": "oi",
    "sum_open_interest_value": "oi_value",
    "count_long_short_ratio": "ls_all",
    "count_toptrader_long_short_ratio": "ls_top_acct",
    "sum_toptrader_long_short_ratio": "ls_top_pos",
    "sum_taker_long_short_vol_ratio": "taker_ls",
}

# Stock variables take the day's closing observation; flow/ratio variables take
# the day's mean, which is less sensitive to a single noisy 5-minute print.
_LAST = ("oi", "oi_value")
_MEAN = ("ls_all", "ls_top_acct", "ls_top_pos", "taker_ls")


_DATED = re.compile(r"-metrics-(\d{4}-\d{2}-\d{2})\.zip$")


def available_from(symbol: str, *, session: requests.Session | None = None) -> date | None:
    """First date with a metrics file for ``symbol``, or None if none exist.

    The dump contains a small number of mislabelled objects — ``MATICUSDT`` and
    ``TRBUSDT`` each carry a ``-metrics-1993.zip``. S3 lists lexicographically,
    so those sort ahead of every real date and land first. Anything that is not
    an ISO date is skipped rather than parsed.
    """
    s = session or requests.Session()
    r = s.get(
        _LIST,
        params={
            "delimiter": "/",
            "prefix": f"data/futures/um/daily/metrics/{symbol}/",
            "max-keys": 20,
        },
        timeout=30,
    )
    r.raise_for_status()
    for key in re.findall(r"<Key>([^<]+)</Key>", r.text):
        m = _DATED.search(key)
        if m:
            try:
                return date.fromisoformat(m.group(1))
            except ValueError:  # pragma: no cover - regex already constrains this
                continue
    return None


def _fetch_day(symbol: str, day: date, session: requests.Session) -> dict | None:
    """Download and collapse one symbol-day to a single daily record."""
    url = f"{VISION}/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{day}.zip"
    try:
        r = session.get(url, timeout=60)
        if r.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            raw = z.read(z.namelist()[0])
    except (requests.RequestException, zipfile.BadZipFile):
        return None

    df = pl.read_csv(raw, ignore_errors=True)
    missing = set(_COLS) - set(df.columns)
    if missing or df.height == 0:
        return None

    df = df.rename(_COLS).select(list(_COLS.values()))
    # Ratio columns arrive as text on some days; coerce and drop non-numeric.
    df = df.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in _COLS.values()]
    )

    rec: dict = {"timestamp": datetime(day.year, day.month, day.day), "symbol": symbol}
    for c in _LAST:
        col = df[c].drop_nulls()
        rec[c] = float(col[-1]) if col.len() else None
    for c in _MEAN:
        col = df[c].drop_nulls()
        # Ratios are strictly positive; zeros and negatives are bad prints.
        col = col.filter((col > 0) & col.is_finite())
        rec[c] = float(col.mean()) if col.len() else None

    # --- intraday structure -------------------------------------------------
    # The file is 5-minute resolution and collapsing it to one number per day
    # discards the *path*. Two symbols can end the day with identical open
    # interest after completely different sessions: one drifted there, the
    # other churned violently. That difference is not recoverable later without
    # re-downloading, so it is computed here.
    oi = df["oi_value"].drop_nulls()
    oi = oi.filter((oi > 0) & oi.is_finite())
    if oi.len() >= 3:
        v = oi.to_numpy()
        rec["oi_open"] = float(v[0])
        # Churn: dispersion of within-day log changes in open interest. High
        # churn means positions are being opened and closed against each other
        # rather than accumulating.
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.diff(np.log(v))
        d = d[np.isfinite(d)]
        rec["oi_churn"] = float(d.std()) if d.size >= 2 else None
        rec["oi_range"] = float((v.max() - v.min()) / v.mean()) if v.mean() > 0 else None
    else:
        rec["oi_open"] = rec["oi_churn"] = rec["oi_range"] = None

    ls = df["ls_all"].drop_nulls()
    ls = ls.filter((ls > 0) & ls.is_finite())
    # Where the crowd's positioning started the day, so its intraday drift can
    # be separated from the overnight gap.
    rec["ls_all_open"] = float(ls[0]) if ls.len() else None
    rec["ls_all_close"] = float(ls[-1]) if ls.len() else None

    tk = df["taker_ls"].drop_nulls()
    tk = tk.filter((tk > 0) & tk.is_finite())
    rec["taker_ls_std"] = float(tk.std()) if tk.len() >= 2 else None
    return rec


def fetch_metrics(
    symbols: list[str],
    start: date | str,
    end: date | str,
    *,
    workers: int = 24,
    progress: bool = True,
) -> pl.DataFrame:
    """Download daily derivative metrics for ``symbols`` over ``[start, end]``.

    Missing symbol-days (before listing, or gaps in the dump) are skipped
    silently rather than raising — coverage varies by symbol and the caller
    aligns to a price panel afterwards.

    Returns a long DataFrame: timestamp, symbol, oi, oi_value, ls_all,
    ls_top_acct, ls_top_pos, taker_ls.
    """
    start = date.fromisoformat(str(start)[:10])
    end = date.fromisoformat(str(end)[:10])
    if start > end:
        raise ValueError("start must be <= end")

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=workers * 2, max_retries=2)
    session.mount("https://", adapter)

    # Only ask for days a symbol could plausibly have.
    # A multi-hour download must not be killable by one unlisted or malformed
    # symbol, so enumeration failures skip that symbol instead of raising.
    jobs: list[tuple[str, date]] = []
    skipped: list[str] = []
    for sym in symbols:
        try:
            first = available_from(sym, session=session)
        except (requests.RequestException, ValueError):
            skipped.append(sym)
            continue
        if first is None:
            skipped.append(sym)
            continue
        d = max(first, start)
        while d <= end:
            jobs.append((sym, d))
            d += timedelta(days=1)
    if progress and skipped:
        print(f"  skipped {len(skipped)} symbols with no usable metrics listing")

    if progress:
        print(f"  {len(jobs):,} symbol-days to fetch across {len(symbols)} symbols")

    rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_fetch_day, s, d, session): (s, d) for s, d in jobs}
        for fut in as_completed(futs):
            rec = fut.result()
            if rec is not None:
                rows.append(rec)
            done += 1
            if progress and done % 2000 == 0:
                print(f"    {done:,}/{len(jobs):,}  ({len(rows):,} rows)", flush=True)

    if not rows:
        raise RuntimeError("no metrics retrieved — check symbols and date range")

    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
        .sort(["symbol", "timestamp"])
    )


FIELDS = (
    "oi", "oi_value", "ls_all", "ls_top_acct", "ls_top_pos", "taker_ls",
    "oi_open", "oi_churn", "oi_range", "ls_all_open", "ls_all_close",
    "taker_ls_std",
)


def align_metrics(
    metrics: pl.DataFrame,
    timestamps: pl.Series,
    symbols: list[str],
    fields: tuple[str, ...] = FIELDS,
    *,
    ffill_limit: int = 3,
    lag: int = 0,
) -> dict[str, np.ndarray]:
    """Align long-format metrics onto a price timeline.

    Args:
        metrics: long DataFrame from :func:`fetch_metrics`.
        timestamps: the price panel's timestamp column.
        symbols: panel symbols, in column order.
        fields: which metric columns to align.
        ffill_limit: how many bars a stale value may be carried forward. Gaps in
            the dump are short; carrying a value indefinitely would invent data
            for delisted or halted symbols.
        lag: shift every series back by this many bars. ``lag=0`` is
            contemporaneous (correct here — end-of-day metrics are published
            intraday via the live endpoint, so they are genuinely known at the
            close that trades on them). ``lag=1`` is the paranoid setting, and
            any real edge should survive it.

    Returns:
        ``{field: (T, N) array}``, NaN where unavailable.
    """
    _np = np

    if lag < 0:
        raise ValueError("lag must be >= 0")

    ts = pl.DataFrame({"timestamp": timestamps}).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone(None)
    )
    m = metrics.with_columns(
        pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone(None)
    )
    # Panel symbols may be bare ("BTC") while metrics are suffixed ("BTCUSDT").
    have = set(m["symbol"].unique().to_list())
    keymap = {s: (s if s in have else f"{s}USDT") for s in symbols}

    T, N = len(ts), len(symbols)
    out: dict[str, _np.ndarray] = {}
    for f in fields:
        if f not in m.columns:
            continue
        wide = m.select("timestamp", "symbol", f).pivot(
            index="timestamp", on="symbol", values=f, aggregate_function="first"
        )
        joined = ts.join(wide, on="timestamp", how="left").sort("timestamp")
        cols = []
        for s in symbols:
            k = keymap[s]
            cols.append(
                joined[k].forward_fill(limit=ffill_limit)
                if k in joined.columns
                else pl.Series(k, [None] * T, dtype=pl.Float64)
            )
        arr = _np.column_stack([c.cast(pl.Float64).to_numpy() for c in cols])
        arr = _np.where(_np.isfinite(arr), arr, _np.nan)
        if lag:
            shifted = _np.full((T, N), _np.nan)
            shifted[lag:] = arr[:-lag]
            arr = shifted
        out[f] = arr
    return out


def save(df: pl.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def load(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(path)
