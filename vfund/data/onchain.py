"""On-chain fundamentals from DefiLlama (free, no API key).

Total Value Locked (TVL) = the real capital deposited in a protocol. It's a
*fundamental usage* signal that has no equivalent in traditional markets and is
far less mined than price data — the kind of differentiated input where a small
player can still find edge. The economic story: protocols gaining real capital
and usage (rising TVL) should see their tokens follow.

Endpoints used:
- https://api.llama.fi/protocols            (list: symbol, slug, tvl)
- https://api.llama.fi/protocol/{slug}      (historical TVL series)
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import requests

_PROTOCOLS_URL = "https://api.llama.fi/protocols"
_PROTOCOL_URL = "https://api.llama.fi/protocol/{slug}"
_FEES_URL = "https://api.llama.fi/summary/fees/{slug}?dataType=dailyFees"


def slugs_by_symbol(session: requests.Session | None = None) -> dict[str, str]:
    """Map each token symbol to its highest-TVL DefiLlama protocol slug."""
    sess = session or requests.Session()
    protocols = sess.get(_PROTOCOLS_URL, timeout=30).json()
    best: dict[str, tuple[float, str]] = {}
    for p in protocols:
        sym, slug, tvl = p.get("symbol"), p.get("slug"), p.get("tvl") or 0.0
        if not sym or sym == "-" or not slug:
            continue
        if sym not in best or tvl > best[sym][0]:
            best[sym] = (tvl, slug)
    return {sym: slug for sym, (_, slug) in best.items()}


def fetch_protocol_tvl(slug: str, session: requests.Session | None = None) -> pl.DataFrame:
    """Fetch a protocol's daily historical TVL as (timestamp, tvl)."""
    sess = session or requests.Session()
    data = sess.get(_PROTOCOL_URL.format(slug=slug), timeout=30).json()
    points = data.get("tvl", [])
    if not points:
        raise RuntimeError(f"no TVL history for {slug}")
    return pl.DataFrame(
        {
            "timestamp": [int(p["date"]) * 1000 for p in points],
            "tvl": [float(p.get("totalLiquidityUSD", 0.0)) for p in points],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", time_zone="UTC")))


_STABLECOIN_URL = "https://stablecoins.llama.fi/stablecoincharts/all"


def fetch_stablecoin_supply(
    *, start: datetime | str, end: datetime | str | None = None,
) -> pl.DataFrame:
    """Fetch aggregate stablecoin circulating supply (timestamp, supply) in USD.

    Total stablecoin supply is a macro liquidity gauge: when it grows, capital is
    flowing into crypto (risk-on); when it shrinks, capital is leaving. Not a
    cross-sectional signal — a market-timing / regime overlay.
    """
    data = requests.get(_STABLECOIN_URL, timeout=30).json()

    def circ(x):
        c = x.get("totalCirculating", {})
        return float(c.get("peggedUSD", 0.0)) if isinstance(c, dict) else 0.0

    start_ms, end_ms = _to_ms(start), _to_ms(end) if end else int(datetime.now().timestamp() * 1000)
    return (
        pl.DataFrame({"timestamp": [int(x["date"]) * 1000 for x in data],
                      "supply": [circ(x) for x in data]})
        .with_columns(pl.col("timestamp").cast(pl.Datetime("ms", time_zone="UTC")))
        .filter((pl.col("timestamp") >= _dt(start_ms)) & (pl.col("timestamp") <= _dt(end_ms)))
        .sort("timestamp")
    )


def fetch_protocol_fees(slug: str, session: requests.Session | None = None) -> pl.DataFrame:
    """Fetch a protocol's daily fee (revenue) history as (timestamp, fees)."""
    sess = session or requests.Session()
    data = sess.get(_FEES_URL.format(slug=slug), timeout=30).json()
    chart = data.get("totalDataChart", [])
    if not chart:
        raise RuntimeError(f"no fee history for {slug}")
    return pl.DataFrame(
        {
            "timestamp": [int(p[0]) * 1000 for p in chart],
            "fees": [float(p[1]) for p in chart],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", time_zone="UTC")))


def fetch_fees_panel(
    symbols: list[str],
    *,
    start: datetime | str,
    end: datetime | str | None = None,
) -> pl.DataFrame:
    """Build a long fee-revenue panel (timestamp, symbol, fees) for the symbols.

    Fees measure real protocol *revenue* — distinct from TVL (parked capital) —
    so a fee signal captures usage/earnings the price may not yet reflect.
    """
    sess = requests.Session()
    smap = slugs_by_symbol(sess)
    start_ms, end_ms = _to_ms(start), _to_ms(end) if end else int(datetime.now().timestamp() * 1000)
    frames = []
    for sym in symbols:
        slug = smap.get(sym)
        if slug is None:
            continue
        try:
            df = fetch_protocol_fees(slug, sess)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {sym} ({slug}): {exc}")
            continue
        df = (
            df.filter((pl.col("timestamp") >= _dt(start_ms)) & (pl.col("timestamp") <= _dt(end_ms)))
            .with_columns(pl.col("timestamp").dt.truncate("1d"))
            .group_by("timestamp").agg(pl.col("fees").last())
            .with_columns(pl.lit(sym).alias("symbol"))
        )
        frames.append(df.select(["timestamp", "symbol", "fees"]))
        print(f"  + {sym} ({slug}): {df.height} days")
    if not frames:
        raise RuntimeError("no fees fetched")
    return pl.concat(frames).sort(["symbol", "timestamp"])


def fetch_tvl_panel(
    symbols: list[str],
    *,
    start: datetime | str,
    end: datetime | str | None = None,
) -> pl.DataFrame:
    """Build a long TVL panel (timestamp, symbol, tvl) for the given coin symbols.

    ``symbols`` are bare tokens (e.g. ``"AAVE"``), matched to DefiLlama protocols.
    Timestamps are snapped to daily (00:00 UTC). Missing protocols are skipped.
    """
    sess = requests.Session()
    smap = slugs_by_symbol(sess)
    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end is not None else int(datetime.now().timestamp() * 1000)

    frames = []
    for sym in symbols:
        slug = smap.get(sym)
        if slug is None:
            print(f"  ! no DefiLlama protocol for {sym}")
            continue
        try:
            df = fetch_protocol_tvl(slug, sess)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {sym} ({slug}): {exc}")
            continue
        df = (
            df.filter((pl.col("timestamp") >= _dt(start_ms)) & (pl.col("timestamp") <= _dt(end_ms)))
            .with_columns(pl.col("timestamp").dt.truncate("1d"))
            .group_by("timestamp").agg(pl.col("tvl").last())
            .with_columns(pl.lit(sym).alias("symbol"))
        )
        frames.append(df.select(["timestamp", "symbol", "tvl"]))
        print(f"  + {sym} ({slug}): {df.height} days")

    if not frames:
        raise RuntimeError("no TVL fetched")
    return pl.concat(frames).sort(["symbol", "timestamp"])


def align_tvl(tvl: pl.DataFrame, timestamps: pl.Series, symbols: list[str]) -> "object":
    """Forward-fill a TVL panel onto a price timeline -> (T, N) matrix."""
    import numpy as np

    base = pl.DataFrame({"timestamp": timestamps})
    cols = []
    for sym in symbols:
        f = tvl.filter(pl.col("symbol") == sym).select(["timestamp", "tvl"]).sort("timestamp")
        joined = base.join(f, on="timestamp", how="left").sort("timestamp")
        series = joined["tvl"].fill_null(strategy="forward")
        cols.append(series.to_numpy())
    return np.column_stack(cols) if cols else np.zeros((timestamps.len(), 0))


def save_tvl(df: pl.DataFrame, path) -> object:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def load_tvl(path) -> pl.DataFrame:
    return pl.read_parquet(path)


def _to_ms(dt) -> int:
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _dt(ms: int):
    return pl.lit(ms).cast(pl.Datetime("ms", time_zone="UTC"))
