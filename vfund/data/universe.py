"""Build a tradable universe and pull it as a panel from Binance.

Most crypto edge is *cross-sectional* — it lives in how coins move relative to
each other, not in any one of them. So research starts with a basket of liquid
names, fetched once into a single panel.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import requests

from vfund.data.ingest import fetch_klines
from vfund.data.panel import validate_panel

# A curated default of historically-liquid USDT pairs. Override with your own,
# or pull the current top-by-volume set via ``top_symbols_by_volume``.
DEFAULT_UNIVERSE: list[str] = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "MATICUSDT", "LTCUSDT", "TRXUSDT", "ATOMUSDT", "UNIUSDT",
    "ETCUSDT", "XLMUSDT", "BCHUSDT", "FILUSDT", "APTUSDT",
    "NEARUSDT", "ICPUSDT", "ARBUSDT", "OPUSDT", "INJUSDT",
    "AAVEUSDT", "SANDUSDT", "EOSUSDT", "ALGOUSDT", "EGLDUSDT",
]

_TICKER_24H_URL = "https://api.binance.com/api/v3/ticker/24hr"
_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"

# Curated USDT pairs delisted from Binance spot in the 2021-2025 era, for which
# the public klines endpoint still serves history (the candle series simply ends
# at the delisting date). Including these in a universe is the single biggest
# survivorship-bias fix available for free — these are the coins that died,
# merged, or were removed, and would otherwise silently vanish from a backtest.
# Not exhaustive; a representative sample of meaningful delistings.
KNOWN_DELISTED: list[str] = [
    "SRMUSDT", "WAVESUSDT", "ANTUSDT", "MIRUSDT", "TORNUSDT", "BTGUSDT",
    "MOBUSDT", "OMGUSDT", "AGIXUSDT", "OCEANUSDT", "KEEPUSDT", "AKROUSDT",
    "RGTUSDT", "TRIBEUSDT", "BONDUSDT", "MDTUSDT", "DREPUSDT", "VGXUSDT",
    "WRXUSDT", "LOOMUSDT", "RAMPUSDT", "WNXMUSDT", "BALUSDT", "NULSUSDT",
    "WINGUSDT", "FIROUSDT", "PROSUSDT", "BETAUSDT", "REEFUSDT", "AERGOUSDT",
    "KP3RUSDT", "UNFIUSDT", "BTSUSDT", "COCOSUSDT", "GTOUSDT", "STMXUSDT",
    "VIDTUSDT", "DOCKUSDT", "FORUSDT", "PERLUSDT", "LINAUSDT",
]


# Stablecoins and metal-pegged tokens — not tradable-alpha assets. "Top by
# volume" drags these in, so a clean universe must exclude them.
PEGGED: set[str] = {
    "USDCUSDT", "USD1USDT", "FDUSDUSDT", "EURUSDT", "RLUSDUSDT", "TUSDUSDT",
    "DAIUSDT", "PYUSDUSDT", "USDPUSDT", "BUSDUSDT", "PAXGUSDT", "XAUTUSDT",
}


def clean_universe(panel: pl.DataFrame, *, min_bars: int = 365) -> pl.DataFrame:
    """Drop pegged tokens, non-standard symbols, and coins with too little history.

    A raw "top by volume" fetch pulls in stablecoins, gold tokens, and brand-new
    listings with a handful of bars — none of which belong in a cross-sectional
    book. This filters to real, established USDT pairs.
    """
    counts = panel.group_by("symbol").agg(pl.len().alias("n"))
    enough = counts.filter(pl.col("n") >= min_bars)["symbol"].to_list()
    return panel.filter(
        pl.col("symbol").is_in(enough)
        & ~pl.col("symbol").is_in(list(PEGGED))
        & pl.col("symbol").str.contains(r"^[A-Z0-9]+USDT$")
    )


def top_symbols_by_volume(n: int = 30, quote: str = "USDT") -> list[str]:
    """Return the ``n`` highest 24h quote-volume spot pairs ending in ``quote``."""
    resp = requests.get(_TICKER_24H_URL, timeout=30)
    resp.raise_for_status()
    rows = [
        (r["symbol"], float(r["quoteVolume"]))
        for r in resp.json()
        if r["symbol"].endswith(quote) and not r["symbol"].startswith(quote)
    ]
    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in rows[:n]]


def fetch_universe(
    symbols: list[str],
    interval: str = "1h",
    *,
    start: datetime | str,
    end: datetime | str | None = None,
) -> pl.DataFrame:
    """Fetch bars for each symbol and stack them into one long panel."""
    frames = []
    for sym in symbols:
        try:
            df = fetch_klines(sym, interval, start=start, end=end)
        except Exception as exc:  # noqa: BLE001 - skip a bad symbol, keep going
            print(f"  ! skipping {sym}: {exc}")
            continue
        frames.append(df.with_columns(pl.lit(sym).alias("symbol")))
        print(f"  + {sym}: {df.height} bars")

    if not frames:
        raise RuntimeError("no symbols fetched successfully")
    return validate_panel(pl.concat(frames))


def _to_ms(dt: datetime | str) -> int:
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    return int(dt.timestamp() * 1000)


def fetch_funding(
    symbol: str,
    *,
    start: datetime | str,
    end: datetime | str | None = None,
    session: requests.Session | None = None,
    pause_s: float = 0.2,
) -> pl.DataFrame:
    """Fetch perpetual funding-rate history for ``symbol`` (USD-M futures).

    Funding is the structural carry of crypto: when perp longs crowd in, they pay
    shorts (every 8h on Binance). Harvesting that spread — short the payers, long
    the receivers — is a low-turnover edge that survives costs. Paginates across
    the 1000-row cap. Returns ``(timestamp, symbol, funding_rate)``.
    """
    import time

    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end is not None else int(datetime.now().timestamp() * 1000)
    sess = session or requests.Session()
    rows: list[dict] = []
    cursor = start_ms

    while cursor < end_ms:
        params = {"symbol": symbol.upper(), "startTime": cursor, "endTime": end_ms, "limit": 1000}
        resp = sess.get(_FUNDING_URL, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000:
            break
        time.sleep(pause_s)

    if not rows:
        raise RuntimeError(f"no funding data for {symbol}")
    return (
        pl.DataFrame(
            {
                "timestamp": [int(r["fundingTime"]) for r in rows],
                "symbol": [symbol.upper()] * len(rows),
                "funding_rate": [float(r["fundingRate"]) for r in rows],
            }
        )
        .with_columns(pl.col("timestamp").cast(pl.Datetime("ms", time_zone="UTC")))
        .unique(subset="timestamp", keep="first")
        .sort("timestamp")
    )


def fetch_funding_universe(
    symbols: list[str],
    *,
    start: datetime | str,
    end: datetime | str | None = None,
) -> pl.DataFrame:
    """Fetch funding history for each symbol and stack into one long panel."""
    frames = []
    for sym in symbols:
        try:
            f = fetch_funding(sym, start=start, end=end)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! skipping funding {sym}: {exc}")
            continue
        frames.append(f)
        print(f"  + {sym}: {f.height} funding points")
    if not frames:
        raise RuntimeError("no funding fetched successfully")
    return pl.concat(frames).sort(["symbol", "timestamp"])
