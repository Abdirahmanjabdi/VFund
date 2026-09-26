"""Market adapters — the same five factors, different data sources per market.

The derivative microstructure factors operate on two standardised inputs:

    positioning : (T, N) array
        How crowded is each asset? The magnitude and direction of speculative
        positioning. What this *is* depends on the market:
            - Crypto perps: open interest (USD notional)
            - US equities: short interest (shares short / float)
            - Commodity / financial futures: CFTC net speculative positions
            - FX: IMM speculative positioning

    carry_cost : (T, N) array
        What does it cost to hold the crowded side? Also market-dependent:
            - Crypto perps: funding rate (paid 3x/day by longs when positive)
            - US equities: short borrow fee (paid by shorts)
            - Futures: contango/backwardation (roll cost for longs)

The factors do not know or care which market produced the data. They rank
the cross-section by derived features and trade the extremes. The adapter's
job is to fetch, align, and present the data in this standardised form.

Extending to a new market
--------------------------
1. Subclass :class:`MarketAdapter`.
2. Implement :meth:`fetch` to pull historical data from your market's source.
3. Implement :meth:`positioning` and :meth:`carry_cost` to return aligned arrays.
4. Pass the arrays to :class:`~vfund.deriv_alpha.composer.DerivAlphaComposer`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import polars as pl


class MarketAdapter(ABC):
    """Abstract interface — one per market type."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier: 'crypto_perps', 'us_equities', 'cme_futures'."""
        ...

    @abstractmethod
    def fetch(
        self,
        symbols: list[str],
        start: datetime | str,
        end: datetime | str,
    ) -> None:
        """Fetch raw positioning + carry data from the market's source.

        Implementations should store the fetched data internally so that
        :meth:`positioning` and :meth:`carry_cost` can return it aligned
        to a price timeline.
        """
        ...

    @abstractmethod
    def positioning(
        self, timestamps: pl.Series, symbols: list[str]
    ) -> np.ndarray:
        """(T, N) positioning array aligned to the price timeline.

        Each cell is the positioning metric for that (bar, symbol). NaN where
        data is unavailable.
        """
        ...

    @abstractmethod
    def carry_cost(
        self, timestamps: pl.Series, symbols: list[str]
    ) -> pl.DataFrame:
        """Carry-cost data as a long DataFrame (timestamp, symbol, funding_rate).

        Returns the carry metric in the same format that VFund's backtester
        expects for funding alignment. Column must be ``funding_rate`` for
        compatibility with :func:`~vfund.data.panel.align_funding`.
        """
        ...


# ---------------------------------------------------------------------------
# Crypto perpetual futures (Binance USD-M)
# ---------------------------------------------------------------------------


class CryptoPerpsAdapter(MarketAdapter):
    """Binance USD-M perpetual futures — OI as positioning, funding as carry.

    All data comes from public (no-auth) endpoints. This is the first and
    most complete adapter because crypto markets publish the richest real-time
    positioning data of any asset class.
    """

    name = "crypto_perps"

    def __init__(self) -> None:
        self._funding: pl.DataFrame | None = None
        self._oi: pl.DataFrame | None = None

    def fetch(
        self,
        symbols: list[str],
        start: datetime | str,
        end: datetime | str,
    ) -> None:
        from vfund.deriv_alpha.collector import fetch_funding, fetch_oi

        self._funding = fetch_funding(symbols, start, end)
        self._oi = fetch_oi(symbols, start, end)

    def load(self, funding_path: str, oi_path: str) -> None:
        """Load previously cached parquet files instead of fetching."""
        from vfund.deriv_alpha.collector import load

        self._funding = load(funding_path)
        self._oi = load(oi_path)

    def positioning(
        self, timestamps: pl.Series, symbols: list[str]
    ) -> np.ndarray:
        if self._oi is None:
            raise RuntimeError("call fetch() or load() first")
        from vfund.deriv_alpha.collector import align_oi

        return align_oi(self._oi, timestamps, symbols)

    def carry_cost(
        self, timestamps: pl.Series, symbols: list[str]
    ) -> pl.DataFrame:
        if self._funding is None:
            raise RuntimeError("call fetch() or load() first")
        return self._funding


# ---------------------------------------------------------------------------
# Stubs for future market adapters
# ---------------------------------------------------------------------------


class USEquityAdapter(MarketAdapter):
    """US equities — short interest as positioning, borrow fee as carry.

    Data sources (to implement):
        - Short interest: FINRA/exchange reports (bi-monthly, ~2 week lag)
        - Options OI + put/call ratio: CBOE (daily)
        - Borrow fee: interactive brokers API or securities lending desk
        - Dark pool volume: FINRA ATS data

    The same five factors apply:
        - OI-Price Divergence -> Short-interest-Price Divergence
        - Crowding Pressure -> Short squeeze risk (high SI + high borrow)
        - Liquidation Bounce -> Short-covering bounce after margin calls
        - Funding Momentum -> Borrow-cost momentum
        - OI Acceleration -> Short-interest acceleration
    """

    name = "us_equities"

    def fetch(self, symbols, start, end):
        raise NotImplementedError(
            "US equity adapter not yet implemented. Need: short interest data "
            "(FINRA), options OI (CBOE), borrow fees (broker API)."
        )

    def positioning(self, timestamps, symbols):
        raise NotImplementedError

    def carry_cost(self, timestamps, symbols):
        raise NotImplementedError


class FuturesAdapter(MarketAdapter):
    """Commodity / financial futures — COT net speculative as positioning.

    Data sources (to implement):
        - CFTC Commitment of Traders: weekly, ~3 day lag
        - Exchange OI: CME, ICE (daily)
        - Roll yield / contango: derived from term structure

    The same five factors apply:
        - OI-Price Divergence -> Speculative positioning vs price
        - Crowding Pressure -> Extreme net long/short + high roll cost
        - Liquidation Bounce -> Post-margin-call bounce in commodity blowoffs
        - Funding Momentum -> Contango/backwardation momentum
        - OI Acceleration -> Change in pace of speculative entry
    """

    name = "cme_futures"

    def fetch(self, symbols, start, end):
        raise NotImplementedError(
            "Futures adapter not yet implemented. Need: CFTC COT data, "
            "CME/ICE OI, term structure data."
        )

    def positioning(self, timestamps, symbols):
        raise NotImplementedError

    def carry_cost(self, timestamps, symbols):
        raise NotImplementedError
