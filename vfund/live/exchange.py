"""Binance USD-M Futures API client.

Thin, stateless wrapper around the REST API. Uses ``requests`` (already a
dependency) — no third-party Binance SDK needed. Supports both mainnet and
testnet via a single flag.

Authentication
--------------
Every private endpoint is HMAC-SHA256 signed per Binance's specification.
API keys are passed at construction time, never logged, and never stored to
disk. Read them from environment variables — they must not appear in source,
CLI arguments, or config files.

Safety
------
This module only wraps the API. It does not decide *what* to trade — that is
the job of :mod:`vfund.live.execute`, which adds reconciliation, safety
checks, and logging on top of this client. Calling :meth:`place_market_order`
directly bypasses all of those safeguards and should be done only in tests.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import time
from dataclasses import dataclass
from urllib.parse import urlencode

import requests


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolSpec:
    """Exchange-enforced constraints for a trading pair."""

    symbol: str
    step_size: float
    min_qty: float
    min_notional: float
    qty_precision: int
    price_precision: int


@dataclass(frozen=True)
class OrderResult:
    """The exchange's response to a filled order."""

    symbol: str
    side: str
    quantity: float
    price: float
    status: str
    order_id: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def round_step_down(value: float, step: float, precision: int) -> float:
    """Round *down* to the nearest multiple of ``step``.

    Always rounds toward zero — never orders more than intended. The result
    is rounded to ``precision`` decimal places to avoid float artefacts.
    """
    if step <= 0 or value <= 0:
        return 0.0
    result = math.floor(value / step) * step
    return round(result, precision)


def fmt_qty(qty: float, spec: SymbolSpec) -> str:
    """Format a quantity for the Binance API, respecting step size."""
    stepped = round_step_down(abs(qty), spec.step_size, spec.qty_precision)
    return f"{stepped:.{spec.qty_precision}f}"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class BinanceError(RuntimeError):
    """An error returned by the Binance API."""

    def __init__(self, code: int, msg: str):
        self.code = code
        super().__init__(f"Binance error {code}: {msg}")


class BinanceClient:
    """Authenticated Binance USD-M Futures REST client.

    Args:
        api_key: your Binance API key.
        api_secret: your Binance API secret.
        testnet: if True (default), connect to the testnet — free fake money,
            no financial risk.
    """

    MAINNET_URL = "https://fapi.binance.com"
    TESTNET_URL = "https://testnet.binancefuture.com"

    def __init__(
        self, api_key: str, api_secret: str, *, testnet: bool = True
    ):
        if not api_key or not api_secret:
            raise ValueError(
                "API key and secret are required — set BINANCE_TESTNET_KEY "
                "and BINANCE_TESTNET_SECRET environment variables"
            )
        self._key = api_key
        self._secret = api_secret.encode("utf-8")
        self._base = self.TESTNET_URL if testnet else self.MAINNET_URL
        self._session = requests.Session()
        self._session.headers["X-MBX-APIKEY"] = api_key
        self.testnet = testnet

    def close(self) -> None:
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- signing ----------------------------------------------------------

    def _sign(self, params: dict) -> dict:
        """Add timestamp and HMAC-SHA256 signature to *params* (mutates)."""
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        qs = urlencode(params)
        sig = hmac.new(self._secret, qs.encode("utf-8"), hashlib.sha256)
        params["signature"] = sig.hexdigest()
        return params

    # ---- transport --------------------------------------------------------

    def _get(
        self, path: str, params: dict | None = None, *, signed: bool = False
    ):
        params = dict(params or {})
        if signed:
            params = self._sign(params)
        resp = self._session.get(
            f"{self._base}{path}", params=params, timeout=15
        )
        self._check(resp)
        return resp.json()

    def _post(self, path: str, params: dict):
        params = self._sign(dict(params))
        resp = self._session.post(
            f"{self._base}{path}", params=params, timeout=15
        )
        self._check(resp)
        return resp.json()

    @staticmethod
    def _check(resp: requests.Response) -> None:
        """Raise on HTTP or Binance-level errors."""
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            try:
                body = resp.json()
                raise BinanceError(body.get("code", -1), body.get("msg", str(exc))) from exc
            except (ValueError, KeyError):
                raise exc
        data = resp.json() if resp.content else {}
        if isinstance(data, dict) and data.get("code", 0) < 0:
            raise BinanceError(data["code"], data.get("msg", "unknown"))

    # ---- public endpoints -------------------------------------------------

    def exchange_info(self) -> dict[str, SymbolSpec]:
        """Fetch per-symbol filters (lot size, min notional, precision).

        Returns a dict keyed by symbol (e.g. ``"BTCUSDT"``).
        """
        data = self._get("/fapi/v1/exchangeInfo")
        specs: dict[str, SymbolSpec] = {}
        for s in data["symbols"]:
            if s.get("status") != "TRADING":
                continue
            if s.get("contractType") != "PERPETUAL":
                continue
            filters = {f["filterType"]: f for f in s["filters"]}
            lot = filters.get("LOT_SIZE", {})
            # Market orders may have a separate, tighter lot filter.
            mkt = filters.get("MARKET_LOT_SIZE", lot)
            notn = filters.get("MIN_NOTIONAL", {})
            specs[s["symbol"]] = SymbolSpec(
                symbol=s["symbol"],
                step_size=float(mkt.get("stepSize", lot.get("stepSize", "0.001"))),
                min_qty=float(mkt.get("minQty", lot.get("minQty", "0.001"))),
                min_notional=float(notn.get("notional", "5")),
                qty_precision=int(s.get("quantityPrecision", 3)),
                price_precision=int(s.get("pricePrecision", 2)),
            )
        return specs

    def ticker_prices(self) -> dict[str, float]:
        """Current mark prices for all USD-M perp symbols."""
        data = self._get("/fapi/v1/ticker/price")
        return {d["symbol"]: float(d["price"]) for d in data}

    # ---- private endpoints ------------------------------------------------

    def account_equity(self) -> float:
        """Total wallet balance (USDT) including unrealised P&L."""
        data = self._get("/fapi/v2/account", signed=True)
        return float(data["totalMarginBalance"])

    def wallet_balance(self) -> float:
        """Total wallet balance (USDT) excluding unrealised P&L."""
        data = self._get("/fapi/v2/account", signed=True)
        return float(data["totalWalletBalance"])

    def positions(self) -> dict[str, float]:
        """Current position quantities by symbol.

        Returns only non-zero positions. Positive = long, negative = short.
        """
        data = self._get("/fapi/v2/positionRisk", signed=True)
        out: dict[str, float] = {}
        for p in data:
            qty = float(p["positionAmt"])
            if abs(qty) > 0:
                out[p["symbol"]] = qty
        return out

    def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set the maximum leverage for ``symbol``.

        Acts as a safety rail: even if a bug tries to over-size a position,
        the exchange will reject it.
        """
        self._post("/fapi/v1/leverage", {
            "symbol": symbol,
            "leverage": leverage,
        })

    def place_market_order(
        self, symbol: str, side: str, quantity: str
    ) -> OrderResult:
        """Place a MARKET order. ``quantity`` is a pre-formatted string.

        **Do not call this directly** — use :func:`vfund.live.execute.run`
        which wraps it with reconciliation, safety checks, and logging.
        """
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY or SELL, got {side!r}")
        data = self._post("/fapi/v1/order", {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
        })
        return OrderResult(
            symbol=data["symbol"],
            side=data["side"],
            quantity=float(data["executedQty"]),
            price=float(data.get("avgPrice", 0)),
            status=data["status"],
            order_id=int(data["orderId"]),
        )

    def cancel_all_orders(self, symbol: str) -> None:
        """Cancel all open orders for ``symbol`` (safety measure)."""
        self._post("/fapi/v1/allOpenOrders", {"symbol": symbol})
