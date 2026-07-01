"""A minimal price-time limit order book with a matching engine.

Prices are integer ticks (avoids float-key headaches). Bids and asks are kept as
``price -> quantity`` maps; a market order walks the opposite side from the best
price inward, consuming liquidity. This is deliberately simple — enough to model
how passive quotes get filled — not a production matching engine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Fill:
    price: int      # tick
    qty: float
    side: int       # +1 the aggressor bought (lifted an ask), -1 sold (hit a bid)


class LimitOrderBook:
    def __init__(self) -> None:
        self.bids: dict[int, float] = {}  # price tick -> resting qty
        self.asks: dict[int, float] = {}

    # --- inspection -------------------------------------------------------
    def best_bid(self) -> int | None:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> int | None:
        return min(self.asks) if self.asks else None

    def mid(self) -> float | None:
        b, a = self.best_bid(), self.best_ask()
        return (b + a) / 2.0 if b is not None and a is not None else None

    def spread(self) -> int | None:
        b, a = self.best_bid(), self.best_ask()
        return a - b if b is not None and a is not None else None

    # --- resting liquidity ------------------------------------------------
    def add(self, side: int, price: int, qty: float) -> None:
        """Add resting limit liquidity. ``side`` +1 = bid, -1 = ask."""
        book = self.bids if side > 0 else self.asks
        book[price] = book.get(price, 0.0) + qty

    def cancel(self, side: int, price: int, qty: float | None = None) -> None:
        book = self.bids if side > 0 else self.asks
        if price not in book:
            return
        if qty is None or qty >= book[price]:
            del book[price]
        else:
            book[price] -= qty

    # --- matching ---------------------------------------------------------
    def match_market(self, side: int, qty: float) -> list[Fill]:
        """Execute a market order. ``side`` +1 buys (walks asks up), -1 sells.

        Returns the fills, consuming resting liquidity. Unfilled remainder (book
        exhausted) is dropped.
        """
        book = self.asks if side > 0 else self.bids
        fills: list[Fill] = []
        remaining = qty
        # Buy: cheapest asks first (ascending). Sell: highest bids first (desc).
        levels = sorted(book) if side > 0 else sorted(book, reverse=True)
        for price in levels:
            if remaining <= 0:
                break
            avail = book[price]
            take = min(avail, remaining)
            fills.append(Fill(price=price, qty=take, side=side))
            remaining -= take
            if take >= avail:
                del book[price]
            else:
                book[price] -= take
        return fills
