"""A persistent paper-trading account — forward-tracked, zero real money.

Each ``update`` marks the held book forward by the price move since the last
update, charges turnover to rebalance into the fresh signal, and appends to an
equity history stored as JSON. Run it as new data arrives (e.g. daily) and you
get a genuine out-of-sample record — the honest test a backtest can't give.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from vfund.data.panel import pivot_to_wide, validate_panel
from vfund.live.signal import combined_book


def _latest_prices(panel: pl.DataFrame):
    # Ragged: the true latest bar; skip coins with no price then (delisted).
    wide = pivot_to_wide(panel, "close", drop_incomplete=False)
    symbols = [c for c in wide.columns if c != "timestamp"]
    row = wide.tail(1)
    ts = str(row["timestamp"][0])
    prices = {s: float(row[s][0]) for s in symbols if row[s][0] is not None}
    return ts, prices


class PaperTracker:
    def __init__(self, path: str | Path, start_equity: float = 10_000.0):
        self.path = Path(path)
        self.start_equity = float(start_equity)

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text())

    def save(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, indent=2))

    def update(self, panel: pl.DataFrame, *, cost_bps: float = 10.0, **book_kwargs) -> dict:
        panel = validate_panel(panel)
        book = combined_book(panel, **book_kwargs)
        ts, prices = _latest_prices(panel)
        return self.record(book, ts, prices, cost_bps=cost_bps)

    def update_three_sleeve(
        self, broad: pl.DataFrame, defi: pl.DataFrame, tvl: pl.DataFrame,
        *, cost_bps: float = 10.0, min_short_dollar_volume: float = 5_000_000,
    ) -> dict:
        """Mark the diversified trend+size+on-chain book forward."""
        from vfund.live.signal import three_sleeve_book

        book = three_sleeve_book(broad, defi, tvl,
                                 min_short_dollar_volume=min_short_dollar_volume)
        ts, p_broad = _latest_prices(broad)
        _, p_defi = _latest_prices(defi)

        def strip(d):  # normalise BTCUSDT<->BTC to match the book's keys
            return {(k[:-4] if k.endswith("USDT") else k): v for k, v in d.items()}

        prices = {**strip(p_defi), **strip(p_broad)}  # broad price wins on overlap
        return self.record(book, ts, prices, cost_bps=cost_bps)

    def record(self, book, ts: str, prices: dict, *, cost_bps: float = 10.0) -> dict:
        """Mark the account forward given a precomputed book and latest prices."""
        cost = cost_bps / 10_000.0
        st = self.load()

        if st is None:
            turnover = sum(abs(w) for w in book.weights.values())
            equity = self.start_equity * (1 - turnover * cost)
            st = {
                "created": ts,
                "start_equity": self.start_equity,
                "equity": equity,
                "last_ts": ts,
                "last_prices": prices,
                "weights": book.weights,
                "history": [{"ts": ts, "equity": equity}],
            }
            self.save(st)
            return {"status": "initialized", "asof": ts, "equity": equity,
                    "port_ret": 0.0, "book": book}

        if ts == st["last_ts"]:
            return {"status": "no new data", "asof": ts, "equity": st["equity"],
                    "port_ret": 0.0, "book": book}

        # Mark the held book forward by the realised price move.
        old_w, old_p = st["weights"], st["last_prices"]
        port_ret = sum(
            w * (prices[s] / old_p[s] - 1.0)
            for s, w in old_w.items()
            if s in prices and s in old_p and old_p[s] > 0
        )
        st["equity"] *= 1.0 + port_ret

        # Rebalance into the new signal, paying turnover.
        new_w = book.weights
        names = set(old_w) | set(new_w)
        turnover = sum(abs(new_w.get(s, 0.0) - old_w.get(s, 0.0)) for s in names)
        st["equity"] *= 1.0 - turnover * cost

        st.update(last_ts=ts, last_prices=prices, weights=new_w)
        st["history"].append({"ts": ts, "equity": st["equity"]})
        self.save(st)
        return {"status": "updated", "asof": ts, "equity": st["equity"],
                "port_ret": port_ret, "book": book}
