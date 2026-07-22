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


def _days_between(a: str, b: str) -> float:
    """Days between two 'YYYY-MM-DD ...' timestamp strings (b - a)."""
    from datetime import date

    da = date.fromisoformat(a[:10])
    db = date.fromisoformat(b[:10])
    return (db - da).days


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

    def update_two_engine(
        self,
        broad: pl.DataFrame,
        defi: pl.DataFrame,
        tvl: pl.DataFrame,
        spot: pl.DataFrame,
        perp: pl.DataFrame,
        funding: pl.DataFrame,
        *,
        fees: pl.DataFrame | None = None,
        cost_bps: float = 10.0,
        min_short_dollar_volume: float = 5_000_000,
        carry_weight: float = 0.5,
    ) -> dict:
        """Mark the two-engine book (alpha + carry) forward.

        The two engines are nearly uncorrelated and are tracked as two capital
        pots, rebalanced back to the target split on every update:

        * **alpha** — the 3-sleeve trend + size + on-chain book, a set of spot
          weights marked forward by price moves (identical to
          :meth:`update_three_sleeve`);
        * **carry** — the timed funding-basis pair trade on majors, marked
          forward by realised ``funding - basis change`` (see
          :mod:`vfund.live.carry`), which has no spot-weight representation.

        Args:
            broad: broad spot panel for the alpha sleeves.
            defi: DeFi price panel for the on-chain sleeve.
            tvl: TVL panel for the on-chain sleeve.
            spot: spot panel covering the carry majors.
            perp: USD-M perpetual panel covering the carry majors.
            funding: funding-rate panel covering the carry majors.
            fees: optional protocol fee-revenue panel; supplying it makes the
                alpha engine the 4-sleeve book rather than the 3-sleeve one.
            cost_bps: turnover cost charged on the alpha half.
            min_short_dollar_volume: hard-to-short gate for the alpha half.
            carry_weight: fraction of capital in carry (0.5 = the validated
                50/50 blend).

        Returns:
            A status dict with the alpha book and the carry sleeve attached.
        """
        from vfund.live.carry import carry_sleeve, realised_carry
        from vfund.live.signal import alpha_book

        if not 0.0 <= carry_weight <= 1.0:
            raise ValueError("carry_weight must be between 0 and 1")

        book = alpha_book(
            broad, defi, tvl, fees=fees,
            min_short_dollar_volume=min_short_dollar_volume,
        )
        ts, p_broad = _latest_prices(broad)
        _, p_defi = _latest_prices(defi)

        def strip(d):  # normalise BTCUSDT<->BTC to match the book's keys
            return {(k[:-4] if k.endswith("USDT") else k): v for k, v in d.items()}

        prices = {**strip(p_defi), **strip(p_broad)}  # broad price wins on overlap
        sleeve = carry_sleeve(spot, perp, funding)
        return self.record_two_engine(
            book, sleeve, ts, prices,
            cost_bps=cost_bps,
            carry_weight=carry_weight,
            realised_carry=realised_carry,
        )

    # An update gap larger than this is almost certainly a stale/wrong state
    # file (e.g. a leftover from an old test), not a real holding period.
    # Marking months of price moves in one step on old weights corrupts the
    # forward record, so refuse rather than silently continue.
    MAX_GAP_DAYS = 45

    def _check_gap(self, st: dict | None, ts: str) -> None:
        """Refuse to mark forward across an implausible gap (stale state file)."""
        if st is None:
            return
        gap_days = _days_between(st["last_ts"], ts)
        if gap_days > self.MAX_GAP_DAYS:
            raise RuntimeError(
                f"paper state at {self.path} was last updated {st['last_ts'][:10]} "
                f"({gap_days:.0f} days before {ts[:10]}) - it looks stale. "
                f"Marking such a gap in one step would corrupt the forward "
                f"record. Archive the file and re-initialize, or update with "
                f"intermediate data first."
            )

    def record_two_engine(
        self, book, sleeve, ts: str, prices: dict, *,
        cost_bps: float = 10.0, carry_weight: float = 0.5, realised_carry=None,
    ) -> dict:
        """Mark a two-pot (alpha + carry) account forward. See :meth:`update_two_engine`."""
        if realised_carry is None:  # pragma: no cover - injected by the caller
            from vfund.live.carry import realised_carry
        cost = cost_bps / 10_000.0
        alpha_share = 1.0 - carry_weight
        st = self.load()
        self._check_gap(st, ts)

        if st is None:
            turnover = sum(abs(w) for w in book.weights.values())
            alpha_eq = self.start_equity * alpha_share * (1 - turnover * cost)
            carry_eq = self.start_equity * carry_weight
            equity = alpha_eq + carry_eq
            st = {
                "created": ts,
                "start_equity": self.start_equity,
                "equity": equity,
                "carry_weight": carry_weight,
                "alpha_equity": alpha_eq,
                "carry_equity": carry_eq,
                "last_ts": ts,
                "last_prices": prices,
                "weights": book.weights,
                "carry_basket": sleeve.basket,
                "history": [{"ts": ts, "equity": equity,
                             "alpha_equity": alpha_eq, "carry_equity": carry_eq}],
            }
            self.save(st)
            return {"status": "initialized", "asof": ts, "equity": equity,
                    "port_ret": 0.0, "alpha_ret": 0.0, "carry_ret": 0.0,
                    "book": book, "sleeve": sleeve}

        if ts == st["last_ts"]:
            return {"status": "no new data", "asof": ts, "equity": st["equity"],
                    "port_ret": 0.0, "alpha_ret": 0.0, "carry_ret": 0.0,
                    "book": book, "sleeve": sleeve}

        # --- alpha pot: mark held spot weights forward, then pay turnover ---
        old_w, old_p = st["weights"], st["last_prices"]
        alpha_ret = sum(
            w * (prices[s] / old_p[s] - 1.0)
            for s, w in old_w.items()
            if s in prices and s in old_p and old_p[s] > 0
        )
        alpha_eq = st["alpha_equity"] * (1.0 + alpha_ret)
        new_w = book.weights
        names = set(old_w) | set(new_w)
        turnover = sum(abs(new_w.get(s, 0.0) - old_w.get(s, 0.0)) for s in names)
        alpha_eq *= 1.0 - turnover * cost

        # --- carry pot: accrue realised funding-minus-basis over the window ---
        carry_ret = realised_carry(sleeve, st["last_ts"])
        carry_eq = st["carry_equity"] * (1.0 + carry_ret)

        equity = alpha_eq + carry_eq
        port_ret = equity / st["equity"] - 1.0 if st["equity"] else 0.0

        # Rebalance the two pots back to the target split (the validated blend
        # is a fixed capital allocation, not a let-winners-run drift).
        st["alpha_equity"] = equity * alpha_share
        st["carry_equity"] = equity * carry_weight
        st["equity"] = equity
        st.update(last_ts=ts, last_prices=prices, weights=new_w,
                  carry_basket=sleeve.basket, carry_weight=carry_weight)
        st["history"].append({"ts": ts, "equity": equity,
                              "alpha_equity": st["alpha_equity"],
                              "carry_equity": st["carry_equity"]})
        self.save(st)
        return {"status": "updated", "asof": ts, "equity": equity,
                "port_ret": port_ret, "alpha_ret": alpha_ret,
                "carry_ret": carry_ret, "book": book, "sleeve": sleeve}

    def record(self, book, ts: str, prices: dict, *, cost_bps: float = 10.0) -> dict:
        """Mark the account forward given a precomputed book and latest prices."""
        cost = cost_bps / 10_000.0
        st = self.load()
        self._check_gap(st, ts)

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
