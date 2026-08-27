"""Tests for the execution layer — reconciliation, safety, and quantity rounding.

All tests run offline: they mock the exchange client so no network calls are
made and no API keys are needed. The tests verify the *logic* that decides
what to trade and the safety checks that prevent bad trades.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from vfund.live.exchange import (
    OrderResult,
    SymbolSpec,
    fmt_qty,
    round_step_down,
)
from vfund.live.execute import (
    KILLSWITCH_PATH,
    ExecutionPlan,
    PlannedOrder,
    SkippedOrder,
    check_killswitch,
    preflight,
    reconcile,
)
from vfund.live.signal import Book


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _book(weights: dict[str, float]) -> Book:
    """Build a minimal Book for testing."""
    import numpy as np

    arr = np.array(list(weights.values())) if weights else np.zeros(0)
    return Book(
        weights=weights, asof="2026-08-17",
        gross=float(np.abs(arr).sum()), net=float(arr.sum()),
    )


def _specs() -> dict[str, SymbolSpec]:
    """A handful of realistic symbol specs."""
    return {
        "BTCUSDT": SymbolSpec("BTCUSDT", step_size=0.001, min_qty=0.001,
                              min_notional=5.0, qty_precision=3, price_precision=2),
        "ETHUSDT": SymbolSpec("ETHUSDT", step_size=0.001, min_qty=0.001,
                              min_notional=5.0, qty_precision=3, price_precision=2),
        "SOLUSDT": SymbolSpec("SOLUSDT", step_size=0.1, min_qty=0.1,
                              min_notional=5.0, qty_precision=1, price_precision=2),
        "DOGEUSDT": SymbolSpec("DOGEUSDT", step_size=1.0, min_qty=1.0,
                               min_notional=5.0, qty_precision=0, price_precision=5),
        "ADAUSDT": SymbolSpec("ADAUSDT", step_size=0.1, min_qty=0.1,
                              min_notional=5.0, qty_precision=1, price_precision=4),
    }


def _prices() -> dict[str, float]:
    return {
        "BTCUSDT": 63000.0,
        "ETHUSDT": 1900.0,
        "SOLUSDT": 75.0,
        "DOGEUSDT": 0.07,
        "ADAUSDT": 0.17,
    }


# ---------------------------------------------------------------------------
# round_step_down
# ---------------------------------------------------------------------------

class TestRoundStepDown:
    def test_basic(self):
        assert round_step_down(1.567, 0.01, 2) == 1.56

    def test_exact(self):
        assert round_step_down(1.50, 0.01, 2) == 1.50

    def test_rounds_down_not_up(self):
        assert round_step_down(1.999, 0.001, 3) == 1.999
        assert round_step_down(1.9999, 0.001, 3) == 1.999

    def test_large_step(self):
        assert round_step_down(15.7, 1.0, 0) == 15.0

    def test_zero_value(self):
        assert round_step_down(0.0, 0.001, 3) == 0.0

    def test_zero_step(self):
        assert round_step_down(1.5, 0.0, 3) == 0.0


class TestFmtQty:
    def test_btc(self):
        spec = _specs()["BTCUSDT"]
        assert fmt_qty(0.00567, spec) == "0.005"

    def test_doge(self):
        spec = _specs()["DOGEUSDT"]
        assert fmt_qty(142.7, spec) == "142"

    def test_sol(self):
        spec = _specs()["SOLUSDT"]
        assert fmt_qty(3.78, spec) == "3.7"


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

class TestReconcile:
    def test_open_from_scratch(self):
        """No current positions → all orders are opens."""
        book = _book({"BTC": -0.05, "ETH": 0.10})
        plan = reconcile(book, {}, _prices(), _specs(), 10000.0)
        assert plan.n_orders == 2
        symbols = {o.symbol for o in plan.orders}
        assert symbols == {"BTCUSDT", "ETHUSDT"}
        for o in plan.orders:
            if o.symbol == "BTCUSDT":
                assert o.side == "SELL"
                assert o.reason == "open_short"
            else:
                assert o.side == "BUY"
                assert o.reason == "open_long"

    def test_already_at_target(self):
        """Positions match target → no orders (within step-size tolerance)."""
        book = _book({"BTC": -0.05})
        equity = 10000.0
        target_qty = -0.05 * equity / 63000.0  # ~ -0.00793
        plan = reconcile(book, {"BTCUSDT": target_qty}, _prices(), _specs(), equity)
        # The delta should be zero or below min order size.
        assert plan.n_orders == 0

    def test_close_position_not_in_target(self):
        """Currently holding a coin that's not in the target → close it."""
        book = _book({"ETH": 0.10})  # only ETH in target
        current = {"BTCUSDT": 0.01, "ETHUSDT": 0.5}  # holding BTC too
        plan = reconcile(book, current, _prices(), _specs(), 10000.0)
        close_orders = [o for o in plan.orders if o.reason == "close"]
        assert len(close_orders) == 1
        assert close_orders[0].symbol == "BTCUSDT"
        assert close_orders[0].side == "SELL"

    def test_no_perp_skipped(self):
        """A symbol without a USD-M perp spec is skipped, not errored."""
        book = _book({"GPS": 0.05})  # GPS has no perp
        prices = {**_prices(), "GPSUSDT": 0.50}  # price exists but no spec
        plan = reconcile(book, {}, prices, _specs(), 10000.0)
        assert plan.n_orders == 0
        assert len(plan.skipped) == 1
        assert "no USD-M perp" in plan.skipped[0].reason

    def test_no_price_skipped(self):
        """A symbol with no price feed is skipped."""
        book = _book({"ADA": 0.10})
        prices = {"BTCUSDT": 63000.0}  # ADA price missing
        plan = reconcile(book, {}, prices, _specs(), 10000.0)
        assert plan.n_orders == 0
        assert len(plan.skipped) == 1
        assert "no price" in plan.skipped[0].reason

    def test_order_below_min_notional_skipped(self):
        """Very small orders (below min notional) are skipped."""
        book = _book({"BTC": 0.0001})  # 0.01% of $1000 = $0.10 → below $6 min
        plan = reconcile(book, {}, _prices(), _specs(), 1000.0)
        # 0.0001 × 1000 / 63000 ≈ 0.0000016 → notional $0.10 → skipped
        assert plan.n_orders == 0

    def test_gross_exposure_computed(self):
        book = _book({"BTC": -0.30, "ETH": 0.30})
        plan = reconcile(book, {}, _prices(), _specs(), 10000.0)
        assert abs(plan.target_gross - 0.60) < 0.01


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

class TestPreflight:
    def test_passes_when_clean(self):
        plan = ExecutionPlan(
            orders=[PlannedOrder("BTCUSDT", "SELL", 0.01, 630.0, "open_short")],
            account_equity=10000.0,
            target_gross=0.8,
        )
        book = _book({"BTC": -0.05})
        assert preflight(plan, book) == []

    def test_blocks_excess_gross(self):
        plan = ExecutionPlan(
            orders=[], account_equity=10000.0, target_gross=3.5,
        )
        issues = preflight(plan, _book({}), max_gross=2.0)
        assert any("gross exposure" in i for i in issues)

    def test_blocks_oversized_position(self):
        plan = ExecutionPlan(
            orders=[PlannedOrder("BTCUSDT", "BUY", 1.0, 5000.0, "open_long")],
            account_equity=10000.0,
            target_gross=0.5,
        )
        issues = preflight(plan, _book({}), max_position_pct=0.15)
        assert any("50.0%" in i for i in issues)

    def test_blocks_zero_equity(self):
        plan = ExecutionPlan(
            orders=[PlannedOrder("BTCUSDT", "BUY", 0.01, 630.0, "open_long")],
            account_equity=0.0,
            target_gross=0.5,
        )
        issues = preflight(plan, _book({}))
        assert any("zero or negative" in i for i in issues)


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

class TestKillswitch:
    def test_not_engaged(self, tmp_path):
        assert check_killswitch(tmp_path / "KILLSWITCH") is False

    def test_engaged(self, tmp_path):
        ks = tmp_path / "KILLSWITCH"
        ks.write_text("stop")
        assert check_killswitch(ks) is True


# ---------------------------------------------------------------------------
# Symbol mapping
# ---------------------------------------------------------------------------

class TestSymbolMapping:
    def test_bare_to_perp(self):
        from vfund.live.execute import _symbol_to_perp

        assert _symbol_to_perp("BTC") == "BTCUSDT"
        assert _symbol_to_perp("ETHUSDT") == "ETHUSDT"
        assert _symbol_to_perp("AAVE") == "AAVEUSDT"
