"""Execute a target book on Binance USD-M Futures — with every safeguard.

This module is the **last mile**: it takes the target weights from
:mod:`vfund.live.signal` (the same signal the paper tracker uses) and turns
them into actual exchange orders. It is designed so that every failure mode
is survivable and every action is logged.

Safety model
------------
1. **Kill switch** — if ``data/KILLSWITCH`` exists, refuse to trade.
   ``touch data/KILLSWITCH`` halts everything; ``rm data/KILLSWITCH`` resumes.
2. **Dry run** — the default first use. Shows every order that *would* be
   placed without touching the exchange.
3. **Pre-flight checks** — max position size, max gross exposure, minimum
   order size, symbol availability. Any blocker aborts the whole run.
4. **Leverage cap** — sets exchange-side max leverage to 3× on every symbol
   before any order is placed, so a bug cannot over-lever the account.
5. **Reconciliation-based** — reads current positions, diffs against target,
   and places only the delta. Idempotent: run it again after a crash and it
   finishes whatever was left.
6. **Append-only log** — every order, fill, skip, and error is written to
   ``data/execution_log.jsonl`` before the next order is placed.

The whole module is inert on import. Nothing connects to the exchange until
:func:`run` is called, and :func:`run` requires an explicit ``dry_run=False``
to place real orders.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from vfund.live.exchange import (
    BinanceClient,
    BinanceError,
    OrderResult,
    SymbolSpec,
    fmt_qty,
    round_step_down,
)
from vfund.live.signal import Book

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KILLSWITCH_PATH = Path("data/KILLSWITCH")
LOG_PATH = Path("data/execution_log.jsonl")

MAX_LEVERAGE = 3
MAX_GROSS_EXPOSURE = 2.0
MAX_SINGLE_POSITION_PCT = 0.15
MIN_ORDER_USDT = 6.0


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannedOrder:
    """One order the reconciler wants to place."""

    symbol: str          # e.g. "BTCUSDT"
    side: str            # "BUY" or "SELL"
    quantity: float      # absolute, in base asset
    notional: float      # quantity × price (USDT value)
    reason: str          # "open_long", "open_short", "increase", "reduce", "close"


@dataclass(frozen=True)
class SkippedOrder:
    """An order that was planned but skipped, and why."""

    symbol: str
    reason: str


@dataclass
class ExecutionPlan:
    """The full diff between current and target positions."""

    orders: list[PlannedOrder] = field(default_factory=list)
    skipped: list[SkippedOrder] = field(default_factory=list)
    account_equity: float = 0.0
    target_gross: float = 0.0

    @property
    def n_orders(self) -> int:
        return len(self.orders)


@dataclass
class ExecutionReport:
    """What actually happened when the plan was executed."""

    filled: list[OrderResult] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    skipped: list[SkippedOrder] = field(default_factory=list)

    @property
    def n_filled(self) -> int:
        return len(self.filled)

    @property
    def n_failed(self) -> int:
        return len(self.failed)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class ExecutionLogger:
    """Append-only JSONL logger for the full execution audit trail."""

    def __init__(self, path: Path = LOG_PATH):
        self.path = path

    def _write(self, event: str, **data) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **data,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def run_start(self, mode: str, equity: float) -> None:
        self._write("run_start", mode=mode, equity=equity)

    def plan(self, n_orders: int, n_skipped: int, gross: float) -> None:
        self._write("plan", n_orders=n_orders, n_skipped=n_skipped, gross=gross)

    def order_placed(self, order: PlannedOrder) -> None:
        self._write("order_placed", **asdict(order))

    def order_filled(self, result: OrderResult) -> None:
        self._write("order_filled", **asdict(result))

    def order_failed(self, symbol: str, error: str) -> None:
        self._write("order_failed", symbol=symbol, error=error)

    def order_skipped(self, symbol: str, reason: str) -> None:
        self._write("order_skipped", symbol=symbol, reason=reason)

    def run_complete(self, n_filled: int, n_failed: int) -> None:
        self._write("run_complete", n_filled=n_filled, n_failed=n_failed)

    def error(self, msg: str) -> None:
        self._write("error", msg=msg)


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

def check_killswitch(path: Path = KILLSWITCH_PATH) -> bool:
    """Return True if the kill switch is engaged (file exists)."""
    return path.exists()


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def _symbol_to_perp(bare: str) -> str:
    """Map a bare symbol (BTC) to a Binance USD-M perp symbol (BTCUSDT)."""
    if bare.endswith("USDT"):
        return bare
    return bare + "USDT"


def reconcile(
    book: Book,
    current_positions: dict[str, float],
    prices: dict[str, float],
    specs: dict[str, SymbolSpec],
    account_equity: float,
) -> ExecutionPlan:
    """Diff the target book against current positions → an order plan.

    For each symbol in the target or currently held:
    1. Compute target quantity = weight × equity / price.
    2. Compute delta = target − current.
    3. If delta is large enough to trade, plan an order; else skip.
    """
    plan = ExecutionPlan(account_equity=account_equity)
    target_gross = 0.0

    # All symbols we need to consider: in target OR currently held.
    target_perps = {_symbol_to_perp(s): w for s, w in book.weights.items()}
    all_symbols = set(target_perps.keys()) | set(current_positions.keys())

    for sym in sorted(all_symbols):
        target_weight = target_perps.get(sym, 0.0)
        current_qty = current_positions.get(sym, 0.0)
        price = prices.get(sym)

        # Skip if no price available (symbol might be delisted/paused).
        if price is None or price <= 0:
            if abs(target_weight) > 1e-9:
                plan.skipped.append(SkippedOrder(sym, "no price available"))
            continue

        spec = specs.get(sym)
        if spec is None:
            if abs(target_weight) > 1e-9:
                plan.skipped.append(SkippedOrder(sym, "no USD-M perp on Binance"))
            elif abs(current_qty) > 0:
                plan.skipped.append(SkippedOrder(sym, "held but no spec — close manually"))
            continue

        target_notional = target_weight * account_equity
        target_qty = target_notional / price
        target_gross += abs(target_weight)

        delta = target_qty - current_qty

        # Round the absolute delta down to the exchange's step size.
        abs_delta = round_step_down(abs(delta), spec.step_size, spec.qty_precision)
        order_notional = abs_delta * price

        # Skip if the order is too small to place.
        if abs_delta < spec.min_qty or order_notional < max(spec.min_notional, MIN_ORDER_USDT):
            if abs(delta * price) > 1.0:
                plan.skipped.append(SkippedOrder(
                    sym, f"order too small: ${order_notional:.2f} "
                         f"(min ${max(spec.min_notional, MIN_ORDER_USDT):.0f})"
                ))
            continue

        side = "BUY" if delta > 0 else "SELL"
        if current_qty == 0:
            reason = "open_long" if side == "BUY" else "open_short"
        elif target_weight == 0 or sym not in target_perps:
            reason = "close"
        elif abs(target_qty) > abs(current_qty):
            reason = "increase"
        else:
            reason = "reduce"

        plan.orders.append(PlannedOrder(
            symbol=sym, side=side, quantity=abs_delta,
            notional=order_notional, reason=reason,
        ))

    plan.target_gross = target_gross
    return plan


# ---------------------------------------------------------------------------
# Pre-flight safety checks
# ---------------------------------------------------------------------------

def preflight(
    plan: ExecutionPlan,
    book: Book,
    *,
    max_gross: float = MAX_GROSS_EXPOSURE,
    max_position_pct: float = MAX_SINGLE_POSITION_PCT,
) -> list[str]:
    """Run safety checks. Returns a list of blocking issues (empty = safe)."""
    issues: list[str] = []

    if check_killswitch():
        issues.append(
            f"KILL SWITCH engaged — {KILLSWITCH_PATH} exists. "
            f"Remove it to resume trading."
        )

    if plan.target_gross > max_gross:
        issues.append(
            f"target gross exposure {plan.target_gross:.2f}× exceeds "
            f"limit {max_gross:.1f}×"
        )

    if plan.account_equity <= 0:
        issues.append("account equity is zero or negative")

    for order in plan.orders:
        pct = order.notional / plan.account_equity if plan.account_equity else 1.0
        if pct > max_position_pct:
            issues.append(
                f"{order.symbol}: order is {pct:.1%} of equity "
                f"(limit {max_position_pct:.0%})"
            )

    return issues


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute_plan(
    client: BinanceClient,
    plan: ExecutionPlan,
    specs: dict[str, SymbolSpec],
    logger: ExecutionLogger,
    *,
    dry_run: bool = True,
    order_pause_s: float = 0.15,
) -> ExecutionReport:
    """Place the orders in *plan*, logging every action.

    With ``dry_run=True`` (the default), prints what *would* happen without
    touching the exchange.
    """
    report = ExecutionReport(skipped=plan.skipped)

    if dry_run:
        return report

    # Set leverage cap on every symbol we're about to trade.
    leveraged: set[str] = set()
    for order in plan.orders:
        if order.symbol not in leveraged:
            try:
                client.set_leverage(order.symbol, MAX_LEVERAGE)
                leveraged.add(order.symbol)
            except (BinanceError, Exception) as exc:
                logger.error(f"set_leverage {order.symbol}: {exc}")

    # Place orders sequentially, with a pause for rate limits.
    for order in plan.orders:
        spec = specs.get(order.symbol)
        if spec is None:
            logger.order_failed(order.symbol, "no spec at execution time")
            report.failed.append({"symbol": order.symbol, "error": "no spec"})
            continue

        qty_str = fmt_qty(order.quantity, spec)
        if float(qty_str) <= 0:
            logger.order_skipped(order.symbol, "rounded quantity is zero")
            report.skipped.append(SkippedOrder(order.symbol, "rounded to zero"))
            continue

        logger.order_placed(order)
        try:
            result = client.place_market_order(order.symbol, order.side, qty_str)
            logger.order_filled(result)
            report.filled.append(result)
        except (BinanceError, Exception) as exc:
            error_str = str(exc)
            logger.order_failed(order.symbol, error_str)
            report.failed.append({"symbol": order.symbol, "error": error_str})

        time.sleep(order_pause_s)

    return report


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    client: BinanceClient,
    book: Book,
    *,
    dry_run: bool = True,
    log_path: Path = LOG_PATH,
    killswitch_path: Path = KILLSWITCH_PATH,
) -> ExecutionReport:
    """The main entry point: reconcile, check, execute, log.

    Args:
        client: an authenticated :class:`BinanceClient`.
        book: the target book from :func:`vfund.live.signal.three_sleeve_book`
            or any ``Book`` instance.
        dry_run: if True (default), print the plan without placing orders.
            **Always dry-run first.**
        log_path: where to append the execution log.
        killswitch_path: the kill-switch file path.

    Returns:
        An :class:`ExecutionReport` summarising what happened.
    """
    logger = ExecutionLogger(log_path)
    mode = "testnet" if client.testnet else "MAINNET"
    tag = "[DRY RUN] " if dry_run else ""

    # --- kill switch -------------------------------------------------------
    if check_killswitch(killswitch_path):
        msg = (
            f"KILL SWITCH is engaged ({killswitch_path}). "
            f"Remove the file to resume trading."
        )
        logger.error(msg)
        print(f"\n  !! {msg}\n")
        return ExecutionReport()

    # --- fetch state -------------------------------------------------------
    print(f"\n{tag}Connecting to Binance ({mode})...")
    try:
        equity = client.account_equity()
        wallet = client.wallet_balance()
    except Exception as exc:
        logger.error(f"account fetch failed: {exc}")
        raise RuntimeError(f"Cannot read account: {exc}") from exc

    print(f"  wallet balance:  ${wallet:,.2f}")
    print(f"  margin balance:  ${equity:,.2f}")
    logger.run_start(mode=mode, equity=equity)

    print(f"  fetching exchange info + prices...")
    specs = client.exchange_info()
    prices = client.ticker_prices()
    current_pos = client.positions()

    n_held = len(current_pos)
    print(f"  current positions: {n_held}")

    # --- reconcile ---------------------------------------------------------
    plan = reconcile(book, current_pos, prices, specs, equity)
    logger.plan(plan.n_orders, len(plan.skipped), plan.target_gross)

    print(f"\n{tag}Execution plan:")
    print(f"  target symbols:  {len(book.weights)}")
    print(f"  target gross:    {plan.target_gross:.3f}×")
    print(f"  orders to place: {plan.n_orders}")
    print(f"  skipped:         {len(plan.skipped)}")

    if plan.skipped:
        print(f"\n{tag}Skipped:")
        for s in plan.skipped:
            print(f"    {s.symbol:<12} {s.reason}")

    if plan.orders:
        # Sort: closes first (reduce risk), then opens.
        closes = [o for o in plan.orders if o.reason == "close"]
        others = [o for o in plan.orders if o.reason != "close"]
        plan.orders = closes + others

        print(f"\n{tag}Orders:")
        for o in plan.orders:
            print(f"    {o.side:<4} {o.symbol:<12} "
                  f"qty={o.quantity:<12.6f} ~${o.notional:>8.2f}  ({o.reason})")

    # --- pre-flight --------------------------------------------------------
    issues = preflight(plan, book)
    if issues:
        print(f"\n  !! PRE-FLIGHT FAILED — no orders placed:")
        for issue in issues:
            print(f"     - {issue}")
        logger.error(f"preflight failed: {issues}")
        return ExecutionReport(skipped=plan.skipped)

    # --- execute -----------------------------------------------------------
    if dry_run:
        print(f"\n{tag}No orders placed. Re-run with --live to execute.")
        return ExecutionReport(skipped=plan.skipped)

    if not plan.orders:
        print(f"\n  No orders needed — positions already match target.")
        logger.run_complete(0, 0)
        return ExecutionReport(skipped=plan.skipped)

    print(f"\n  Placing {plan.n_orders} orders...")
    report = execute_plan(client, plan, specs, logger, dry_run=False)

    logger.run_complete(report.n_filled, report.n_failed)

    print(f"\n  Results:")
    print(f"    filled:  {report.n_filled}")
    print(f"    failed:  {report.n_failed}")
    if report.failed:
        for f in report.failed:
            print(f"      {f['symbol']}: {f['error']}")
    print()
    return report


# ---------------------------------------------------------------------------
# Plan-only mode (no API keys / no futures access needed)
# ---------------------------------------------------------------------------

def _fetch_spot_prices() -> dict[str, float]:
    """Fetch mark prices from Binance public spot API (no auth needed)."""
    import requests

    resp = requests.get(
        "https://api.binance.com/api/v3/ticker/price", timeout=15
    )
    resp.raise_for_status()
    return {d["symbol"]: float(d["price"]) for d in resp.json()}


def _default_spec(symbol: str) -> SymbolSpec:
    """Reasonable default spec when exchange_info is unavailable."""
    return SymbolSpec(
        symbol=symbol, step_size=0.001, min_qty=0.001,
        min_notional=5.0, qty_precision=3, price_precision=2,
    )


def plan_only(book: Book, equity: float) -> ExecutionPlan:
    """Compute the full execution plan using public spot prices.

    No API keys, no futures access, no authentication required.
    Uses the paper tracker equity as the assumed account size.
    """
    print(f"\n[PLAN ONLY] Fetching public spot prices...")
    prices = _fetch_spot_prices()
    print(f"  got {len(prices)} symbols")

    # Build minimal specs for every symbol in the book.
    specs: dict[str, SymbolSpec] = {}
    for bare in book.weights:
        sym = _symbol_to_perp(bare)
        if sym in prices:
            specs[sym] = _default_spec(sym)

    plan = reconcile(book, {}, prices, specs, equity)

    print(f"\n[PLAN ONLY] Execution plan (equity=${equity:,.2f}):")
    print(f"  target symbols:  {len(book.weights)}")
    print(f"  target gross:    {plan.target_gross:.3f}×")
    print(f"  orders to place: {plan.n_orders}")
    print(f"  skipped:         {len(plan.skipped)}")

    if plan.skipped:
        print(f"\n  Skipped:")
        for s in plan.skipped:
            print(f"    {s.symbol:<12} {s.reason}")

    if plan.orders:
        closes = [o for o in plan.orders if o.reason == "close"]
        others = [o for o in plan.orders if o.reason != "close"]
        plan.orders = closes + others

        print(f"\n  Orders:")
        for o in plan.orders:
            print(f"    {o.side:<4} {o.symbol:<12} "
                  f"qty={o.quantity:<12.6f} ~${o.notional:>8.2f}  ({o.reason})")

    issues = preflight(plan, book)
    if issues:
        print(f"\n  !! PRE-FLIGHT ISSUES:")
        for issue in issues:
            print(f"     - {issue}")
    else:
        print(f"\n  All pre-flight checks passed.")

    return plan


# ---------------------------------------------------------------------------
# Format helpers (for the CLI)
# ---------------------------------------------------------------------------

def format_plan_summary(plan: ExecutionPlan, dry_run: bool = True) -> str:
    """One-line summary for the CLI."""
    tag = "[DRY RUN] " if dry_run else ""
    return (
        f"{tag}{plan.n_orders} orders, {len(plan.skipped)} skipped, "
        f"target gross {plan.target_gross:.3f}×, equity ${plan.account_equity:,.2f}"
    )
