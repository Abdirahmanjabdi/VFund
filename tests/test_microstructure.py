"""Tests for the order book and the adverse-selection simulation."""

from vfund.microstructure import LimitOrderBook, MarketMakingSim
from vfund.microstructure.simulator import _mm_loop, mm_loop_py


def test_order_book_basics():
    ob = LimitOrderBook()
    ob.add(+1, 99, 5)   # bid
    ob.add(-1, 101, 5)  # ask
    assert ob.best_bid() == 99 and ob.best_ask() == 101
    assert ob.mid() == 100.0 and ob.spread() == 2


def test_market_order_consumes_liquidity():
    ob = LimitOrderBook()
    ob.add(-1, 101, 3)
    ob.add(-1, 102, 3)
    fills = ob.match_market(+1, 4)  # buy 4: takes 3@101 then 1@102
    assert [(f.price, f.qty) for f in fills] == [(101, 3), (102, 1)]
    assert ob.asks == {102: 2.0}


def test_noise_flow_maker_earns_spread():
    # No informed traders -> the maker should capture ~the spread (profit).
    res = MarketMakingSim(n_steps=30_000, half_spread=2, informed_frac=0.0,
                          sigma=2.0, seed=1).run()
    assert res.total_pnl > 0


def test_mm_loop_dispatch_matches_reference():
    # The dispatched loop (Rust if built, else Python) must match the reference
    # exactly given identical flow.
    sim = MarketMakingSim(n_steps=15_000, half_spread=2, informed_frac=0.5, seed=3)
    flow = sim._flow()
    ref_pnl, ref_fills = mm_loop_py(*flow, 2)
    pnl, fills = _mm_loop(*flow, 2)
    assert fills == ref_fills
    assert abs(pnl - ref_pnl) < 1e-6


def test_informed_flow_causes_losses():
    # Volatility comparable to the spread: informed flow picks the maker off.
    clean = MarketMakingSim(n_steps=30_000, half_spread=1, informed_frac=0.0,
                            sigma=2.0, seed=2).run()
    picked = MarketMakingSim(n_steps=30_000, half_spread=1, informed_frac=1.0,
                             sigma=2.0, seed=2).run()
    assert clean.total_pnl > 0            # noise flow: earn the spread
    assert picked.total_pnl < 0           # informed flow: picked off, lose money
    assert picked.adverse_selection > clean.adverse_selection
