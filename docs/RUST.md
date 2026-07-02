# The Rust core (optional native acceleration)

VFund's backtester is pure Python and works with no extra toolchain. But the
innermost simulation loop — walk forward bar by bar, earn returns, drift weights,
pay turnover — runs `T` times per backtest and is re-run thousands of times in
robustness bootstraps and walk-forwards. That loop is a natural candidate for a
native implementation.

`rust/` contains a [PyO3](https://pyo3.rs) extension, `vfund_core`, that
implements exactly the same loop as `vfund/backtest/sim.py::simulate_py` (which
is the specification). When built, `vfund.backtest.sim.simulate` uses it
automatically; when not, it falls back to the Python reference. **Nothing
requires the Rust core** — it's a performance option.

## Design

- `vfund/backtest/sim.py` — `simulate_py`, the reference loop, and `simulate`,
  which dispatches to Rust if available.
- `vfund/backtest/_accel.py` — tries `import vfund_core`; exposes `HAVE_RUST`.
- `rust/src/lib.rs` — the Rust implementation (identical arithmetic).
- `tests/test_sim.py` — pins the reference to the full engine's output, so the
  Rust core is verified against the same invariants.

Python computes the per-rebalance target weights (calling strategies, applying
costs/overlays); Rust runs the heavy `T`-length numeric loop. This split keeps
the strategy logic in Python while moving the hot arithmetic to native code.

## Building it

Prerequisites:
1. **Rust** — install from <https://rustup.rs>.
2. **A C linker** — on Windows, the *Visual Studio Build Tools* (MSVC); on
   macOS, `xcode-select --install`; on Linux, `build-essential`.

Then, from the repo root:

```bash
pip install maturin
cd rust
maturin develop --release      # builds vfund_core and installs it into the venv
```

Verify and benchmark:

```bash
python -c "from vfund.backtest._accel import HAVE_RUST; print('rust:', HAVE_RUST)"
python examples/bench_sim.py
```

`bench_sim.py` cross-checks that the Rust and Python results are identical, then
reports the speedup. A measured run (43,800 bars × 50 assets, 260 rebalances)
on Windows with the GNU toolchain:

```
  Python reference :    326.4 ms
  Rust core        :      4.2 ms   (77.2x faster, results match)
```

~77× on the hot loop — the kind of speedup that turns an overnight robustness
sweep into a coffee break.

### Engine integration

`CrossSectionalBacktester.run()` uses the native core automatically when it's
built (and the drawdown circuit-breaker, which needs running equity, isn't in
use): it precomputes the rebalance schedule in Python, then hands the T-length
loop to Rust. Full-backtest speedup is smaller than the raw-loop number because
strategy scoring still runs in Python each rebalance — ~2.2× on a 40-asset ×
4,000-bar daily backtest, and more on longer or higher-frequency series where
the simulation loop dominates. Results are bit-identical to the Python path
(max abs difference ~1e-12), and it falls back to pure Python when unbuilt.

## What the core accelerates

Two hot loops run natively when the extension is built:

- **`simulate`** — the cross-sectional backtest's per-bar loop (`vfund/backtest/`).
  Used automatically by `CrossSectionalBacktester.run()`.
- **`mm_loop`** — the market-making simulation's matching loop
  (`vfund/microstructure/simulator.py`). ~57× faster on 500k steps
  (541 ms → 9.5 ms), identical results — enough to run adverse-selection Monte
  Carlo at millions of steps. Python generates the random flow (vectorised); Rust
  runs the sequential matching.

Both keep a pure-Python reference that is the specification, verified by parity
tests, and both fall back to Python when the extension isn't built.

## Why this is worth doing

Low-latency, cache-friendly systems code in Rust is among the highest-leverage
skills in quantitative trading. This module is a self-contained, real-world place
to learn PyO3, ndarray interop, and Python↔native boundaries — on a loop whose
correctness is already nailed down by the test suite.
