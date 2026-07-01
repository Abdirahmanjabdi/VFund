"""Benchmark the simulation hot loop: Python reference vs the Rust core.

Runs a large simulation (long history, many assets, many rebalances) and times
it. If the Rust extension is built (`maturin develop --release` in `rust/`), it
also times that and reports the speedup; otherwise it just shows the Python
baseline and how to enable the native path.

    python examples/bench_sim.py
"""

import time

import numpy as np

from vfund.backtest import sim
from vfund.backtest._accel import HAVE_RUST
from vfund.backtest.sim import simulate_py

# A big but realistic problem: ~5 years of hourly bars, 50 assets, weekly rebal.
rng = np.random.default_rng(0)
T, N = 43_800, 50
rets = rng.standard_normal((T, N)) * 0.01
rets[0] = 0.0
reb_idx = np.arange(168, T, 168, dtype=np.int64)  # weekly on hourly bars
reb_w = rng.standard_normal((reb_idx.size, N))
reb_w -= reb_w.mean(axis=1, keepdims=True)  # dollar-neutral
reb_w /= np.abs(reb_w).sum(axis=1, keepdims=True)


def timed(fn, n=3):
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


print(f"Problem: {T:,} bars x {N} assets, {reb_idx.size} rebalances\n")

py = timed(lambda: simulate_py(rets, reb_idx, reb_w, initial=1e5, cost_rate=1e-3))
print(f"  Python reference : {py*1000:8.1f} ms")

if HAVE_RUST:
    from vfund.backtest._accel import rust_simulate

    # correctness check
    a = simulate_py(rets, reb_idx, reb_w, initial=1e5, cost_rate=1e-3)[0]
    b = rust_simulate(rets, reb_idx, reb_w, initial=1e5, cost_rate=1e-3)[0]
    assert np.allclose(a, b), "Rust and Python disagree!"
    rs = timed(lambda: sim.simulate(rets, reb_idx, reb_w, initial=1e5, cost_rate=1e-3))
    print(f"  Rust core        : {rs*1000:8.1f} ms   ({py/rs:.1f}x faster, results match)")
else:
    print("  Rust core        : not built")
    print("\n  To enable: install Rust (https://rustup.rs) + a C linker, then:")
    print("    pip install maturin && cd rust && maturin develop --release")
