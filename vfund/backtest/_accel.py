"""Optional native acceleration.

Tries to import the compiled Rust core (`vfund_core`, built from `rust/`). If
present, ``rust_simulate`` wraps it; otherwise it is ``None`` and callers fall
back to the pure-Python reference. Nothing here is required for VFund to work —
the Rust core is a performance option, not a dependency.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - depends on whether the extension was built
    import vfund_core  # type: ignore

    HAVE_RUST = True
except ImportError:
    vfund_core = None
    HAVE_RUST = False


def rust_simulate(
    rets: np.ndarray,
    reb_indices: np.ndarray,
    reb_weights: np.ndarray,
    *,
    initial: float = 10_000.0,
    cost_rate: float = 0.0,
    short_cost_per_bar: float = 0.0,
    funding: np.ndarray | None = None,
):  # pragma: no cover - only runs when the extension is built
    """Call the Rust simulation core, marshalling arrays to the expected dtypes."""
    if vfund_core is None:
        return None
    rets = np.ascontiguousarray(rets, dtype=np.float64)
    reb_indices = np.ascontiguousarray(reb_indices, dtype=np.int64)
    reb_weights = np.ascontiguousarray(reb_weights, dtype=np.float64)
    if funding is not None:
        funding = np.ascontiguousarray(funding, dtype=np.float64)
    return vfund_core.simulate(
        rets, reb_indices, reb_weights,
        float(initial), float(cost_rate), float(short_cost_per_bar), funding,
    )


# Exposed as None when the extension isn't built, so sim.simulate falls back.
if not HAVE_RUST:
    rust_simulate = None  # type: ignore
