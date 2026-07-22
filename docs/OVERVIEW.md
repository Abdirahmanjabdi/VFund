# Project overview (for contributors)

VFund is a local-first quantitative research platform for systematic crypto
trading. Its guiding principle is **honesty over flattery**: every feature exists
to make it *harder* to fool yourself about whether an edge is real.

## Philosophy

A backtest is a hypothesis, not a result. The platform is organised around the
ways backtests lie, and the defences against each:

| Failure mode | Defence in VFund |
|---|---|
| Look-ahead bias | Event-driven engine: decide on close `t`, fill at `t+1` |
| Unpaid costs | Commission, slippage, short financing modelled throughout |
| Overfitting | Walk-forward selection; in-sample vs out-of-sample split |
| Multiple testing | Probabilistic & Deflated Sharpe (`research/robustness.py`) |
| Fragility | Sub-period stability + universe (coin-dropping) bootstrap |
| Survivorship bias | Ragged engine + `KNOWN_DELISTED` (dead coins re-included) |
| Un-executable trades | Hard-to-short gate, capacity limits, maker/fill model |
| Ignoring capacity | Position caps by share of daily volume |

## Package layout

```
vfund/
├── data/         ingestion, storage, universes
├── strategy/     single-asset + cross-sectional/time-series signals
├── backtest/     event-driven + ragged cross-sectional engines, construction
├── research/     splits, walk-forward, robustness harness
├── live/         signal generation + forward paper tracker
├── microstructure/  order book + market-making (adverse selection)
└── analytics/    performance metrics, reports, charts
```

## The core engine (`backtest/cross_sectional.py`)

`CrossSectionalBacktester` is the heart of the platform. It:

- pivots a long panel into a **ragged** `time × symbol` matrix — coins enter and
  exit the cross-section as they were actually listed/delisted (nulls outside
  their life), which is the survivorship-reducing choice;
- runs a **return-based simulation** with one-bar execution lag and weight drift;
- charges **turnover cost**, **short financing**, and enforces a **shortability
  mask** (can't short illiquid names), a **capacity cap** (position ≤ a share of
  daily volume), a **drawdown circuit-breaker**, and a **fill-rate** (maker) model;
- optionally **vol-targets** the book and applies an **on-chain / funding overlay**.

The innermost loop is isolated in `backtest/sim.py` (`simulate_py`, the spec) and
dispatched to a native Rust core when built (`backtest/_accel.py`, `rust/`). A
parity test (`tests/test_sim.py`) proves `run()` produces identical results either
way. `run()` uses the fast path automatically when the Rust core is present and no
running-equity overlay (drawdown breaker) is active.

## Adding a strategy

Cross-sectional strategies implement one method — return a per-asset score:

```python
from vfund.strategy.cross_sectional import CrossSectionalStrategy, PanelContext
import numpy as np

class MyFactor(CrossSectionalStrategy):
    def scores(self, ctx: PanelContext) -> np.ndarray:
        closes = ctx.closes          # (n_seen, n_assets), truncated at 'now'
        # ctx.volumes, ctx.funding, ctx.tvl are also available when supplied
        return -(closes[-1] / closes[-2] - 1.0)   # e.g. one-bar reversal
```

The engine converts scores → dollar-neutral (or directional) weights, applies all
overlays, and simulates. Keep strategies **pure signal**; keep sizing and costs in
the engine.

## Validation tooling (`research/`)

- `splits.py` — forward-only train/test and walk-forward windows.
- `walkforward.py` — pick params in-sample, judge out-of-sample, report the gap.
- `robustness.py` — Probabilistic & Deflated Sharpe (Bailey & López de Prado),
  sub-period stability, universe bootstrap, `alpha_beta` (regress on a benchmark
  to separate skill from market exposure).

## Reproducing the research

The `examples/` scripts tell the whole story in order — see
[EXAMPLES.md](EXAMPLES.md). Headline arc: honest backtesting → most factors die
out-of-sample → trend's crisis alpha + size + on-chain combine into a small-cap
book → survivorship/cost/capacity de-biasing → a majors funding carry for
capacity → a two-engine (alpha + yield) book → a self-audit proving the engine
has no look-ahead. No edge is *live-confirmed*; a forward paper account (running
since 2026-07-01) is the judge — see the log in [../ROADMAP.md](../ROADMAP.md).

## Contributing

See [../CONTRIBUTING.md](../CONTRIBUTING.md). The bar for a new feature: does it
make a backtest *harder to fool yourself with*, or add a well-motivated, testable
hypothesis? All PRs must keep `pytest` green (CI runs it on 3.11 & 3.12).

## Not financial advice

VFund is a research tool. Nothing here is a recommendation to trade. Past — and
especially backtested — performance does not predict future results.
