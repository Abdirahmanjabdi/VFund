# VFund — Project Overview (for contributors)

VFund is a local-first quantitative research platform for systematic crypto
trading. Its guiding principle is **honesty over flattery**: every feature exists
to make it *harder* to fool yourself about whether an edge is real.

## Philosophy

A backtest is a hypothesis, not a result. The platform is organised around the
ways backtests lie, and the defenses against each:

| Failure mode | Defense in VFund |
|---|---|
| Look-ahead bias | Event-driven engine: decide on close `t`, fill at `t+1` |
| Unpaid costs | Commission + slippage + short financing modelled throughout |
| Overfitting | Walk-forward selection; in-sample vs out-of-sample split |
| Multiple testing | Probabilistic & Deflated Sharpe (`research/robustness.py`) |
| Fragility | Sub-period stability + universe (coin-dropping) bootstrap |
| Survivorship bias | Ragged engine + `KNOWN_DELISTED` (dead coins re-included) |
| Un-executable shorts | Hard-to-short gate (`min_short_dollar_volume`) |

## Architecture

```
vfund/
├── data/        # OHLCV + multi-asset panel schema, Binance/universe/funding/
│                #   delisted ingest, Parquet storage, synthetic generators
├── strategy/    # single-asset + cross-sectional signals (reversal, momentum,
│                #   size, low-vol, value, trend, ensemble, carry, MAX, illiquidity)
├── backtest/    # event-driven engine, ragged cross-sectional L/S engine,
│                #   portfolio construction (weights, vol-target, shortability)
├── research/    # time-series splits, walk-forward, robustness harness
├── live/        # combined-book signal + forward paper-account tracker
└── analytics/   # Sharpe/Sortino/drawdown/CAGR/alpha-beta, reports, charts
```

The cross-sectional engine (`backtest/cross_sectional.py`) is the core. It:
- pivots a long panel to a ragged `time × symbol` matrix (coins enter/exit as
  listed/delisted),
- return-based simulation with one-bar execution lag and weight drift,
- charges turnover cost, short financing, and enforces a shortability mask,
- optionally vol-targets the book.

## Reproducing the research

The `examples/` scripts tell the whole story, in order:

1. `demo` / `compare_strategies` — the honest single-asset loop.
2. `hypothesis_gauntlet.py` — many signals through walk-forward + robustness.
3. `trend_cycle.py` — trend's crisis alpha, beta-adjusted, over a full cycle.
4. `build_edge.py` — vol-targeted trend + size, combined.
5. `robustness_combined.py` — the combined book through the full gauntlet.
6. `oos_gauntlet.py` — select in-sample, judge out-of-sample.
7. `survivorship_check.py` — dead coins + short costs + hard-to-short.
8. `account_100k.py` / `edge_stats.py` — the full performance profile.

Headline finding: after every honesty adjustment (broad + delisted universe,
short costs, no-borrow), a combined trend+size book lands around Sharpe ~1,
CAGR ~25%, max drawdown ~−42%, and *makes money in crypto crashes* — a candidate,
not a confirmed edge. Only forward paper-trading (`vfund paper`) can confirm it.

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md). The bar for a new feature: does it make
a backtest *harder to fool yourself with*, or add a well-motivated, testable
hypothesis? Keep strategies as pure signal; keep sizing/costs in the engine.

## Not financial advice

VFund is a research tool. Nothing here is a recommendation to trade. Past (and
especially backtested) performance does not predict future results.
