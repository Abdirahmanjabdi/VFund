# VFund

[![CI](https://github.com/Abdirahmanjabdi/VFund/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdirahmanjabdi/VFund/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![tests](https://img.shields.io/badge/tests-115%20passing-brightgreen)](tests/)

**An open-source quant research & trading platform for crypto — built around a
single principle: make it as hard as possible to fool yourself.**

VFund is a local-first Python toolkit (with an optional Rust core) for systematic
trading research. It ingests market data, backtests strategies *without the usual
self-deception*, validates them out-of-sample, models real-world frictions, and
runs a live forward paper-trading loop. The tools are open; any edge you find with
them is yours.

📖 **Start with the [case study](docs/CASE_STUDY.md)** — the honest story of how
this platform rigorously killed most of its own best ideas and what survived. Then
[docs/OVERVIEW.md](docs/OVERVIEW.md) (architecture) and
[docs/EXAMPLES.md](docs/EXAMPLES.md) (every example explained).

---

## Why VFund exists

Most people who "backtest" a strategy accidentally cheat — they use future
information, ignore costs, test only on coins that still exist, overfit to the
past, or get excited about the best of a hundred noisy tries. They ship a
beautiful fake result and lose money.

VFund is built to prevent exactly that. Every feature exists to make a backtest
*more honest*, and the platform's real value is that it **kills bad ideas quickly
and cheaply, on paper, before any money is at risk.**

| How backtests lie | VFund's defense |
|---|---|
| Look-ahead bias | Event-driven engine: decide on close `t`, fill at `t+1` — and [*proven*](#verification-the-past-cannot-see-the-future) by a test that corrupts the future and asserts the past is byte-identical |
| Stale/slow data used too early | On-chain metrics are lagged a bar (`tvl_lag=1`) — you can't trade a number you don't have yet |
| Unpaid costs | Commission, slippage, short financing, hard-to-short all modelled |
| Overfitting | Walk-forward selection; strict in-sample vs out-of-sample split |
| Multiple testing | Probabilistic & Deflated Sharpe (`research/robustness.py`) |
| Fragility | Sub-period stability + universe (coin-dropping) bootstrap |
| Survivorship bias | Ragged engine + delisted coins re-included (`KNOWN_DELISTED`) |
| Un-executable trades | Hard-to-short gate, capacity limits, maker/fill modelling |
| Ignoring capacity | Position caps by share of daily volume — the edge decays with size |

## Install

```bash
git clone https://github.com/Abdirahmanjabdi/VFund
cd VFund
python -m venv .venv && . .venv/Scripts/activate   # Windows
# source .venv/bin/activate                        # macOS/Linux
pip install -e ".[dev]"
pytest -q                                          # 115 tests, no network needed
```

The optional native Rust core (a ~77× faster simulation loop) is separate; see
[docs/RUST.md](docs/RUST.md). Everything works in pure Python without it.

## 60-second demo (offline, no API key)

```bash
vfund demo
```

Generates synthetic price data, runs a moving-average strategy, and prints a
fund-style report (Sharpe, Sortino, CAGR, max drawdown, profit factor). The demo
strategy *loses* money — the engine tells the truth from the first run.

---

## The current result (honestly measured)

The platform's leading candidate is a **two-engine book**:

- **Alpha engine** — a small-cap cross-sectional book (trend + size + on-chain),
  higher return but capacity-limited to ~$2–5M.
- **Yield engine** — a majors funding-basis carry, modest return but scalable,
  run cross-margined.

They're nearly uncorrelated (≈ −0.02), which is why blending them roughly halves
the drawdown. Backtested 2021–2026 on $100k, after every de-biasing and friction,
with on-chain data lagged one bar:

| Book | $100k → | CAGR | Sharpe | Max DD | OOS Sharpe |
|---|---|---|---|---|---|
| 3-sleeve alpha *(the live config — see below)* | $388k | 27.9% | 1.35 | −21% | 1.00 |
| 4-sleeve alpha (adds an on-chain **fees** sleeve) | $409k | 29.2% | 1.52 | −15% | 1.23 |
| **4-sleeve alpha + carry (50/50)** | $260k | 19.0% | **1.98** | **−7%** | **1.23** |

Note the trade-off: the alpha alone earns the most money; blending in the carry
earns *less* but with a far better ride (Sharpe 1.98, −7% drawdown).

The original forward account was deliberately left on the earlier 3-sleeve config
rather than restarted every time research improved, because restarting a forward
test destroys its value. That kept the record honest but left the *best* book
untested forward — so as of 2026-07-22 a **second, parallel** account runs the
4-sleeve + carry candidate alongside it, and neither is restarted again. See
[live / forward paper trading](#live--forward-paper-trading).

**Nothing here is a confirmed, live-tradable edge.** These are backtested and
out-of-sample results with known optimism. The only real test is the live forward
account below. See [docs/CASE_STUDY.md](docs/CASE_STUDY.md) and the
[limitations](#known-limitations).

## Verification: the past cannot see the future

The failure that invalidates *every* number above is look-ahead bias. So it's
tested directly, not assumed — `tests/test_lookahead.py` runs the engine with
every overlay active (vol-targeting, shortability, capacity, funding, short
costs), then **violently corrupts all data after a cut point and asserts the
equity curve before it is byte-identical**:

```python
assert np.array_equal(base[:421], corrupted[:421])   # past is untouched
assert not np.allclose(base[421:], corrupted[421:])  # future did change (sanity)
```

It passes with a maximum difference of exactly 0.0. A full audit also confirmed
the on-chain edges survive (and *improve* under) conservative lagging, and that
blend-weight leakage is immaterial. Details in [ROADMAP.md](ROADMAP.md).

---

## Architecture

```
vfund/
├── data/            # ingestion, storage, universes
│   ├── models.py         canonical OHLCV bar schema
│   ├── panel.py          multi-asset (ragged) panels + funding alignment
│   ├── ingest.py         Binance spot & perp klines (paginated)
│   ├── universe.py       liquid universes, funding, delisted coins (KNOWN_DELISTED)
│   ├── onchain.py        DefiLlama TVL, fees & stablecoin supply (on-chain data)
│   ├── synthetic.py      GBM price/panel generators (offline demo & tests)
│   └── storage.py        Parquet read/write
├── strategy/        # signals
│   ├── base.py           single-asset Strategy interface
│   └── cross_sectional.py 13 cross-sectional + time-series strategies (see below)
├── backtest/        # engines
│   ├── engine.py         single-asset event-driven backtester
│   ├── cross_sectional.py ragged long/short engine (costs, funding, vol-target,
│   │                      shortability, capacity, drawdown breaker, fill model)
│   ├── construct.py      scores→weights, vol-targeting, shortability
│   ├── sim.py / _accel.py the hot-loop primitive + Rust dispatch
│   └── broker.py, portfolio.py, result.py
├── factors/         # formulaic alphas
│   ├── operators.py      19 look-ahead-proof operators (rank/ts_*/delta/decay)
│   ├── purity.py         AST gate: an alpha that could peek fails at import
│   ├── zoo.py            41 published alphas (Kakushadze 101 subset + academic)
│   └── bench.py          IC / IR + alive-reversed-dead categorisation
├── research/        # validation
│   ├── splits.py         time-series train/test + walk-forward windows
│   ├── walkforward.py    walk-forward optimisation (in-sample select, OOS judge)
│   └── robustness.py     Probabilistic & Deflated Sharpe, bootstraps, alpha/beta
├── live/            # forward trading
│   ├── signal.py         today's target book (single / 3- / 4-sleeve alpha)
│   ├── carry.py          funding-basis carry sleeve (the non-spot-weight engine)
│   └── paper.py          persistent forward paper-account tracker
├── microstructure/  # order book & market-making
│   ├── orderbook.py      price-time limit order book + matching engine
│   └── simulator.py      market-making sim with adverse selection
└── analytics/       # Sharpe/Sortino/drawdown/CAGR/alpha-beta, reports, charts
```

## Data sources (all free, no keys)

- **Binance spot klines** — OHLCV for any pair (`vfund fetch` / `fetch-universe`).
- **Binance perpetual klines** — `fetch_klines(..., futures=True)` (for basis).
- **Binance funding rates** — perp funding history (`vfund fetch-funding`).
- **Delisted coins** — Binance still serves klines for delisted symbols; a curated
  `KNOWN_DELISTED` list re-includes dead coins (LUNC, SRM, WAVES, …) to fix
  survivorship bias.
- **DefiLlama on-chain data** — total-value-locked per protocol (`vfund fetch-tvl`),
  plus protocol **fees/revenue** and aggregate **stablecoin supply** (a macro
  risk-appetite proxy). All lagged a bar before use.

## Strategies

Cross-sectional (rank a universe, dollar-neutral long/short):
`CrossSectionalReversal`, `CrossSectionalMomentum`, `CrossSectionalSize`,
`CrossSectionalValue`, `CrossSectionalLowVol`, `CrossSectionalMaxReturn`
(lottery), `CrossSectionalResidualMomentum`, `CrossSectionalIlliquidity`
(Amihud), `FundingCarry`, `TVLMomentum`, `TVLDivergence` (on-chain value).

Time-series / directional: `TimeSeriesTrend`, `TimeSeriesTrendEnsemble`.

Baselines: `MACrossover`, `BuyAndHold`. Writing your own is one method — see
[docs/OVERVIEW.md](docs/OVERVIEW.md).

## The research workflow

```python
from vfund.data.synthetic import generate_gbm_panel
from vfund.research import walk_forward
from vfund.strategy import CrossSectionalReversal

panel = generate_gbm_panel(20, 6000, reversion=0.15, seed=1)
wf = walk_forward(
    panel,
    lambda lookback: CrossSectionalReversal(lookback),
    [{"lookback": lb} for lb in (1, 2, 3, 6)],
    train_size=2000, test_size=800, backtest_kwargs={"cost_bps": 10},
)
print(wf.summary())     # reports the overfitting gap: in-sample minus out-of-sample
```

The discipline: **select parameters in-sample, judge on data never seen, and
correct for how many things you tried** (Deflated Sharpe). A result that survives
that is worth a second look; most don't.

## CLI reference

```bash
# Data
vfund fetch-universe --top 60 --interval 1d --start 2021-01-01 --out data/uni.parquet
vfund fetch-funding  --symbols BTCUSDT ETHUSDT --start 2021-01-01 --out data/f.parquet
vfund fetch-tvl      --start 2021-01-01 --out data/tvl.parquet

# Backtest / research
vfund demo                                              # offline synthetic
vfund backtest --data data/uni.parquet --strategy ma --fast 20 --slow 50
vfund research --data data/uni.parquet --walkforward --hypothesis reversal
vfund research --data uni.parquet --funding f.parquet --hypothesis carry --walkforward

# Live
vfund signal --data data/uni.parquet                   # today's target book
vfund paper  --data data/uni.parquet --state data/paper.json --start-equity 100000
vfund paper  --three-sleeve --data uni.parquet --defi-data defi.parquet --tvl-data tvl.parquet ...
```

Run `vfund <command> -h` for full options.

## Live / forward paper trading

The only honest test of an edge is data it has never seen. `vfund paper` marks a
persistent hypothetical account forward as new data arrives — re-fetch and re-run
periodically (a Windows scheduled task in `scripts/` automates it weekly) to
accumulate a genuine, untouched out-of-sample record. The live signal runs the
*validated* configuration (broad universe, hard-to-short gate).

**A live account has been running since 2026-07-01**, $100k notional, on the
3-sleeve book (trend 35% / size 33% / on-chain 32% by risk):

| Date | Equity | Since start |
|---|---|---|
| 2026-07-01 | $99,917 | −0.08% |
| 2026-07-08 | $99,906 | −0.09% |
| 2026-07-13 | $98,301 | −1.70% |
| 2026-07-22 | $95,487 | **−4.51%** |

It is currently **in a drawdown**, and that is published here rather than quietly
omitted. Context: over the same window the majors rallied **+8.1%**, and the book
stayed market-neutral (net exposure +0.008) — so this is a *factor* drawdown, the
known weakness of cross-sectional long/short in a broad junk rally, not a
market-tracking loss. Historically **5% of rolling 3-week windows were this bad
or worse** (worst: −13.6%), so it is well inside normal variance.

Three weeks proves nothing either way. The record is left untouched and unadjusted
— which is the entire point.

### A second account, on the leading candidate

Leaving the original account alone is right — restarting a forward test destroys
it — but it left the *best* configuration untested forward. So since **2026-07-22**
a **second, parallel** account runs the full two-engine book at $100k:

| Account | State file | Book |
|---|---|---|
| Original *(untouched)* | `data/paper.json` | 3-sleeve alpha |
| New | `data/paper_two_engine.json` | 4-sleeve alpha + funding carry, 50/50 |

Running both is deliberate. The alpha engine is common to each, so the difference
between the two curves *isolates what the carry engine actually contributes
forward* — rather than leaving that to a backtest to claim. Carry's standalone
backtest Sharpe is ~5, which is exactly the kind of number that should be
distrusted until forward data speaks: it comes from low volatility that masks
thin-margin and squeeze tail-risk (see `examples/carry_liquidation.py`).

Carry cannot be expressed as spot weights — it is a long-spot / short-perp pair
whose return is `funding − basis change` — so `vfund/live/carry.py` accrues it on
its own path, verified bit-identical (max diff 8.7e-19 over 1,977 bars) to the
`examples/two_engine.py` arithmetic it was extracted from.

```bash
vfund paper --two-engine --data data/live.parquet --defi-data data/live_defi.parquet \
  --tvl-data data/live_tvl.parquet --fees-data data/live_fees.parquet \
  --perp-data data/live_perp.parquet --funding-data data/live_funding.parquet \
  --state data/paper_two_engine.json --start-equity 100000
```

## The Rust core (optional)

The innermost simulation loop can run natively via a [PyO3](https://pyo3.rs)
extension in `rust/` — ~77× faster on the raw loop, identical results, with a pure
Python fallback when unbuilt. `CrossSectionalBacktester.run()` uses it
automatically when present. Build and details: [docs/RUST.md](docs/RUST.md).

## Microstructure

`vfund/microstructure/` provides a limit order book and a market-making simulator
that reproduces **adverse selection** from first principles — the effect that
makes naive maker backtests overstate. It's the foundation for honestly evaluating
short-horizon / market-making edges (like reversal) that die as a taker.

## Examples

29 runnable scripts in `examples/` reproduce the entire research journey, from
the first honest backtest to the full composed book. **Each is explained in
[docs/EXAMPLES.md](docs/EXAMPLES.md).** Highlights:

- `oos_gauntlet.py` — select in-sample, judge out-of-sample (the core discipline)
- `robustness_combined.py` — the full gauntlet (bootstrap + Deflated Sharpe)
- `trend_cycle.py` — trend's crisis alpha, beta-adjusted, over a full cycle
- `survivorship_check.py` — dead coins + short costs + hard-to-short
- `capacity_curve.py` — how the edge decays as capital grows
- `two_engine.py` — the combined alpha + carry book
- `crypto_alpha_study.py` — 41 equity alphas on crypto, with a null control
- `full_book.py` — the full composed book (4 alpha sleeves + carry + macro)

## Known limitations

The results are backtested and out-of-sample, **not** live-confirmed. Honest
caveats, in order of severity:

1. **No meaningful live track record.** The forward account is three weeks old and
   currently down −4.5% — far too short to conclude anything in either direction.
   Every backtested number predates the strategy's own design. Months are needed.
2. **Survivorship is reduced, not eliminated.** Dead coins are re-included, but
   the current-liquid universe still has selection bias.
3. **Multiple testing.** Deflated Sharpe adjusts for configs in one study, not the
   whole research search — true significance is lower.
4. **Capacity.** The small-cap edge caps at ~$2–5M; it's not a large-AUM strategy.
5. **Execution realism.** Real slippage, borrow availability, and (for the carry)
   intraday liquidation will shave results further.

## Develop

```bash
pip install -e ".[dev]"
pytest -q          # 115 tests, network-free
```

CI (`.github/workflows/ci.yml`) runs the suite on Python 3.11 & 3.12 for every
push. See [CONTRIBUTING.md](CONTRIBUTING.md) — the bar for a feature is: does it
make a backtest *harder to fool yourself with*, or add a well-motivated hypothesis?

## Not financial advice

VFund is a research tool. Nothing here is a recommendation to trade. Past — and
especially backtested — performance does not predict future results. See
[LICENSE](LICENSE) (MIT).
