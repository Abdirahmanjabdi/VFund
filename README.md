# VFund

[![CI](https://github.com/Abdirahmanjabdi/VFund/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdirahmanjabdi/VFund/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![tests](https://img.shields.io/badge/tests-290%20passing-brightgreen)](tests/)

**An open-source quant research & trading platform — built around a single
principle: make it as hard as possible to fool yourself.**

VFund is a local-first Python toolkit (with an optional Rust core) for systematic
trading research across crypto, FX, commodities and equities. It ingests market
data, backtests strategies *without the usual self-deception*, validates them
out-of-sample, models real-world frictions, and runs a live forward paper-trading
loop. The tools are open; any edge you find with them is yours.

📖 **Start with the [case study](docs/CASE_STUDY.md)** — the honest story of how
this platform rigorously killed most of its own best ideas and what survived.
Then the [crypto alpha study](docs/CRYPTO_ALPHA_STUDY.md) — 41 published equity
alphas tested on crypto, where the platform caught *itself* mistaking a strong
information coefficient for a tradable edge. Then the
[multi-market study](docs/MULTI_MARKET_STUDY.md) — 35 hypotheses across crypto,
FX and gold, **all of which failed**, including one that passed every statistical
test before a leakage audit destroyed it. Reference:
[docs/OVERVIEW.md](docs/OVERVIEW.md) (architecture) and
[docs/EXAMPLES.md](docs/EXAMPLES.md) (every example explained).

> **What this repo is honest about.** Most of the research here produced negative
> results, and they are documented as prominently as the positive ones —
> including retracted findings and my own bugs. A platform whose purpose is to
> resist self-deception has no business hiding the times it failed.

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
| **Live book ≠ backtested book** | The live signal *is* the engine: `signal.py` asks `CrossSectionalBacktester._target_weights`, never rebuilds the overlay chain. Asserted by [`tests/test_parity.py`](tests/test_parity.py) |
| **Silent ops failure** | `vfund status` grades account staleness and exits non-zero; the weekly task gates on it |

## Install

```bash
git clone https://github.com/Abdirahmanjabdi/VFund
cd VFund
python -m venv .venv && . .venv/Scripts/activate   # Windows
# source .venv/bin/activate                        # macOS/Linux
pip install -e ".[dev]"
pytest -q                                          # 290 tests, no network needed
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

## What has been tested and rejected

Most of this repo's research produced negative results. They are listed because
a rejected hypothesis with a measured reason is more useful than an untested one,
and because the failure modes generalise. Full detail in the
[multi-market study](docs/MULTI_MARKET_STUDY.md).

| Family | Verdict |
|---|---|
| Derivative microstructure factors (5, crypto perps) | 4 show no predictive information; the 5th has the **opposite sign** to its hypothesis and no tradeable spread |
| Informed-vs-crowd positioning divergence | Reversed and weak — the mechanism is not in the data |
| Four-quadrant flow decomposition (price × ΔOI) | Dies once controlled for reversal |
| Illiquid-tail reversal | Rank IC t = 9.48, **mean spread t = 1.51** — unmonetisable |
| Delta-neutral funding carry | Real until 2024; **−2.0% over the last six months** |
| Month-end FX hedge rebalancing | Real (t = 3.84), **decayed after the 2015 fix reform** |
| 106 calendar/session buckets (FX, gold) | None survive family-wise correction |
| Trend following on GBPUSD / gold | OOS Sharpe −0.24 / 0.26 — gold **loses to buy-and-hold** |
| Gold-complex pair reversion | Stale-price artefact — dies at a one-day skip |
| OI-verified liquidity sweep | **Retracted — lookahead bias** |
| Gold variance risk premium | **Real** (t = 8.24, 79% hit rate), with caveats |

Two findings from those failures now shape the rest of the platform:

**Rank IC is not a screen.** It endorsed a factor at t = 9.48 whose tradeable
decile spread was t = 1.51. Rank correlation is blind to magnitude, and in a
fat-tailed market magnitude *is* the P&L. `research/ic.py` gates on the mean
decile spread and flags the mean/median sign disagreement that identifies an
unmonetisable factor.

**Cross-validation does not protect against leakage.** One signal cleared
Bonferroni across 32 configurations, a tail check, breadth across 14 of 15
instruments, *and* a train/test split — then failed a leakage audit outright. All
of those tests ask whether a pattern is real; a leak makes it genuinely real, and
it replicates in every fold. The audit is now a gate, not an afterthought.

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
│   ├── yahoo.py          FX, commodity futures, equities & indices (all markets)
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
│   ├── ic.py             Newey-West IC + the mean-spread gate rank IC cannot give
│   ├── seasonality.py    calendar/session buckets w/ family-wise correction
│   └── robustness.py     Probabilistic & Deflated Sharpe, bootstraps, alpha/beta
├── deriv_alpha/     # derivative microstructure (tested; see study — mostly dead)
│   ├── vision.py         deep open-interest history (the REST endpoint caps at 30d)
│   ├── crowding.py       OI acceleration, corrected sign
│   ├── flow.py           four-quadrant price × ΔOI decomposition
│   ├── illiquid.py       liquidity-provision reversal in the thin tail
│   ├── tiers.py          informed-vs-crowd positioning divergence
│   └── factors.py, composer.py, regime.py, markets.py, collector.py
├── vol/             # volatility risk premium
│   └── vrp.py            implied-vs-realised premium, tail capping, sizing
├── live/            # forward trading & execution
│   ├── signal.py         today's target book - engine-computed (parity)
│   ├── carry.py          funding-basis carry sleeve (the non-spot-weight engine)
│   ├── health.py         account staleness ladder; `vfund status` exits non-zero
│   ├── paper.py          persistent forward paper-account tracker
│   ├── exchange.py       Binance USD-M Futures API client (HMAC-SHA256, testnet/mainnet)
│   └── execute.py        reconciliation-based executor with full safety model
├── microstructure/  # order book & market-making
│   ├── orderbook.py      price-time limit order book + matching engine
│   └── simulator.py      market-making sim with adverse selection
└── analytics/       # Sharpe/Sortino/drawdown/CAGR/alpha-beta, reports, charts
```

## Data sources (all free, no keys)

- **Binance spot klines** — OHLCV for any pair (`vfund fetch` / `fetch-universe`).
- **Binance perpetual klines** — `fetch_klines(..., futures=True)` (for basis).
- **Binance funding rates** — perp funding history (`vfund fetch-funding`).
- **Binance Vision archive** — deep **open-interest** history plus participant-tier
  long/short ratios and taker flow, back to 2020-09 (`vfund.deriv_alpha.vision`).
  The obvious REST endpoint (`openInterestHist`) is hard-capped at **30 days** and
  rejects older requests outright; this routes around it.
- **Yahoo Finance** — FX pairs, commodity futures, equity tickers and indices
  (`vfund.data.yahoo`). Daily history is deep given explicit date bounds; hourly
  caps at ~730 days and finer intervals at ~60. Asking for `range=max` with a
  daily interval silently returns *monthly* bars, which is documented in the
  module because it is an easy way to run a study on the wrong data.
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

Derivative microstructure (`vfund/deriv_alpha/`) — **all tested, all rejected**,
retained as documented negative results: `OIPriceDivergence`, `CrowdingPressure`,
`LiquidationBounce`, `FundingMomentum`, `OIAcceleration`, `CrowdingAcceleration`,
`TierDivergence`, `FlowConfirmation`, `ForcedFlowReversal`, `IlliquidReversal`.
Each carries its measured result in its docstring. See the
[multi-market study](docs/MULTI_MARKET_STUDY.md).

## Formulaic alphas

Hand-writing a class per idea bounds how many hypotheses you can afford to test —
the wrong constraint for a platform premised on most ideas dying. `vfund/factors/`
lets an alpha be a formula instead:

```python
from vfund.factors import Panel, alpha, bench, all_alphas, panel_from_long
from vfund.factors.operators import rank, ts_corr

@alpha("my_alpha", formula="-1 * rank(ts_corr(close, volume, 10))", source="me")
def my_alpha(p: Panel):
    return -1.0 * rank(ts_corr(p.close, p.volume, 10))

print(bench(all_alphas(), panel_from_long(my_panel)))
```

**Look-ahead is inexpressible in the vocabulary**: `ts_*` operators read backwards
only, `delta`/`delay` refuse a non-positive lag, and no negative-shift operator
exists. An AST purity gate (`vfund/factors/purity.py`) rejects anything routing
around it — imports, `eval`, dunder access, reversed slices, `np.roll`, method
calls — and runs at registration, so a peeking alpha fails at import rather than
producing a flattering backtest. A test corrupts the future and asserts all 41
bundled alphas produce a byte-identical past.

Ships with 41 alphas (a Kakushadze-101 OHLCV subset + academic proxies) and an IC
bench with alive/reversed/dead categorisation. Operator semantics follow
[HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading)'s Alpha Zoo (MIT);
implementations are independent (pandas there, numpy here).

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

# Execution (testnet or mainnet)
vfund execute --data data/live.parquet --three-sleeve \
  --defi-data data/live_defi.parquet --tvl-data data/live_tvl.parquet --plan-only
vfund execute ... --testnet                           # dry run against testnet (default)
vfund execute ... --testnet --live                    # place real orders on testnet
vfund execute ... --mainnet --live                    # REAL MONEY (use with extreme caution)
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
| 2026-07-13 | $98,301 | −1.70% |
| 2026-07-22 | $95,487 | −4.51% |
| 2026-08-10 | $102,085 | +2.08% |
| 2026-08-17 | $102,892 | **+2.89% (peak)** |
| 2026-08-31 | $96,399 | −3.60% |
| 2026-09-04 | $94,649 | **−5.35% (trough)** |
| 2026-09-16 | $94,798 | −5.20% |
| 2026-09-19 | $96,419 | −3.58% |
| 2026-09-26 | $98,181 | **−1.82%** |

Thirteen weeks in, and **still below water**. The book peaked at +2.89% in
mid-August, then gave back 8.0 points over the following three weeks to a −5.35%
trough on 04 Sep, and has recovered in three consecutive up weeks to −1.82%. It
is currently 4.6% off its peak.

Two drawdowns now, both survived: −4.5% at week 3 (a junk rally where majors ran
+8% while the book stayed market-neutral) and −5.4% at week 10. Historically 5%
of rolling 3-week windows were ≤ −4.5% with a worst of −13.6%, so both sit inside
normal variance — but thirteen weeks of a net-negative forward record is exactly
the kind of result that should be reported plainly rather than framed around the
peak. The record is left untouched and unadjusted.

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

The two-engine account after 9 updates:

| Date | Equity | Since start |
|---|---|---|
| 2026-07-22 | $99,961 | −0.04% |
| 2026-08-10 | $103,068 | +3.07% |
| 2026-08-17 | $103,358 | **+3.36% (peak)** |
| 2026-08-31 | $99,839 | −0.16% |
| 2026-09-04 | $98,980 | −1.02% |
| 2026-09-16 | $98,701 | **−1.30% (trough)** |
| 2026-09-19 | $99,721 | −0.28% |
| 2026-09-26 | $101,007 | **+1.01%** |

Ten weeks in and **marginally positive**, having recovered from a −1.30% trough.
It is 2.3% off its August peak.

**What the comparison actually shows.** The two accounts share the same alpha
engine, so the gap between the curves is the closest available read on what carry
contributes forward. Over the drawdown it is consistent and in carry's favour:
the two-engine book gave up upside in the August rally (+3.07% vs the alpha
book's +2.08% — actually ahead, since carry also rose) and then lost far less on
the way down (−1.30% trough vs −5.35%). As of today the spread is **2.8 points**
in the blend's favour. That is the behaviour the 50/50 blend was designed for, on
a sample far too short to be conclusive.

One caveat on reading the state files: `alpha_equity` and `carry_equity` are
rebalanced to exactly half of total at every update, so those fields track the
blend rather than each engine's standalone contribution. Per-engine attribution
would need the sleeves tracked separately, which is not currently done. Carry's
standalone backtest Sharpe of ~5 therefore remains untested forward in isolation.

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

## Execution layer

When the forward test convinces, `vfund execute` turns the same target book into
real exchange orders on Binance USD-M Perpetual Futures. It is designed so that
every failure mode is survivable:

| Safety layer | How it works |
|---|---|
| **Dry run default** | No orders unless you pass `--live` explicitly |
| **Plan-only mode** | `--plan-only` shows the full order plan using public prices — no API keys or futures access needed |
| **Kill switch** | `touch data/KILLSWITCH` halts all execution instantly; `rm` resumes |
| **Reconciliation-based** | Reads current positions, diffs against target, places only the delta — idempotent and crash-safe |
| **Pre-flight checks** | Max gross exposure (2×), max single position (15%), min order size ($6), zero-equity guard |
| **Leverage cap** | Sets exchange-side max leverage to 3× on every symbol before any order |
| **Append-only log** | Every order, fill, skip, and error is written to `data/execution_log.jsonl` before the next order |
| **Closes first** | Reducing orders execute before opening orders to minimise risk during rebalance |

API keys are read from environment variables only — never from CLI arguments,
config files, or source code. Testnet and mainnet use separate key pairs.

```bash
# Plan only (no keys needed)
python -m vfund execute --data data/live.parquet --three-sleeve \
  --defi-data data/live_defi.parquet --tvl-data data/live_tvl.parquet --plan-only

# Testnet (free fake money, requires Binance futures testnet access)
export BINANCE_TESTNET_KEY="..."  BINANCE_TESTNET_SECRET="..."
python -m vfund execute ... --testnet --live
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

32 runnable scripts in `examples/` reproduce the entire research journey, from
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
- `deriv_alpha_study.py` — derivative microstructure factors (a negative result)
- `gold_vrp_study.py` — the gold variance risk premium, with its tail arithmetic

## Known limitations

The results are backtested and out-of-sample, **not** live-confirmed. Honest
caveats, in order of severity:

1. **The forward record is negative or flat, not positive.** Thirteen weeks in,
   the 3-sleeve account is **−1.8%** and the two-engine account **+1.0%** — both
   below their August peaks. Every backtested number predates the strategy's own
   design. Nothing here is confirmed, and the forward data so far does not
   support the backtested CAGR.
2. **Leakage is the failure mode that statistics will not catch.** One finding in
   this repo passed Bonferroni, a tail check, breadth testing *and* an
   out-of-sample split before a leakage audit destroyed it (see the
   [multi-market study](docs/MULTI_MARKET_STUDY.md)). Cross-validation defends
   against overfitting, not against future information inside an input.
3. **Survivorship is reduced, not eliminated.** Dead coins are re-included, but
   the current-liquid universe still has selection bias.
4. **Multiple testing across the whole search.** Deflated Sharpe adjusts for
   configs within one study, not for the ~35 hypotheses tried across the project.
   True significance is lower than any single study reports.
5. **Capacity.** The small-cap edge caps at ~$2–5M; it's not a large-AUM strategy.
6. **Execution realism.** The execution layer is built and tested but not yet
   live-proven. Real slippage, borrow availability, and (for the carry)
   intraday liquidation will shave results further.
7. **Carry has decayed.** Standalone funding-basis carry measured **−2.0% over the
   last six months** as institutional basis capital arrived. Its backtested
   Sharpe of ~5 is a historical artefact of a period that has ended, and the
   forward two-engine account has never isolated it.

## Develop

```bash
pip install -e ".[dev]"
pytest -q          # 290 tests, network-free
```

CI (`.github/workflows/ci.yml`) runs the suite on Python 3.11 & 3.12 for every
push. See [CONTRIBUTING.md](CONTRIBUTING.md) — the bar for a feature is: does it
make a backtest *harder to fool yourself with*, or add a well-motivated hypothesis?

## Not financial advice

VFund is a research tool. Nothing here is a recommendation to trade. Past — and
especially backtested — performance does not predict future results. See
[LICENSE](LICENSE) (MIT).
