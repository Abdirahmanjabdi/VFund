# VFund

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![tests](https://img.shields.io/badge/tests-60%20passing-brightgreen)](tests/)

<!-- CI badge (uncomment once GitHub Actions is enabled on the account):
[![CI](https://github.com/Abdirahmanjabdi/VFund/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdirahmanjabdi/VFund/actions/workflows/ci.yml)
-->

> Continuous integration is configured in `.github/workflows/ci.yml` and runs the
> full suite on Python 3.11 & 3.12. It will activate once GitHub Actions is
> enabled for the account.

**An open-source quant research & trading platform — open tools, private edge.**

VFund is a local-first toolkit for systematic trading research. Ingest market
data, backtest strategies *without fooling yourself*, and measure performance
the way a real fund does. The tools are open source. The edge you discover with
them is yours to keep.

📖 **New here? Read the [case study](docs/CASE_STUDY.md)** — the honest story of
building a platform that rigorously killed most of its own best ideas (and found
a differentiated on-chain lead). See also [docs/OVERVIEW.md](docs/OVERVIEW.md).

> **Status:** a working research platform — honest backtesting, walk-forward +
> robustness validation, survivorship-corrected data, and a live forward
> paper-trading loop. No confirmed edge yet; one promising on-chain lead.

---

## Why VFund exists

Serious trading infrastructure is almost all proprietary. The open-source corner
is either toy-grade or a single heavyweight incumbent. VFund is a third path: a
clean, modern, *learnable* platform where each component teaches one role of the
quant stack — and where contributors improve the **microscope**, never the
discoveries made with it.

The design principle that matters most: **you cannot accidentally cheat.** The
backtester hands your strategy a view of history that physically excludes the
future, and fills your orders one bar late, at realistic prices, after costs.

## Install

```bash
git clone https://github.com/Abdirahmanjabdi/VFund
cd VFund
python -m venv .venv && . .venv/Scripts/activate   # Windows
# source .venv/bin/activate                        # macOS/Linux
pip install -e ".[dev]"
```

## 60-second demo (no API key, fully offline)

```bash
vfund demo
```

This generates synthetic price data, runs a moving-average crossover, and prints
a fund-style performance report:

```
==============================================
  VFund backtest report — MACrossover
==============================================
  Bars                               2,000
  Trades                                ...
  Initial equity               $10,000.00
  Final equity                        ...
----------------------------------------------
  Total return                        ...
  CAGR                                ...
  Ann. volatility                     ...
  Sharpe (ann.)                       ...
  Sortino (ann.)                      ...
  Max drawdown                        ...
==============================================
```

## Backtest on real crypto

```bash
# Pull two years of hourly BTC bars from Binance (no key required)
vfund fetch --symbol BTCUSDT --interval 1h --start 2023-01-01 --out data/btc.parquet

# Backtest a 20/50 MA crossover, save an equity + drawdown chart
vfund backtest --data data/btc.parquet --interval 1h \
    --strategy ma --fast 20 --slow 50 --plot results/btc.png
```

## Use it as a library

```python
from vfund.data.synthetic import generate_gbm_bars
from vfund.backtest import Backtester
from vfund.strategy import MACrossover

data = generate_gbm_bars(2000, interval="1h", seed=1)
result = Backtester(data, MACrossover(fast=20, slow=50), interval="1h").run()

print(result.summary())
print(result.metrics()["sharpe"])
```

Writing your own strategy is one method:

```python
from vfund.strategy.base import Strategy, BarContext
import numpy as np

class Momentum(Strategy):
    def __init__(self, lookback: int = 100):
        self.lookback = lookback

    def on_bar(self, ctx: BarContext) -> float:
        if ctx.n_seen <= self.lookback:
            return 0.0
        past = ctx.closes[-self.lookback - 1]
        return 1.0 if ctx.bar.close > past else 0.0   # ride positive momentum
```

`on_bar` returns a **target weight** in `[-1, 1]` (`1.0` = fully long, `0.0` =
flat, `-1.0` = fully short). The engine handles rebalancing, slippage,
commission, and accounting.

## Cross-sectional research (v0.1)

Real crypto edge is usually *relative* — how coins move against each other — and
only counts if it survives out-of-sample and after costs. VFund tests exactly
that.

```bash
# Offline: does a short-term reversal signal survive out-of-sample?
# (synthetic data with a real reversal effect baked in, so you can see the
#  machine detect genuine edge — and watch costs destroy it)
vfund research --demo --walkforward --hypothesis reversal \
    --grid 1 2 3 6 --train-size 2000 --test-size 800 --cost-bps 0    # signal is real
vfund research --demo --walkforward --hypothesis reversal \
    --grid 1 2 3 6 --train-size 2000 --test-size 800 --cost-bps 10   # costs kill it

# On real data: pull a 30-coin universe, then research it
vfund fetch-universe --top 30 --interval 1h --start 2023-01-01 --out data/uni.parquet
vfund research --data data/uni.parquet --walkforward --hypothesis reversal

# Funding carry: harvest the perp funding spread (a low-turnover, cost-surviving edge)
vfund fetch-funding --start 2023-06-01 --end 2024-06-01 --out data/funding.parquet
vfund research --data data/uni.parquet --funding data/funding.parquet --walkforward \
    --hypothesis carry --rebalance-every 24 --top-k 5 --cost-bps 5
```

Two hypotheses ship in v0.1, and they teach opposite lessons:

* **Short-term reversal** — a real signal, but so fast it dies below ~1 bp of
  cost. A cautionary tale about the backtest-to-reality gap.
* **Funding carry** — lower-turnover and structural. A naive walk-forward looked
  promising (OOS Sharpe ~1.0 at 5 bp), but the robustness harness **rejected**
  it: under a coin-dropping bootstrap it was positive only ~35% of the time, and
  its Deflated Sharpe (adjusting for how many configs were tried) fell to ~15%. A
  textbook case of walk-forward flattering a fragile result — and exactly why the
  harness exists. See [`examples/robustness_carry.py`](examples/robustness_carry.py).

Neither hypothesis is a tradable edge yet. That's the normal state of honest
research: most ideas die, and the value is in killing them quickly.

The walk-forward report prints the **overfitting gap** (in-sample minus
out-of-sample) — the number that tells you whether you found an edge or just fit
noise.

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
print(wf.summary())
```

## From research to a book

Building on the literature (Liu-Tsyvinski-Wu size factor; Hurst-Ooi-Pedersen
trend), a **combined trend + size book** survived the full gauntlet on 2021-2024
data — Sharpe ~1.6, alpha t ≈ 2.7, Deflated Sharpe ~99% over the configs tried
([`examples/build_edge.py`](examples/build_edge.py),
[`examples/robustness_combined.py`](examples/robustness_combined.py)).

The live signal runs the **validated** configuration — a broad, cleaned universe
(stablecoins/pegs/thin listings removed) with the hard-to-short gate (only short
names with enough recent liquidity).

```bash
# 1. Fetch a broad current universe
vfund fetch-universe --top 60 --interval 1d --start 2021-01-01 --out data/live.parquet

# 2. What should I hold right now? (long small-caps, short liquid majors)
vfund signal --data data/live.parquet

# 3. Forward-track a hypothetical account. Re-fetch (step 1) and re-run this
#    periodically to accumulate a clean, never-seen out-of-sample record — the
#    only honest test left once the historical OOS window has been studied.
vfund paper --data data/live.parquet --state data/paper.json --start-equity 100000
```

## Known limitations — read before trusting any number

The combined book passed *in-sample* robustness. That is **not** proof it makes
money live. Honest caveats, in order of severity:

1. **Survivorship bias (addressed, not perfect).** The engine is ragged (coins
   enter/exit as listed/delisted), short costs are modelled, and — crucially —
   Binance still serves *delisted* coins' candles, so `KNOWN_DELISTED` feeds a
   survivorship-corrected universe of coins that actually died. This *revealed*
   the bias: on the hand-picked survivor universe, adding dead coins cut
   in-sample Sharpe 1.30→0.87 and pushed OOS to −1.46. The broader universe held
   up (broad+dead, with short costs: IS 1.00 / OOS 0.46). Residual gap: shorting
   coins into delisting is often impossible, so those short profits are
   optimistic. See [`examples/survivorship_check.py`](examples/survivorship_check.py).
2. **One bear cycle (n=1).** The crisis-alpha rests on a single 2022 crash.
3. **Multiple testing.** The Deflated Sharpe adjusts for configs in one script,
   not the whole research search — true significance is lower.
4. **Short-side frictions** (borrow, funding, liquidation) are not modelled; the
   size factor uses a dollar-volume proxy, not true market cap.
5. **Capital reality.** A ~20-name long/short book needs ~$10k+ and a futures
   account; it can't run on a $100 spot account
   ([`examples/account_sim.py`](examples/account_sim.py)). Paper-trade first.

## Architecture

```
vfund/
├── data/        # OHLCV + multi-asset panel schema, Binance/universe/funding ingest,
│                #   Parquet storage, synthetic GBM (single & panel)
├── strategy/    # single-asset Strategy + cross-sectional (reversal, momentum) strategies
├── backtest/    # event-driven engine, cross-sectional L/S engine, broker (costs),
│                #   portfolio, portfolio construction, result
├── research/    # walk-forward, robustness harness (bootstrap, deflated Sharpe)
├── live/        # combined-book signal generation + paper-account tracker
└── analytics/   # Sharpe/Sortino/drawdown/CAGR/alpha-beta, reports, charts
```

Each layer maps to a job on a quant desk — that's deliberate. See
[ROADMAP.md](ROADMAP.md) for where it's headed (order-book data, live execution,
a research copilot) and how the build doubles as a curriculum.

## Learn more

[docs/OVERVIEW.md](docs/OVERVIEW.md) — architecture, the anti-self-deception
philosophy, and how to reproduce the research end to end.

## Develop

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE). Contributions welcome.
