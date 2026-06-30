# VFund

**An open-source quant research & trading platform — open tools, private edge.**

VFund is a local-first toolkit for systematic trading research. Ingest market
data, backtest strategies *without fooling yourself*, and measure performance
the way a real fund does. The tools are open source. The edge you discover with
them is yours to keep.

> **Status:** v0 — the honest data → backtest → report loop works end to end on
> free crypto data. This is the foundation everything else is built on.

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
git clone https://github.com/your-handle/vfund
cd vfund
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

## Architecture

```
vfund/
├── data/        # canonical OHLCV schema, Binance ingest, Parquet storage, synthetic GBM
├── strategy/    # the Strategy interface + baselines (MA crossover, buy & hold)
├── backtest/    # event-driven engine, simulated broker (costs), portfolio, result
└── analytics/   # Sharpe/Sortino/drawdown/CAGR, text reports, equity charts
```

Each layer maps to a job on a quant desk — that's deliberate. See
[ROADMAP.md](ROADMAP.md) for where it's headed (order-book data, live execution,
a research copilot) and how the build doubles as a curriculum.

## Develop

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE). Contributions welcome.
