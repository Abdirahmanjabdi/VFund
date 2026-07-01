# Examples guide

Every script in `examples/` is runnable and reproduces part of VFund's research
journey. Most need data in `data/` first (fetch commands noted). They're grouped
here in the order the research actually unfolded — reading them top to bottom is a
tour of how an edge is found, stress-tested, and mostly killed.

Run any with `python examples/<name>.py`.

## 1. The honest core loop

| Script | What it shows |
|---|---|
| `compare_strategies.py` | A moving-average crossover vs buy-and-hold on synthetic data — the engine tells the truth (the strategy loses). |
| `account_sim.py` | A single strategy's full performance report on a $-account basis. |

## 2. Cross-sectional research + out-of-sample discipline

| Script | What it shows |
|---|---|
| `hypothesis_gauntlet.py` | Six hypotheses run through walk-forward + robustness at once. On 2023-24 data, no market-neutral edge passed. |
| `oos_gauntlet.py` | **The core discipline.** Select each config in-sample (2021-24), judge on held-out 2025-26. Verdicts by cross-period consistency (overfit / noise / dead / candidate). |
| `robustness_carry.py` | The robustness harness *rejecting* funding carry (universe bootstrap + Deflated Sharpe ~15%). |
| `robustness_combined.py` | The full gauntlet on the combined book — the decisive honesty test. |

## 3. Finding an edge: trend, size, and combining them

| Script | What it shows |
|---|---|
| `trend_cycle.py` | Time-series trend's **crisis alpha**, beta-adjusted over 2021-2024. Trend made +7% in the 2022 crash vs −74% buy-and-hold; alpha t-stat ≈ 2. |
| `build_edge.py` | Vol-targeted trend + the size factor, combined at equal risk (uncorrelated → higher Sharpe). |
| `account_100k.py` | A $100k account 2021→2026, split in-sample vs out-of-sample — where the naive combined book *failed* OOS. |

## 4. De-biasing: survivorship, costs, capacity

| Script | What it shows |
|---|---|
| `survivorship_check.py` | Adds delisted (dead) coins, short-financing costs, and the hard-to-short gate. Reveals the survivor universe's in-sample Sharpe was inflated. |
| `capacity.py` / `capacity_curve.py` | How the edge decays as account size grows (position caps by share of daily volume). The small-cap edge caps around $2–5M. |
| `edge_stats.py` | The full statistical profile: return, risk, profit factor, turnover, holding period, per-year. |
| `progression.py` | **The big-picture arc:** naive → honest → honest+on-chain, on a $100k / 2021-today basis. De-biasing then genuine improvement. |

## 5. On-chain fundamentals (a differentiated data source)

| Script | What it shows |
|---|---|
| `onchain_tvl.py` | TVL momentum vs TVL *divergence* (usage grew but price lagged). Divergence is positive in-sample **and** out-of-sample — the first differentiated lead. |
| `diversified.py` | Adding the on-chain sleeve to trend+size. Near-uncorrelated; raises Sharpe and halves drawdown, out-of-sample. |

## 6. Execution & microstructure

| Script | What it shows |
|---|---|
| `maker_reversal.py` | Reversal under taker vs maker execution. Dead as a taker; only alive earning the rebate — a market-making edge. |
| `microstructure_adverse_selection.py` | An order-book market-making sim reproducing **adverse selection** — why the naive maker Sharpes overstate. |
| `bench_sim.py` | Benchmarks the simulation hot loop, Python vs the Rust core (~77×). |

## 7. Large-cap / high-capacity search

| Script | What it shows |
|---|---|
| `largecap_search.py` | Price-based strategies on 13 majors — **no** out-of-sample edge (majors are efficient). |
| `funding_carry_major.py` | Naive funding carry on majors (funding-only). High but optimistic Sharpe. |
| `funding_carry_basis.py` | Carry with the **real** perp basis. The hedge contains drawdowns; the real killers are thin margins (cost-sensitivity) and funding-regime dependence. |
| `carry_liquidation.py` | Intraday squeeze risk via daily perp highs. Siloed leverage gets wiped out (DOGE +412% intraday); cross-margin is safe. Survival depends on margin setup. |

## 8. The combined book

| Script | What it shows |
|---|---|
| `two_engine.py` | Small-cap alpha + majors carry, combined. Uncorrelated engines; a 50/50 blend lifts Sharpe to 1.80 and halves drawdown, out-of-sample. |

## Reproducing from scratch

Most scripts expect data. A typical setup:

```bash
vfund fetch-universe --top 60 --interval 1d --start 2021-01-01 --end 2026-07-01 --out data/uni_broad.parquet
vfund fetch-funding  --symbols BTCUSDT ETHUSDT ... --start 2021-01-01 --out data/funding_major.parquet
vfund fetch-tvl      --start 2021-01-01 --out data/tvl.parquet
# perp prices & delisted coins: see universe.py (KNOWN_DELISTED) and ingest futures=True
```

The offline scripts (`compare_strategies`, `microstructure_adverse_selection`,
`bench_sim`) need no data and run anywhere.
