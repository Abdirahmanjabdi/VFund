# VFund roadmap

VFund is built in the order a quant desk is built, so that **shipping each layer
teaches the role it represents.** The tools are open; any edge found with them
stays private.

## v0 — the honest loop ✅ (this release)
Data → backtest → report, end to end, on free crypto data.
- Canonical OHLCV schema + Parquet storage (local-first)
- Binance ingest (no API key) + offline synthetic generator
- Event-driven backtester with one-bar execution lag (no look-ahead)
- Simulated broker: slippage + commission
- Performance analytics: return, CAGR, vol, Sharpe, Sortino, max drawdown
- CLI (`demo` / `fetch` / `backtest`) + equity/drawdown charts

**Teaches:** data engineering, market microstructure basics, and the single most
important quant skill — *not fooling yourself*.

## v0.1 — cross-sectional research + out-of-sample validation ✅ (this release)
- Multi-asset **panel** data model, storage, and Binance universe ingest
- Perp **funding-rate** ingest (for the funding-carry hypothesis)
- **Cross-sectional long/short engine**: dollar-neutral books, turnover-based
  costs, weight drift between rebalances
- Portfolio construction split from signal (`scores_to_weights`)
- First hypotheses: **cross-sectional reversal** and momentum
- **Walk-forward** optimisation: pick params in-sample, judge out-of-sample,
  and report the overfitting gap
- `vfund fetch-universe` and `vfund research [--walkforward]` CLI

**Teaches:** the quant-researcher workflow; that costs and out-of-sample decay,
not backtest returns, decide whether an edge is real.

### v0.1.x — research rigour ✅ / still to come
- ✅ Funding-rate ingest + funding-aware P&L in the cross-sectional engine
- ✅ Funding-carry strategy wired end to end (`vfund research --hypothesis carry`)
- ✅ Robustness harness: sub-period stability, universe (coin-dropping) bootstrap,
  Probabilistic & Deflated Sharpe. It **rejected** the funding-carry candidate
  (positive in only ~35% of coin subsets; Deflated Sharpe ~15%). Working exactly
  as intended — it killed a false positive that walk-forward alone had flattered.
- ✅ Hypothesis library + gauntlet: cross-sectional momentum, low-vol,
  value (MA-distance), and directional time-series trend, each run through
  walk-forward + robustness (`examples/hypothesis_gauntlet.py`). On 2023-24 data
  no *market-neutral* edge passed; time-series trend "passed" only because it
  rode market beta (Sharpe 1.66 vs 1.33 buy-and-hold) — needs a bear-market
  sample and beta-adjustment to judge as alpha.
- ✅ Bull+bear cycle test + beta-adjusted evaluation (`alpha_beta`,
  `examples/trend_cycle.py`). **Finding:** over 2021-2024, time-series trend
  shows *real* crisis alpha — beta ≈ 0, alpha ≈ 62%/yr (t ≈ 2), and it made
  +7% through the 2022 bear while buy-and-hold lost -74%. Grounded in the
  trend-following / managed-futures literature (Hurst-Ooi-Pedersen; crypto
  momentum in Liu-Tsyvinski-Wu, *J. Finance* 2022).
- ✅ `TimeSeriesTrendEnsemble` — multi-horizon trend (fixes single-lookback
  fragility); slightly lower point estimate (alpha t ≈ 1.7) but more robust.
- ✅ Vol-targeting (managed-futures risk control) in the engine; `CrossSectionalSize`
  (size factor, Liu-Tsyvinski-Wu); `examples/build_edge.py` combines trend+size.
  **Combined portfolio (full cycle, beta-adjusted): Sharpe 1.58, max DD -29%
  (vs -79% buy&hold), alpha ~33%/yr with t ≈ 2.7**, trend/size correlation ≈ 0.
  Two weakly-correlated, beta-adjusted-real edges beat either alone.
- ✅ Combined book through the robustness harness (`examples/robustness_combined.py`):
  **PASSED** — universe bootstrap 100% positive (5th-pct Sharpe 0.96), 5/6
  sub-periods positive, alpha t 2.8, Deflated Sharpe 99% over 18 configs. First
  candidate all project to survive the gauntlet that rejected carry.
- **REALITY CHECK — the edge FAILED out-of-sample.** Extending to 2025-2026
  (data the config never saw), the combined book returned **-13% (Sharpe -0.45)**
  vs +235% in-sample (`examples/account_100k.py`). Both sleeves broke: trend
  flatlined (no crash to exploit), and the size premium *reversed* (big caps
  led). In-sample robustness — even Deflated Sharpe 99% — did **not** survive
  live time. This is the whole point of the platform: we found it on paper, not
  with real money.
- ✅ **Survivorship + execution realism (partial).** Ragged engine (coins
  enter/exit as listed/delisted — `pivot_to_wide(drop_incomplete=False)`);
  short-financing costs (`short_cost_bps_annual`); broader auto-fetched universe
  with a stablecoin/peg/min-history filter (`examples/survivorship_check.py`).
  **Finding:** the survivor universe's in-sample Sharpe (1.51) was inflated — on
  a broad 39-coin universe it falls to ~0.9; short costs shave another ~0.2. But
  the broad universe held up out-of-sample (OOS Sharpe ~0.46 after short costs)
  where the narrow one went negative — modest, consistent, more believable.
- ✅ **Point-in-time / delisted coins (the real survivorship fix).** Binance's
  public klines still serve delisted symbols (series ends at delisting), and the
  ragged engine consumes that natively. `KNOWN_DELISTED` (41 curated dead USDT
  pairs: SRM, TORN, WAVES, ANT, OMG, WRX, …) feeds a survivorship-corrected
  universe. **Finding:** on the narrow survivor universe, adding dead coins
  slashed in-sample Sharpe 1.30→0.87 and OOS to −1.46 (the survivor result was
  badly inflated). On the broad universe it held up — broad+dead (80 coins,
  10%/yr shorts): full 0.86, IS 1.00, **OOS 0.46** — positive in both periods,
  the most credible number yet.
- ✅ **Hard-to-short model** (`min_short_dollar_volume`): a coin is shortable only
  with enough trailing liquidity — no look-ahead, and fading coins block their
  own shorts. **Surprise finding:** forbidding illiquid shorts *improved* the
  broad+dead book (OOS Sharpe 0.46 → 0.53-0.57, IS ~1.2). The edge lives in
  liquid shorts (majors) + small-cap longs, not in shorting dust. It survived
  every honesty adjustment thrown at it.
- **Where it stands:** strongest candidate found — broad + dead coins + short
  costs + no-borrow → IS ~1.2, OOS ~0.55, positive in both periods. BUT the
  2025-26 OOS has been examined many times (credibility burned); the only clean
  test left is forward paper-trading on untouched 2026+ data.
- Next: forward paper-trade; maker-fill modelling; expand delisted set; model
  small-cap long-side liquidity/capacity limits.

## v0.2 — large-cap / high-capacity search (in progress)
- Price-based large-cap strategies (trend, momentum, reversal, low-vol on 13
  majors) found NO out-of-sample edge — majors are efficient/crowded
  (`examples/largecap_search.py`).
- **Funding-basis carry** (structural, high-capacity): delta-neutral long-spot/
  short-perp harvests the perp premium. Timed (harvest only positive-funding
  coins) was strongly positive IS *and* OOS (`examples/funding_carry_major.py`).
  **BUT the reported Sharpe (4-7) is wildly optimistic** — the model ignores
  basis-blowup risk in crashes, real two-leg execution, margin/borrow costs. True
  net Sharpe is more like 1-2; the tail risk (funding/basis turns violently
  negative in deleveraging events) is the thing to model next. Still: the right
  *kind* of edge — structural, deep, scalable — to complement the small-cap book.
- Basis/crash risk modelled with real perp prices (`examples/funding_carry_basis.py`):
  delta-neutral hedge contains drawdowns (MaxDD a few %); real killers are thin
  margin (cost-sensitive) + funding-regime dependence — a modest yield, not a
  7-Sharpe.
- Intraday liquidation modelled via daily perp highs (`examples/carry_liquidation.py`):
  **the carry's survival hinges entirely on margin setup.** Siloed leveraged
  shorts get wiped out by squeezes (DOGE +412% intraday; siloed 10x → −100%).
  Cross-margined (spot collateralises perp, how desks run it) → barely liquidates,
  safe. Verdict: viable ONLY cross-margined, modest leverage, ideally BTC/ETH.
- Next: combine carry (high-capacity yield) with the small-cap 3-sleeve (alpha)
  as a two-engine book; intraday data for a finer liquidation model.

## v0.2 — on-chain fundamentals (in progress)
- ✅ DefiLlama TVL ingest (`data/onchain.py`, free, no key) + engine/context
  integration (`tvl=...`); `TVLMomentum` and `TVLDivergence` signals.
- **Finding (26 DeFi coins, select IS 2021-24, judge OOS 2025-26):** naive TVL
  momentum is dead (IS 0.23 / OOS 0.09); **TVL *divergence*** (long coins whose
  on-chain usage grew but whose price lagged — a fundamental value bet) is the
  first *new* signal positive in BOTH periods (IS 1.09 / OOS 0.44) — differentiated,
  less-crowded data. Promising lead, not confirmed (small universe, OOS peeked).
- Next on-chain: fees/revenue and stablecoin-flow signals; larger coin coverage;
  combine TVL-value with the trend/size book.

## v0.2 — hardening (realism)
- ✅ Position **capacity limits** (`capacity_aum`, `max_participation`): cap each
  name to a share of its daily dollar volume. **Finding:** the edge is
  capacity-limited to ~$100k-$500k — by $1M Sharpe falls 1.0 → 0.26; it's a
  small-capital strategy, not a big-fund one. At $100M+ only liquid majors remain
  (scalable but low return). See `examples/capacity.py`.
- ✅ Drawdown **circuit-breaker** (`dd_derisk_*`): de-risk exposure in deep
  drawdowns — cut worst drawdown ~47% → ~30%, trading some return for safety.
- ✅ Diversification: added the on-chain sleeve to trend+size (near-uncorrelated).
  3-sleeve book raised Sharpe (IS 1.19→1.60, OOS 0.57→0.66) and halved OOS
  drawdown (−24%→−11%) for ~the same return — held out-of-sample. See
  `examples/diversified.py`.
- ✅ Maker execution model (`fill_rate`, negative `cost_bps` for rebates):
  reversal is a *market-making* edge — dead as a taker (Sharpe −37 at 10bp),
  strong only when earning the rebate (Sharpe ~4.8 at −1bp / 50% fill), dead
  again at a +1bp maker fee. See `examples/maker_reversal.py`. **Caveat: the fill
  model ignores adverse selection (you get filled exactly when the market moves
  against you) — the real maker killer — so those Sharpes are optimistic. Needs
  a proper order-book / tick simulator to validate.**
- ✅ Microstructure layer (`vfund/microstructure/`): a price-time `LimitOrderBook`
  + matching engine, and a `MarketMakingSim` that models informed/uninformed
  flow. It reproduces **adverse selection** from first principles (realised-spread
  P&L): a maker earns the spread on noise flow but gets picked off by informed
  flow — losing money at tight spreads, exactly why the naive maker-reversal
  Sharpes were optimistic. See `examples/microstructure_adverse_selection.py`.
- Next: replay real L2/tick data through the book; simulate reversal as a maker
  under adverse selection to get its true Sharpe; port the book to Rust.

## v0.3 — native core (Rust, optional)
- ✅ Isolated the simulation hot loop into `vfund/backtest/sim.py` (`simulate_py`,
  the spec) with a dispatch to a native core if present. `tests/test_sim.py`
  proves the primitive reproduces the full engine's equity curve exactly.
- ✅ `rust/` — a PyO3 (`vfund_core`) extension implementing the identical loop;
  `_accel.py` auto-detects it; `examples/bench_sim.py` benchmarks + cross-checks.
  **Built and verified: ~77× faster (326ms → 4.2ms on 43.8k bars × 50 assets),
  results identical.** Build with `maturin develop --release` (GNU toolchain
  works with no MSVC). See [docs/RUST.md](docs/RUST.md). Falls back to pure
  Python when not built.
- Next: wire `simulate` into the engine's `run()` behind the fallback; port the
  order-book / tick reconstruction path; broaden the native surface.
- Vectorised indicator library (RSI, ATR, z-score, …)
- Cleaner carry model (perp prices / true delta-neutral basis) before any retest
- Notebook examples

## v0.2 — portfolio & risk
- Multi-asset portfolios and rebalancing
- Position sizing: fixed-fraction, volatility-targeting, Kelly
- Risk metrics: exposure, correlation, VaR, turnover

**Teaches:** the portfolio-manager role.

## v0.3 — data depth (the hard systems layer)
- Tick & trade ingestion; order-book reconstruction
- Rust core for the hot paths via PyO3 (introduced once the need is proven)

**Teaches:** low-latency systems, the highest-leverage engineering skill in the
field.

## v0.4 — live execution
- Paper trading against live feeds; broker/exchange adapters
- Execution algos (TWAP/VWAP), latency accounting

**Teaches:** the trader role; the gap between backtest and reality.

## v0.5 — research copilot
- Local-LLM assistant for strategy ideation, code, and report critique
- Strictly a *research* aid — never a source of the edge itself

**Teaches:** applied LLM tooling on top of real infrastructure.

---

### Guiding constraints
- **Local-first, private by default.** Your data and your strategies never leave
  your machine.
- **Open microscope, closed discoveries.** Infrastructure is shared; alpha isn't.
- **Honesty over flattery.** Every feature should make it *harder*, not easier,
  to fool yourself about a strategy's edge.
