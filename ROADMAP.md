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

### v0.1.x — still to come
- Vectorised indicator library (RSI, ATR, z-score, …)
- Funding-carry strategy wired end to end
- Deflated Sharpe / multiple-testing corrections
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
