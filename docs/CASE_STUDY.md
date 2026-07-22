# Case study: building a quant platform that kills its own best ideas

This is the story of VFund — not a victory lap, but an honest account of how a
systematic crypto research platform was built, and what it found when held to
professional standards. The punchline up front: **most of the "edges" it
discovered turned out to be illusions, and the platform's real value is that it
proved that — on paper, before any money was at risk.**

That is exactly what a research platform is *for*.

## The premise

The goal was a systematic (rule-based) trading strategy with a real, repeatable
edge — the kind you could eventually run a fund on. The trap everyone falls into:
write a backtest, see a beautiful equity curve, trade it, lose money. Backtests
lie in a dozen ways. So the platform was built around a single principle:

> **Make it as hard as possible to fool yourself.**

## What "honest" required (each defense earned its place)

Every layer below was added because, without it, a backtest flatters a strategy
that couldn't be traded:

- **No look-ahead.** Decisions use data through today's close and execute at the
  *next* bar. You cannot act on information you didn't have.
- **Real costs.** Commission, slippage, short-financing — modelled everywhere.
  Many "edges" are just unpaid transaction costs.
- **Out-of-sample discipline.** Design on 2021–2024; judge on 2025–2026, which the
  strategy never saw. Walk-forward selection simulates real deployment.
- **Multiple-testing correction.** Try enough ideas and one looks great by luck.
  The Deflated Sharpe Ratio discounts a result by how many things were tried.
- **Robustness.** Does it survive dropping random coins? Across sub-periods?
- **Survivorship correction.** The single biggest cheat: testing only on coins
  that still exist. We discovered Binance still serves *delisted* coins' data and
  added 41 coins that actually died back into the universe.
- **Hard-to-short.** You can't borrow illiquid coins to short them. Enforced with
  a liquidity gate that never peeks at the future.

## The graveyard of ideas (this is the point)

Idea after idea passed a naive test and then died under scrutiny:

| Idea | What happened |
|---|---|
| Moving-average crossover | Lost money — the engine told the truth on day one |
| Short-term reversal | Real signal, but died below ~1bp cost — a market-maker's edge, not ours |
| Funding carry | Looked good (Sharpe ~1), then the robustness harness rejected it (Deflated Sharpe ~15%) |
| Cross-sectional momentum | Robust *sign*, but ~zero magnitude |
| Low-vol, value, illiquidity, size, residual momentum | Strong in-sample, **dead out-of-sample** |

The recurring lesson, learned over and over: **in-sample brilliance — even after
passing robustness tests — routinely dies out-of-sample.** Well-known factors get
crowded and arbitraged away. The academic Sharpe was real *then*; it doesn't mean
it's tradable *now*.

## The one that held up (mostly): trend + size, and crisis alpha

Time-series **trend** (ride up-trends long, down-trends short) stood out — but only
after a crucial check. In a bull market a long-biased strategy *looks* brilliant;
that's **beta**, not skill. Stripping out beta with a regression revealed genuine
**alpha**: in the 2022 crash, trend made **+7%** while buy-and-hold lost **−74%**.
That downside protection — "crisis alpha" — is trend-following's documented,
durable property across a century of markets.

Combined with a **size** factor at equal risk (the two are nearly uncorrelated —
which is how real multi-strategy funds are built), the book survived *every*
honesty adjustment: broad universe, dead coins, short costs, and a no-borrow
constraint. Its honest profile: **~Sharpe 1, ~25%/yr, −42% worst drawdown, and it
makes money in crashes.** Notably, forbidding illiquid shorts *improved* it — the
edge lives in liquid shorts and small-cap longs, not in shorting dying dust.

But the reality check was humbling: run naively on 2025–2026 the combined book
*lost* money. The size premium had reversed; trend flatlined with no crash to
exploit. A candidate, not a confirmation.

## A new frontier: on-chain fundamentals

Since crowded price factors kept decaying, the search moved to less-mined data:
**on-chain TVL** (total value locked — the real capital using a protocol), free
from DefiLlama. Naive TVL momentum was dead. But **TVL *divergence*** — long coins
whose real usage grew while their price lagged, a fundamental value bet — was the
first *new* signal positive in **both** in-sample (1.09) and out-of-sample (0.44).
Modest, unconfirmed, but differentiated — and exactly where a small player has an
informational edge classic quants don't.

Combining the size factor and the on-chain sleeve with trend gave a diversified
**small-cap 3-sleeve book** — the near-uncorrelated sleeves lifted the Sharpe and
halved the drawdown, and it held out-of-sample (Sharpe ≈ 0.9). But it has a hard
limit: capacity. Modelling position caps by share of daily volume showed the edge
lives in small coins that can't absorb money — it fades past a few million dollars.

## Fixing survivorship for real

The biggest bias flatters every crypto backtest: testing only on coins that still
exist. It turned out Binance still serves klines for *delisted* symbols, so a
curated list of coins that actually died (LUNC, SRM, WAVES, …) could be added
back. The verdict was clear — the survivor-only in-sample Sharpe had been inflated;
the broader, dead-coin-inclusive universe was more honest and, usefully, more
robust out-of-sample. Adding real short-financing costs and a hard-to-short gate
(you can't borrow illiquid names) completed the de-biasing.

## Searching for capacity: the funding carry

A small-cap edge can't run a fund on its own, so the search turned to a
*high-capacity* complement. Price-based strategies on the majors found nothing —
BTC/ETH and friends are efficient and crowded. But a **structural** edge did
appear: **funding-basis carry**. Perpetual futures trade at a premium (leveraged
longs pay), so a delta-neutral long-spot/short-perp book harvests that funding
with huge capacity.

Three layers of modelling turned an exciting-but-fake number into an honest one:

1. **Naive** funding-only: Sharpe 4–7. Too good to be true.
2. **Real basis** (actual perp prices): the delta-neutral hedge genuinely contains
   drawdowns — but it's a *thin-margin yield*, so realistic costs flip the recent
   period negative. A modest carry, not a money printer. (This corrected the
   author's own prediction that basis blowups would dominate — the data said costs
   and funding-regime dependence do.)
3. **Intraday liquidation** (via daily perp highs): the carry's survival depends
   *entirely* on margin setup. Run with siloed leverage, a squeeze (DOGE spiked
   +412% intraday) wipes you out. Run cross-margined (spot collateralises the
   perp), it's safe. The operational setup *is* the risk management.

## The two-engine book

The two edges are nearly uncorrelated: a small-cap **alpha** engine and a majors
**yield** engine. Combined 50/50, the book keeps a strong Sharpe (~2.0) with about
a third of the drawdown of the alpha alone, and holds out-of-sample. That is a
realistic fund structure — a high-return small book plus a scalable, modest carry —
with every number honestly stress-tested.

A second on-chain sleeve was later added from protocol **fees** (revenue is harder
to fake than TVL), and a macro overlay from aggregate **stablecoin supply** was
tried and **rejected** — it looked plausible in-sample and earned nothing
out-of-sample (0.19). The negative result is kept in `examples/stablecoin_macro.py`
rather than deleted, because a research log that only records wins is a sales
pitch, not a record.

## Auditing the platform against itself

Late in the project the obvious question was asked: *is any of this actually real,
or has some subtle bug been flattering everything?* One bug — look-ahead, the past
seeing the future — would invalidate every number produced. So instead of trusting
the design, it was tested: corrupt all data after a cut point, and assert the
equity curve **before** that point is byte-identical. With every overlay active,
the difference was exactly **0.0**. That test now lives permanently in
`tests/test_lookahead.py`.

The audit found one genuine problem. On-chain metrics for day *t* aren't reliably
known until after *t*, so trading them same-day was a subtle look-ahead. Lagging
them a bar was the honest fix — and the edge **survived and improved** (the
headline book's out-of-sample Sharpe went 0.92 → 1.23). That's the outcome you
hope for: the signal was a slow fundamental one, not a data artifact. A more
convenient result would have been more suspicious.

## The forward test, in public

A $100k paper account has run since 2026-07-01 on the 3-sleeve book. Three weeks
in it is **down −4.5%** — published rather than quietly dropped. The market rose
+8.1% over the same window while the book held market-neutral, so this is a
*factor* drawdown (the known weakness of cross-sectional long/short in a broad
junk rally), and historically 5% of rolling 3-week windows were this bad or worse.

Two operational failures also surfaced immediately, which is its own lesson: the
weekly task once marked a **stale state file** forward by 22 months while
reporting success, and another run died because the machine was asleep at the
trigger. Both are fixed (a gap guard in `paper.py`, restart-on-failure on the
task). In a real fund, the ops layer fails long before the alpha does.

## What actually got built

A local-first Python platform (73 tests, CI on 3.11 & 3.12), with an optional
native Rust core (~77× on the hot loop). Market + perp + funding + delisted-coin +
on-chain (TVL, fees, stablecoin supply) ingestion; an event-driven, ragged
cross-sectional long/short engine modelling costs, short financing, shortability,
capacity, drawdown control, and maker fills — with its causality proven by test;
a research suite (walk-forward, robustness, Probabilistic/Deflated Sharpe,
alpha/beta); a market-microstructure layer (order book + adverse-selection sim); a
live signal + forward paper-trading loop; and 29 `examples/` that reproduce the
entire journey. It's public: <https://github.com/Abdirahmanjabdi/VFund>.

## The takeaway

VFund did not find a money-printing machine. It found a *candidate* and a
*frontier*, and — more importantly — it **rejected a long list of ideas that would
have lost real money**, quickly and cheaply. The most valuable output of quant
research isn't a strategy; it's the disciplined machine that tells you the truth,
and the judgment to trust it over a pretty backtest.

That discipline — not any single edge — is the real asset. The only clean test
left is the future, so a paper account now runs forward on data that didn't exist
when the strategy was designed. It is currently losing. That number is published
here unedited, and it will keep being published whichever way it goes, because a
research record you'd only show when it flatters you isn't a record at all.

Time, not another backtest, gets the final word.
