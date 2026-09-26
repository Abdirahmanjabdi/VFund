# Does a positioning edge exist outside crypto?

*~35 hypotheses across crypto perpetuals, FX and gold. 2000–2026.
Reproduce with `examples/deriv_alpha_study.py` and `examples/gold_vrp_study.py`.*

---

## TL;DR

**None of them worked.** Thirty-five hypotheses were tested across three asset
classes with deep data and consistent methodology. Zero produced a tradeable
edge after transaction costs and multiple-testing correction.

That is the result, and it is worth publishing. Four hypothesis families failed
for four *different structural* reasons, and each reason constrains what is
worth trying next far more usefully than another backtest would.

Two methodological findings came out of it that now apply to everything else in
this repo:

1. **Rank IC is not a screen.** It reported a t-statistic of 9.48 on a factor
   whose tradeable spread was t = 1.51. `vfund.research.ic.decile_spread` now
   gates on the mean decile spread instead, and flags the mean/median sign
   disagreement that identifies an unmonetisable factor.
2. **Cross-validation does not protect against leakage.** A signal with future
   information in it cleared Bonferroni across 32 configurations, a tail check,
   breadth across 14 of 15 instruments, and a train/test split. All of those
   test whether a pattern is *real*; a leak makes it genuinely real.

---

## What was tested, and how each one died

| Hypothesis | Market | Outcome |
|---|---|---|
| OI–price divergence, crowding pressure, liquidation bounce, funding momentum | Crypto perps | 4 of 5 show no predictive information |
| OI acceleration (2nd derivative of positioning) | Crypto perps | Real IC (t = −5.06), **sign opposite to hypothesis**, no tradeable spread (t = 0.20) |
| Informed-vs-crowd tier divergence (top-trader vs all-account L/S) | Crypto perps | Reversed and weak; the mechanism is not in the data |
| Four-quadrant flow decomposition (price × ΔOI) | Crypto perps | Dies entirely once controlled for reversal |
| Illiquid-tail reversal | Crypto perps | Rank IC t = 9.48, mean spread t = 1.51 — unmonetisable |
| Delta-neutral funding carry | Crypto perps | Real until 2024, **−2.0% over the last six months** |
| Verified liquidity sweep (OI-confirmed stop run) | Crypto perps | **Retracted — lookahead bias.** See below |
| 106 calendar and session buckets | FX, gold | None survive family-wise correction |
| Month-end equity-hedge rebalancing | FX | Real (t = 3.84), **decayed after the 2015 fix reform**; ~1.3%/yr |
| Commodity-index roll window (business days 5–9) | Gold | −0.70 bps vs other days, t = −0.22 |
| Gold vs real rates | Gold | Real contemporaneously (t = −3.48), **not predictive** lagged (t = −0.97) |
| Trend following | FX, gold | GBPUSD OOS Sharpe −0.24; gold 0.26, **loses to buy-and-hold at 0.70** |
| Gold-complex pair reversion (6 pairs) | Gold | Stale-price artefact — dies at a one-day skip |
| Gold-miner residual momentum | Gold equities | t = 2.83 against a 2.94 threshold; dies on small-cap costs |
| Gold variance risk premium | Gold options | **Real and persistent** — see caveats |

---

## The four structural reasons

Each family failed for its own reason, and the reasons are more durable than the
results.

**Crypto: the payoff shape defeats long/short.** In the thin tail of the
perpetual universe, *every* decile has a negative median and a positive-ish
mean — most alt perps bleed and a few go +180% in a day. A short leg eats those
explosions. This is not a missing factor; the return distribution is hostile to
the entire strategy family, which explains four separate failures at once.

**FX and gold: breadth of two.** Portfolio Sharpe scales as single-market Sharpe
× √N across uncorrelated sleeves. With two instruments the ceiling is roughly
0.4 regardless of signal quality. The two sleeves tested correlated at +0.054 —
beautifully uncorrelated, and still not enough of them. This is why
managed-futures funds run 50–200 markets rather than picking the best two.

**Carry: crowding.** The basis trade was real and paid well through 2024, then
went negative as institutional capital entered crypto basis via ETFs and basis
funds. The decay is visible year by year and mechanistically explained.

**The fix effect: regulation.** Month-end FX hedge-rebalancing flow was real and
significant (t = 3.84 on a USD basket) and decayed to t = 0.76 over the most
recent five years. Banks were fined for front-running the 4pm WM/Reuters fix and
the calculation window was widened from one minute to five in February 2015. The
data matches the reform.

---

## The retraction, in full

One finding looked genuine long enough to be written up before it failed.

The setup: a liquidity sweep (price breaks a prior 24-bar extreme, then closes
back inside it) confirmed by open interest *falling* — evidence that resting
stops were genuinely cleared rather than new positions opened. Measured on
806,427 hourly bars across 20 instruments, it returned **+33.2 bps over 24
hours at t = 8.00**, with the median (+44.09) *above* the mean, 14 of 15
instruments clearing cost individually, and out-of-sample significance intact.

It was lookahead bias. Daily open interest is an **end-of-day** value; a sweep
firing at 05:00 was being classified using open interest measured at 23:59 —
nineteen hours in its own future. Worse, the leak is circular: "open interest
fell over day D" partly encodes "price moved in direction X during day D", and
day D is largely the window the forward return was measured over. The filter was
selecting the trades that had already worked.

| Verification | n | Net mean | t |
|---|---|---|---|
| Same-day OI change — uses the future | 10,191 | +23.20 bps | 5.59 |
| **Prior-day OI change — causal** | 10,221 | **−3.90 bps** | **−0.88** |
| No verification | 17,190 | −8.19 bps | −2.41 |

The unverified sweep is worth its own note, because it is the setup as commonly
taught: all 16 configurations lose money after costs, with win rates of **52–54%
and negative expectancy**. It wins often and loses big, which is a reasonable
explanation for why people believe in it.

A separate check answers the natural follow-up. Converted to a stop-and-target
strategy, the best of twelve configurations (2 ATR stop, 1:3 target) returns
**+0.006 R at t = 0.36** — statistically zero, with a 29.4% win rate against a
25% breakeven. The edge was an *average drift*; a stop samples a single path.

---

## The one thing that is real

The **gold variance risk premium** is not a prediction, and that is why it
survived. Options on gold systematically price more volatility than gold
subsequently delivers:

| | |
|---|---|
| Mean implied (GVZ) | 18.94 |
| Mean subsequent realised | 16.29 |
| **Premium** | **+2.65 vol points (+14% of implied)** |
| Implied > realised | **79.0% of the time** |
| Newey-West t (lag 21) | **8.24** |

It is positive and significant in every sub-period — +3.82 (2008–13), +2.34
(2014–19), +1.94 (2020–26) — where carry went negative and the fix effect went
to zero. An insurance premium stops paying only when buyers stop wanting
insurance.

The tail is the whole problem: the worst month cost 55 vol points against a mean
gain of 1.77, a ratio of 31:1, and the worst ten of 217 months put 55% of
cumulative profit at risk. Capping the loss at 5 vol points — which is what
buying the protective wing of a spread does — raises the t-statistic from 3.45
to **6.94** while keeping 81% of the mean, because in this distribution the tail
is nearly all of the variance.

**Three caveats that matter.** It is measured from a volatility *index*, not
traded option prices, so it is not a backtest of an executable structure.
Daily-sampled realised variance understates true variance, biasing the estimate
upward. And the premium has been shrinking — 2024 +1.17, 2025 +0.22, 2026
−1.24 — with the worst month in eighteen years occurring in January 2026.

Reproduce with `python examples/gold_vrp_study.py`.

---

## What this cost, and what it bought

Thirty-five hypotheses, four bug fixes to pre-existing code (a biased midrank
formula, degenerate-variance z-scores, inverted quadrant signs, and a data job
that died on two malformed files in the exchange's own archive), and three
errors of my own — the rank-IC screen, a return-smearing bug that produced a
spurious Sharpe of 13.5, and the lookahead above.

What it bought:

- **`vfund.deriv_alpha.vision`** — deep open-interest history. The obvious
  endpoint (`openInterestHist`) is hard-capped at 30 days and rejects older
  requests outright; the Vision archive reaches back to 2020-09 and also carries
  participant-tier long/short ratios and taker flow.
- **`vfund.data.yahoo`** — one interface for FX, commodity futures, equities and
  indices, with the interval limits documented so a study is not silently run on
  monthly bars.
- **`vfund.research.ic`** — Newey-West corrected IC *plus* the mean decile
  spread gate and its tail-driven flag.
- **`vfund.research.seasonality`** — calendar testing with family-wise
  correction and a cost hurdle enforced rather than optional.
- **`vfund.vol.vrp`** — variance risk premium measurement with the tail-capping
  arithmetic.

---

## The honest conclusion

Across three asset classes and thirty-five hypotheses, no causal signal survived
cost and correction. The marginal return on further hypothesis-hunting in liquid
public markets, using public data, appears low.

The finding that generalises is the failure mode. A repeatedly-validated result
can still be wrong in a way that no amount of cross-validation detects, and the
only defence is auditing what each input knew and when it knew it. That audit
now runs before publication in this repo rather than after.
