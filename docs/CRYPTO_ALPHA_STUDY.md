# Do equity formulaic alphas work on crypto?

*41 published equity alphas, 80 coins including 41 delisted, 2021–2026.
Reproduce with `python examples/crypto_alpha_study.py`.*

---

## TL;DR

Of 41 published equity alphas benchmarked on a survivorship-corrected crypto
universe, **21 (51%) look alive on the full window — but only 5 (12%) hold their
verdict out-of-sample**, and the strongest survivors are ones VFund's own
PnL-level testing has *already killed once transaction costs are charged*.

The headline number to take away is **12%, not 51%**, and even that is a
signal-quality result, not a profitability result.

The most useful output of this study is not a list of alphas. It is a
demonstration, on our own data, that **a strong information coefficient and a
profitable strategy are different things** — a gap that is easy to state and
apparently easy to forget.

---

## Why run this at all

The published formulaic-alpha catalogues — Kakushadze's 101, GTJA's 191, Qlib's
158 — are the most systematic public libraries of short-horizon factors that
exist. Every one of them was written for **equities**. As far as we can find,
nobody has published an honest test of them on crypto.

Crypto is not a change of venue so much as a change of physics. No earnings, no
sectors worth the name, no comparable market-cap tiers, 24/7 trading with no
overnight gap, and a flow structure dominated by retail rather than
institutions. A factor built on the microstructure of US equities has no
particular reason to survive that.

So the honest prior was that almost nothing transfers, and a negative result
would be perfectly publishable. That prior turned out to be roughly right — but
not for the reason expected.

## Method

| | |
|---|---|
| **Universe** | 80 coins: the broad liquid set **plus 41 delisted coins** re-included so the dead do not silently vanish. The panel is ragged — a coin is NaN before listing and after death, never forward-filled. |
| **Window** | 2021-01-01 → 2026-07-01, 2,008 daily bars |
| **Alphas** | 41 — a 32-formula OHLCV-only subset of Kakushadze (2015) plus 9 academic price-based proxies. Formulas needing sector tags or market cap are **omitted, not approximated**: crypto has no sector data, and inventing one would manufacture a result. |
| **Signal** | Cross-sectional Spearman IC vs the next bar's return |
| **Split** | 2021–2024 in-sample, 2025+ out-of-sample |
| **Verdict** | alive: IC > 0.02, t > 2, hit ≥ 55% · reversed: IC < −0.02, t < −2 · dead: otherwise |

Every alpha is written against an operator vocabulary in which look-ahead is
*inexpressible* (`ts_*` read backwards only, no negative shift exists), and an
AST purity gate rejects anything reaching around it. A test corrupts the future
and asserts all 41 alphas produce a byte-identical past.

## Results

### Full window

| Verdict | Count | Share |
|---|---|---|
| alive | 21 | 51% |
| reversed | 5 | 12% |
| dead | 15 | 37% |

Strongest by |IC|:

| Alpha | mean IC | t-stat | IR | hit | verdict |
|---|---|---|---|---|---|
| `academic_ivol` | +0.0880 | 15.51 | 0.349 | 64.6% | alive |
| `academic_maxret` | +0.0772 | 15.92 | 0.358 | 65.5% | alive |
| `academic_decay_mom` | −0.0623 | −12.65 | −0.284 | 36.6% | reversed |
| `academic_strev` | +0.0553 | 11.35 | 0.254 | 61.1% | alive |
| `alpha101_044` | +0.0529 | 14.57 | 0.326 | 63.4% | alive |
| `alpha101_033` | +0.0456 | 9.69 | 0.216 | 59.9% | alive |

### A 51% survival rate is not believable

The comparable study on equities (HKUDS/Vibe-Trading's GTJA-191 benchmark, CSI
300, 2018–2025) found **5% alive**. Getting 51% by taking *foreign* alphas to a
*harder* market should be disbelieved before it is celebrated.

Two checks were run against it.

**1. Null control.** The same 41 alphas were benchmarked on synthetic panels —
independent zero-drift GBM with volatility deliberately varied across assets, so
no cross-sectional edge exists by construction. The hypothesis was that ranking
by volatility produces a mechanical IC on skewed returns.

**The hypothesis was wrong.** The null control produced **0 false positives out
of 41**, and the real ICs stand 18–74× clear of their null counterparts:

| Alpha | real IC | null IC | ratio |
|---|---|---|---|
| `academic_ivol` | +0.0880 | +0.0018 | 48× |
| `academic_maxret` | +0.0772 | +0.0010 | 74× |
| `academic_strev` | +0.0553 | −0.0011 | 49× |
| `alpha101_044` | +0.0529 | +0.0030 | 18× |

So the metric is not manufacturing the signal. **That validates the measurement,
not the edge** — an important distinction, and the reason the control is now
built into the script rather than run once and forgotten.

**2. Out-of-sample split.** This is where 51% collapses.

| | Full window | Held verdict IS **and** OOS |
|---|---|---|
| alive or reversed | 26 | **5** |

Only five alphas kept their verdict across the split: `academic_ivol`,
`academic_maxret`, `alpha101_044`, `alpha101_016`, `academic_illiq`. Everything
else was in-sample decoration. Typical decay was brutal — `alpha101_033` fell
from +0.058 to +0.014, `alpha101_001` from +0.037 to −0.012.

### Survival by theme

| Theme | n | alive | reversed | rate |
|---|---|---|---|---|
| volatility | 4 | 4 | 0 | 100% |
| lottery | 1 | 1 | 0 | 100% |
| volume | 14 | 8 | 1 | 57% |
| reversal | 20 | 10 | 2 | 50% |
| momentum | 10 | 4 | 1 | 40% |
| liquidity | 1 | 0 | 1 | 0% |

Volatility and lottery themes lead — which is consistent with the well-documented
crypto pattern of retail overpaying for high-variance names. Momentum is the
weakest surviving theme, matching the equity finding that momentum is the most
thoroughly arbitraged family.

## The finding that actually matters

Four of the five OOS-consistent survivors are volatility, lottery, or
illiquidity alphas.

**VFund has already tested those, with costs, and found them dead.** From
[the case study](CASE_STUDY.md): *"Low-vol, value, illiquidity, size, residual
momentum — strong in-sample, dead out-of-sample."* That verdict came from the
full backtest path: real commissions, short financing, the hard-to-short gate,
walk-forward selection, deflated Sharpe.

So we have two internally consistent results that point opposite ways:

| | low-vol / illiquidity |
|---|---|
| IC scan (this study, no costs) | **alive**, in-sample *and* out-of-sample |
| PnL gauntlet (with costs) | **dead** out-of-sample |

Both are correct. They measure different things. The IC says these alphas *rank
tomorrow's winners better than chance*; the gauntlet says you *cannot collect it*
after paying to trade it. These alphas rebalance daily, and a daily-turnover
book at 10bp round-trip needs vastly more than IC 0.02 to clear its own costs.

This is the single most useful output of the study, and it cost nothing to
obtain because both halves already existed in the repo. An IC scan is a cheap
filter for deciding what deserves an expensive backtest. It is not evidence of
anything tradable, and a project that reported only the IC table here would be
publishing a result its own machinery had already refuted.

## Limitations

In rough order of severity:

1. **No transaction costs anywhere in this study.** See above — this is the
   dominant caveat, not a footnote.
2. **Multiple testing.** 41 alphas tested; at a 5% false-positive rate ~2 would
   look alive by luck alone. No deflated-Sharpe correction is applied to the IC
   statistics, so the true significance is lower than the t-stats suggest.
3. **The OOS window is short** — roughly 18 months, and it has now been looked
   at, so it is no longer virgin for future work.
4. **80 coins is a small cross-section.** Rank correlations over ~40 live names
   per bar are noisy, and the study requires only 5 valid names to compute an IC.
5. **OHLCV-only subset.** The ~70% of published formulas needing sector or
   fundamental data are untested here, not shown to fail.
6. **Residual survivorship.** 41 delisted coins are re-included, but the liquid
   universe is still chosen with hindsight.
7. **Daily bars only.** Several of these alphas were designed for intraday
   horizons and are being evaluated outside their intended frequency.

## Reproduce

```bash
python examples/crypto_alpha_study.py
```

Needs `data/uni_broad.parquet` and `data/delisted.parquet` — build them with
`vfund fetch-universe` (see [EXAMPLES.md](EXAMPLES.md)). The null control is
synthetic and always runs.

Contributions welcome: more of the Kakushadze 101, the GTJA 191, other universes
(majors-only, a CSI-300-equivalent cut), a longer OOS window, and above all a
**cost-charged backtest of the five survivors** — which is the study that would
turn any of this into an actual answer.
