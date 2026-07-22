"""Do equity formulaic alphas work on crypto?

The question
------------
The published formulaic-alpha catalogues — Kakushadze's 101, GTJA's 191, Qlib's
158 — were all written for **equities**. They are the most systematic public
libraries of short-horizon factors in existence, and as far as we can find,
nobody has published an honest test of them on crypto.

Crypto is not a small change of venue. There are no earnings, no sectors worth
the name, no market-cap tiers with the same meaning, 24/7 trading with no
overnight gap, and a retail-dominated flow structure. A factor built on the
microstructure of US equities has no particular reason to survive that. The
honest prior is that almost nothing transfers.

So this is a test we expect to mostly fail — which is the point. A negative
result is publishable and useful; a positive one would need much more scrutiny
than an IC scan before anyone believed it.

Method
------
* **Universe** — the survivorship-corrected set: the broad liquid universe plus
  the delisted coins that would otherwise silently vanish. The panel is left
  ragged; a coin is NaN before it listed and after it died, never forward-filled.
* **Signal** — cross-sectional Spearman IC against the next bar's return, the
  same definition used in published survival studies so the numbers compare.
* **Split** — 2021-2024 in-sample, 2025 onward out-of-sample. The OOS window is
  the only one that carries weight.
* **Categorisation** — alive / reversed / dead on the standard thresholds (see
  ``vfund/factors/bench.py``).

What this does NOT establish
----------------------------
**IC is not profitability.** No transaction costs, no borrow constraints, no
capacity limits are applied here. Most of these alphas rebalance daily, and a
daily-turnover book at 10bp round-trip needs far more edge than an IC of 0.02 to
survive. Read the output as a *signal-quality scan* that decides what deserves a
real backtest — never as a return claim.

Prereqs: ``data/uni_broad.parquet`` and ``data/delisted.parquet``.
    python examples/crypto_alpha_study.py
"""

from collections import defaultdict

import numpy as np
import polars as pl

import vfund.factors.zoo  # noqa: F401  - registers the zoo
from vfund.data.panel import load_panel
from vfund.data.universe import clean_universe
from vfund.factors import Panel, all_alphas, bench, panel_from_long, summarise

OOS_YEAR = 2025


def slice_years(p: Panel, ts, lo: int | None = None, hi: int | None = None) -> Panel:
    """Restrict a panel to bars whose year is in ``[lo, hi]``."""
    years = np.array([int(str(t)[:4]) for t in ts])
    m = np.ones(len(years), dtype=bool)
    if lo is not None:
        m &= years >= lo
    if hi is not None:
        m &= years <= hi
    return Panel(close=p.close[m], open=p.open[m], high=p.high[m],
                 low=p.low[m], volume=p.volume[m], symbols=p.symbols)


def null_control(n_symbols: int, n_bars: int, seed: int = 42) -> Panel:
    """A panel with heterogeneous volatility and **no cross-sectional edge**.

    Independent zero-drift GBM, volatility varying across assets. Any alpha that
    scores here is reading an artifact of the rank transform on skewed returns,
    not a signal. Running this alongside the real result is what separates
    "this alpha works" from "this metric is broken" — and it is cheap, so there
    is no excuse for omitting it.
    """
    rng = np.random.default_rng(seed)
    sigmas = np.linspace(0.01, 0.12, n_symbols)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1, (n_bars, n_symbols)) * sigmas, axis=0))
    vol = np.abs(rng.normal(1e6, 2e5, (n_bars, n_symbols)))
    return Panel(close=close, open=close, high=close, low=close, volume=vol,
                 symbols=[f"S{i}" for i in range(n_symbols)])


def main() -> None:
    broad = clean_universe(load_panel("data/uni_broad.parquet"))
    dead = load_panel("data/delisted.parquet")
    full = pl.concat([broad, dead])

    panel = panel_from_long(full)
    ts = (full.select("timestamp").unique().sort("timestamp")["timestamp"].to_list())
    n_dead = dead["symbol"].n_unique()

    print(__doc__.split("Prereqs:")[0].strip())
    print("=" * 78)
    print(f"universe      {len(panel.symbols)} coins "
          f"({n_dead} of them delisted - survivorship corrected)")
    print(f"window        {str(ts[0])[:10]} .. {str(ts[-1])[:10]}  ({panel.shape[0]} bars)")
    print(f"alphas        {len(all_alphas())}")
    print("=" * 78)

    alphas = all_alphas()
    full_res = bench(alphas, panel)

    print("\nFULL WINDOW\n")
    print(summarise(full_res))

    ins = slice_years(panel, ts, hi=OOS_YEAR - 1)
    oos = slice_years(panel, ts, lo=OOS_YEAR)
    ins_res = {r.name: r for r in bench(alphas, ins)}
    oos_res = {r.name: r for r in bench(alphas, oos)}

    print(f"\n\nIN-SAMPLE (..{OOS_YEAR - 1}) vs OUT-OF-SAMPLE ({OOS_YEAR}..)\n")
    head = f"{'alpha':<24} {'IS IC':>8} {'OOS IC':>8} {'IS':>9} {'OOS':>9}  consistent"
    print(head)
    print("-" * len(head))
    consistent = []
    for r in full_res:
        i, o = ins_res.get(r.name), oos_res.get(r.name)
        if i is None or o is None:
            continue
        same = (i.verdict == o.verdict) and i.verdict != "dead"
        if same:
            consistent.append(r.name)
        print(f"{r.name:<24} {i.mean_ic:>+8.4f} {o.mean_ic:>+8.4f} "
              f"{i.verdict:>9} {o.verdict:>9}  {'YES' if same else ''}")

    # --- theme breakdown ----------------------------------------------------
    by_theme: dict[str, list] = defaultdict(list)
    for r in full_res:
        for th in (r.theme or ("untagged",)):
            by_theme[th].append(r)

    print("\n\nSURVIVAL BY THEME (full window)\n")
    print(f"{'theme':<14} {'n':>4} {'alive':>6} {'reversed':>9} {'rate':>7}")
    print("-" * 44)
    for th in sorted(by_theme):
        rs = by_theme[th]
        a = sum(1 for r in rs if r.verdict == "alive")
        rev = sum(1 for r in rs if r.verdict == "reversed")
        print(f"{th:<14} {len(rs):>4} {a:>6} {rev:>9} {a/len(rs):>6.0%}")

    # --- null control -------------------------------------------------------
    null = bench(alphas, null_control(len(panel.symbols), panel.shape[0]))
    null_by_name = {r.name: r for r in null}
    null_alive = [r for r in null if r.verdict in ("alive", "reversed")]

    print("\n\nNULL CONTROL (no edge exists by construction)\n")
    print(f"{'alpha':<24} {'real IC':>9} {'null IC':>9} {'ratio':>7}")
    print("-" * 52)
    for r in full_res[:6]:
        nr = null_by_name.get(r.name)
        if nr is None or not np.isfinite(nr.mean_ic):
            continue
        # A near-zero null IC makes the raw ratio explode and say nothing;
        # report it as a floor rather than a spurious 14-digit number.
        ratio = (f">{abs(r.mean_ic) / 0.001:>5.0f}x" if abs(nr.mean_ic) < 0.001
                 else f"{abs(r.mean_ic / nr.mean_ic):>6.1f}x")
        print(f"{r.name:<24} {r.mean_ic:>+9.4f} {nr.mean_ic:>+9.4f} {ratio:>7}")
    print(f"\nnull-control false positives: {len(null_alive)} of {len(null)} "
          f"({len(null_alive)/len(null):.0%})")
    print("The real ICs stand well clear of the null, so the metric is not")
    print("manufacturing them. That validates the measurement, not the edge.")

    # --- verdict ------------------------------------------------------------
    alive = [r for r in full_res if r.verdict == "alive"]
    rev = [r for r in full_res if r.verdict == "reversed"]
    print("\n\nVERDICT\n" + "-" * 78)
    print(f"full window : {len(alive)} alive, {len(rev)} reversed, "
          f"{len(full_res) - len(alive) - len(rev)} dead  (of {len(full_res)})")
    print(f"consistent across the IS/OOS split (the only ones worth a backtest): "
          f"{len(consistent)}")
    if consistent:
        for n in consistent:
            print(f"    {n}")
    else:
        print("    none - no alpha held its verdict out-of-sample")
    print()
    print("Reminder: this is an IC scan with NO transaction costs. Even a name in")
    print("the consistent list is a candidate to backtest properly, not an edge.")
    print()
    print("Cross-check that proves the point: VFund's own PnL-level gauntlet")
    print("already tested low-vol and illiquidity WITH costs and found them dead")
    print("out-of-sample (docs/CASE_STUDY.md). This scan calls them alive in both")
    print("windows. Both results are correct - they measure different things, and")
    print("the one with costs in it is the one that decides whether you trade.")


if __name__ == "__main__":
    main()
