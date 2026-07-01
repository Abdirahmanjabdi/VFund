"""Why a maker backtest that ignores adverse selection lies.

A naive maker model assumes you earn the half-spread on every fill. This sim
adds informed traders who pick you off. We sweep the fraction of informed flow
(rows) against the quoted half-spread (cols) and print net P&L — the naive model
would predict *all cells profitable*.

    python examples/microstructure_adverse_selection.py
"""

from vfund.microstructure import MarketMakingSim

INFORMED = [0.0, 0.2, 0.4, 0.6, 0.8]
SPREADS = [1, 2, 4, 8]
SIGMA = 3.0  # fundamental vol per step (ticks) — comparable to the spreads

header = "informed \\ half-spread"
print("Market-maker NET P&L (naive model says every cell is profitable)\n")
print(f"{header:>22} |" + "".join(f"{s:>9}t" for s in SPREADS))
print("-" * 66)
for inf in INFORMED:
    cells = []
    for h in SPREADS:
        res = MarketMakingSim(n_steps=80_000, half_spread=h, informed_frac=inf,
                              sigma=SIGMA, seed=7).run()
        cells.append(res.total_pnl)
    print(f"{inf:>21.0%} |" + "".join(f"{c:>10.0f}" for c in cells))

print("\nBreakdown at half_spread=4:\n")
for inf in INFORMED:
    print("  " + MarketMakingSim(n_steps=80_000, half_spread=4, informed_frac=inf,
                                 sigma=SIGMA, seed=7).run().summary())

print("\nRead: profit needs the captured spread to EXCEED adverse selection.")
print("With enough informed flow, no spread is wide enough - the naive maker")
print("backtest (which assumes pure spread capture) massively overstates.")
