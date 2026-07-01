"""Funding-carry research: is the perp funding spread a tradable edge?

Run after fetching data:

    vfund fetch-universe --interval 1h --start 2023-06-01 --end 2024-06-01 --out data/uni.parquet
    vfund fetch-funding             --start 2023-06-01 --end 2024-06-01 --out data/funding.parquet
    python examples/funding_carry_study.py

The strategy shorts the coins paying the highest funding and longs those paying
the least — a dollar-neutral book that harvests the funding differential. We
sweep rebalance frequency, concentration (top_k), and cost, reporting the
OUT-OF-SAMPLE Sharpe from a walk-forward study.

Takeaway from the 2023-06..2024-06 / 30-coin run: the edge lives in the
*extremes* (top_k ~5, not the whole cross-section) and rebalanced *daily*
(funding is persistent, so trading it every 8h just burns costs). In that corner
it survived realistic costs (OOS Sharpe ~1.0 at 5 bps). Treat as a *candidate*,
not a validated edge — see the caveats printed at the end.
"""

from vfund.data.panel import load_funding, load_panel
from vfund.research import walk_forward
from vfund.strategy import FundingCarry

panel = load_panel("data/uni.parquet")
funding = load_funding("data/funding.parquet")

SMOOTH_GRID = [{"lookback": s} for s in (1, 3, 8)]  # funding-signal smoothing (bars)


def oos_sharpe(rebalance, top_k, cost):
    wf = walk_forward(
        panel,
        lambda lookback: FundingCarry(smooth=lookback),
        SMOOTH_GRID,
        train_size=4000,
        test_size=1500,
        interval="1h",
        backtest_kwargs=dict(rebalance_every=rebalance, top_k=top_k, cost_bps=cost),
        funding=funding,
    )
    return wf.oos_sharpe()


print(f"{'reb/h':>5} {'top_k':>6} | {'0bp':>7} {'5bp':>7} {'10bp':>7}")
print("-" * 40)
for rebalance in (8, 24):
    for top_k in (3, 5, None):
        row = [oos_sharpe(rebalance, top_k, c) for c in (0, 5, 10)]
        tk = "all" if top_k is None else str(top_k)
        print(f"{rebalance:>5} {tk:>6} |" + "".join(f"{x:>8.2f}" for x in row))

print(
    "\nCAVEATS (why this is a candidate, not a confirmed edge):\n"
    "  - one period, one 30-coin universe, only ~3 walk-forward windows\n"
    "  - funding applied to spot returns (a hybrid, not a true delta-neutral carry)\n"
    "  - the winning config was chosen from this very sweep (multiple testing)\n"
    "  Next: more periods/universes, a cleaner carry model, deflated Sharpe."
)
