"""Walk-forward optimisation — choose parameters on the past, judge on the future.

For each rolling window we pick the best parameter set on the *train* segment,
then record how it does on the *test* segment it never saw. Stitching those
out-of-sample segments together gives an equity curve you can actually believe.

The number that matters is the gap between in-sample and out-of-sample
performance. A strategy that shines in-sample and dies out-of-sample isn't edge —
it's a curve fit. Watching that gap is the whole point.

Note: each test segment starts flat for the first ``lookback`` bars (no warmup
is carried across the boundary), so OOS results here are mildly conservative —
the honest direction to err.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl

from vfund.backtest.cross_sectional import CrossSectionalBacktester
from vfund.data.intervals import bars_per_year as _bars_per_year
from vfund.data.panel import pivot_to_wide, validate_panel
from vfund.research.splits import walk_forward_windows


@dataclass
class WalkForwardResult:
    windows: pl.DataFrame        # per-window: params, train metric, test metric
    oos_equity: pl.DataFrame     # stitched out-of-sample equity curve
    metric: str
    bars_per_year: float

    @property
    def oos_returns(self) -> np.ndarray:
        eq = self.oos_equity["equity"].to_numpy()
        return eq[1:] / eq[:-1] - 1.0 if eq.size > 1 else np.zeros(0)

    def oos_sharpe(self) -> float:
        from vfund.analytics.performance import sharpe_ratio

        return sharpe_ratio(self.oos_returns, self.bars_per_year)

    def oos_total_return(self) -> float:
        eq = self.oos_equity["equity"].to_numpy()
        return float(eq[-1] / eq[0] - 1.0) if eq.size else 0.0

    def summary(self) -> str:
        w = self.windows
        train_col = f"train_{self.metric}"
        test_col = f"test_{self.metric}"
        mean_train = float(w[train_col].mean())
        mean_test = float(w[test_col].mean())
        lines = [
            "=" * 58,
            "  Walk-forward validation",
            "=" * 58,
            f"  Windows                        {w.height:>10}",
            f"  Mean in-sample  {self.metric:<8}       {mean_train:>10.2f}",
            f"  Mean out-of-sample {self.metric:<8}    {mean_test:>10.2f}",
            f"  Overfitting gap (IS - OOS)     {mean_train - mean_test:>10.2f}",
            "-" * 58,
            f"  Stitched OOS Sharpe            {self.oos_sharpe():>10.2f}",
            f"  Stitched OOS total return      {self.oos_total_return() * 100:>9.2f}%",
            "=" * 58,
        ]
        return "\n".join(lines)


def walk_forward(
    panel: pl.DataFrame,
    build_strategy: Callable[..., object],
    param_grid: list[dict],
    *,
    train_size: int,
    test_size: int,
    step: int | None = None,
    metric: str = "sharpe",
    interval: str = "1h",
    backtest_kwargs: dict | None = None,
    funding: pl.DataFrame | None = None,
) -> WalkForwardResult:
    """Run a walk-forward study of ``build_strategy`` over ``param_grid``.

    Pass ``funding`` to validate a funding-carry hypothesis; it is sliced to each
    window alongside the price panel.
    """
    if not param_grid:
        raise ValueError("param_grid is empty")
    panel = validate_panel(panel)
    bt_kwargs = {**(backtest_kwargs or {}), "interval": interval}

    ts_master = pivot_to_wide(panel, "close")["timestamp"]  # aligned timeline
    n = ts_master.len()
    windows = walk_forward_windows(n, train_size, test_size, step)
    if not windows:
        raise ValueError(
            f"timeline of {n} aligned bars too short for "
            f"train_size={train_size} + test_size={test_size}"
        )

    def run(sub_panel, sub_funding, params) -> tuple[float, object]:
        strat = build_strategy(**params)
        res = CrossSectionalBacktester(
            sub_panel, strat, funding=sub_funding, **bt_kwargs
        ).run()
        return res.metrics()[metric], res

    rows: list[dict] = []
    oos_ret_chunks: list[np.ndarray] = []
    oos_ts_chunks: list[np.ndarray] = []

    for wi, (train, test) in enumerate(windows):
        train_ts = ts_master.slice(train.start, len(train))
        test_ts = ts_master.slice(test.start, len(test))
        train_panel = panel.filter(pl.col("timestamp").is_in(train_ts.implode()))
        test_panel = panel.filter(pl.col("timestamp").is_in(test_ts.implode()))
        if funding is not None:
            train_fund = funding.filter(pl.col("timestamp").is_in(train_ts.implode()))
            test_fund = funding.filter(pl.col("timestamp").is_in(test_ts.implode()))
        else:
            train_fund = test_fund = None

        # Pick the parameter set that does best in-sample.
        best_score, best_params = -np.inf, param_grid[0]
        for params in param_grid:
            score, _ = run(train_panel, train_fund, params)
            if score > best_score:
                best_score, best_params = score, params

        # Judge it out-of-sample.
        test_score, test_res = run(test_panel, test_fund, best_params)

        row = {"window": wi}
        row.update({f"param_{k}": v for k, v in best_params.items()})
        row[f"train_{metric}"] = best_score
        row[f"test_{metric}"] = test_score
        row["test_return"] = test_res.metrics()["total_return"]
        rows.append(row)

        eq = test_res.equity_curve["equity"].to_numpy()
        if eq.size > 1:
            oos_ret_chunks.append(eq[1:] / eq[:-1] - 1.0)
            oos_ts_chunks.append(
                test_res.equity_curve["timestamp"].to_numpy()[1:]
            )

    rets = np.concatenate(oos_ret_chunks) if oos_ret_chunks else np.zeros(0)
    ts = np.concatenate(oos_ts_chunks) if oos_ts_chunks else np.array([])
    oos_equity_vals = 10_000.0 * np.cumprod(1.0 + rets)
    oos_equity = pl.DataFrame(
        {
            "timestamp": pl.Series(ts).cast(pl.Datetime("ms", time_zone="UTC")),
            "equity": oos_equity_vals,
        }
    )

    return WalkForwardResult(
        windows=pl.DataFrame(rows),
        oos_equity=oos_equity,
        metric=metric,
        bars_per_year=_bars_per_year(interval),
    )
