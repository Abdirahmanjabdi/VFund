"""Cross-sectional long/short backtester.

Where the single-asset engine trades one instrument to a target weight, this one
holds a whole book. It uses the standard return-based simulation for factor
portfolios:

* weights decided at the close of bar ``t`` (from data through ``t``) earn the
  return from ``t`` to ``t+1`` — the same one-bar lag that keeps v0 honest;
* between rebalances, weights *drift* with prices (a winner grows its share);
* at each rebalance you pay a cost proportional to **turnover** — the sum of the
  weight changes you actually trade.

That turnover cost is where most cross-sectional "edges" quietly die, so it is
front and centre rather than an afterthought.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from vfund.backtest.construct import scores_to_weights, vol_scale_weights
from vfund.backtest.result import BacktestResult
from vfund.data.intervals import bars_per_year as _bars_per_year
from vfund.data.panel import align_funding, pivot_to_wide, validate_panel
from vfund.strategy.cross_sectional import CrossSectionalStrategy, PanelContext


class CrossSectionalBacktester:
    def __init__(
        self,
        panel: pl.DataFrame,
        strategy: CrossSectionalStrategy,
        *,
        rebalance_every: int = 1,
        leverage: float = 1.0,
        top_k: int | None = None,
        neutralize: bool = True,
        cost_bps: float = 10.0,
        interval: str = "1h",
        initial_cash: float = 10_000.0,
        funding: pl.DataFrame | None = None,
        vol_target: float | None = None,
        vol_lookback: int = 30,
        max_leverage: float = 3.0,
        short_cost_bps_annual: float = 0.0,
        allow_ragged: bool = True,
    ):
        panel = validate_panel(panel)
        # Ragged by default: coins enter/exit the cross-section as they were
        # actually listed/delisted (nulls outside their life), instead of forcing
        # every coin to span all history — the survivorship-reducing choice.
        wide = pivot_to_wide(panel, "close", drop_incomplete=not allow_ragged)
        self.timestamps = wide["timestamp"]
        self.symbols = [c for c in wide.columns if c != "timestamp"]
        self.closes = wide.select(self.symbols).to_numpy()  # (T, N), may hold NaN

        # Volumes aligned to the same timeline (for the size factor).
        vwide = pivot_to_wide(panel, "volume", drop_incomplete=False)
        vwide = wide.select("timestamp").join(vwide, on="timestamp", how="left")
        self.volumes = vwide.select(self.symbols).fill_null(0.0).to_numpy()

        # Optional funding overlay: `event` pays into P&L on funding bars,
        # `prevailing` is the forward-filled rate the strategy trades on.
        if funding is not None:
            self.funding_event, self.funding_prevailing = align_funding(
                funding, self.timestamps, self.symbols
            )
        else:
            self.funding_event = self.funding_prevailing = None

        self.strategy = strategy
        self.rebalance_every = max(1, int(rebalance_every))
        self.leverage = leverage
        self.top_k = top_k
        self.neutralize = neutralize
        self.vol_target = vol_target
        self.vol_lookback = max(5, int(vol_lookback))
        self.max_leverage = max_leverage
        self.short_cost_annual = short_cost_bps_annual / 10_000.0
        self.cost_rate = cost_bps / 10_000.0
        self.interval = interval
        self.bars_per_year = _bars_per_year(interval)
        self.initial_cash = float(initial_cash)

    def run(self) -> BacktestResult:
        C = self.closes
        T, N = C.shape
        if T < 3:
            raise ValueError("need at least 3 bars to backtest")

        # Per-bar simple returns; row t is the return from t-1 to t. On a ragged
        # panel, returns touching a null (pre-listing or post-delisting) are set
        # to 0 — a held coin that delists simply stops contributing, and we exit
        # it below. Fully-aligned panels have no nulls, so behaviour is unchanged.
        rets = np.zeros_like(C)
        rets[1:] = C[1:] / C[:-1] - 1.0
        rets = np.where(np.isfinite(rets), rets, 0.0)
        tradable = np.isfinite(C)

        equity = np.empty(T)
        equity[0] = self.initial_cash
        eq = self.initial_cash

        w_active = np.zeros(N)  # weights currently held (start flat)
        gross_exposure = np.zeros(T)
        per_bar_short_cost = self.short_cost_annual / self.bars_per_year

        reb_idx, reb_turn, reb_cost, reb_eq = [], [], [], []

        for t in range(1, T):
            # Earn the return on the weights set at the previous bar: price move,
            # any funding cashflow (a short in positive funding *receives* it,
            # hence the minus sign), minus financing paid to hold shorts.
            price_ret = float(w_active @ rets[t])
            funding_ret = (
                -float(w_active @ self.funding_event[t])
                if self.funding_event is not None
                else 0.0
            )
            short_ret = -float(np.abs(np.minimum(w_active, 0.0)).sum()) * per_bar_short_cost
            port_ret = price_ret + funding_ret + short_ret
            eq *= 1.0 + port_ret
            growth = 1.0 + price_ret  # weights drift with prices, not cashflows

            # Drift the weights forward with realised returns, then force-exit any
            # coin that is no longer tradable this bar (delisted -> to cash).
            if growth != 0.0:
                w_drifted = w_active * (1.0 + rets[t]) / growth
            else:  # pragma: no cover - portfolio wiped out
                w_drifted = w_active.copy()
            w_drifted = np.where(tradable[t], w_drifted, 0.0)

            # Rebalance on schedule.
            if t % self.rebalance_every == 0:
                ctx = PanelContext(
                    t, C, self.symbols,
                    funding=self.funding_prevailing, volumes=self.volumes,
                )
                scores = self.strategy.scores(ctx)
                w_target = scores_to_weights(
                    scores, leverage=self.leverage, top_k=self.top_k,
                    neutralize=self.neutralize,
                )
                if self.vol_target is not None:
                    w_target = self._vol_scale(w_target, rets, t)
                turnover = float(np.abs(w_target - w_drifted).sum())
                cost = turnover * self.cost_rate
                eq *= 1.0 - cost
                w_active = w_target
                if turnover > 0:
                    reb_idx.append(t)
                    reb_turn.append(turnover)
                    reb_cost.append(cost * eq)
                    reb_eq.append(eq)
            else:
                w_active = w_drifted

            equity[t] = eq
            gross_exposure[t] = float(np.abs(w_active).sum())

        equity_curve = pl.DataFrame(
            {
                "timestamp": self.timestamps,
                "equity": equity,
                "gross_exposure": gross_exposure,
            }
        )
        trades = pl.DataFrame(
            {
                "timestamp": self.timestamps.gather(reb_idx)
                if reb_idx
                else pl.Series("timestamp", [], dtype=self.timestamps.dtype),
                "turnover": reb_turn,
                "cost": reb_cost,
                "equity": reb_eq,
            }
        )

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            initial_cash=self.initial_cash,
            bars_per_year=self.bars_per_year,
            meta={
                "strategy": type(self.strategy).__name__,
                "interval": self.interval,
                "n_bars": T,
                "n_assets": N,
                "rebalance_every": self.rebalance_every,
            },
        )

    def _vol_scale(self, weights: np.ndarray, rets: np.ndarray, t: int) -> np.ndarray:
        """Scale weights so the book's predicted volatility hits ``vol_target``.

        Estimates ex-ante portfolio vol from the recent return covariance
        (``sqrt(w' Σ w)``, annualised). High recent vol -> smaller book, and vice
        versa — the standard managed-futures risk control. Capped at
        ``max_leverage`` gross so a calm patch can't over-lever.
        """
        lo = max(1, t - self.vol_lookback + 1)
        return vol_scale_weights(
            weights, rets[lo : t + 1],
            vol_target=self.vol_target, bars_per_year=self.bars_per_year,
            max_leverage=self.max_leverage,
        )
