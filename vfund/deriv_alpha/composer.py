"""Adaptive multi-factor combination with regime-aware weighting.

:class:`DerivAlphaComposer` is itself a
:class:`~vfund.strategy.cross_sectional.CrossSectionalStrategy`, so it plugs
directly into the existing backtester. Internally it:

1. Evaluates each of the five derivative factors.
2. Detects the current market regime from funding dispersion.
3. Combines the factor scores with regime-adaptive weights.
4. Returns a single cross-sectional score vector.

A human trader sees the same information via :func:`format_dashboard`: top
longs, top shorts, and the current regime.
"""

from __future__ import annotations

import numpy as np

from vfund.strategy.cross_sectional import CrossSectionalStrategy, PanelContext
from vfund.deriv_alpha.factors import (
    CrowdingPressure,
    FundingMomentum,
    LiquidationBounce,
    OIAcceleration,
    OIPriceDivergence,
    _rank_cs,
)
from vfund.deriv_alpha.regime import FundingDispersionRegime, Regime


class DerivAlphaComposer(CrossSectionalStrategy):
    """The DMA strategy: five derivative factors, regime-adaptive weights.

    Args:
        oi: (T, N) open-interest array aligned to the price timeline.
        oi_div_lookback: lookback for the OI-price divergence factor.
        crowding_lookback: lookback for the crowding-pressure factor.
        liq_lookback: lookback for the liquidation-bounce factor.
        funding_mom_lookback: lookback for the funding-momentum factor.
        oi_accel_lookback: lookback for the OI-acceleration factor.
        regime_window: trailing window for the regime detector.
    """

    def __init__(
        self,
        oi: np.ndarray,
        *,
        oi_div_lookback: int = 10,
        crowding_lookback: int = 14,
        liq_lookback: int = 5,
        funding_mom_lookback: int = 7,
        oi_accel_lookback: int = 7,
        regime_window: int = 30,
    ):
        self._factors = {
            "oi_price_divergence": OIPriceDivergence(oi, lookback=oi_div_lookback),
            "crowding_pressure": CrowdingPressure(oi, lookback=crowding_lookback),
            "liquidation_bounce": LiquidationBounce(oi, lookback=liq_lookback),
            "funding_momentum": FundingMomentum(lookback=funding_mom_lookback),
            "oi_acceleration": OIAcceleration(oi, lookback=oi_accel_lookback),
        }
        self._regime = FundingDispersionRegime(window=regime_window)

    def scores(self, ctx: PanelContext) -> np.ndarray:
        N = len(ctx.symbols)

        factor_scores: dict[str, np.ndarray] = {}
        for name, factor in self._factors.items():
            try:
                s = factor.scores(ctx)
            except ValueError:
                s = np.full(N, np.nan)
            factor_scores[name] = s

        funding = ctx.funding
        regime = self._regime.detect(funding) if funding is not None else Regime.NEUTRAL
        w_map = self._regime.weights(regime)

        combined = np.zeros(N)
        total_w = np.zeros(N)
        for name, raw in factor_scores.items():
            ranked = _rank_cs(raw) - 0.5  # center around 0
            valid = np.isfinite(ranked)
            w = w_map.get(name, 1.0)
            combined[valid] += w * ranked[valid]
            total_w[valid] += w

        with np.errstate(divide="ignore", invalid="ignore"):
            combined = np.where(total_w > 0, combined / total_w, np.nan)

        return combined

    @property
    def factor_names(self) -> list[str]:
        return list(self._factors.keys())


def format_dashboard(
    scores: np.ndarray,
    symbols: list[str],
    regime: Regime,
    factor_scores: dict[str, np.ndarray] | None = None,
    *,
    top_k: int = 5,
) -> str:
    """Human-readable dashboard of the DMA signal.

    Args:
        scores: combined score per symbol from the composer.
        symbols: symbol names matching the score array.
        regime: current regime.
        factor_scores: optional per-factor scores for the "drivers" column.
        top_k: number of longs/shorts to show.
    """
    valid = [(sym, float(s)) for sym, s in zip(symbols, scores) if np.isfinite(s)]
    valid.sort(key=lambda x: -x[1])

    longs = valid[:top_k]
    shorts = valid[-top_k:][::-1]

    def _drivers(sym: str) -> str:
        if factor_scores is None:
            return ""
        idx = symbols.index(sym)
        ranked = sorted(
            (
                (n, float(factor_scores[n][idx]))
                for n in factor_scores
                if np.isfinite(factor_scores[n][idx])
            ),
            key=lambda x: -abs(x[1]),
        )[:2]
        if not ranked:
            return ""
        return "  (" + ", ".join(f"{n[:10]}={v:+.2f}" for n, v in ranked) + ")"

    bar = "=" * 56
    lines = [
        bar,
        f"  DMA Dashboard    Regime: {regime.value.upper():>10}",
        bar,
        "",
        f"  TOP {top_k} LONGS",
        "  " + "-" * 40,
    ]
    for sym, sc in longs:
        lines.append(f"    {sym:<12} {sc:+.3f}{_drivers(sym)}")
    lines += ["", f"  TOP {top_k} SHORTS", "  " + "-" * 40]
    for sym, sc in shorts:
        lines.append(f"    {sym:<12} {sc:+.3f}{_drivers(sym)}")
    lines += ["", bar]
    return "\n".join(lines)
