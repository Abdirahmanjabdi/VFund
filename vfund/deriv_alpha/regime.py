"""Market regime detection from derivative microstructure.

The cross-sectional dispersion of funding rates is a crypto-native regime
signal: it measures how much the market *disagrees* about direction.

High dispersion
    Strong, divergent directional views. Different coins carry very different
    funding rates — some deeply positive (crowded longs), some deeply negative
    (crowded shorts). Conviction is driving prices, and momentum factors tend
    to work.

Low dispersion
    Funding rates are uniform across coins. The market is herding — everyone
    positioned the same way. Mean-reversion factors tend to work because the
    consensus trade is fragile.

This regime classification has no equivalent in equity markets because equity
funding/borrow data is private, delayed, and sparse. In crypto it is a
real-time public signal refreshed every 8 hours.
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class Regime(Enum):
    TRENDING = "trending"
    REVERTING = "reverting"
    NEUTRAL = "neutral"


class FundingDispersionRegime:
    """Classify market regime from cross-sectional funding-rate dispersion.

    Computes the cross-sectional std of funding rates at each bar, then
    compares the current dispersion to its own trailing history.

    Args:
        window: trailing bars over which to rank dispersion.
        high_pct: percentile above which dispersion is "high" (trending).
        low_pct: percentile below which dispersion is "low" (reverting).
    """

    def __init__(
        self, window: int = 30, high_pct: float = 70.0, low_pct: float = 30.0
    ):
        if window < 5:
            raise ValueError("window must be >= 5")
        self.window = window
        self.high_pct = high_pct
        self.low_pct = low_pct

    def detect(self, funding: np.ndarray) -> Regime:
        """Classify the regime from the funding array (already truncated to now).

        Args:
            funding: (n_seen, N) array of prevailing funding rates, as provided
                by ``PanelContext.funding``.
        """
        if funding is None:
            return Regime.NEUTRAL
        n = funding.shape[0]
        if n < self.window:
            return Regime.NEUTRAL

        dispersions = np.empty(self.window)
        for k in range(self.window):
            row = funding[n - self.window + k]
            valid = row[np.isfinite(row)]
            dispersions[k] = float(valid.std()) if len(valid) > 1 else 0.0

        current = dispersions[-1]
        high_thresh = float(np.percentile(dispersions, self.high_pct))
        low_thresh = float(np.percentile(dispersions, self.low_pct))

        if current >= high_thresh:
            return Regime.TRENDING
        if current <= low_thresh:
            return Regime.REVERTING
        return Regime.NEUTRAL

    def weights(self, regime: Regime) -> dict[str, float]:
        """Factor weight multipliers per regime.

        Trending -> momentum + acceleration up; crowding + liquidation down.
        Reverting -> the reverse.
        """
        if regime == Regime.TRENDING:
            return {
                "oi_price_divergence": 1.0,
                "crowding_pressure": 0.5,
                "liquidation_bounce": 0.5,
                "funding_momentum": 1.5,
                "oi_acceleration": 1.5,
            }
        if regime == Regime.REVERTING:
            return {
                "oi_price_divergence": 1.0,
                "crowding_pressure": 1.5,
                "liquidation_bounce": 1.5,
                "funding_momentum": 0.5,
                "oi_acceleration": 0.5,
            }
        return {
            "oi_price_divergence": 1.0,
            "crowding_pressure": 1.0,
            "liquidation_bounce": 1.0,
            "funding_momentum": 1.0,
            "oi_acceleration": 1.0,
        }
