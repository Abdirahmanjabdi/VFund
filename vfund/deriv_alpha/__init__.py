"""Derivative Microstructure Alpha (DMA) — a tested and rejected hypothesis.

.. warning::

   **These five factors do not work.** They are kept because the negative
   result is the finding, and because the data layer beneath them
   (:mod:`~vfund.deriv_alpha.vision`) is genuinely useful.

   Measured on Binance USD-M perpetuals, 2021-12 to 2026-08: four of the five
   factors show no predictive information. The fifth (OI acceleration) has a
   real information coefficient with the *opposite* sign to the one
   hypothesised below, and still has no tradeable decile spread (t = 0.20).
   The composed strategy loses money out of sample, and its deflated Sharpe
   ratio across the configurations searched is 0.0.

   What is worth reusing: :mod:`~vfund.deriv_alpha.vision` (deep open-interest
   history), and :mod:`vfund.research.ic` (the mean-spread gate that caught
   these factors after rank IC had endorsed them).

The thesis, as originally stated and since refuted: crypto perpetual futures
broadcast real-time positioning data
(funding rates, open interest) that traditional markets keep private. This
module systematically ranks the entire cross-section by derivative micro-
structure features and trades the extremes — with regime-adaptive factor
weighting driven by funding dispersion, a crypto-native signal with no
equity-market equivalent.

Five factors:
    1. OI-Price Divergence — conviction outpacing price (or lagging it)
    2. Crowding Pressure — elevated funding x elevated OI = squeeze risk
    3. Liquidation Bounce — post-forced-selling mean-reversion
    4. Funding Momentum — the change in crowdedness, not the level
    5. OI Acceleration — the second derivative of positioning commitment

Regime layer:
    Cross-sectional funding dispersion classifies the market as trending
    (high dispersion -> momentum factors weighted higher) or reverting
    (low dispersion -> contrarian factors weighted higher).
"""

from vfund.deriv_alpha.factors import (
    CrowdingPressure,
    FundingMomentum,
    LiquidationBounce,
    OIAcceleration,
    OIPriceDivergence,
)
from vfund.deriv_alpha.regime import FundingDispersionRegime, Regime
from vfund.deriv_alpha.composer import DerivAlphaComposer

__all__ = [
    "OIPriceDivergence",
    "CrowdingPressure",
    "LiquidationBounce",
    "FundingMomentum",
    "OIAcceleration",
    "FundingDispersionRegime",
    "Regime",
    "DerivAlphaComposer",
]
