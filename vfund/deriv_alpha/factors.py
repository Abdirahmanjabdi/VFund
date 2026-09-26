"""Derivative microstructure factors — five novel cross-sectional alphas.

Each factor is a :class:`~vfund.strategy.cross_sectional.CrossSectionalStrategy`
that carries its own derivative data (open interest) and uses
:class:`~vfund.strategy.cross_sectional.PanelContext` for prices and funding.
This keeps the existing backtester and PanelContext unchanged — the factors
are fully self-contained.

The edge thesis
---------------
Crypto perpetual futures broadcast positioning data (funding rates, open
interest) that traditional markets keep hidden. Most participants trade
individual coin signals; almost nobody systematically RANKS the cross-section
by derivative microstructure and trades the extremes — the approach that has
driven factor investing in equities for decades, applied to crypto-native
data nobody else is using this way.

Factor catalogue
----------------
1. **OI-Price Divergence** — conviction outpacing price, or lagging it.
2. **Crowding Pressure** — elevated funding x elevated OI = squeeze risk.
3. **Liquidation Bounce** — post-forced-selling mean-reversion.
4. **Funding Momentum** — the change in crowdedness, not the level.
5. **OI Acceleration** — the second derivative of positioning commitment.
"""

from __future__ import annotations

import numpy as np

from vfund.strategy.cross_sectional import CrossSectionalStrategy, PanelContext


def _rank_cs(x: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank of a 1-D array. NaN preserved, (0, 1]."""
    valid = np.isfinite(x)
    if not valid.any():
        return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    v = x[valid]
    n = len(v)
    less = (v[:, None] > v[None, :]).sum(axis=1)
    equal = (v[:, None] == v[None, :]).sum(axis=1)
    # Midrank: half of each tie group, self included. Using (equal + 1) / 2
    # counts the self-comparison twice and shifts every rank up by 0.5/n —
    # harmless where two ranks are differenced, but not where the composer
    # centres a single rank by subtracting 0.5.
    out[valid] = (less + equal / 2.0) / n
    return out


# ---------------------------------------------------------------------------
# Factor 1: OI-Price Divergence
# ---------------------------------------------------------------------------


class OIPriceDivergence(CrossSectionalStrategy):
    """Long where conviction (OI) outpaces price; short the reverse.

    Open interest measures capital committed to positions. When OI rises faster
    than price, informed money is building positions that haven't moved the
    market yet. When price runs ahead of OI, the move is driven by existing-
    position mark-to-market, not fresh conviction.

    Score = rank(OI change) - rank(price change).
    """

    def __init__(self, oi: np.ndarray, lookback: int = 10):
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        self._oi = oi
        self.lookback = lookback

    def scores(self, ctx: PanelContext) -> np.ndarray:
        i = ctx.i
        N = len(ctx.symbols)
        if i < self.lookback:
            return np.full(N, np.nan)

        oi_now = self._oi[i]
        oi_past = self._oi[i - self.lookback]
        with np.errstate(divide="ignore", invalid="ignore"):
            oi_chg = np.where(oi_past > 0, oi_now / oi_past - 1.0, np.nan)
            px_chg = ctx.closes[-1] / ctx.closes[-1 - self.lookback] - 1.0

        return _rank_cs(oi_chg) - _rank_cs(px_chg)


# ---------------------------------------------------------------------------
# Factor 2: Crowding Pressure
# ---------------------------------------------------------------------------


class CrowdingPressure(CrossSectionalStrategy):
    """Short the most crowded coins; long the least crowded.

    Crowding = elevated funding x elevated OI growth. When both are high, the
    trade is maximally consensus: everyone is piled in on the same side, paying
    high carry, with OI at record levels. These crowded positions are vulnerable
    to liquidation cascades — the exit is narrow and the triggers hit the whole
    herd at once.

    Score = -(rank(avg funding) * rank(OI growth)).
    Needs funding data in the backtest panel.
    """

    def __init__(self, oi: np.ndarray, lookback: int = 14):
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        self._oi = oi
        self.lookback = lookback

    def scores(self, ctx: PanelContext) -> np.ndarray:
        i = ctx.i
        N = len(ctx.symbols)
        funding = ctx.funding
        if funding is None:
            raise ValueError(
                "CrowdingPressure needs funding data — pass funding=... "
                "to CrossSectionalBacktester"
            )
        if i < self.lookback:
            return np.full(N, np.nan)

        avg_funding = funding[-self.lookback :].mean(axis=0)
        oi_now = self._oi[i]
        oi_past = self._oi[i - self.lookback]
        with np.errstate(divide="ignore", invalid="ignore"):
            oi_growth = np.where(oi_past > 0, oi_now / oi_past - 1.0, np.nan)

        return -(_rank_cs(avg_funding) * _rank_cs(oi_growth))


# ---------------------------------------------------------------------------
# Factor 3: Liquidation Bounce
# ---------------------------------------------------------------------------


class LiquidationBounce(CrossSectionalStrategy):
    """Long coins after forced-liquidation events.

    When OI drops sharply alongside a sharp price drop, it signals forced
    selling — not fundamental revaluation. Forced sellers create prices below
    fair value because they sell at any price to meet margin calls. The bounce
    is strongest for the most-liquidated coins.

    Score = max(0, -OI change) * max(0, -price change). Positive only after
    simultaneous OI and price drops. A coin that fell on news (price down, OI
    flat) scores zero — that is fundamental, not forced.
    """

    def __init__(self, oi: np.ndarray, lookback: int = 5):
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        self._oi = oi
        self.lookback = lookback

    def scores(self, ctx: PanelContext) -> np.ndarray:
        i = ctx.i
        N = len(ctx.symbols)
        if i < self.lookback:
            return np.full(N, np.nan)

        oi_now = self._oi[i]
        oi_past = self._oi[i - self.lookback]
        with np.errstate(divide="ignore", invalid="ignore"):
            oi_chg = np.where(oi_past > 0, oi_now / oi_past - 1.0, np.nan)
            px_chg = ctx.closes[-1] / ctx.closes[-1 - self.lookback] - 1.0

        oi_drop = np.maximum(0.0, np.where(np.isfinite(oi_chg), -oi_chg, 0.0))
        px_drop = np.maximum(0.0, np.where(np.isfinite(px_chg), -px_chg, 0.0))
        return oi_drop * px_drop


# ---------------------------------------------------------------------------
# Factor 4: Funding Momentum
# ---------------------------------------------------------------------------


class FundingMomentum(CrossSectionalStrategy):
    """Short coins where the funding rate is accelerating upward.

    Not the funding *level* (that is FundingCarry, already in VFund) but the
    *change*. Accelerating funding means the crowded trade is getting MORE
    crowded and the carrying cost for longs is rising. When the cost of holding
    rises, the marginal holder exits — and in levered markets, exits cascade.

    Score = -(recent avg funding - past avg funding).
    """

    def __init__(self, lookback: int = 7):
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        self.lookback = lookback

    def scores(self, ctx: PanelContext) -> np.ndarray:
        N = len(ctx.symbols)
        funding = ctx.funding
        if funding is None:
            raise ValueError(
                "FundingMomentum needs funding data — pass funding=... "
                "to CrossSectionalBacktester"
            )
        n = ctx.n_seen
        lb = self.lookback
        if n < lb + 1:
            return np.full(N, np.nan)

        recent = funding[-lb:].mean(axis=0)
        past_start = max(0, n - 2 * lb)
        past_end = n - lb
        past = funding[past_start:past_end].mean(axis=0)
        return -(recent - past)


# ---------------------------------------------------------------------------
# Factor 5: OI Acceleration
# ---------------------------------------------------------------------------


class OIAcceleration(CrossSectionalStrategy):
    """Trade the second derivative of open interest.

    First derivative (OI momentum) tells you if positions are growing. Second
    derivative tells you if that growth is *accelerating* or *decelerating*.

    .. note::

       The original hypothesis here — that accelerating growth means fresh
       conviction and so a tailwind — is **wrong, and measurably so.** Tested
       on Binance perpetuals 2021-12 to 2026-08, the sign is the opposite:
       accelerating position build-up marks late-stage crowding and
       *underperforms* (IC -0.029 at a 1-day horizon, Newey-West t = -5.06).

       See :class:`vfund.deriv_alpha.crowding.CrowdingAcceleration` for the
       corrected sign, and note that even with the sign right the factor has
       no tradeable decile spread (t = 0.20) and lost money out of sample.
       This class is retained as the raw, unsigned construction.

    Score = (recent OI growth) - (past OI growth).
    """

    def __init__(self, oi: np.ndarray, lookback: int = 7):
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        self._oi = oi
        self.lookback = lookback

    def scores(self, ctx: PanelContext) -> np.ndarray:
        i = ctx.i
        N = len(ctx.symbols)
        lb = self.lookback
        if i < 2 * lb:
            return np.full(N, np.nan)

        oi_now = self._oi[i]
        oi_mid = self._oi[i - lb]
        oi_far = self._oi[i - 2 * lb]
        with np.errstate(divide="ignore", invalid="ignore"):
            growth_recent = np.where(oi_mid > 0, oi_now / oi_mid - 1.0, np.nan)
            growth_past = np.where(oi_far > 0, oi_mid / oi_far - 1.0, np.nan)

        return growth_recent - growth_past
