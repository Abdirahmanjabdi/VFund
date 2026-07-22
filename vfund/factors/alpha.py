"""Formulaic alphas: a registry, a panel container, and a strategy adapter.

An alpha here is a pure function of a :class:`Panel` returning a ``(T, N)`` score
matrix — one score per symbol per bar, higher = more attractive. That is the
shape the published formulaic-alpha literature works in, and it is deliberately
*not* the shape VFund's engine consumes; :class:`FormulaicStrategy` bridges the
two so a paper formula can be backtested by the same honest machinery as a
hand-written strategy, with no separate code path to keep in sync.

Registering an alpha runs the :mod:`vfund.factors.purity` gate immediately, so a
formula that could look ahead fails at import time rather than silently
producing a flattering backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from vfund.factors import operators as ops
from vfund.factors.purity import assert_pure
from vfund.strategy.cross_sectional import CrossSectionalStrategy, PanelContext


@dataclass(frozen=True)
class Panel:
    """The OHLCV inputs an alpha may read, as aligned ``(T, N)`` arrays.

    Attributes:
        close, open, high, low, volume: price/size panels, time down axis 0.
        symbols: column labels, positionally aligned with the panels.
    """

    close: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    volume: np.ndarray
    symbols: list[str] = field(default_factory=list)

    @property
    def returns(self) -> np.ndarray:
        """One-bar simple returns."""
        return ops.returns(self.close, 1)

    @property
    def vwap(self) -> np.ndarray:
        """Typical-price VWAP proxy — see :func:`vfund.factors.operators.vwap`."""
        return ops.vwap(self.high, self.low, self.close)

    def adv(self, n: int) -> np.ndarray:
        """Average dollar volume over ``n`` bars."""
        return ops.adv(self.close, self.volume, n)

    @property
    def shape(self) -> tuple[int, int]:
        return self.close.shape


@dataclass(frozen=True)
class Alpha:
    """A registered formulaic alpha."""

    name: str
    fn: Callable[[Panel], np.ndarray]
    formula: str
    source: str
    theme: tuple[str, ...] = ()
    warmup: int = 0

    def compute(self, panel: Panel) -> np.ndarray:
        """Evaluate the alpha, validating the output shape and finiteness."""
        out = np.asarray(self.fn(panel), dtype=np.float64)
        if out.shape != panel.shape:
            raise ValueError(
                f"alpha {self.name} returned shape {out.shape}, expected {panel.shape}"
            )
        # Infinities are never a valid score and would poison downstream ranking.
        return np.where(np.isfinite(out), out, np.nan)


REGISTRY: dict[str, Alpha] = {}


def alpha(name: str, *, formula: str, source: str, theme: tuple[str, ...] = (),
          warmup: int = 0) -> Callable:
    """Decorator registering a pure formulaic alpha.

    Args:
        name: unique id, e.g. ``"alpha101_012"``.
        formula: the formula as stated in the source paper, for auditability.
        source: citation of where the formula comes from.
        theme: coarse tags (``"reversal"``, ``"momentum"``, …) for grouping.
        warmup: bars required before the output is meaningful.

    Raises:
        ValueError: on a duplicate name.
        PurityError: if the body violates the purity rules — checked at import.
    """
    def deco(fn: Callable[[Panel], np.ndarray]) -> Callable[[Panel], np.ndarray]:
        if name in REGISTRY:
            raise ValueError(f"alpha {name} is already registered")
        assert_pure(fn, name=name)
        REGISTRY[name] = Alpha(name=name, fn=fn, formula=formula, source=source,
                               theme=theme, warmup=warmup)
        return fn
    return deco


def get(name: str) -> Alpha:
    """Look up a registered alpha by name."""
    if name not in REGISTRY:
        raise KeyError(f"unknown alpha {name!r}; {len(REGISTRY)} registered")
    return REGISTRY[name]


def all_alphas() -> list[Alpha]:
    """Every registered alpha, name-sorted."""
    return [REGISTRY[k] for k in sorted(REGISTRY)]


class FormulaicStrategy(CrossSectionalStrategy):
    """Adapt a registered :class:`Alpha` to VFund's cross-sectional interface.

    The engine asks for scores one bar at a time; the alpha computes over the
    whole history. This recomputes on the visible prefix and returns its final
    row — correct by construction (the alpha literally cannot see beyond the
    prefix) at the cost of being O(T) per call. For scoring many alphas, prefer
    :func:`vfund.factors.bench.information_coefficient`, which evaluates the
    full panel once.
    """

    def __init__(self, alpha_name: str, *, sign: float = 1.0):
        self.alpha = get(alpha_name)
        self.sign = float(sign)

    def scores(self, ctx: PanelContext) -> np.ndarray:
        # ctx.closes / ctx.volumes are already truncated at the current bar, so
        # the alpha is handed a panel that physically cannot contain the future.
        closes = np.asarray(ctx.closes, dtype=np.float64)
        if ctx.n_seen <= self.alpha.warmup:
            return np.full(closes.shape[1], np.nan)

        volumes = ctx.volumes
        volumes = (np.full_like(closes, np.nan) if volumes is None
                   else np.asarray(volumes, dtype=np.float64))
        # PanelContext carries close and volume only; OHLC-hungry alphas degrade
        # to close for open/high/low rather than silently fabricating a range.
        panel = Panel(close=closes, open=closes, high=closes, low=closes,
                      volume=volumes, symbols=list(ctx.symbols))
        return self.sign * self.alpha.compute(panel)[-1]
