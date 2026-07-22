"""Formulaic-alpha layer: operators, a purity gate, a zoo, and IC benching.

Express a factor as a formula instead of a class::

    from vfund.factors import Panel, get, operators as ops

    @alpha("my_alpha", formula="-1 * rank(delta(close, 5))", source="me")
    def my_alpha(p: Panel):
        return -1.0 * ops.rank(ops.delta(p.close, 5))

Registration runs the AST purity gate immediately, so a formula that could look
ahead fails at import rather than producing a flattering backtest.
"""

from vfund.factors.alpha import (
    REGISTRY,
    Alpha,
    FormulaicStrategy,
    Panel,
    all_alphas,
    alpha,
    get,
    panel_from_long,
)
from vfund.factors.bench import ICResult, bench, information_coefficient, summarise
from vfund.factors.purity import PurityError, assert_pure, check_source

__all__ = [
    "Alpha", "Panel", "FormulaicStrategy", "REGISTRY",
    "alpha", "get", "all_alphas", "panel_from_long",
    "ICResult", "bench", "information_coefficient", "summarise",
    "PurityError", "assert_pure", "check_source",
]
