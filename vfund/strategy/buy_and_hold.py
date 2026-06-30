"""Buy-and-hold — the benchmark every active strategy must justify itself against.

If your clever signal can't beat simply buying the asset and sitting still, the
signal isn't edge. This strategy goes fully long on the first opportunity and
never trades again.
"""

from __future__ import annotations

from vfund.strategy.base import BarContext, Strategy


class BuyAndHold(Strategy):
    def on_bar(self, ctx: BarContext) -> float | None:
        return 1.0
