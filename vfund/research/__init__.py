"""Research tooling: the discipline that separates edge from wishful thinking.

Anyone can find a rule that fit the past. The question that matters is whether
it holds on data it has never seen. These utilities make that test the default,
not an afterthought.
"""

from vfund.research.splits import train_test_split, walk_forward_windows
from vfund.research.walkforward import walk_forward, WalkForwardResult

__all__ = [
    "train_test_split",
    "walk_forward_windows",
    "walk_forward",
    "WalkForwardResult",
]
