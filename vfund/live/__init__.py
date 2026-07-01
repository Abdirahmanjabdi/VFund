"""Live/paper layer: turn the researched edge into an actionable book.

`combined_book` computes today's target positions from the latest data; the
`PaperTracker` marks a hypothetical account forward as new data arrives — the
only honest out-of-sample test short of real money.
"""

from vfund.live.signal import combined_book, format_book
from vfund.live.paper import PaperTracker

__all__ = ["combined_book", "format_book", "PaperTracker"]
