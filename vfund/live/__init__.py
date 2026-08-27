"""Live/paper layer: turn the researched edge into an actionable book.

`combined_book` computes today's target positions from the latest data; the
`PaperTracker` marks a hypothetical account forward as new data arrives — the
only honest out-of-sample test short of real money.

The execution sub-layer (:mod:`.exchange` + :mod:`.execute`) turns the same
target book into real (or testnet) exchange orders with full safety checks.
"""

from vfund.live.signal import combined_book, format_book
from vfund.live.paper import PaperTracker
from vfund.live.exchange import BinanceClient, BinanceError, SymbolSpec
from vfund.live.execute import run as execute_run

__all__ = [
    "combined_book", "format_book", "PaperTracker",
    "BinanceClient", "BinanceError", "SymbolSpec", "execute_run",
]
