"""Time-series splits — always forward, never shuffled.

Financial data must be split in time order: train on the past, test on the
future. Random k-fold splits leak future information into training and are one
of the most common ways a backtest lies to you.
"""

from __future__ import annotations


def train_test_split(n: int, train_frac: float = 0.6) -> tuple[range, range]:
    """Split ``n`` bars into a leading train range and a trailing test range."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    cut = int(n * train_frac)
    if cut < 1 or cut >= n:
        raise ValueError("split leaves an empty side; adjust train_frac or n")
    return range(0, cut), range(cut, n)


def walk_forward_windows(
    n: int,
    train_size: int,
    test_size: int,
    step: int | None = None,
) -> list[tuple[range, range]]:
    """Generate rolling ``(train, test)`` index windows over ``n`` bars.

    Each window trains on ``train_size`` bars and tests on the ``test_size``
    bars immediately after. The window then advances by ``step`` (default:
    ``test_size``, i.e. non-overlapping test segments that tile the timeline).
    """
    if train_size < 1 or test_size < 1:
        raise ValueError("train_size and test_size must be >= 1")
    step = step or test_size
    windows: list[tuple[range, range]] = []
    start = 0
    while start + train_size + test_size <= n:
        train = range(start, start + train_size)
        test = range(start + train_size, start + train_size + test_size)
        windows.append((train, test))
        start += step
    return windows
