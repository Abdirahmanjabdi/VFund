"""Local-first storage for market data.

VFund stores everything as Parquet on your own disk — columnar, compressed, and
fast to scan. No database to run, no data leaving your machine.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from vfund.data.models import validate_bars


def save_parquet(df: pl.DataFrame, path: str | Path) -> Path:
    """Validate ``df`` against the canonical bar schema and write it to Parquet."""
    df = validate_bars(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def load_parquet(path: str | Path) -> pl.DataFrame:
    """Load a bar Parquet file and validate it on the way in."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no data file at {path}")
    return validate_bars(pl.read_parquet(path))
