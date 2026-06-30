import polars as pl
import pytest

from vfund.data.models import validate_bars
from vfund.data.synthetic import generate_gbm_bars
from vfund.data.storage import save_parquet, load_parquet


def test_synthetic_is_deterministic():
    a = generate_gbm_bars(500, seed=7)
    b = generate_gbm_bars(500, seed=7)
    assert a.equals(b)
    assert a.height == 500
    assert list(a.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_synthetic_ohlc_invariants():
    df = generate_gbm_bars(1000, seed=1)
    # high is the max and low is the min of the bar, always.
    assert (df["high"] >= df["open"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["open"]).all()
    assert (df["low"] <= df["close"]).all()
    assert (df["low"] > 0).all()


def test_validate_rejects_missing_columns():
    with pytest.raises(ValueError):
        validate_bars(pl.DataFrame({"timestamp": [1, 2], "open": [1.0, 2.0]}))


def test_validate_rejects_duplicate_timestamps():
    df = generate_gbm_bars(10, seed=1)
    dupe = pl.concat([df, df.head(1)])
    with pytest.raises(ValueError):
        validate_bars(dupe)


def test_parquet_roundtrip(tmp_path):
    df = generate_gbm_bars(100, seed=3)
    path = save_parquet(df, tmp_path / "bars.parquet")
    back = load_parquet(path)
    assert back.equals(df)
