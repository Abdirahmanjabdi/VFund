"""Market data: canonical schema, ingestion, storage, and synthetic generation."""

from vfund.data.models import Bar, BAR_SCHEMA, validate_bars
from vfund.data.storage import save_parquet, load_parquet

__all__ = ["Bar", "BAR_SCHEMA", "validate_bars", "save_parquet", "load_parquet"]
