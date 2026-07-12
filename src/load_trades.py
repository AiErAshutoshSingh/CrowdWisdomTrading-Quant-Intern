"""CSV trade-log ingestion."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError
from .constants import REQUIRED_TRADE_COLUMNS
from .database import Database


def load_trades(path: Path, database: Database, logger: logging.Logger) -> int:
    """Validate a CSV, create deterministic IDs, and insert trades."""
    if not path.exists():
        logger.warning("Trade CSV not found at %s; ingestion skipped.", path)
        return 0
    if path.stat().st_size == 0:
        logger.warning(
            "Trade CSV is empty at %s; ingestion skipped. Add a header and trade rows.",
            path,
        )
        return 0
    try:
        frame = pd.read_csv(path)
        missing = REQUIRED_TRADE_COLUMNS - set(frame.columns)
        if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        if frame["timestamp"].isna().any(): raise ValueError("One or more timestamps are invalid.")
        for column in ("quantity", "price", "pnl"): frame[column] = pd.to_numeric(frame[column], errors="raise")
        if "trade_id" not in frame:
            frame["trade_id"] = frame.apply(lambda row: hashlib.sha256("|".join(map(str, row.values)).encode()).hexdigest()[:32], axis=1)
        return database.upsert_trades(frame[["trade_id", "timestamp", "direction", "quantity", "price", "pnl", "account", "simulation"]])
    except EmptyDataError:
        logger.warning("Trade CSV contains no readable columns at %s; ingestion skipped.", path)
        return 0
    except Exception as exc:
        logger.exception("Trade ingestion failed: %s", exc)
        raise
