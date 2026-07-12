"""Apify macro-calendar ingestion."""
from __future__ import annotations

import logging
from typing import Any
import pandas as pd
from apify_client import ApifyClient
from apify_client.errors import NotFoundError

from .database import Database


class MacroScraper:
    """Fetch a calendar actor dataset and persist standardized events."""
    def __init__(self, token: str | None, actor_id: str | None, database: Database, logger: logging.Logger) -> None:
        self.token, self.actor_id, self.database, self.logger = token, actor_id, database, logger

    def fetch_and_store(self) -> int:
        """Fetch configured Apify actor output; return number of inserted events."""
        if not self.token or not self.actor_id:
            self.logger.warning("Apify credentials absent; macro ingestion skipped.")
            return 0
        try:
            run = ApifyClient(self.token).actor(self.actor_id).call(run_input={})
            items = list(ApifyClient(self.token).dataset(run["defaultDatasetId"]).iterate_items())
            frame = self._standardize(items)
            return self.database.upsert_events(frame) if not frame.empty else 0
        except NotFoundError:
            self.logger.error(
                "Apify actor '%s' was not found or is not accessible to this token. "
                "Set APIFY_ACTOR_ID to a public actor ID or an actor owned by this account, "
                "or leave it blank to skip macro ingestion.",
                self.actor_id,
            )
            return 0
        except Exception as exc:
            self.logger.error("Macro scraping failed: %s", exc)
            return 0

    @staticmethod
    def _standardize(items: list[dict[str, Any]]) -> pd.DataFrame:
        """Map common calendar field variants into the database schema."""
        if not items: return pd.DataFrame()
        raw = pd.DataFrame(items)
        def field(*names: str) -> pd.Series:
            for name in names:
                if name in raw: return raw[name]
            return pd.Series([None] * len(raw))
        frame = pd.DataFrame({"event_name": field("event_name", "event", "title", "name"), "country": field("country", "currency"), "actual": field("actual"), "forecast": field("forecast", "consensus"), "previous": field("previous"), "event_time": pd.to_datetime(field("event_time", "date", "datetime", "time"), utc=True, errors="coerce"), "timezone": field("timezone")})
        for column in ("actual", "forecast", "previous"):
            frame[column] = pd.to_numeric(frame[column].astype(str).str.replace(r"[^0-9.\\-]", "", regex=True), errors="coerce")
        return frame.dropna(subset=["event_name", "event_time"])
