"""Typed configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Runtime configuration and project paths."""
    root: Path
    database_url: str
    trade_input_csv: Path
    apify_token: str | None
    apify_actor_id: str | None
    timezone: str
    active_accounts: tuple[str, ...]
    remove_outliers: bool
    group_window_ms: int

    @property
    def raw_dir(self) -> Path: return self.root / "data" / "raw"
    @property
    def processed_dir(self) -> Path: return self.root / "data" / "processed"
    @property
    def database_dir(self) -> Path: return self.root / "data" / "database"
    @property
    def outputs_dir(self) -> Path: return self.root / "outputs"
    @property
    def reports_dir(self) -> Path: return self.root / "reports"

    def create_directories(self) -> None:
        """Create all runtime directories."""
        for directory in (self.raw_dir, self.processed_dir, self.database_dir, self.outputs_dir, self.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Load settings from .env in the repository root."""
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    db_url = os.getenv("DATABASE_URL", "sqlite:///data/database/crowd_wisdom.db")
    if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:////"):
        db_url = f"sqlite:///{root / db_url.removeprefix('sqlite:///')}"
    csv_value = os.getenv("TRADE_INPUT_CSV", "data/raw/trading_logs.csv")
    accounts = tuple(value.strip() for value in os.getenv("ACTIVE_ACCOUNTS", "").split(",") if value.strip())
    remove_outliers = os.getenv("REMOVE_OUTLIERS", "true").lower() in {"1", "true", "yes"}
    group_window_ms = int(os.getenv("GROUP_WINDOW_MS", "500"))
    return Settings(root, db_url, root / csv_value, os.getenv("APIFY_TOKEN"), os.getenv("APIFY_ACTOR_ID"), os.getenv("TIMEZONE", "America/New_York"), accounts, remove_outliers, group_window_ms)
