"""SQLAlchemy persistence layer."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable

import pandas as pd
from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    """Base ORM model."""


class MacroEvent(Base):
    """Macroeconomic calendar event."""
    __tablename__ = "macro_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_name: Mapped[str] = mapped_column(String(255), nullable=False)
    country: Mapped[str | None] = mapped_column(String(64))
    actual: Mapped[float | None] = mapped_column(Float)
    forecast: Mapped[float | None] = mapped_column(Float)
    previous: Mapped[float | None] = mapped_column(Float)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    timezone: Mapped[str | None] = mapped_column(String(64))
    __table_args__ = (UniqueConstraint("event_name", "event_time", "country", name="uq_macro_event"),)


class Trade(Base):
    """Executed or simulated trade."""
    __tablename__ = "trades"
    trade_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(16), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    pnl: Mapped[float] = mapped_column(Float, nullable=False)
    account: Mapped[str] = mapped_column(String(128), nullable=False)
    simulation: Mapped[str] = mapped_column(String(128), nullable=False, index=True)


class Database:
    """Database lifecycle and idempotent DataFrame ingestion."""
    def __init__(self, url: str, logger: logging.Logger) -> None:
        self.engine = create_engine(url, future=True)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)
        self.logger = logger

    def create_tables(self) -> None:
        """Create database schema if absent."""
        Base.metadata.create_all(self.engine)

    def upsert_trades(self, frame: pd.DataFrame) -> int:
        """Insert trades, ignoring records already present by trade ID."""
        records = frame.to_dict("records")
        with self.session_factory() as session:
            existing = {row[0] for row in session.query(Trade.trade_id).filter(Trade.trade_id.in_([r["trade_id"] for r in records])).all()} if records else set()
            session.add_all([Trade(**record) for record in records if record["trade_id"] not in existing])
            session.commit()
        return len(records) - len(existing)

    def upsert_events(self, frame: pd.DataFrame) -> int:
        """Insert macro events, ignoring event-name/time/country duplicates."""
        records = frame.to_dict("records")
        inserted = 0
        with self.session_factory() as session:
            for record in records:
                exists = session.query(MacroEvent.id).filter_by(event_name=record["event_name"], event_time=record["event_time"], country=record.get("country")).first()
                if not exists:
                    session.add(MacroEvent(**record)); inserted += 1
            session.commit()
        return inserted

    def read_trades(self) -> pd.DataFrame:
        """Load all trades ordered by timestamp."""
        return pd.read_sql("SELECT * FROM trades ORDER BY timestamp", self.engine, parse_dates=["timestamp"])

    def read_events(self) -> pd.DataFrame:
        """Load all macro events ordered by event time."""
        return pd.read_sql("SELECT * FROM macro_events ORDER BY event_time", self.engine, parse_dates=["event_time"])
