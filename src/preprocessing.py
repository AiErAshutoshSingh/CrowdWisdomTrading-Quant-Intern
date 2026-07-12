"""Quality-controlled preprocessing for historical trading logs."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytz

from .constants import REQUIRED_TRADE_COLUMNS


class TradePreprocessor:
    """Validate and cleanse trades before feature engineering."""

    _NUMERIC_COLUMNS = ("price", "quantity", "pnl")
    _CATEGORICAL_COLUMNS = ("direction", "simulation", "account")
    _VALID_DIRECTIONS = {"BUY", "SELL", "LONG", "SHORT"}

    def __init__(self, timezone: str, logger: logging.Logger, active_accounts: list[str] | tuple[str, ...] | None = None, remove_outliers: bool = True, group_window_ms: int = 500) -> None:
        """Configure preprocessing behavior."""
        self.timezone, self.logger = timezone, logger
        self.active_accounts = tuple(active_accounts or ())
        self.remove_outliers, self.group_window_ms = remove_outliers, group_window_ms
        self.quality: dict[str, Any] = {}

    def load_trades(self, source: Path | str | pd.DataFrame) -> pd.DataFrame:
        """Load trades from CSV or copy an in-memory frame."""
        try:
            data = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
        except (OSError, pd.errors.ParserError) as exc:
            raise ValueError(f"Unable to load trading data: {exc}") from exc
        self.quality["total_rows"] = len(data)
        self.logger.info("Loaded %d trades for preprocessing.", len(data))
        return data

    def validate_schema(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure every required trading-log column is present."""
        missing = sorted(REQUIRED_TRADE_COLUMNS - set(data.columns))
        if missing:
            raise ValueError(f"Trading log is missing required columns: {', '.join(missing)}")
        return data

    def normalize_timezones(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert UTC, EST, EDT, and naive timestamps into aware UTC values."""
        result = data.copy()
        default_zone = pytz.timezone(self.timezone)

        def convert(value: object) -> pd.Timestamp | pd.NaT:
            if pd.isna(value):
                return pd.NaT
            text = str(value).upper().strip()
            try:
                local_text = re.sub(r"\s+(?:EST|EDT)$", "", text)
                stamp = pd.Timestamp(local_text)
                if stamp.tzinfo is not None:
                    return stamp.tz_convert(pytz.UTC)
                zone = pytz.timezone("US/Eastern") if "EST" in text or "EDT" in text else default_zone
                is_dst = True if "EDT" in text else False if "EST" in text else None
                return pd.Timestamp(zone.localize(stamp.to_pydatetime(), is_dst=is_dst)).tz_convert(pytz.UTC)
            except (TypeError, ValueError, pytz.AmbiguousTimeError, pytz.NonExistentTimeError):
                return pd.NaT

        result["timestamp"] = result["timestamp"].map(convert)
        self.quality["invalid_timestamps"] = int(result["timestamp"].isna().sum())
        self.quality["timezone_summary"] = f"Converted to UTC; default source timezone: {self.timezone}."
        self.logger.info("Timezone conversion complete; %d invalid timestamps found.", self.quality["invalid_timestamps"])
        return result

    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate trade IDs and duplicate timestamps, keeping earliest rows."""
        original = len(data)
        result = data.drop_duplicates(subset=["trade_id"], keep="first")
        result = result.drop_duplicates(subset=["timestamp"], keep="first")
        self.quality["duplicates_removed"] = original - len(result)
        self.logger.info("Removed %d duplicate trades.", self.quality["duplicates_removed"])
        return result

    def group_transactions_500ms(self, data: pd.DataFrame) -> pd.DataFrame:
        """Keep only earliest transactions per account inside the grouping window."""
        ordered = data.sort_values(["account", "timestamp"], kind="stable").copy()
        elapsed_ms = ordered.groupby("account")["timestamp"].diff().dt.total_seconds() * 1000
        result = ordered[elapsed_ms.isna() | elapsed_ms.gt(self.group_window_ms)].copy()
        self.quality["grouping_original_rows"] = len(data)
        self.quality["grouping_remaining_rows"] = len(result)
        self.quality["grouping_rows_removed"] = len(data) - len(result)
        self.logger.info("Grouped same-account trades within %d ms: %d -> %d rows.", self.group_window_ms, len(data), len(result))
        return result

    def filter_accounts(self, data: pd.DataFrame) -> pd.DataFrame:
        """Retain configured active accounts when an allow-list exists."""
        if not self.active_accounts:
            return data
        result = data[data["account"].isin(self.active_accounts)].copy()
        self.logger.info("Filtered to %d active accounts; %d rows remain.", len(self.active_accounts), len(result))
        return result

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drop missing timestamps, median-fill numeric data, and mode-fill categories."""
        result = data.dropna(subset=["timestamp"]).copy()
        self.quality["rows_dropped_missing_timestamp"] = len(data) - len(result)
        missing = result.isna().sum().to_dict()
        for column in self._NUMERIC_COLUMNS:
            result[column] = pd.to_numeric(result[column], errors="coerce")
            result[column] = result[column].fillna(result[column].median()).fillna(0.0)
        for column in self._CATEGORICAL_COLUMNS:
            result[column] = result[column].replace("", pd.NA)
            mode = result[column].mode(dropna=True)
            result[column] = result[column].fillna(mode.iloc[0] if not mode.empty else "UNKNOWN")
        self.quality["missing_values_before_imputation"] = {key: int(value) for key, value in missing.items() if value}
        self.logger.info("Missing values handled; %d timestamp rows dropped.", self.quality["rows_dropped_missing_timestamp"])
        return result

    def detect_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove only extreme (3×IQR) outliers from numeric trade fields."""
        self.quality["outlier_statistics_before"] = data.loc[:, self._NUMERIC_COLUMNS].describe().to_dict()
        if not self.remove_outliers or data.empty:
            self.quality["outliers_removed"] = 0
            return data
        keep = pd.Series(True, index=data.index)
        for column in self._NUMERIC_COLUMNS:
            q1, q3 = data[column].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                keep &= data[column].between(q1 - 3 * iqr, q3 + 3 * iqr)
        result = data[keep].copy()
        self.quality["outliers_removed"] = len(data) - len(result)
        self.quality["outlier_statistics_after"] = result.loc[:, self._NUMERIC_COLUMNS].describe().to_dict()
        self.logger.info("Removed %d extreme IQR outliers.", self.quality["outliers_removed"])
        return result

    def clean_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reject non-positive price/quantity and invalid categorical trade values."""
        result = data.copy()
        result["direction"] = result["direction"].astype(str).str.upper().str.strip()
        result["simulation"] = result["simulation"].astype(str).str.strip()
        valid = (result["price"] > 0) & (result["quantity"] > 0)
        valid &= result["direction"].isin(self._VALID_DIRECTIONS)
        valid &= result["simulation"].notna() & result["simulation"].ne("") & result["simulation"].ne("UNKNOWN")
        self.quality["invalid_price_data_removed"] = len(result) - int(valid.sum())
        self.logger.info("Removed %d invalid price/data rows.", self.quality["invalid_price_data_removed"])
        return result[valid].sort_values("timestamp").reset_index(drop=True)

    def generate_quality_report(self, output: Path) -> None:
        """Write a concise Markdown quality report."""
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Data Quality Report", "", f"- **Total rows:** {self.quality.get('total_rows', 0)}", f"- **Invalid timestamps:** {self.quality.get('invalid_timestamps', 0)}", f"- **Rows dropped for missing timestamps:** {self.quality.get('rows_dropped_missing_timestamp', 0)}", f"- **Duplicates removed:** {self.quality.get('duplicates_removed', 0)}", f"- **500 ms grouped rows removed:** {self.quality.get('grouping_rows_removed', 0)}", f"- **Outliers removed:** {self.quality.get('outliers_removed', 0)}", f"- **Invalid price/data rows removed:** {self.quality.get('invalid_price_data_removed', 0)}", f"- **Missing values before imputation:** {self.quality.get('missing_values_before_imputation', {})}", f"- **Accounts:** {', '.join(self.quality.get('accounts', [])) or 'None'}", f"- **Date range (UTC):** {self.quality.get('date_range', 'No valid timestamps')}", f"- **Timezone conversion:** {self.quality.get('timezone_summary', '')}"]
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.logger.info("Saved data quality report to %s.", output)

    def save_processed_data(self, data: pd.DataFrame, output: Path) -> None:
        """Persist clean trades for downstream feature engineering."""
        output.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output, index=False)
        self.logger.info("Saved %d processed trades to %s.", len(data), output)

    def transform(self, source: Path | str | pd.DataFrame, output: Path, report_path: Path | None = None) -> pd.DataFrame:
        """Run the ordered Day 2 preprocessing pipeline."""
        try:
            data = self.validate_schema(self.load_trades(source))
            data = self.normalize_timezones(data)
            data = self.remove_duplicates(data)
            data = self.group_transactions_500ms(data)
            data = self.filter_accounts(data)
            data = self.handle_missing_values(data)
            data = self.detect_outliers(data)
            data = self.clean_price_data(data)
            self.quality["accounts"] = sorted(data["account"].dropna().unique().tolist())
            self.quality["date_range"] = "No valid timestamps" if data.empty else f"{data.timestamp.min()} to {data.timestamp.max()}"
            self.save_processed_data(data, output)
            if report_path:
                self.generate_quality_report(report_path)
            return data
        except Exception:
            self.logger.exception("Preprocessing failed.")
            raise
