"""Leakage-aware quantitative features for historical trading logs."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Build machine-learning features using only information known at trade time."""

    _LAGS = (1, 2, 3, 5, 10)
    _WINDOWS = (5, 10, 20, 50, 100)

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.report: dict[str, object] = {}

    def load_processed_trades(self, source: Path | str | pd.DataFrame) -> pd.DataFrame:
        """Load and chronologically order the Day 2 processed trade dataset."""
        try:
            data = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
            data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
            data = data.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            self.logger.info("Loaded %d processed trades.", len(data))
            return data
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise ValueError(f"Unable to load processed trades: {exc}") from exc

    def load_macro_events(self, source: pd.DataFrame | None) -> pd.DataFrame:
        """Normalize a macro-events frame loaded from the SQLite database."""
        if source is None or source.empty:
            self.logger.warning("No macro events available. Macro features skipped.")
            return pd.DataFrame()
        events = source.copy()
        events["event_time"] = pd.to_datetime(events["event_time"], utc=True, errors="coerce")
        events = events.dropna(subset=["event_time", "event_name"]).sort_values("event_time")
        for column in ("actual", "forecast", "previous"):
            events[column] = pd.to_numeric(events.get(column), errors="coerce")
        return events

    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create UTC calendar fields available at the trade timestamp."""
        self.logger.info("Creating time features.")
        result = data.copy()
        time = pd.to_datetime(result["timestamp"], utc=True)
        iso = time.dt.isocalendar()
        result["hour"] = time.dt.hour
        result["minute"] = time.dt.minute
        result["weekday"] = time.dt.weekday
        result["day_name"] = time.dt.day_name()
        result["month"] = time.dt.month
        result["quarter"] = time.dt.quarter
        result["week_of_year"] = iso.week.astype("int16")
        result["day_of_month"] = time.dt.day
        result["is_weekend"] = (time.dt.weekday >= 5).astype("int8")
        result["is_month_end"] = time.dt.is_month_end.astype("int8")
        result["is_month_start"] = time.dt.is_month_start.astype("int8")
        return result

    def create_session_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add UTC Asia, London, New York, and London/New York overlap flags."""
        result = data.copy()
        hour = result["hour"]
        result["asia_session"] = hour.between(0, 7).astype("int8")
        result["london_session"] = hour.between(7, 15).astype("int8")
        result["new_york_session"] = hour.between(13, 21).astype("int8")
        result["overlap_session"] = (result["london_session"].eq(1) & result["new_york_session"].eq(1)).astype("int8")
        return result

    def create_trade_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create direct trade values and history-safe cumulative PnL features."""
        result = data.copy()
        direction = result["direction"].astype(str).str.upper()
        result["trade_direction_encoded"] = direction.map({"BUY": 1, "LONG": 1, "SELL": -1, "SHORT": -1}).fillna(0).astype("int8")
        result["buy_sell_flag"] = direction.isin(["BUY", "LONG"]).astype("int8")
        result["position_size"] = result["quantity"].abs()
        result["trade_value"] = result["position_size"] * result["price"]
        result["absolute_pnl"] = result["pnl"].abs()
        result["positive_trade"] = result["pnl"].gt(0).astype("int8")
        result["negative_trade"] = result["pnl"].lt(0).astype("int8")
        result["cumulative_pnl"] = result["pnl"].cumsum().shift(1).fillna(0.0)
        result["trade_number"] = np.arange(1, len(result) + 1, dtype=np.int64)
        return result

    def create_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create globally chronological lag features for selected past trades."""
        self.logger.info("Creating lag features.")
        result = data.copy()
        for lag in self._LAGS:
            for source, name in (("pnl", "pnl"), ("price", "price"), ("quantity", "quantity"), ("trade_direction_encoded", "direction"), ("trade_value", "trade_value"), ("positive_trade", "trade_result")):
                result[f"previous_{name}_{lag}"] = result[source].shift(lag)
        result["previous_pnl"] = result["previous_pnl_1"]
        result["previous_price"] = result["previous_price_1"]
        result["previous_quantity"] = result["previous_quantity_1"]
        result["previous_direction"] = result["previous_direction_1"]
        result["previous_trade_value"] = result["previous_trade_value_1"]
        result["previous_trade_result"] = result["previous_trade_result_1"]
        return result

    def create_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create shifted rolling features so no current or future PnL leaks in."""
        self.logger.info("Creating rolling features.")
        result = data.copy()
        history = result[["pnl", "positive_trade", "quantity", "price"]].shift(1)
        for window in self._WINDOWS:
            rolling = history.rolling(window, min_periods=1)
            result[f"rolling_mean_pnl_{window}"] = rolling["pnl"].mean()
            result[f"rolling_std_pnl_{window}"] = rolling["pnl"].std()
            result[f"rolling_win_rate_{window}"] = rolling["positive_trade"].mean()
            result[f"rolling_trade_count_{window}"] = rolling["pnl"].count()
            result[f"rolling_average_quantity_{window}"] = rolling["quantity"].mean()
            result[f"rolling_average_price_{window}"] = rolling["price"].mean()
            result[f"rolling_volatility_{window}"] = rolling["pnl"].std()
            result[f"rolling_max_pnl_{window}"] = rolling["pnl"].max()
            result[f"rolling_min_pnl_{window}"] = rolling["pnl"].min()
        return result

    def create_simulation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode simulation and compute its expanding history excluding current trade."""
        result = data.copy()
        result["simulation_encoded"] = pd.factorize(result["simulation"], sort=True)[0]
        group = result.groupby("simulation", group_keys=False)
        result["simulation_trade_count"] = group.cumcount()
        result["simulation_total_pnl"] = group["pnl"].cumsum() - result["pnl"]
        result["simulation_average_pnl"] = result["simulation_total_pnl"] / result["simulation_trade_count"].replace(0, np.nan)
        result["simulation_win_rate"] = group["positive_trade"].transform(lambda values: values.shift(1).expanding(min_periods=1).mean())
        account_group = result.groupby("account", group_keys=False)
        result["account_trade_count"] = account_group.cumcount()
        result["account_total_pnl"] = account_group["pnl"].cumsum() - result["pnl"]
        result["account_average_pnl"] = result["account_total_pnl"] / result["account_trade_count"].replace(0, np.nan)
        result["account_win_rate"] = account_group["positive_trade"].transform(lambda values: values.shift(1).expanding(min_periods=1).mean())
        return result

    def create_macro_features(self, data: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """Attach past macro outcomes and distance to the next scheduled event."""
        self.logger.info("Creating macro features.")
        result = data.copy()
        if events.empty:
            result["minutes_since_previous_event"] = np.nan
            result["minutes_until_next_event"] = np.nan
            return result
        base = result.sort_values("timestamp")
        prior = pd.merge_asof(base, events, left_on="timestamp", right_on="event_time", direction="backward")
        future = pd.merge_asof(base[["timestamp"]], events[["event_time"]], left_on="timestamp", right_on="event_time", direction="forward")
        prior["minutes_since_previous_event"] = (prior["timestamp"] - prior["event_time"]).dt.total_seconds() / 60
        prior["minutes_until_next_event"] = (future["event_time"].to_numpy() - prior["timestamp"].to_numpy()) / np.timedelta64(1, "m")
        name = prior["event_name"].astype(str).str.lower()
        for label, terms in {"cpi": ("cpi",), "fomc": ("fomc",), "employment": ("employment", "payroll", "jobs")}.items():
            source = events[events["event_name"].astype(str).str.lower().str.contains("|".join(terms), na=False)][["event_time", "actual"]]
            attached = pd.merge_asof(base[["timestamp"]], source, left_on="timestamp", right_on="event_time", direction="backward")
            prior[f"previous_{label}"] = attached["actual"].to_numpy()
        prior["actual_minus_forecast"] = prior["actual"] - prior["forecast"]
        prior["forecast_error"] = prior["actual_minus_forecast"].abs()
        prior["surprise_score"] = prior["actual_minus_forecast"] / prior["forecast"].abs().replace(0, np.nan)
        prior["positive_surprise"] = prior["actual_minus_forecast"].gt(0).astype("int8")
        prior["negative_surprise"] = prior["actual_minus_forecast"].lt(0).astype("int8")
        self.report["macro_features_created"] = True
        return prior.drop(columns=["id"], errors="ignore")

    def create_target(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create the only forward-looking values: next-trade PnL and sign."""
        result = data.copy()
        result["future_pnl"] = result["pnl"].shift(-1)
        result["future_positive_trade"] = result["future_pnl"].gt(0).astype("int8")
        result.loc[result["future_pnl"].isna(), "future_positive_trade"] = np.nan
        return result

    def validate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Log feature-quality warnings and sanitize infinity values without targets."""
        duplicate_columns = data.columns[data.columns.duplicated()].tolist()
        if duplicate_columns:
            self.logger.warning("Duplicate columns detected: %s", duplicate_columns)
        numeric = data.select_dtypes(include=np.number)
        infinity_count = int(np.isinf(numeric.to_numpy()).sum())
        if infinity_count:
            self.logger.warning("Replacing %d infinite feature values with NaN.", infinity_count)
            data = data.replace([np.inf, -np.inf], np.nan)
        missing = data.isna().sum()
        self.report["missing_values"] = {key: int(value) for key, value in missing.items() if value}
        if self.report["missing_values"]:
            self.logger.warning("Features contain missing values; first-row historical features are expected.")
        constants = [column for column in numeric if numeric[column].nunique(dropna=False) <= 1]
        if constants:
            self.logger.warning("Constant numeric columns detected: %s", constants)
        correlation = numeric.corr().abs()
        pairs = (correlation.where(np.triu(np.ones(correlation.shape), 1).astype(bool)).stack() > 0.98)
        if pairs.any():
            self.logger.warning("Highly correlated numeric feature pairs detected: %d", int(pairs.sum()))
        self.report["constant_columns"] = constants
        return data

    def save_dataset(self, data: pd.DataFrame, output: Path) -> None:
        """Write final ML-ready dataset to CSV."""
        output.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output, index=False)
        self.logger.info("Saved final dataset to %s.", output)

    def _write_report(self, data: pd.DataFrame, output: Path) -> None:
        """Write feature inventory and quality information as Markdown."""
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Feature Engineering Report", "", f"- **Number of columns:** {len(data.columns)}", f"- **Target variables:** future_pnl, future_positive_trade", f"- **Macro features created:** {bool(self.report.get('macro_features_created', False))}", f"- **Rolling windows:** {', '.join(map(str, self._WINDOWS))}", f"- **Lag periods:** {', '.join(map(str, self._LAGS))}", f"- **Categorical columns:** account, simulation, direction, day_name", f"- **Missing values:** {self.report.get('missing_values', {})}", "", "## Feature List", ""]
        lines.extend(f"- `{column}`" for column in data.columns)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def transform(self, trades: Path | str | pd.DataFrame, events: pd.DataFrame | None, output: Path, report_path: Path | None = None) -> pd.DataFrame:
        """Run Day 3 feature generation with history-only predictors."""
        try:
            data = self.load_processed_trades(trades)
            data = self.create_time_features(data)
            data = self.create_session_features(data)
            data = self.create_trade_features(data)
            data = self.create_lag_features(data)
            data = self.create_rolling_features(data)
            data = self.create_simulation_features(data)
            data = self.create_macro_features(data, self.load_macro_events(events))
            data = self.create_target(data)
            data = self.validate_features(data)
            self.save_dataset(data, output)
            if report_path:
                self._write_report(data, report_path)
            self.logger.info("Feature engineering complete.")
            return data
        except Exception:
            self.logger.exception("Feature engineering failed.")
            raise
