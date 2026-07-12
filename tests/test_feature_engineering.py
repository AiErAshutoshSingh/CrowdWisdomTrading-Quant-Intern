"""Unit tests for leakage-aware feature engineering."""
from __future__ import annotations

import logging
import unittest

import pandas as pd

from src.feature_engineering import FeatureEngineer


def trades() -> pd.DataFrame:
    """Return a compact chronological trade sample."""
    return pd.DataFrame(
        {
            "trade_id": ["t1", "t2", "t3"],
            "timestamp": pd.to_datetime(["2025-01-01T12:00:00Z", "2025-01-01T12:01:00Z", "2025-01-01T12:02:00Z"]),
            "direction": ["BUY", "SELL", "BUY"],
            "quantity": [1.0, 2.0, 3.0],
            "price": [100.0, 101.0, 102.0],
            "pnl": [10.0, -5.0, 20.0],
            "account": ["ACC001", "ACC001", "ACC002"],
            "simulation": ["SIM_A", "SIM_A", "SIM_B"],
        }
    )


class FeatureEngineerTests(unittest.TestCase):
    """Verify required Day 3 feature categories and target leakage controls."""

    def setUp(self) -> None:
        self.engineer = FeatureEngineer(logging.getLogger("feature-engineering-test"))

    def test_time_features(self) -> None:
        """UTC timestamp fields should be extracted correctly."""
        result = self.engineer.create_time_features(trades())
        self.assertEqual(result.loc[0, "hour"], 12)
        self.assertEqual(result.loc[0, "weekday"], 2)
        self.assertEqual(result.loc[0, "day_name"], "Wednesday")

    def test_lag_and_rolling_features_are_shifted(self) -> None:
        """First observation has no history and second uses only first PnL."""
        result = self.engineer.create_trade_features(self.engineer.create_time_features(trades()))
        result = self.engineer.create_lag_features(result)
        result = self.engineer.create_rolling_features(result)
        self.assertTrue(pd.isna(result.loc[0, "previous_pnl"]))
        self.assertEqual(result.loc[1, "previous_pnl"], 10.0)
        self.assertEqual(result.loc[1, "rolling_mean_pnl_5"], 10.0)

    def test_macro_merge_uses_previous_event(self) -> None:
        """An event before trades should populate past macro fields."""
        events = pd.DataFrame({"id": [1], "event_name": ["US CPI"], "event_time": pd.to_datetime(["2025-01-01T11:30:00Z"]), "actual": [3.2], "forecast": [3.0], "previous": [3.1], "country": ["US"], "timezone": ["UTC"]})
        result = self.engineer.create_macro_features(trades(), self.engineer.load_macro_events(events))
        self.assertEqual(result.loc[0, "previous_cpi"], 3.2)
        self.assertAlmostEqual(result.loc[0, "actual_minus_forecast"], 0.2)
        self.assertEqual(result.loc[0, "minutes_since_previous_event"], 30.0)

    def test_target_is_next_trade_pnl(self) -> None:
        """Only target columns may use future observations."""
        result = self.engineer.create_target(trades())
        self.assertEqual(result.loc[0, "future_pnl"], -5.0)
        self.assertEqual(result.loc[0, "future_positive_trade"], 0)
        self.assertTrue(pd.isna(result.loc[2, "future_pnl"]))


if __name__ == "__main__":
    unittest.main()
