"""Unit tests for trading-log preprocessing."""
from __future__ import annotations

import logging
import unittest

import pandas as pd

from src.preprocessing import TradePreprocessor


def make_frame() -> pd.DataFrame:
    """Create a minimal valid trading-log frame."""
    return pd.DataFrame(
        {
            "trade_id": ["a", "b", "c"],
            "timestamp": ["2025-01-01 09:00:00 EST", "2025-01-01 09:00:00.400 EST", "2025-01-01 09:00:01 EST"],
            "direction": ["BUY", "SELL", "BUY"],
            "quantity": [1, 2, 3],
            "price": [100.0, 101.0, 102.0],
            "pnl": [1.0, 2.0, 3.0],
            "account": ["ACC001", "ACC001", "ACC001"],
            "simulation": ["SIM_A", "SIM_A", "SIM_A"],
        }
    )


class TradePreprocessorTests(unittest.TestCase):
    """Verify required Day 2 preprocessing rules."""

    def setUp(self) -> None:
        self.processor = TradePreprocessor("UTC", logging.getLogger("preprocessing-test"), remove_outliers=False)

    def test_timezone_conversion(self) -> None:
        """EST timestamps should become UTC-aware timestamps."""
        result = self.processor.normalize_timezones(make_frame())
        self.assertEqual(str(result.loc[0, "timestamp"].tz), "UTC")
        self.assertEqual(result.loc[0, "timestamp"].hour, 14)

    def test_duplicate_removal(self) -> None:
        """Duplicate trade IDs should keep only the earliest record."""
        frame = make_frame()
        frame.loc[1, "trade_id"] = "a"
        result = self.processor.remove_duplicates(frame)
        self.assertEqual(len(result), 2)

    def test_500ms_grouping(self) -> None:
        """Only the earliest same-account trade in a 500ms window remains."""
        normalized = self.processor.normalize_timezones(make_frame())
        result = self.processor.group_transactions_500ms(normalized)
        self.assertEqual(result["trade_id"].tolist(), ["a", "c"])

    def test_missing_value_handling(self) -> None:
        """Numeric gaps use median and category gaps use the mode."""
        frame = make_frame()
        frame.loc[1, "price"] = None
        frame.loc[2, "direction"] = None
        result = self.processor.handle_missing_values(self.processor.normalize_timezones(frame))
        self.assertFalse(result[["price", "direction"]].isna().any().any())
        self.assertEqual(result.loc[1, "price"], 101.0)
        self.assertEqual(result.loc[2, "direction"], "BUY")


if __name__ == "__main__":
    unittest.main()
