"""Small reusable pipeline utilities."""
from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while mapping zero denominators to missing values."""
    return numerator / denominator.replace(0, np.nan)


def numeric_or_nan(values: Iterable[object]) -> pd.Series:
    """Convert an iterable to a numeric Series without raising."""
    return pd.to_numeric(pd.Series(values), errors="coerce")
