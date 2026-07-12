"""Strict date-based walk-forward validation."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

@dataclass(frozen=True)
class Fold: train_index: pd.Index; test_index: pd.Index
class WalkForwardValidator:
    """Generate 30-calendar-day train / 7-day test rolling folds."""
    def __init__(self, train_days: int = 30, test_days: int = 7) -> None: self.train_days, self.test_days = train_days, test_days
    def split(self, frame: pd.DataFrame) -> list[Fold]:
        data = frame.sort_values("timestamp"); dates = pd.to_datetime(data.timestamp, utc=True); start, last = dates.min().normalize(), dates.max()
        folds=[]
        while start + pd.Timedelta(days=self.train_days + self.test_days) <= last + pd.Timedelta(days=1):
            train_end=start + pd.Timedelta(days=self.train_days); test_end=train_end + pd.Timedelta(days=self.test_days)
            folds.append(Fold(data.index[(dates >= start)&(dates < train_end)], data.index[(dates >= train_end)&(dates < test_end)])); start += pd.Timedelta(days=self.test_days)
        return folds
    def plot(self, frame: pd.DataFrame, folds: list[Fold], path: Path) -> None:
        fig, ax=plt.subplots(figsize=(12, max(3,len(folds)*.35))); dates=pd.to_datetime(frame.timestamp,utc=True)
        for i,f in enumerate(folds): ax.scatter(dates.loc[f.train_index], [i]*len(f.train_index), marker="|", color="steelblue"); ax.scatter(dates.loc[f.test_index], [i]*len(f.test_index), marker="|", color="darkorange")
        ax.set(xlabel="Time", ylabel="Fold", title="Walk-forward folds"); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)
