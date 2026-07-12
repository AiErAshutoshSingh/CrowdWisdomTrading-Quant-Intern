"""Fold-wise LightGBM/XGBoost PnL modelling."""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from .constants import FEATURE_EXCLUSIONS
from .walk_forward import Fold

class ModelTrainer:
    """Train one model per strict walk-forward fold and persist predictions."""
    def __init__(self, logger: logging.Logger, model_name: str = "lightgbm") -> None:
        self.logger, self.model_name = logger, model_name
        self._fallback_warned = False
    def _model(self):
        try:
            if self.model_name == "xgboost":
                from xgboost import XGBRegressor; return XGBRegressor(n_estimators=300,max_depth=5,learning_rate=.04,n_jobs=1,random_state=42)
            from lightgbm import LGBMRegressor; return LGBMRegressor(n_estimators=300,num_leaves=31,learning_rate=.04,random_state=42,verbosity=-1)
        except ImportError:
            if not self._fallback_warned:
                self.logger.warning(
                    "Optional gradient booster unavailable; using sklearn fallback. "
                    "Install LightGBM or XGBoost to use the configured model."
                )
                self._fallback_warned = True
            return HistGradientBoostingRegressor(max_iter=300,learning_rate=.04,random_state=42)
    def fit_predict(self, data: pd.DataFrame, folds: list[Fold], output: Path) -> pd.DataFrame:
        """Fit fold models without temporal leakage and write predictions."""
        columns=[c for c in data.select_dtypes(include=np.number).columns if c not in FEATURE_EXCLUSIONS]
        results=[]
        for number, fold in enumerate(folds, 1):
            train, test=data.loc[fold.train_index], data.loc[fold.test_index]
            if train.empty or test.empty: continue
            model=self._model(); model.fit(train[columns], train.pnl); prediction=model.predict(test[columns])
            results.append(pd.DataFrame({"Timestamp":test.timestamp,"Simulation":test.simulation,"Actual":test.pnl,"Prediction":prediction,"Fold":number}))
        final=pd.concat(results,ignore_index=True) if results else pd.DataFrame(columns=["Timestamp","Simulation","Actual","Prediction","Fold"])
        output.parent.mkdir(parents=True,exist_ok=True); final.to_csv(output,index=False); return final
