"""Predictive and trading-performance evaluation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(predictions: pd.DataFrame, output: Path) -> dict[str, float]:
    """Compute and write robust regression and strategy metrics."""
    if predictions.empty: raise ValueError("No predictions available for evaluation.")
    actual, predicted=predictions.Actual, predictions.Prediction; returns=np.where(predicted >= 0, actual, -actual); cumulative=np.cumsum(returns); peak=np.maximum.accumulate(cumulative); drawdown=cumulative-peak
    downside=returns[returns<0]; gains=returns[returns>0].sum(); losses=abs(returns[returns<0].sum())
    metrics={"RMSE":float(mean_squared_error(actual,predicted)**.5),"MAE":float(mean_absolute_error(actual,predicted)),"R2":float(r2_score(actual,predicted)),"Sharpe Ratio":float(np.mean(returns)/np.std(returns)*np.sqrt(252)) if np.std(returns) else 0.,"Sortino Ratio":float(np.mean(returns)/np.std(downside)*np.sqrt(252)) if len(downside) and np.std(downside) else 0.,"Calmar Ratio":float(np.sum(returns)/abs(drawdown.min())) if drawdown.min() else 0.,"Maximum Drawdown":float(drawdown.min()),"Profit Factor":float(gains/losses) if losses else float("inf"),"Average Return":float(np.mean(returns)),"Win Rate":float(np.mean(returns>0)),"Cumulative Return":float(cumulative[-1])}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text("# Evaluation Report\n\n"+"\n".join(f"- **{key}**: {value:.6f}" for key,value in metrics.items()),encoding="utf-8"); return metrics
