"""Strategy recommendation and equity visualizations."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class MatrixGenerator:
    """Select highest predicted strategy for each weekday/hour bucket."""
    def generate(self, predictions: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
        """Write matrix CSV, heatmap, and model-vs-baseline equity curve."""
        output_dir.mkdir(parents=True,exist_ok=True); data=predictions.copy(); t=pd.to_datetime(data.Timestamp,utc=True); data["weekday"],data["hour"]=t.dt.day_name(),t.dt.hour
        best=data.loc[data.groupby(["weekday","hour"])["Prediction"].idxmax()]; matrix=best.pivot(index="weekday",columns="hour",values="Simulation").reindex(["Monday","Tuesday","Wednesday","Thursday","Friday"]); matrix.to_csv(output_dir/"recommendation_matrix.csv")
        simulation_codes = {
            simulation: code
            for code, simulation in enumerate(matrix.stack().unique())
        }
        codes = matrix.replace(simulation_codes).astype(float)
        fig, ax = plt.subplots(figsize=(16, 5))
        sns.heatmap(
            codes,
            annot=matrix,
            fmt="",
            cbar=False,
            mask=matrix.isna(),
            ax=ax,
        )
        ax.set(title="Recommended simulation by weekday and hour")
        fig.tight_layout()
        fig.savefig(output_dir / "heatmap.png", dpi=150)
        plt.close(fig)
        data["model_return"]=data.Actual.where(data.Prediction>=0,-data.Actual); data=data.sort_values("Timestamp"); fig,ax=plt.subplots(figsize=(12,5)); data.model_return.cumsum().plot(ax=ax,label="Model strategy"); data.Actual.cumsum().plot(ax=ax,label="Baseline strategy"); ax.legend(); ax.set(title="Equity curve",ylabel="Cumulative PnL"); fig.tight_layout(); fig.savefig(output_dir/"equity_curve.png",dpi=150); plt.close(fig); return matrix
