"""Execute the complete CrowdWisdomTrading pipeline."""
from src.config import get_settings
from src.database import Database
from src.feature_engineering import FeatureEngineer
from src.load_trades import load_trades
from src.logger import get_logger
from src.matrix_generator import MatrixGenerator
from src.preprocessing import TradePreprocessor
from src.scraper import MacroScraper
from src.train_model import ModelTrainer
from src.walk_forward import WalkForwardValidator
from src.evaluation import evaluate

def main() -> None:
    """Run ingestion through recommendation artifacts."""
    settings=get_settings(); settings.create_directories(); logger=get_logger("crowd_wisdom",settings.outputs_dir); db=Database(settings.database_url,logger); db.create_tables()
    MacroScraper(settings.apify_token,settings.apify_actor_id,db,logger).fetch_and_store(); load_trades(settings.trade_input_csv,db,logger)
    trades=db.read_trades()
    if trades.empty: logger.warning("No trades available. Add data/raw/trading_logs.csv and rerun."); return
    preprocessor = TradePreprocessor(settings.timezone, logger, settings.active_accounts, settings.remove_outliers, settings.group_window_ms)
    clean = preprocessor.transform(trades, settings.processed_dir / "processed_trades.csv", settings.reports_dir / "data_quality_report.md")
    features = FeatureEngineer(logger).transform(clean, db.read_events(), settings.processed_dir / "final_dataset.csv", settings.reports_dir / "feature_engineering_report.md"); validator=WalkForwardValidator(); folds=validator.split(features); validator.plot(features,folds,settings.outputs_dir/"walk_forward_folds.png")
    if not folds: logger.warning("At least 37 days of data are required for walk-forward training."); return
    predictions=ModelTrainer(logger).fit_predict(features,folds,settings.outputs_dir/"prediction_table.csv"); evaluate(predictions,settings.reports_dir/"evaluation_report.md"); MatrixGenerator().generate(predictions,settings.outputs_dir); logger.info("Pipeline completed successfully.")
if __name__ == "__main__": main()
