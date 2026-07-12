"""Project-wide constants."""

REQUIRED_TRADE_COLUMNS = {"timestamp", "direction", "quantity", "price", "pnl", "account", "simulation"}
FEATURE_EXCLUSIONS = {"timestamp", "pnl", "trade_id", "event_name", "country", "event_time", "timezone", "future_pnl", "future_positive_trade"}
MACRO_KEYWORDS = ("cpi", "fomc", "employment", "nonfarm", "payroll", "gdp")
