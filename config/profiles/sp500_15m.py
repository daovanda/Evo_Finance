"""SPY plus 50 diversified S&P 500 companies on regular-session 15m bars."""

PROFILE_NAME = "sp500_15m"
DEFAULT_DATA_DIR = "data/raw/sp500"
DATA_FILE_SUFFIX = "_15Min"
TIME_UNIT = "bar"
BAR_INTERVAL_MINUTES = 15

VAL_START = "2024-01-01"
TEST_START = "2025-01-01"
TEST_END = None

# Signal at close(t), enter open(t+1), exit close(t+8): about two hours.
HOLDING_HORIZON = 8
WF_END = TEST_START
WF_MIN_TRAIN_MONTHS = 36
WF_VAL_MONTHS = 6
WF_STEP_MONTHS = 6
WF_PURGE_DAYS = HOLDING_HORIZON
# Fit WF models on every third timestamp while retaining all 50 stocks in each
# selected LightGBM ranking group. WF validation and final evaluation stay full.
WF_TRAIN_BAR_STRIDE = 100

WINDOWS = [2, 4, 8, 13, 26, 52, 130, 260]
DEFAULT_WINDOW = 1
MARKET_INDEX_TICKER = "SPY"

SECTORS = {
    "Information_Technology": [
        "NVDA", "AAPL", "MSFT", "AVGO", "AMD", "ORCL", "CSCO", "AMAT", "TXN", "IBM",
    ],
    "Communication_Services": ["GOOGL", "META", "NFLX", "DIS"],
    "Consumer_Discretionary": ["AMZN", "TSLA", "HD", "MCD", "BKNG"],
    "Consumer_Staples": ["WMT", "COST", "PG", "KO"],
    "Financials": ["JPM", "BRK.B", "V", "MA", "BAC", "GS"],
    "Health_Care": ["LLY", "JNJ", "UNH", "ABBV", "MRK", "TMO"],
    "Industrials": ["GE", "CAT", "RTX", "UNP", "BA"],
    "Energy": ["XOM", "CVX", "COP"],
    "Utilities": ["NEE", "SO"],
    "Real_Estate": ["PLD", "AMT"],
    "Materials": ["LIN", "SHW", "FCX"],
}

# The panel has millions of rows, so correlation guards use deterministic
# sampling. Startup precompute warms only the 100 corr-relevant formulas below;
# the remaining domain formulas stay lazy. Model training is configured above.
CORR_CHECK_MAX_ROWS = 30_000
DOMAIN_CORR_MAX_CHECKS = 100
DOMAIN_PRECOMPUTE_ON_START = True

# Bounded LRU caches prevent multi-million-row Series from accumulating over
# long evolutionary runs. FEATURE_MAX is 30, so 32 retains one full individual.
TRAINER_FEATURE_CACHE_MAX_ITEMS = 32
TRAINER_SPLIT_CACHE_MAX_ITEMS = 64
