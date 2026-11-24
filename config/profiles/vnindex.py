"""Vietnam daily-stock dataset profile (backward-compatible default)."""

PROFILE_NAME = "vnindex"
DEFAULT_DATA_DIR = "data/raw"
DATA_FILE_SUFFIX = ""
TIME_UNIT = "trading_day"
BAR_INTERVAL_MINUTES = None

VAL_START = "2023-05-12"
TEST_START = "2025-01-01"
TEST_END = None

HOLDING_HORIZON = 10
WF_END = TEST_START
WF_MIN_TRAIN_MONTHS = 48
WF_VAL_MONTHS = 6
WF_STEP_MONTHS = 6
WF_PURGE_DAYS = HOLDING_HORIZON
# Keep every training timestamp. This preserves the established VN behavior.
WF_TRAIN_BAR_STRIDE = 1

WINDOWS = [3, 5, 10, 14, 20, 30, 60, 120]
DEFAULT_WINDOW = 1
MARKET_INDEX_TICKER = "VNINDEX"

SECTORS = {
    "Banking": ["ACB", "BID", "CTG", "HDB", "MBB", "STB", "TCB", "VCB", "VPB", "TPB"],
    "Real_Estate": ["KDH", "NVL", "VHM", "VIC", "VRE"],
    "Industry": ["FPT", "GAS", "GVR", "HPG", "MSN", "PDR", "PLX", "POW"],
    "Consumer": ["BVH", "MWG", "PNJ", "SAB", "SSI", "VJC", "VNM"],
}

# Preserve the established VNINDEX execution/correlation behavior.
CORR_CHECK_MAX_ROWS = 0
DOMAIN_CORR_MAX_CHECKS = 0
DOMAIN_PRECOMPUTE_ON_START = True

# Daily VNINDEX panels are much smaller, but caches are still bounded so a
# multi-day run cannot grow memory indefinitely.
TRAINER_FEATURE_CACHE_MAX_ITEMS = 128
TRAINER_SPLIT_CACHE_MAX_ITEMS = 64
