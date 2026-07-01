"""Configuration for the BTC/USDT evolutionary pipeline."""

from __future__ import annotations

from pathlib import Path


# Data
DATA_PATH: Path = Path("data/crypto/BTCUSDT_15m.csv")
DATE_COLUMN: str = "date"

# Output
RESULTS_DIR: Path = Path("crypto/results")
DEFAULT_ARCHIVE_PATH: Path = RESULTS_DIR / "crypto_btc_archive.json"

# Multi-horizon binary labels. Edit this list freely, for example [3, 7, 10, 20].
HOLDING_HORIZONS: list[int] = [3, 7, 10]
LABEL_THRESHOLD: float = 0.001  # label=1 when future return > 0.1%

# Final split, kept separate from the stock settings.
VAL_START: str = "2024-01-01"
TEST_START: str = "2025-01-01"
TEST_END: str | None = None

# Walk-forward folds used during evolution. WF_END defaults to TEST_START.
WF_END: str = TEST_START
WF_MIN_TRAIN_MONTHS: int = 36
WF_VAL_MONTHS: int = 6
WF_STEP_MONTHS: int = 6
WF_PURGE_BARS: int | None = None  # None => max(HOLDING_HORIZONS) + 1

# Safe feature construction. All features are time-series/ratio normalized;
# raw price/volume scale columns are intentionally not selectable.
WINDOWS: list[int] = [3, 5, 10, 14, 20, 30, 60, 120, 240, 480]
FEATURE_MIN_VALID_RATIO: float = 0.70
FEATURE_MAX_DOMINANT_VALUE_RATIO: float = 0.985
FEATURE_CORR_THRESHOLD: float = 0.95
EXPR_MAX_DEPTH: int = 4
EXPR_MAX_LENGTH: int = 240
EXPR_MAX_ABS_QUANTILE: float = 50.0

# Individual/evolution knobs.
FEATURE_MIN: int = 4
FEATURE_MAX: int = 24
ARCHIVE_SIZE: int = 50
TIME_BUDGET_SECONDS: float = 3600.0
RESTART_PROB: float = 0.001
CHECKPOINT_EVERY_SECONDS: float = 12 * 60 * 60

# Fitness. RETURN_SCORE_SCALE normalizes mean trade return so that one metric
# cannot dominate merely by being on a wider numerical scale.
TRADE_TOP_FRACTION: float = 0.20
MIN_TRADES_PER_SPLIT: int = 20
TRADE_COST: float = 0.0
RETURN_SCORE_SCALE: float = 0.01
BAD_AUC_THRESHOLD: float = 0.50

FITNESS_WEIGHTS: dict[str, float] = {
    "auc_edge": 0.40,
    "precision_excess": 0.25,
    "trade_return_score": 0.25,
    "auc_std": -0.20,
    "overfit_gap": -0.30,
    "bad_fold_ratio": -0.30,
}

# Binary LightGBM. These are deliberately conservative because evolution itself
# is an optimizer and BTC 15m data is noisy.
LGBM_PARAMS: dict = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "feature_fraction": 0.70,
    "bagging_fraction": 0.80,
    "bagging_freq": 1,
    "min_data_in_leaf": 300,
    "lambda_l1": 2.0,
    "lambda_l2": 10.0,
    "force_col_wise": True,
    "verbose": -1,
    "seed": 42,
    "feature_fraction_seed": 42,
    "bagging_seed": 42,
    "data_random_seed": 42,
}
LGBM_NUM_BOOST_ROUND: int = 250
LGBM_EARLY_STOPPING: int = 20
EARLY_STOP_VALID_FRACTION: float = 0.20
EARLY_STOP_MIN_VALID_SAMPLES: int = 100


def purge_bars_for_horizons(horizons: list[int] | tuple[int, ...]) -> int:
    if WF_PURGE_BARS is not None:
        return int(WF_PURGE_BARS)
    return max(int(h) for h in horizons) + 1


def validate_config() -> None:
    if not HOLDING_HORIZONS:
        raise ValueError("HOLDING_HORIZONS must not be empty.")
    if any(int(h) < 1 for h in HOLDING_HORIZONS):
        raise ValueError("HOLDING_HORIZONS must contain positive integers.")
    if LABEL_THRESHOLD < 0:
        raise ValueError("LABEL_THRESHOLD must be non-negative.")
    if FEATURE_MIN < 1 or FEATURE_MAX < FEATURE_MIN:
        raise ValueError("Require 1 <= FEATURE_MIN <= FEATURE_MAX.")
    if EXPR_MAX_DEPTH < 1:
        raise ValueError("EXPR_MAX_DEPTH must be positive.")
    if EXPR_MAX_LENGTH < 20:
        raise ValueError("EXPR_MAX_LENGTH must be at least 20.")
    if EXPR_MAX_ABS_QUANTILE <= 0:
        raise ValueError("EXPR_MAX_ABS_QUANTILE must be positive.")
    if not 0 < TRADE_TOP_FRACTION <= 1:
        raise ValueError("TRADE_TOP_FRACTION must be in (0, 1].")
    if MIN_TRADES_PER_SPLIT < 1:
        raise ValueError("MIN_TRADES_PER_SPLIT must be positive.")
    if RETURN_SCORE_SCALE <= 0:
        raise ValueError("RETURN_SCORE_SCALE must be positive.")
    if ARCHIVE_SIZE < 1:
        raise ValueError("ARCHIVE_SIZE must be positive.")
    if CHECKPOINT_EVERY_SECONDS < 0:
        raise ValueError("CHECKPOINT_EVERY_SECONDS must be non-negative.")
    if not 0.0 <= EARLY_STOP_VALID_FRACTION < 0.5:
        raise ValueError("EARLY_STOP_VALID_FRACTION must be in [0, 0.5).")
    if EARLY_STOP_MIN_VALID_SAMPLES < 1:
        raise ValueError("EARLY_STOP_MIN_VALID_SAMPLES must be positive.")
