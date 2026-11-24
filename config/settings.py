"""
Evo_Finance — Central Configuration
All tunable knobs live here.
"""

from __future__ import annotations
from typing import List, Callable

from config.profiles import active_profile_name, load_active_profile


_PROFILE = load_active_profile()
ACTIVE_PROFILE: str = active_profile_name()
DEFAULT_DATA_DIR: str = _PROFILE.DEFAULT_DATA_DIR
DATA_FILE_SUFFIX: str = _PROFILE.DATA_FILE_SUFFIX
TIME_UNIT: str = _PROFILE.TIME_UNIT
BAR_INTERVAL_MINUTES: int | None = _PROFILE.BAR_INTERVAL_MINUTES

# ─── Data split — nhập ngày cụ thể ──────────────────────────────────────────
# TRAIN : [đầu data]   → VAL_START   (exclusive)
# VAL   : VAL_START    → TEST_START  (exclusive)
# TEST  : TEST_START   → TEST_END    (inclusive, None = hết data)

VAL_START: str = _PROFILE.VAL_START
TEST_START: str = _PROFILE.TEST_START
TEST_END: str | None = _PROFILE.TEST_END

# ─── Holding horizon ──────────────────────────────────────────────────────────

HOLDING_HORIZON: int = _PROFILE.HOLDING_HORIZON

def default_label(df, h: int = HOLDING_HORIZON):
    """
    label(t) = (close(t+h) - open(t+1)) / open(t+1)
    df must have columns: open, close  (already shifted by caller).
    Returns a Series aligned to df.index.
    """
    future_close = df["close"].shift(-h)
    next_open    = df["open"].shift(-1)
    return (future_close - next_open) / next_open

LABEL_FN: Callable = default_label

# Walk-forward folds used inside the evolutionary loop. Final validation/test
# still use VAL_START / TEST_START / TEST_END after the time budget ends.
WF_END: str = _PROFILE.WF_END
WF_MIN_TRAIN_MONTHS: int = _PROFILE.WF_MIN_TRAIN_MONTHS
WF_VAL_MONTHS: int = _PROFILE.WF_VAL_MONTHS
WF_STEP_MONTHS: int = _PROFILE.WF_STEP_MONTHS
# Kept under the historical name for API compatibility. It counts unique
# timestamps, so it represents trading days for VNINDEX and bars for intraday.
WF_PURGE_DAYS: int = _PROFILE.WF_PURGE_DAYS
# Applied only to fold-train during walk-forward evolution. A value of 3 keeps
# every third timestamp as a complete cross-sectional LightGBM query group.
WF_TRAIN_BAR_STRIDE: int = _PROFILE.WF_TRAIN_BAR_STRIDE

# ─── Feature limits ───────────────────────────────────────────────────────────

FEATURE_MIN: int = 3
FEATURE_MAX: int = 30
EXPR_MAX_DEPTH: int = 4
EXPR_MAX_LENGTH: int = 320
FEATURE_MIN_VALID_RATIO: float = 0.70  # reject new features with too many NaN rows
FEATURE_MAX_DOMINANT_VALUE_RATIO: float = 0.98  # reject near-constant new features

# ─── Correlation threshold ────────────────────────────────────────────────────

CORR_THRESHOLD: float = 0.70       # used for domain & individual dedup
# 0 = use all valid rows; >0 = deterministic row sample for faster Spearman checks.
CORR_CHECK_MAX_ROWS: int = _PROFILE.CORR_CHECK_MAX_ROWS
# 0 = check all existing domain formulas; >0 = max domain formulas checked per new gene.
DOMAIN_CORR_MAX_CHECKS: int = _PROFILE.DOMAIN_CORR_MAX_CHECKS

# Warm the formulas used by domain correlation checks at startup. If
# DOMAIN_CORR_MAX_CHECKS=N>0, only those N formulas are materialized; 0 means
# the full non-constant domain. This changes timing/memory only, not selection.
DOMAIN_PRECOMPUTE_ON_START: bool = _PROFILE.DOMAIN_PRECOMPUTE_ON_START

# 0 means unlimited. Positive values use least-recently-used eviction; eviction
# only causes exact recomputation and therefore does not alter model results.
TRAINER_FEATURE_CACHE_MAX_ITEMS: int = _PROFILE.TRAINER_FEATURE_CACHE_MAX_ITEMS
TRAINER_SPLIT_CACHE_MAX_ITEMS: int = _PROFILE.TRAINER_SPLIT_CACHE_MAX_ITEMS

# ─── Window whitelist ─────────────────────────────────────────────────────────

WINDOWS: List[int] = list(_PROFILE.WINDOWS)

DEFAULT_WINDOW: int = _PROFILE.DEFAULT_WINDOW

# CSV ticker used as the broad market benchmark. The active profile supplies
# the directory and optional filename suffix; loader exposes it as market_*.
MARKET_INDEX_TICKER: str = _PROFILE.MARKET_INDEX_TICKER

# Edit this mapping before each run if you want a different sector universe.
# Tickers are matched case-insensitively by the evaluator.
SECTORS: dict[str, list[str]] = {
    sector: list(tickers) for sector, tickers in _PROFILE.SECTORS.items()
}
REQUIRE_SECTOR_MAPPING: bool = True
MIN_SECTOR_MEMBERS_IN_UNIVERSE: int = 2

# ─── Mutator probabilities ────────────────────────────────────────────────────

MUTATOR_PROBS = {
    "c1": 0.40,   # add/remove a feature
    "c2": 0.35,   # change window
    "c3": 0.25,   # transform a gene
}
# must sum to 1.0 — validated at runtime

MAX_RETRY: int = 5                 # max retries before fallback in c1/c2/c3

# ─── Restart ──────────────────────────────────────────────────────────────────

RESTART_PROB: float = 0.0001       # 0.01 % — restart from normalized seed features

# ─── Archive ──────────────────────────────────────────────────────────────────

ARCHIVE_SIZE: int = 50             # max individuals kept

# ─── LightGBM ─────────────────────────────────────────────────────────────────

LGBM_WF_PARAMS: dict = {
    "objective":        "lambdarank",
    "metric":           "ndcg",
    "eval_at":          [10],
    "learning_rate":    0.03,
    "num_leaves":       15,
    "max_depth":        4,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "bagging_freq":     1,
    "min_data_in_leaf": 300,
    "lambda_l1":        5.0,
    "lambda_l2":        20.0,
    "verbose":          -1,
    "seed":             42,
    "feature_fraction_seed": 42,
    "bagging_seed":     42,
    "data_random_seed": 42,
}

LGBM_WF_NUM_BOOST_ROUND: int = 250
LGBM_WF_EARLY_STOPPING: int = 20
# WF early stopping uses the tail of each fold-train; fold-val is fitness-only.
WF_EARLY_STOP_VALID_FRACTION: float = 0.20
WF_EARLY_STOP_MIN_VALID_DATES: int = 20

LGBM_FINAL_PARAMS: dict = {
    "objective":        "lambdarank",
    "metric":           "ndcg",
    "eval_at":          [10],
    "learning_rate":    0.03,
    "num_leaves":       15,
    "max_depth":        4,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "bagging_freq":     1,
    "min_data_in_leaf": 300,
    "lambda_l1":        5.0,
    "lambda_l2":        20.0,
    "verbose":          -1,
    "seed":             42,
    "feature_fraction_seed": 42,
    "bagging_seed":     42,
    "data_random_seed": 42,
}

LGBM_FINAL_NUM_BOOST_ROUND: int = 250
LGBM_FINAL_EARLY_STOPPING: int = 20

# ─── Time budget ──────────────────────────────────────────────────────────────

TIME_BUDGET_SECONDS: float = 3600.0   # 1 hour default
CHECKPOINT_EVERY_SECONDS: float = 12 * 60 * 60  # autosave archive every 12h

# ─── Hit-rate ─────────────────────────────────────────────────────────────────

HIT_RATE_TOP_K: int = 10

# ─── Fitness weights ──────────────────────────────────────────────────────────

FITNESS_WEIGHTS = {
    "wf_mean_ic":      0.38,
    "wf_icir":         0.18,
    "wf_hit_excess":   0.18,
    "wf_ic_std":      -0.20,
    "bad_fold_ratio": -0.18,
    "wf_overfit_gap": -0.25,
}


def validate_config() -> None:
    val_start_ts = _parse_date_setting("VAL_START", VAL_START)
    test_start_ts = _parse_date_setting("TEST_START", TEST_START)
    wf_end_ts = _parse_date_setting("WF_END", WF_END)
    test_end_ts = _parse_optional_date_setting("TEST_END", TEST_END)

    if TIME_UNIT not in {"trading_day", "bar"}:
        raise ValueError("TIME_UNIT must be 'trading_day' or 'bar'.")
    if TIME_UNIT == "bar" and (
        BAR_INTERVAL_MINUTES is None or int(BAR_INTERVAL_MINUTES) < 1
    ):
        raise ValueError("BAR_INTERVAL_MINUTES must be positive for bar profiles.")
    if any(char in DATA_FILE_SUFFIX for char in ("/", "\\")):
        raise ValueError("DATA_FILE_SUFFIX must not contain path separators.")

    if val_start_ts >= test_start_ts:
        raise ValueError("VAL_START must be before TEST_START.")
    if test_end_ts is not None and test_end_ts < test_start_ts:
        raise ValueError("TEST_END must be >= TEST_START.")
    if wf_end_ts > test_start_ts:
        raise ValueError("WF_END must be <= TEST_START to keep evolution out of final test.")

    required_mutators = {"c1", "c2", "c3"}
    if set(MUTATOR_PROBS) != required_mutators:
        raise ValueError(f"MUTATOR_PROBS keys must be {sorted(required_mutators)}.")
    if any(value < 0 for value in MUTATOR_PROBS.values()):
        raise ValueError("MUTATOR_PROBS values must be non-negative.")
    if abs(sum(MUTATOR_PROBS.values()) - 1.0) > 1e-9:
        raise ValueError("MUTATOR_PROBS must sum to 1.0.")

    if FEATURE_MIN < 1 or FEATURE_MAX < FEATURE_MIN:
        raise ValueError("Feature limits require 1 <= FEATURE_MIN <= FEATURE_MAX.")
    if int(EXPR_MAX_DEPTH) < 1:
        raise ValueError("EXPR_MAX_DEPTH must be >= 1.")
    if int(EXPR_MAX_LENGTH) < 1:
        raise ValueError("EXPR_MAX_LENGTH must be >= 1.")
    if not 0.0 <= float(FEATURE_MIN_VALID_RATIO) <= 1.0:
        raise ValueError("FEATURE_MIN_VALID_RATIO must be in [0, 1].")
    if not 0.0 <= float(FEATURE_MAX_DOMINANT_VALUE_RATIO) <= 1.0:
        raise ValueError("FEATURE_MAX_DOMINANT_VALUE_RATIO must be in [0, 1].")
    if ARCHIVE_SIZE < 1:
        raise ValueError("ARCHIVE_SIZE must be positive.")
    if not WINDOWS or any(int(w) <= 0 for w in WINDOWS):
        raise ValueError("WINDOWS must contain positive integers.")
    if len(set(WINDOWS)) != len(WINDOWS):
        raise ValueError("WINDOWS must not contain duplicates.")
    if not 0.0 <= CORR_THRESHOLD <= 1.0:
        raise ValueError("CORR_THRESHOLD must be in [0, 1].")
    if int(CORR_CHECK_MAX_ROWS) < 0:
        raise ValueError("CORR_CHECK_MAX_ROWS must be non-negative.")
    if int(DOMAIN_CORR_MAX_CHECKS) < 0:
        raise ValueError("DOMAIN_CORR_MAX_CHECKS must be non-negative.")
    if int(TRAINER_FEATURE_CACHE_MAX_ITEMS) < 0:
        raise ValueError("TRAINER_FEATURE_CACHE_MAX_ITEMS must be non-negative.")
    if int(TRAINER_SPLIT_CACHE_MAX_ITEMS) < 0:
        raise ValueError("TRAINER_SPLIT_CACHE_MAX_ITEMS must be non-negative.")

    if HOLDING_HORIZON < 1:
        raise ValueError("HOLDING_HORIZON must be positive.")
    if WF_MIN_TRAIN_MONTHS < 1 or WF_VAL_MONTHS < 1 or WF_STEP_MONTHS < 1:
        raise ValueError("WF month settings must be positive.")
    if WF_PURGE_DAYS < 0:
        raise ValueError("WF_PURGE_DAYS must be non-negative.")
    if WF_PURGE_DAYS < HOLDING_HORIZON:
        raise ValueError("WF_PURGE_DAYS must be >= HOLDING_HORIZON to avoid split leakage.")
    if int(WF_TRAIN_BAR_STRIDE) < 1:
        raise ValueError("WF_TRAIN_BAR_STRIDE must be >= 1.")
    if not 0.0 <= float(WF_EARLY_STOP_VALID_FRACTION) < 0.5:
        raise ValueError("WF_EARLY_STOP_VALID_FRACTION must be in [0, 0.5).")
    if int(WF_EARLY_STOP_MIN_VALID_DATES) < 1:
        raise ValueError("WF_EARLY_STOP_MIN_VALID_DATES must be >= 1.")
    if HIT_RATE_TOP_K < 1:
        raise ValueError("HIT_RATE_TOP_K must be positive.")
    if CHECKPOINT_EVERY_SECONDS < 0:
        raise ValueError("CHECKPOINT_EVERY_SECONDS must be non-negative.")
    if not isinstance(REQUIRE_SECTOR_MAPPING, bool):
        raise ValueError("REQUIRE_SECTOR_MAPPING must be a bool.")
    if int(MIN_SECTOR_MEMBERS_IN_UNIVERSE) < 1:
        raise ValueError("MIN_SECTOR_MEMBERS_IN_UNIVERSE must be >= 1.")

    _validate_sector_mapping()


def _parse_date_setting(name: str, value: str):
    import pandas as pd

    try:
        return pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a valid date string.") from exc


def _parse_optional_date_setting(name: str, value: str | None):
    if value is None:
        return None
    return _parse_date_setting(name, value)


def _validate_sector_mapping() -> None:
    if not isinstance(SECTORS, dict):
        raise ValueError("SECTORS must be a dict[str, list[str]].")

    seen: dict[str, str] = {}
    sector_counts: dict[str, int] = {}
    for sector_name, tickers in SECTORS.items():
        sector = str(sector_name).strip()
        if not sector:
            raise ValueError("SECTORS contains an empty sector name.")
        if not isinstance(tickers, (list, tuple, set)):
            raise ValueError(f"SECTORS[{sector!r}] must be a ticker list.")

        sector_counts.setdefault(sector, 0)
        for ticker in tickers:
            symbol = str(ticker).strip().upper()
            if not symbol:
                raise ValueError(f"SECTORS[{sector!r}] contains an empty ticker.")
            previous = seen.get(symbol)
            if previous is not None:
                raise ValueError(
                    f"Ticker {symbol!r} appears in multiple sectors: "
                    f"{previous!r} and {sector!r}."
                )
            seen[symbol] = sector
            sector_counts[sector] += 1

    if REQUIRE_SECTOR_MAPPING and int(MIN_SECTOR_MEMBERS_IN_UNIVERSE) > 1:
        small = {
            sector: count
            for sector, count in sector_counts.items()
            if count < int(MIN_SECTOR_MEMBERS_IN_UNIVERSE)
        }
        if small:
            raise ValueError(
                "Each sector must contain at least "
                f"{MIN_SECTOR_MEMBERS_IN_UNIVERSE} tickers; too small: {small}."
            )


validate_config()
