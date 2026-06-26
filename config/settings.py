"""
Evo_Finance — Central Configuration
All tunable knobs live here.
"""

from __future__ import annotations
from typing import List, Callable

# ─── Data split — nhập ngày cụ thể ──────────────────────────────────────────
# TRAIN : [đầu data]   → VAL_START   (exclusive)
# VAL   : VAL_START    → TEST_START  (exclusive)
# TEST  : TEST_START   → TEST_END    (inclusive, None = hết data)

VAL_START:  str = "2023-05-12"   # ngày đầu tiên của val  = ngày kết thúc train
TEST_START: str = "2025-01-01"   # ngày đầu tiên của test = ngày kết thúc val
TEST_END:   str | None = None    # ngày cuối test (None = hết data)

# ─── Holding horizon ──────────────────────────────────────────────────────────

HOLDING_HORIZON: int = 10          # h days forward

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
WF_END: str = TEST_START
WF_MIN_TRAIN_MONTHS: int = 36
WF_VAL_MONTHS: int = 6
WF_STEP_MONTHS: int = 6
WF_PURGE_DAYS: int = HOLDING_HORIZON

# ─── Feature limits ───────────────────────────────────────────────────────────

FEATURE_MIN: int = 3
FEATURE_MAX: int = 30

# ─── Correlation threshold ────────────────────────────────────────────────────

CORR_THRESHOLD: float = 0.70       # used for domain & individual dedup

# Full startup precompute can be very slow once the domain contains many
# sector/market primitives. Keep it lazy by default; set True for debugging.
DOMAIN_PRECOMPUTE_ON_START: bool = False

# ─── Window whitelist ─────────────────────────────────────────────────────────

WINDOWS: List[int] = [3, 5, 10, 14, 20, 30, 60, 120]

DEFAULT_WINDOW: int = 1            # initial {O,H,C,L,V} window

# CSV ticker used as the broad market benchmark. The loader reads
# data/raw/{MARKET_INDEX_TICKER}.csv and exposes it as market_* columns.
MARKET_INDEX_TICKER: str = "VNINDEX"

# Edit this mapping before each run if you want a different sector universe.
# Tickers are matched case-insensitively by the evaluator.
SECTORS: dict[str, list[str]] = {
    "Banking":     ["ACB", "BID", "CTG", "HDB", "MBB", "STB", "TCB", "VCB", "VPB", "TPB"],
    "Real_Estate": ["KDH", "NVL", "VHM", "VIC", "VRE"],
    "Industry":    ["FPT", "GAS", "GVR", "HPG", "MSN", "PDR", "PLX", "POW"],
    "Consumer":    ["BVH", "MWG", "PNJ", "SAB", "SSI", "VJC", "VNM"],
}

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
LGBM_WF_EARLY_STOPPING: int = 20    # stop if fold val NDCG@10 doesn't improve

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
    if ARCHIVE_SIZE < 1:
        raise ValueError("ARCHIVE_SIZE must be positive.")
    if not WINDOWS or any(int(w) <= 0 for w in WINDOWS):
        raise ValueError("WINDOWS must contain positive integers.")
    if len(set(WINDOWS)) != len(WINDOWS):
        raise ValueError("WINDOWS must not contain duplicates.")
    if not 0.0 <= CORR_THRESHOLD <= 1.0:
        raise ValueError("CORR_THRESHOLD must be in [0, 1].")

    if HOLDING_HORIZON < 1:
        raise ValueError("HOLDING_HORIZON must be positive.")
    if WF_MIN_TRAIN_MONTHS < 1 or WF_VAL_MONTHS < 1 or WF_STEP_MONTHS < 1:
        raise ValueError("WF month settings must be positive.")
    if WF_PURGE_DAYS < 0:
        raise ValueError("WF_PURGE_DAYS must be non-negative.")
    if WF_PURGE_DAYS < HOLDING_HORIZON:
        raise ValueError("WF_PURGE_DAYS must be >= HOLDING_HORIZON to avoid split leakage.")
    if HIT_RATE_TOP_K < 1:
        raise ValueError("HIT_RATE_TOP_K must be positive.")
    if CHECKPOINT_EVERY_SECONDS < 0:
        raise ValueError("CHECKPOINT_EVERY_SECONDS must be non-negative.")


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


validate_config()
