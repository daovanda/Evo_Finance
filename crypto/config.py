"""Configuration for the BTC/USDT evolutionary pipeline."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd


# Data
DATA_PATH: Path = Path("data/crypto/BTCUSDT_15m.csv")
DATE_COLUMN: str = "date"

# Output
RESULTS_DIR: Path = Path("crypto/results")
DEFAULT_ARCHIVE_PATH: Path = RESULTS_DIR / "crypto_btc_archive.json"

# Multi-horizon binary labels. Edit this list freely, for example [3, 7, 10, 20].
HOLDING_HORIZONS: list[int] = [5]
LABEL_THRESHOLD: float = 0.004  # label=1 when future_return > threshold
LABEL_MODE: str = "mfe"  # "close_exit", "mfe", "payoff", or "exit_all"
PAYOFF_TP: float = 0.004  # only used by LABEL_MODE="payoff"


def close_exit_future_return(df: Any, horizon: int) -> Any:
    """
    Default label return.

    future_return(t, h) = (close(t+h) - open(t+1)) / open(t+1)

    Select LABEL_MODE below when you want a different label family.
    """
    h = int(horizon)
    entry_open = df["open"].shift(-1)
    exit_close = df["close"].shift(-h)
    return (exit_close - entry_open) / entry_open


def mfe_future_return(df: Any, horizon: int) -> Any:
    """
    Max favorable excursion label return.

    future_return(t, h) = (max(high(t+1)..high(t+h)) - open(t+1)) / open(t+1)
    """
    h = int(horizon)
    entry_open = df["open"].shift(-1)
    future_highs = pd.concat(
        [df["high"].shift(-offset) for offset in range(1, h + 1)],
        axis=1,
    )
    max_high = future_highs.max(axis=1, skipna=False)
    return (max_high - entry_open) / entry_open


def exit_all_future_return(df: Any, horizon: int) -> Any:
    """
    Exit-model MFE label return.

    This mode is aligned to a row that is evaluated after the entry/H1 candle
    has closed. The feature row may therefore use data <= close(t), while the
    trade entry anchor remains open(t). H1 hits are filtered out in
    crypto.data.add_binary_labels() using the active label threshold, because
    in live trading this exit model is only evaluated when the take profit was
    not reached during H1.

    future_return(t, h) = (max(high(t+1)..high(t+h-1)) - open(t)) / open(t)
    """
    h = int(horizon)
    if h < 2:
        return pd.Series(pd.NA, index=df.index, dtype="float64")
    entry_open = df["open"]
    future_highs = pd.concat(
        [df["high"].shift(-offset) for offset in range(1, h)],
        axis=1,
    )
    max_high = future_highs.max(axis=1, skipna=False)
    return (max_high - entry_open) / entry_open


def payoff_future_return(df: Any, horizon: int) -> Any:
    """
    Strategy payoff label return.

    future_return(t, h) =
        PAYOFF_TP if max(high(t+1)..high(t+h)) reaches PAYOFF_TP,
        otherwise close_exit_future_return(t, h).

    This is a gross payoff. The payoff label should normally use
    default_label_threshold("payoff") == TRADE_COST, so label=1 means the
    rule's gross payoff is above the estimated round-trip cost.
    """
    mfe = mfe_future_return(df, horizon)
    close_return = close_exit_future_return(df, horizon)
    payoff = close_return.where(mfe < float(PAYOFF_TP), float(PAYOFF_TP))
    return payoff.where(mfe.notna() & close_return.notna())


LABEL_RETURN_FNS: dict[str, Callable[[Any, int], Any]] = {
    "close_exit": close_exit_future_return,
    "mfe": mfe_future_return,
    "payoff": payoff_future_return,
    "exit_all": exit_all_future_return,
}


def get_label_return_fn(mode: str | None = None) -> Callable[[Any, int], Any]:
    selected_mode = str(mode or LABEL_MODE).strip().lower()
    if selected_mode not in LABEL_RETURN_FNS:
        allowed = ", ".join(sorted(LABEL_RETURN_FNS))
        raise ValueError(f"Unknown LABEL_MODE={selected_mode!r}. Allowed: {allowed}.")
    return LABEL_RETURN_FNS[selected_mode]


def default_label_threshold(mode: str | None = None, threshold: float | None = None) -> float:
    if threshold is not None:
        return float(threshold)
    selected_mode = str(mode or LABEL_MODE).strip().lower()
    if selected_mode == "payoff":
        return float(TRADE_COST)
    return float(LABEL_THRESHOLD)


LABEL_RETURN_FN: Callable[[Any, int], Any] = get_label_return_fn()

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
WINDOWS: list[int] = [1, 2, 3, 4, 5, 7, 10, 14, 20, 30, 40, 50, 60, 80, 120, 160, 240, 320, 400, 480]
FEATURE_MIN_VALID_RATIO: float = 0.70
FEATURE_MAX_DOMINANT_VALUE_RATIO: float = 0.985
FEATURE_CORR_THRESHOLD: float = 0.70
EXPR_MAX_DEPTH: int = 6
EXPR_MAX_LENGTH: int = 480
EXPR_MAX_ABS_QUANTILE: float = 50.0

# Individual/evolution knobs.
FEATURE_MIN: int = 4
FEATURE_MAX: int = 24
ARCHIVE_SIZE: int = 50
TIME_BUDGET_SECONDS: float = 3600.0
RESTART_PROB: float = 0.001
CHECKPOINT_EVERY_SECONDS: float = 12 * 60 * 60
MUTATOR_PROBS: dict[str, float] = {
    "c1": 0.40,  # add/remove a feature
    "c2": 0.35,  # change one window inside a feature
    "c3": 0.25,  # replace a gene by a transformed gene
}
MAX_RETRY: int = 5

# Fitness. RETURN_SCORE_SCALE normalizes mean trade return so that one metric
# cannot dominate merely by being on a wider numerical scale.
FITNESS_HORIZON_MODE: str = "mean"  # "mean" keeps old behavior; "ensemble" requires all H signals
TRADE_TOP_FRACTION: float = 0.2

MIN_TRADES_PER_SPLIT: int = 20
TRADE_COST: float = 0.002  # 0.2% breakeven round-trip cost per selected trade
RETURN_SCORE_SCALE: float = 0.01
BAD_AUC_THRESHOLD: float = 0.50

FITNESS_WEIGHTS: dict[str, float] = {
    "auc_edge": 0.40,
    "precision_excess": 0.50,  #old: 0.30
    "trade_return_score": 0.20,
    "auc_std": -0.20,
    "overfit_gap": -0.25,
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
    get_label_return_fn()
    if PAYOFF_TP <= 0:
        raise ValueError("PAYOFF_TP must be positive.")
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
    if FITNESS_HORIZON_MODE not in {"mean", "ensemble"}:
        raise ValueError("FITNESS_HORIZON_MODE must be 'mean' or 'ensemble'.")
    if MIN_TRADES_PER_SPLIT < 1:
        raise ValueError("MIN_TRADES_PER_SPLIT must be positive.")
    if TRADE_COST < 0:
        raise ValueError("TRADE_COST must be non-negative.")
    if RETURN_SCORE_SCALE <= 0:
        raise ValueError("RETURN_SCORE_SCALE must be positive.")
    if ARCHIVE_SIZE < 1:
        raise ValueError("ARCHIVE_SIZE must be positive.")
    if CHECKPOINT_EVERY_SECONDS < 0:
        raise ValueError("CHECKPOINT_EVERY_SECONDS must be non-negative.")
    required_mutators = {"c1", "c2", "c3"}
    if set(MUTATOR_PROBS) != required_mutators:
        raise ValueError(f"MUTATOR_PROBS keys must be {sorted(required_mutators)}.")
    if any(value < 0 for value in MUTATOR_PROBS.values()):
        raise ValueError("MUTATOR_PROBS values must be non-negative.")
    if abs(sum(MUTATOR_PROBS.values()) - 1.0) > 1e-9:
        raise ValueError("MUTATOR_PROBS must sum to 1.0.")
    if MAX_RETRY < 1:
        raise ValueError("MAX_RETRY must be positive.")
    if not 0.0 <= EARLY_STOP_VALID_FRACTION < 0.5:
        raise ValueError("EARLY_STOP_VALID_FRACTION must be in [0, 0.5).")
    if EARLY_STOP_MIN_VALID_SAMPLES < 1:
        raise ValueError("EARLY_STOP_MIN_VALID_SAMPLES must be positive.")
