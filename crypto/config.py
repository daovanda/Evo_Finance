"""Configuration for the BTC/USDT evolutionary pipeline."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
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
LABEL_THRESHOLD: float = 0.0  # label=1 when future_return > threshold
LABEL_MODE: str = "close_path_mean"
LABEL_DIRECTION: str = "Long"  # "Long" => price up is favorable, "Short" => price down is favorable
PAYOFF_TP: float = 0.004  # only used by LABEL_MODE="payoff"
TP_SAFE_PATH: float = 0.003  # TP used by LABEL_MODE="safe_path_mfe"
SAFE_ADVERSE_FLOOR: float = -0.0015  # stop-first low/high floor for safe_path_mfe
SAFE_PATH_RULE: str = "adverse_stop_first_v1"


LABEL_DIRECTION_ALIASES: dict[str, str] = {
    "long": "long",
    "l": "long",
    "buy": "long",
    "short": "short",
    "s": "short",
    "sell": "short",
}


def canonical_label_direction(direction: str | None = None) -> str:
    selected = str(direction or LABEL_DIRECTION).strip().lower()
    selected = LABEL_DIRECTION_ALIASES.get(selected, selected)
    if selected not in {"long", "short"}:
        raise ValueError("LABEL_DIRECTION must be 'Long' or 'Short'.")
    return selected


def directional_price_return(price: Any, entry: Any, direction: str | None = None) -> Any:
    """Return price move in the selected trade direction.

    Long:  price / entry - 1
    Short: 1 - price / entry
    """
    try:
        ratio = price.div(entry, axis=0)
    except AttributeError:
        ratio = price / entry
    selected = canonical_label_direction(direction)
    if selected == "short":
        return 1.0 - ratio
    return ratio - 1.0


def close_exit_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> Any:
    """
    Default label return.

    future_return(t, h) = (close(t+h) - open(t+1)) / open(t+1)

    Select LABEL_MODE below when you want a different label family.
    """
    h = int(horizon)
    entry_open = df["open"].shift(-1)
    exit_close = df["close"].shift(-h)
    return directional_price_return(exit_close, entry_open, direction)


def mfe_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> Any:
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
    future_lows = pd.concat(
        [df["low"].shift(-offset) for offset in range(1, h + 1)],
        axis=1,
    )
    if canonical_label_direction(direction) == "short":
        min_low = future_lows.min(axis=1, skipna=False)
        return directional_price_return(min_low, entry_open, direction)
    max_high = future_highs.max(axis=1, skipna=False)
    return directional_price_return(max_high, entry_open, direction)


def safe_path_mfe_outcome(
    df: Any,
    horizon: int,
    adverse_floor: float | None = None,
    direction: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Return stop-first strategy returns and the first-hit safe-path label.

    For a signal at t, entry is open(t+1). The first horizon whose favorable
    extreme reaches ``TP_SAFE_PATH`` is a positive label only when the
    adverse extreme has remained strictly above ``adverse_floor`` through the
    first-hit candle:

        Long:  first_hit_h = first h where high_h / open_H1 - 1 >= TP_SAFE_PATH
               stopped when low_h / open_H1 - 1 <= adverse_floor
        Short: first_hit_h = first h where 1 - low_h / open_H1 >= TP_SAFE_PATH
               stopped when 1 - high_h / open_H1 <= adverse_floor

    The adverse extreme is evaluated before the favorable extreme on every
    candle. If both TP and stop are touched in one OHLC candle, the sample is
    conservatively treated as stopped and receives label 0.

    The returned strategy outcome is TP on a valid first hit, adverse_floor on
    a stop, or the final close return when neither event occurs. A complete
    h-candle path is required.
    """
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be positive for safe_path_mfe.")

    floor = float(SAFE_ADVERSE_FLOOR if adverse_floor is None else adverse_floor)
    entry_open = df["open"].shift(-1)
    future_highs = pd.concat(
        [df["high"].shift(-offset) for offset in range(1, h + 1)],
        axis=1,
    )
    future_lows = pd.concat(
        [df["low"].shift(-offset) for offset in range(1, h + 1)],
        axis=1,
    )
    future_closes = pd.concat(
        [df["close"].shift(-offset) for offset in range(1, h + 1)],
        axis=1,
    )
    if canonical_label_direction(direction) == "short":
        hit_returns = directional_price_return(future_lows, entry_open, direction)
        adverse_returns = directional_price_return(future_highs, entry_open, direction)
    else:
        hit_returns = directional_price_return(future_highs, entry_open, direction)
        adverse_returns = directional_price_return(future_lows, entry_open, direction)
    close_returns = directional_price_return(future_closes, entry_open, direction)
    complete_path = (
        entry_open.notna()
        & future_highs.notna().all(axis=1)
        & future_lows.notna().all(axis=1)
        & future_closes.notna().all(axis=1)
    )

    active = complete_path.copy()
    label = pd.Series(False, index=df.index, dtype="bool")
    outcome = pd.Series(float("nan"), index=df.index, dtype="float64")
    for position in range(h):
        stopped_now = active & adverse_returns.iloc[:, position].le(floor)
        outcome.loc[stopped_now] = floor
        active.loc[stopped_now] = False

        hit_now = active & hit_returns.iloc[:, position].ge(float(TP_SAFE_PATH))
        label.loc[hit_now] = True
        outcome.loc[hit_now] = float(TP_SAFE_PATH)
        active.loc[hit_now] = False

    outcome.loc[active] = close_returns.iloc[:, -1].loc[active]
    label_float = label.astype("float64").where(complete_path)
    return outcome.where(complete_path), label_float


def safe_path_mfe_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> Any:
    """Default-threshold strategy return for ``safe_path_mfe``."""
    outcome, _ = safe_path_mfe_outcome(df, horizon, direction=direction)
    return outcome


def close_path_mean_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> Any:
    """Mean future close-path return relative to the next-candle entry open.

    For a signal at t:

        entry = open(t+1)
        future_return(t, h) = mean(close(t+1)..close(t+h)) / entry - 1

    With LABEL_THRESHOLD=0, label=1 means the mean future close path lies
    above the trade entry. A complete h-candle path is required; incomplete
    rows at the end of the dataset remain NaN.
    """
    h = int(horizon)
    entry_open = df["open"].shift(-1)
    future_closes = pd.concat(
        [df["close"].shift(-offset) for offset in range(1, h + 1)],
        axis=1,
    )
    mean_future_close = future_closes.mean(axis=1, skipna=False)
    return directional_price_return(mean_future_close, entry_open, direction)


def exit_after_h1_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> Any:
    """
    Exit model evaluated after H1 has closed.

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
    future_lows = pd.concat(
        [df["low"].shift(-offset) for offset in range(1, h)],
        axis=1,
    )
    if canonical_label_direction(direction) == "short":
        min_low = future_lows.min(axis=1, skipna=False)
        return directional_price_return(min_low, entry_open, direction)
    max_high = future_highs.max(axis=1, skipna=False)
    return directional_price_return(max_high, entry_open, direction)


exit_all_future_return = exit_after_h1_future_return


def exit_after_h2_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> Any:
    """
    Exit model evaluated after H2 has closed.

    This mode is aligned to a row s that represents H2. The feature row may use
    data <= close(s), while the original trade entry remains open(s-1), i.e.
    open H1. H1/H2 hits are filtered out in crypto.data.add_binary_labels()
    using the active label threshold.

    For h=5:
        entry = open(s-1)
        future_return = (max(high(s+1), high(s+2), high(s+3)) - entry) / entry

    In base-signal coordinates, this is:
        base signal t, entry open(t+1), model row s=t+2,
        label=1 if no H1/H2 hit and there is a hit in H3-H5.
    """
    h = int(horizon)
    if h < 3:
        return pd.Series(pd.NA, index=df.index, dtype="float64")
    entry_open = df["open"].shift(1)
    future_highs = pd.concat(
        [df["high"].shift(-offset) for offset in range(1, h - 1)],
        axis=1,
    )
    future_lows = pd.concat(
        [df["low"].shift(-offset) for offset in range(1, h - 1)],
        axis=1,
    )
    if canonical_label_direction(direction) == "short":
        min_low = future_lows.min(axis=1, skipna=False)
        return directional_price_return(min_low, entry_open, direction)
    max_high = future_highs.max(axis=1, skipna=False)
    return directional_price_return(max_high, entry_open, direction)


def payoff_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> Any:
    """
    Strategy payoff label return.

    future_return(t, h) =
        PAYOFF_TP if max(high(t+1)..high(t+h)) reaches PAYOFF_TP,
        otherwise close_exit_future_return(t, h).

    This is a gross payoff. The payoff label should normally use
    default_label_threshold("payoff") == TRADE_COST, so label=1 means the
    rule's gross payoff is above the estimated round-trip cost.
    """
    mfe = mfe_future_return(df, horizon, direction=direction)
    close_return = close_exit_future_return(df, horizon, direction=direction)
    payoff = close_return.where(mfe < float(PAYOFF_TP), float(PAYOFF_TP))
    return payoff.where(mfe.notna() & close_return.notna())


LABEL_RETURN_FNS: dict[str, Callable[[Any, int], Any]] = {
    "close_exit": close_exit_future_return,
    "close_path_mean": close_path_mean_future_return,
    "mfe": mfe_future_return,
    "safe_path_mfe": safe_path_mfe_future_return,
    "payoff": payoff_future_return,
    "exit_after_h1": exit_after_h1_future_return,
    "exit_after_h2": exit_after_h2_future_return,
}

LABEL_MODE_ALIASES: dict[str, str] = {
    "exit_all": "exit_after_h1",
    "first_hit_safe_close": "safe_path_mfe",
    "safe_close": "safe_path_mfe",
}


def canonical_label_mode(mode: str | None = None) -> str:
    selected_mode = str(mode or LABEL_MODE).strip().lower()
    selected_mode = LABEL_MODE_ALIASES.get(selected_mode, selected_mode)
    if selected_mode not in LABEL_RETURN_FNS:
        allowed = ", ".join(sorted([*LABEL_RETURN_FNS, *LABEL_MODE_ALIASES]))
        raise ValueError(f"Unknown LABEL_MODE={selected_mode!r}. Allowed: {allowed}.")
    return selected_mode


def get_label_return_fn(mode: str | None = None) -> Callable[[Any, int], Any]:
    selected_mode = canonical_label_mode(mode)
    return LABEL_RETURN_FNS[selected_mode]


def default_label_threshold(mode: str | None = None, threshold: float | None = None) -> float:
    if threshold is not None:
        return float(threshold)
    selected_mode = canonical_label_mode(mode)
    if selected_mode == "payoff":
        return float(TRADE_COST)
    if selected_mode == "safe_path_mfe":
        return float(SAFE_ADVERSE_FLOOR)
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
# WINDOWS: list[int] = [3, 5, 10, 15, 30, 60, 120, 240, 480, 960, 1440]
FEATURE_MIN_VALID_RATIO: float = 0.70
FEATURE_MAX_DOMINANT_VALUE_RATIO: float = 0.985
FEATURE_CORR_THRESHOLD: float = 0.70
EXPR_MAX_DEPTH: int = 6
EXPR_MAX_LENGTH: int = 480
EXPR_MAX_ABS_QUANTILE: float = 50.0
# Each generated expression can hold one full-length float Series. Keep this
# cache bounded so long evolution runs do not grow RAM without limit.
EXPR_CACHE_MAX_ITEMS: int = 128
EVOLUTION_GC_EVERY: int = 25  # iterations; 0 disables explicit garbage collection

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
TRADE_TOP_FRACTION: float = 0.1

MIN_TRADES_PER_SPLIT: int = 20
TRADE_COST: float = 0.0005  # 0.1% Futures round-trip fee plus slippage allowance
RETURN_SCORE_SCALE: float = 0.01
BAD_AUC_THRESHOLD: float = 0.50

FITNESS_WEIGHTS: dict[str, float] = {
    "auc_edge": 0.40,
    "precision_excess": 0.30,  #old: 0.30
    "trade_return_score": 0.20, #old: 0.20
    "auc_std": -0.20, #old: -0.20
    "overfit_gap": -0.25, #old: -0.25
    "bad_fold_ratio": -0.30 #old: -0.30
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
    if not isfinite(float(LABEL_THRESHOLD)):
        raise ValueError("LABEL_THRESHOLD must be finite.")
    get_label_return_fn()
    canonical_label_direction()
    if PAYOFF_TP <= 0:
        raise ValueError("PAYOFF_TP must be positive.")
    if not isfinite(float(TP_SAFE_PATH)) or TP_SAFE_PATH <= 0:
        raise ValueError("TP_SAFE_PATH must be finite and positive.")
    if not isfinite(float(SAFE_ADVERSE_FLOOR)):
        raise ValueError("SAFE_ADVERSE_FLOOR must be finite.")
    if FEATURE_MIN < 1 or FEATURE_MAX < FEATURE_MIN:
        raise ValueError("Require 1 <= FEATURE_MIN <= FEATURE_MAX.")
    if EXPR_MAX_DEPTH < 1:
        raise ValueError("EXPR_MAX_DEPTH must be positive.")
    if EXPR_MAX_LENGTH < 20:
        raise ValueError("EXPR_MAX_LENGTH must be at least 20.")
    if EXPR_MAX_ABS_QUANTILE <= 0:
        raise ValueError("EXPR_MAX_ABS_QUANTILE must be positive.")
    if EXPR_CACHE_MAX_ITEMS < 1:
        raise ValueError("EXPR_CACHE_MAX_ITEMS must be positive.")
    if EVOLUTION_GC_EVERY < 0:
        raise ValueError("EVOLUTION_GC_EVERY must be non-negative.")
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
