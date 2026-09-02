"""Configuration for the BTC/USDT evolutionary pipeline."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Data
DATA_PATH: Path = Path("data/crypto/BTCUSDT_15m.csv")
MFE_ENTRY_M2_DATA_PATH: Path = Path("data/crypto/BTCUSDT_1m.csv")
MFE_ENTRY_M2_RULE: str = "entry_open_second_minute_path_from_m2_to_close_h_v1"
DATE_COLUMN: str = "date"

# Output
RESULTS_DIR: Path = Path("crypto/results")
DEFAULT_ARCHIVE_PATH: Path = RESULTS_DIR / "crypto_btc_archive.json"

# Multi-horizon binary labels. Edit this list freely, for example [3, 7, 10, 20].
HOLDING_HORIZONS: list[int] = [5]
LABEL_THRESHOLD: float = 0.0  # label=1 when future_return > threshold
LABEL_MODE: str = "close_path_mean"
LABEL_DIRECTION: str = (
    "Long"  # "Long" => price up is favorable, "Short" => price down is favorable
)

# Optional causal row filter applied after labels are built and before the
# existing chronological final/WF splits are materialized.
SAMPLE_FILTER: str = "none"
SAMPLE_FILTER_SLOPE_WINDOW: int = 3
SAMPLE_FILTER_SLOPE_PREVIOUS_WINDOWS: int = 3
SAMPLE_FILTER_SLOPE_RATIO: float = 0.15
SAMPLE_FILTER_SLOPE_RULE: str = "ols_log_close_max_previous_abs_v1"
PAYOFF_TP: float = 0.002  # legacy payoff archives were evolved with TP=0.40%
# Effectively disables the newer adverse-path filter for legacy payoff archives.
PAYOFF_ADVERSE_FLOOR: float = -999.0
TP_SAFE_PATH: float = 0.004  # TP used by LABEL_MODE="safe_path_mfe"
SAFE_ADVERSE_FLOOR: float = -0.0015  # stop-first low/high floor for safe_path_mfe
SAFE_PATH_RULE: str = "adverse_stop_first_v1"
SLOPE_LOOKBACK: int = 2
SLOPE_SLOWDOWN_THRESHOLD: float = 0.0002  # 0.03% log-OLS slope per candle
SLOPE_MIN_INITIAL: float = 0.0001  # 0.02% log-OLS slope per candle
SLOPE_PRICE_COLUMN: str = "high"
SLOPE_SLOWDOWN_RULE: str = "eligible_initial_slope_only_v2"
SLOPE_SLOWDOWN_ALL_RULE: str = "all_initial_slopes_expanded_path_v1"
# Fast moving-average slope reversal target. The current MA3 slope compares
# MA3(t) with MA3(t-2); its future value is observed two candles later.
MA_SLOPE_FAST_WINDOW: int = 3
MA_SLOPE_FAST_SHIFT: int = 2
MA_SLOPE_FUTURE_SHIFT: int = 2
MA_SLOPE_REVERSAL_RULE: str = "ma3_sign_reversal_future_shift_v2"
# Strict directional close path from entry open through every future close.
# Long requires open(H1) < close(H1) < ... < close(Horizon); Short uses the
# sign-symmetric decreasing chain. This is a classification-only target.
MONOTONIC_CLOSE_PATH_RULE: str = "strict_directional_close_chain_v1"
# Offline close-ZigZag target used by LABEL_MODE="bear". A candle is positive
# only when it lies strictly between a confirmed peak and trough whose decline
# satisfies both the duration and drop filters. These future-confirmed labels
# are targets only; feature construction still receives raw data up to row t.
BEAR_ZIGZAG_TOLERANCE: float = 0.003
BEAR_MIN_DROP: float = 0.004
BEAR_MIN_BARS: int = 5
BEAR_LABEL_RULE: str = "confirmed_close_zigzag_body_v1"
# Symmetric trough-to-peak body target used by LABEL_MODE="bull".
BULL_ZIGZAG_TOLERANCE: float = 0.003
BULL_MIN_RISE: float = 0.004
BULL_MIN_BARS: int = 5
BULL_LABEL_RULE: str = "confirmed_close_zigzag_body_v1"
# Exact confirmed pivot zones. Peak reuses the Bear swing definition and
# trough reuses the Bull swing definition; each valid pivot labels t-1..t+1.
PEAK_ZONE_RADIUS: int = 1
PEAK_LABEL_RULE: str = "confirmed_close_zigzag_peak_zone_v1"
TROUGH_ZONE_RADIUS: int = 1
TROUGH_LABEL_RULE: str = "confirmed_close_zigzag_trough_zone_v1"

# Pure quantile-regression target used by LABEL_MODE="quantile_trade".
# MFE is the maximum upward excursion, MAE the maximum downward excursion, and
# close is the signed final close return over each configured horizon.
# This mode predicts one selected quantile and does not turn
# that prediction into TP/SL, direction, EV, or a trading decision.
QUANTILE_TARGET: str = "mfe"  # "mfe", "mae", or "close"
QUANTILE_ALPHA: float = 0.80  # any finite quantile strictly between 0 and 1
QUANTILE_TRADE_RULE: str = "mfe_mae_close_shared_quantile_fitness_purged_stop_v11"
QUANTILE_BAD_COVERAGE_ERROR: float = 0.10
QUANTILE_BAD_SPEARMAN_IC: float = 0.0
QUANTILE_EXIT_RULE: str = "close_quantile_argmax_no_trade_v1"
# None uses TRADE_COST after all configuration values have been loaded.
QUANTILE_EXIT_MIN_RETURN: float | None = None

# OOF dynamic-TP classification targets used by the three meta label modes.
# The base archive is retrained inside every original walk-forward train fold;
# its MFE quantile prediction on the corresponding OOF validation rows becomes
# the per-row TP. The prediction is a target-construction input only and is not
# exposed as a feature to the evolved meta model.
META_LEARNER_BASE_ARCHIVE: Path = (
    RESULTS_DIR / "crypto_btc_5m_quantile_mfe_q20_h3_seed1_1h.json"
)
META_LEARNER_BASE_RANK: int = 1
META_LEARNER_MIN_PREDICTION: float = 0.0002
# Fixed return added to every base MFE prediction before constructing the
# dynamic TP label and hit return. Zero preserves the original behavior.
META_LEARNER_TP_OFFSET: float = 0.0
META_STRATEGY_STOP_LOSS: float = 0.0
META_LEARNER_META_VAL_FRACTION: float = 0.20
META_LEARNER_RULE: str = "oof_mfe_dynamic_tp_binary_v1"
META_CLOSE_EXIT_RULE: str = "oof_mfe_dynamic_tp_close_exit_binary_v1"
META_STRATEGY_PROFIT_RULE: str = (
    "oof_mfe_dynamic_tp_strategy_profit_fixed_sl_stop_first_binary_v2"
)
META_LEARNER_LABEL_MODES: frozenset[str] = frozenset(
    {"meta_learner", "meta_close_exit", "meta_strategy_profit"}
)
META_PREDICTION_FEATURE_MODES: frozenset[str] = frozenset(
    {"meta_close_exit", "meta_strategy_profit"}
)
# Optional lower-timeframe feature source. None keeps the original behavior
# where base targets and meta features use the same OHLCV file.
META_LEARNER_FEATURE_DATA: Path | None = None
# Number of lower-timeframe candles observed after open H1 before prediction.
# Their highs remain part of the original H1..H target in this test mode.
META_LEARNER_FEATURE_LOOKAHEAD_BARS: int = 0
# Observe all lower-timeframe candles inside H1; TP-hit targets then start at H2.
META_LEARNER_FEATURE_INCLUDE_H1: bool = False
META_LEARNER_FEATURE_ALIGNMENT_RULE: str = (
    "target_open_plus_target_interval_minus_feature_interval_v1"
)
META_LEARNER_TARGET_START_STEP: int = 1
META_LEARNER_TARGET_INTERVAL_SECONDS: float | None = None
META_LEARNER_FEATURE_INTERVAL_SECONDS: float | None = None

# OOF Bull/Bear regime-exit meta learner. The two directional archives are
# retrained inside every original WF train block. Their exclusive signals
# define persistent position episodes; an early meta exit locks that episode
# until the corresponding base signal turns off.
META_REGIME_EXIT_BULL_ARCHIVE: Path = (
    RESULTS_DIR / "crypto_btc_5m_bull_w60_top40_seed1_8h.json"
)
META_REGIME_EXIT_BEAR_ARCHIVE: Path = (
    RESULTS_DIR / "crypto_btc_5m_bear_w60_top40_seed1_8h.json"
)
META_REGIME_EXIT_BULL_RANK: int = 1
META_REGIME_EXIT_BEAR_RANK: int = 1
META_REGIME_EXIT_BASE_TOP_FRACTION: float | None = None
META_REGIME_EXIT_THRESHOLD: float = 0.001
META_REGIME_EXIT_RULE: str = (
    "oof_bull_bear_intrabar_directional_close_episode_lock_v1"
)
META_REGIME_EXIT_LABEL_MODES: frozenset[str] = frozenset({"meta_regime_exit"})
# OOF Bull/Bear episode-entry filter. One sample is created only where the
# executable no-reverse strategy can open a new position. The target is 1 when
# that complete trade is net-positive after the optional fixed SL and cost.
META_REGIME_ENTRY_STOP_LOSS: float = 0.0015
META_REGIME_ENTRY_RULE: str = (
    "oof_bull_bear_episode_start_net_win_fixed_sl_lock_v1"
)
META_REGIME_ENTRY_LABEL_MODES: frozenset[str] = frozenset({"meta_regime_entry"})


def canonical_quantile_target(target: str | None = None) -> str:
    selected = str(target or QUANTILE_TARGET).strip().lower()
    if selected not in {"mfe", "mae", "close"}:
        raise ValueError("QUANTILE_TARGET must be 'mfe', 'mae', or 'close'.")
    return selected


def validate_quantile_alpha(alpha: float | None = None) -> float:
    selected = float(QUANTILE_ALPHA if alpha is None else alpha)
    if not isfinite(selected) or not 0.0 < selected < 1.0:
        raise ValueError("QUANTILE_ALPHA must be finite and strictly between 0 and 1.")
    return selected


def is_meta_learner_label_mode(mode: str | None = None) -> bool:
    """Return whether a mode uses the OOF dynamic-TP meta pipeline."""
    return canonical_label_mode(mode) in META_LEARNER_LABEL_MODES


def is_meta_regime_exit_label_mode(mode: str | None = None) -> bool:
    """Return whether a mode uses the Bull/Bear episode-exit pipeline."""
    return canonical_label_mode(mode) in META_REGIME_EXIT_LABEL_MODES


def is_meta_regime_entry_label_mode(mode: str | None = None) -> bool:
    """Return whether a mode filters executable Bull/Bear episode entries."""
    return canonical_label_mode(mode) in META_REGIME_ENTRY_LABEL_MODES


def meta_learner_rule(mode: str | None = None) -> str:
    """Return the target-construction policy recorded in archive metadata."""
    selected = canonical_label_mode(mode)
    rules = {
        "meta_learner": META_LEARNER_RULE,
        "meta_close_exit": META_CLOSE_EXIT_RULE,
        "meta_strategy_profit": META_STRATEGY_PROFIT_RULE,
    }
    if selected not in rules:
        raise ValueError(f"LABEL_MODE={selected!r} does not use the meta pipeline.")
    return rules[selected]


def meta_prediction_is_feature(mode: str | None = None) -> bool:
    """Return whether the leakage-safe OOF dynamic TP is a model feature."""
    return canonical_label_mode(mode) in META_PREDICTION_FEATURE_MODES


# Decision candle for LABEL_MODE="exit_after_k". For a base trade whose entry
# is open H1, k=1 evaluates after H1 closes, k=2 after H2 closes, and so on.
EXIT_AFTER_K: int = 1


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


def directional_price_return(
    price: Any, entry: Any, direction: str | None = None
) -> Any:
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


def monotonic_close_path_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> pd.Series:
    """Return a strict monotonic-close-path binary target.

    At signal row ``t``, entry is ``open(t+1)``. Long is positive only when
    every future close is strictly above the preceding price, starting with
    ``close(t+1) > open(t+1)``. Short applies the symmetric strict decrease.
    Rows without the complete H1..Horizon path remain NaN.
    """
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be positive for monotonic_close_path.")

    previous = pd.to_numeric(df["open"], errors="coerce").shift(-1)
    complete = previous.notna()
    monotonic = pd.Series(True, index=df.index, dtype=bool)
    is_short = canonical_label_direction(direction) == "short"
    for step in range(1, h + 1):
        current = pd.to_numeric(df["close"], errors="coerce").shift(-step)
        complete &= current.notna()
        monotonic &= current.lt(previous) if is_short else current.gt(previous)
        previous = current

    return monotonic.astype("float64").where(complete)


def high_exit_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> Any:
    """Favorable extreme of the exact exit candle relative to entry open.

    For a signal at t, entry is open(t+1). Long uses high(t+h), while Short
    uses low(t+h) and converts the move into a positive directional return.
    Unlike ``mfe_future_return``, this does not take an extreme over H1..Hh.
    """
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be positive for high_exit.")
    entry_open = df["open"].shift(-1)
    exit_price = (
        df["low"].shift(-h)
        if canonical_label_direction(direction) == "short"
        else df["high"].shift(-h)
    )
    return directional_price_return(exit_price, entry_open, direction)


def _rolling_log_ols_slope(price: pd.Series, window: int) -> pd.Series:
    """OLS log-price slope converted to an approximate return per candle."""
    width = int(window)
    if width < 2:
        raise ValueError("slope window must be at least 2.")

    numeric = pd.to_numeric(price, errors="coerce").to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_price = np.log(numeric)

    x = np.arange(width, dtype="float64")
    centered_x = x - x.mean()
    denominator = float(np.square(centered_x).sum())
    result = np.full(len(log_price), np.nan, dtype="float64")
    if len(log_price) < width:
        return pd.Series(result, index=price.index, dtype="float64")

    beta = np.convolve(log_price, centered_x[::-1], mode="valid") / denominator
    finite_count = np.convolve(
        np.isfinite(log_price).astype("int64"),
        np.ones(width, dtype="int64"),
        mode="valid",
    )
    beta[finite_count != width] = np.nan
    result[width - 1 :] = np.expm1(beta)
    return pd.Series(result, index=price.index, dtype="float64")


def slope_slowdown_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> pd.Series:
    """Directional slope-slowdown strength based only on candle highs.

    At decision row t, the initial OLS slope uses high(t-lookback+1..t).
    The expanded OLS slope uses high(t-lookback+1..t+h), so the future h
    candles must be strong enough to change the trend fitted over the complete
    path. Both slopes are fitted to log(high) and expressed as return per bar.

    Long identifies a rising high-price trend that slows:
        initial_slope > SLOPE_MIN_INITIAL
        strength = initial_slope - expanded_slope

    Short identifies a falling high-price trend that recovers:
        initial_slope < -SLOPE_MIN_INITIAL
        strength = expanded_slope - initial_slope

    Rows without the required initial direction receive NaN and are excluded
    from training/evaluation. This prevents the model from earning easy
    discrimination by merely relearning the observable initial-slope gate.
    The caller labels eligible rows where strength exceeds the active label
    threshold.
    """
    h = int(horizon)
    lookback = int(SLOPE_LOOKBACK)
    if h < 1:
        raise ValueError("horizon must be positive for slope_slowdown.")
    if lookback < 2:
        raise ValueError("SLOPE_LOOKBACK must be at least 2.")
    if SLOPE_PRICE_COLUMN != "high":
        raise ValueError("SLOPE_PRICE_COLUMN must remain 'high'.")

    high = pd.to_numeric(df[SLOPE_PRICE_COLUMN], errors="coerce")
    initial_slope = _rolling_log_ols_slope(high, lookback)
    expanded_at_end = _rolling_log_ols_slope(high, lookback + h)
    expanded_at_t = expanded_at_end.shift(-h)
    complete = initial_slope.notna() & expanded_at_t.notna()

    if canonical_label_direction(direction) == "short":
        eligible = initial_slope.lt(-float(SLOPE_MIN_INITIAL))
        strength = expanded_at_t - initial_slope
    else:
        eligible = initial_slope.gt(float(SLOPE_MIN_INITIAL))
        strength = initial_slope - expanded_at_t

    return strength.where(eligible & complete)


def slope_slowdown_all_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> pd.Series:
    """Slope change over the expanded future path without an initial gate.

    Long measures ``initial_slope - expanded_slope`` and therefore labels a
    sufficiently large downward pull in slope. Short uses the symmetric
    ``expanded_slope - initial_slope`` change. Unlike ``slope_slowdown``, all
    rows with complete lookback and future data remain eligible regardless of
    the sign or magnitude of the observable initial slope.
    """
    h = int(horizon)
    lookback = int(SLOPE_LOOKBACK)
    if h < 1:
        raise ValueError("horizon must be positive for slope_slowdown_all.")
    if lookback < 2:
        raise ValueError("SLOPE_LOOKBACK must be at least 2.")
    if SLOPE_PRICE_COLUMN != "high":
        raise ValueError("SLOPE_PRICE_COLUMN must remain 'high'.")

    high = pd.to_numeric(df[SLOPE_PRICE_COLUMN], errors="coerce")
    initial_slope = _rolling_log_ols_slope(high, lookback)
    expanded_at_end = _rolling_log_ols_slope(high, lookback + h)
    expanded_at_t = expanded_at_end.shift(-h)
    complete = initial_slope.notna() & expanded_at_t.notna()
    if canonical_label_direction(direction) == "short":
        strength = expanded_at_t - initial_slope
    else:
        strength = initial_slope - expanded_at_t
    return strength.where(complete)


def ma_slope_reversal_labels(
    df: Any,
    direction: str | None = None,
) -> pd.Series:
    """Label a future sign reversal of the fast close-MA slope.

    Long follows the requested peak-style definition::

        fast_slope(t) > 0
        fast_slope(t + future_shift) < 0

    Short is its sign-symmetric counterpart. Moving averages and slopes use
    only close prices; future data is used solely by this supervised target.
    Warm-up rows and rows without the complete future shift remain NaN.
    """
    fast_window = int(MA_SLOPE_FAST_WINDOW)
    fast_shift = int(MA_SLOPE_FAST_SHIFT)
    future_shift = int(MA_SLOPE_FUTURE_SHIFT)
    if fast_window < 1:
        raise ValueError("MA slope fast window must be positive.")
    if min(fast_shift, future_shift) < 1:
        raise ValueError("MA slope shifts must be positive.")

    close = pd.to_numeric(df["close"], errors="coerce")
    fast_ma = close.rolling(fast_window, min_periods=fast_window).mean()
    fast_slope = fast_ma - fast_ma.shift(fast_shift)
    future_fast_slope = fast_slope.shift(-future_shift)
    complete = fast_slope.notna() & future_fast_slope.notna()

    if canonical_label_direction(direction) == "short":
        reversal = fast_slope.lt(0.0) & future_fast_slope.gt(0.0)
    else:
        reversal = fast_slope.gt(0.0) & future_fast_slope.lt(0.0)
    return reversal.astype("float64").where(complete)


def ma_slope_reversal_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> pd.Series:
    """Registry adapter for the horizon-neutral MA slope-reversal target."""
    del horizon
    return ma_slope_reversal_labels(df, direction=direction)


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


def mfe_entry_m2_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> pd.Series:
    """MFE from open minute 2 of H1 through the end of H."""
    selected = canonical_label_direction(direction)
    column = f"mfe_entry_m2_{selected}_h{int(horizon)}"
    if column not in df:
        raise ValueError(
            f"Missing {column!r}; provide --mfe-entry-data with 1m OHLCV."
        )
    return pd.to_numeric(df[column], errors="coerce")


def mfe_entry_m2_close_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> pd.Series:
    """Final close-H return relative to open minute 2 of H1."""
    selected = canonical_label_direction(direction)
    column = f"close_entry_m2_{selected}_h{int(horizon)}"
    if column not in df:
        raise ValueError(
            f"Missing {column!r}; provide --mfe-entry-data with 1m OHLCV."
        )
    return pd.to_numeric(df[column], errors="coerce")


def mfe_ahead_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> Any:
    """Look-ahead MFE measured from the current candle's open.

    The model row represents H1 and may use all information through close H1.
    Entry remains open H1, and an h-candle target spans H1..Hh:

        Long:  max(high(H1)..high(Hh)) / open(H1) - 1
        Short: open(H1) / min(low(H1)..low(Hh)) - 1

    This intentionally exposes H1 to the model and must not be interpreted as
    a causal entry-at-open strategy.
    """
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be positive for mfe_ahead.")

    entry_open = pd.to_numeric(df["open"], errors="coerce")
    future_highs = pd.concat(
        [df["high"].shift(-offset) for offset in range(0, h)],
        axis=1,
    )
    future_lows = pd.concat(
        [df["low"].shift(-offset) for offset in range(0, h)],
        axis=1,
    )
    if canonical_label_direction(direction) == "short":
        favorable_price = future_lows.min(axis=1, skipna=False)
    else:
        favorable_price = future_highs.max(axis=1, skipna=False)
    return directional_price_return(favorable_price, entry_open, direction)


def mfe_ahead_close_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> Any:
    """Directional close-Hh return relative to the current H1 open."""
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be positive for mfe_ahead.")
    entry_open = pd.to_numeric(df["open"], errors="coerce")
    final_close = pd.to_numeric(df["close"], errors="coerce").shift(-(h - 1))
    return directional_price_return(final_close, entry_open, direction)


def adverse_floor_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> Any:
    """Worst adverse excursion from the next-candle entry open.

    Long uses the minimum low from H1 through Hh. Short uses the maximum
    high over the same path and converts it to a directional return. The
    result is normally non-positive; ``crypto.data.add_binary_labels`` uses
    it only to build the adverse-floor label and passes a zero return to
    fitness.
    """
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be positive for adverse_floor.")

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
        adverse_price = future_highs.max(axis=1, skipna=False)
    else:
        adverse_price = future_lows.min(axis=1, skipna=False)
    return directional_price_return(adverse_price, entry_open, direction)


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


def exit_after_k_outcome(
    df: Any,
    horizon: int,
    threshold: float,
    direction: str | None = None,
    exit_after_k: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Executable TP-or-final-close payoff after decision candle Hk.

    The row s used by the model represents base trade candle Hk. Features may
    use data through close(s), while the original trade entry remains open H1:

        entry = open(s-k+1)
        label = remaining MFE over H(k+1)..Hh reaches threshold
        payoff = threshold on a hit, otherwise close Hh versus entry

    Earlier H1..Hk TP hits are excluded in crypto.data.add_binary_labels().
    A useful exit decision requires 1 <= k < horizon.
    """
    h = int(horizon)
    k = int(EXIT_AFTER_K if exit_after_k is None else exit_after_k)
    tp = float(threshold)
    if k < 1:
        raise ValueError("exit_after_k must be at least 1.")
    if not isfinite(tp) or tp <= 0.0:
        raise ValueError("exit_after_k label_threshold must be finite and positive.")
    if h <= k:
        empty = pd.Series(np.nan, index=df.index, dtype="float64")
        return empty.copy(), empty

    entry_open = df["open"].shift(k - 1)
    future_highs = pd.concat(
        [df["high"].shift(-offset) for offset in range(1, h - k + 1)],
        axis=1,
    )
    future_lows = pd.concat(
        [df["low"].shift(-offset) for offset in range(1, h - k + 1)],
        axis=1,
    )
    final_close = df["close"].shift(-(h - k))
    if canonical_label_direction(direction) == "short":
        favorable_price = future_lows.min(axis=1, skipna=False)
    else:
        favorable_price = future_highs.max(axis=1, skipna=False)

    remaining_mfe = directional_price_return(
        favorable_price,
        entry_open,
        direction,
    )
    close_return = directional_price_return(final_close, entry_open, direction)
    complete = (
        entry_open.notna()
        & future_highs.notna().all(axis=1)
        & future_lows.notna().all(axis=1)
        & final_close.notna()
    )
    hit = remaining_mfe.ge(tp)
    payoff = close_return.where(~hit, tp).where(complete)
    label = hit.astype("float64").where(complete)
    return payoff, label


def exit_after_k_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
    exit_after_k: int | None = None,
) -> Any:
    """Default-threshold payoff adapter for the label-mode registry."""
    payoff, _ = exit_after_k_outcome(
        df,
        horizon,
        threshold=float(LABEL_THRESHOLD),
        direction=direction,
        exit_after_k=exit_after_k,
    )
    return payoff


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


def two_sided_tp_outcome(
    df: Any,
    horizon: int,
    threshold: float,
) -> tuple[pd.Series, pd.Series]:
    """Gross payoff and label for simultaneous Long and Short positions.

    Both positions use ``open(t+1)`` as entry and the same absolute TP. The
    position that does not hit TP is closed at ``close(t+h)``. If neither side
    hits, the two close returns cancel, so the combined gross payoff is zero.
    Direction is deliberately absent because this strategy trades both sides.
    """
    h = int(horizon)
    tp = float(threshold)
    if h < 1:
        raise ValueError("horizon must be positive for two_sided_tp.")
    if not isfinite(tp) or tp <= 0.0:
        raise ValueError("two_sided_tp label_threshold must be finite and positive.")

    entry_open = df["open"].shift(-1)
    future_highs = pd.concat(
        [df["high"].shift(-offset) for offset in range(1, h + 1)],
        axis=1,
    )
    future_lows = pd.concat(
        [df["low"].shift(-offset) for offset in range(1, h + 1)],
        axis=1,
    )
    max_high_return = future_highs.max(axis=1, skipna=False).div(entry_open) - 1.0
    min_low_return = future_lows.min(axis=1, skipna=False).div(entry_open) - 1.0
    close_return = df["close"].shift(-h).div(entry_open) - 1.0
    complete = (
        entry_open.notna()
        & max_high_return.notna()
        & min_low_return.notna()
        & close_return.notna()
    )

    up_hit = max_high_return.gt(tp)
    down_hit = min_low_return.lt(-tp)
    both_hit = up_hit & down_hit
    payoff = pd.Series(0.0, index=df.index, dtype="float64")
    payoff = payoff.mask(up_hit & ~down_hit, tp - close_return)
    payoff = payoff.mask(~up_hit & down_hit, tp + close_return)
    payoff = payoff.mask(both_hit, 2.0 * tp).where(complete)
    label = both_hit.astype("float64").where(complete)
    return payoff, label


def two_sided_tp_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> Any:
    """Default-threshold return adapter for the label-mode registry."""
    del direction
    payoff, _ = two_sided_tp_outcome(df, horizon, float(LABEL_THRESHOLD))
    return payoff


def _confirmed_close_zigzag_pivots(
    df: Any,
    tolerance: float,
) -> tuple[np.ndarray, list[tuple[int, float, str]]]:
    """Return close prices and reversal-confirmed ZigZag pivots."""
    close = pd.to_numeric(df["close"], errors="coerce")
    prices = close.to_numpy(dtype="float64")
    if len(prices) < 2:
        return prices, []

    pivots: list[tuple[int, float, str]] = []
    direction: str | None = None
    anchor_price = float(prices[0])
    candidate_idx = 0
    candidate_price = anchor_price

    for idx in range(1, len(prices)):
        price = float(prices[idx])
        if not isfinite(price):
            continue

        if direction is None:
            relative_move = price / anchor_price - 1.0
            if relative_move >= tolerance:
                direction = "up"
                candidate_idx, candidate_price = idx, price
            elif relative_move <= -tolerance:
                direction = "down"
                candidate_idx, candidate_price = idx, price
            continue

        if direction == "up":
            if price >= candidate_price:
                candidate_idx, candidate_price = idx, price
            elif price <= candidate_price * (1.0 - tolerance):
                pivots.append((candidate_idx, candidate_price, "peak"))
                direction = "down"
                candidate_idx, candidate_price = idx, price
        else:
            if price <= candidate_price:
                candidate_idx, candidate_price = idx, price
            elif price >= candidate_price * (1.0 + tolerance):
                pivots.append((candidate_idx, candidate_price, "trough"))
                direction = "up"
                candidate_idx, candidate_price = idx, price

    return prices, pivots


def _confirmed_zigzag_body_labels(
    df: Any,
    *,
    tolerance: float,
    min_move: float,
    min_bars: int,
    start_kind: str,
) -> pd.Series:
    """Label bars strictly inside valid confirmed peak/trough swing bodies."""
    prices, pivots = _confirmed_close_zigzag_pivots(df, float(tolerance))
    labels = np.zeros(len(prices), dtype="float64")
    end_kind = "trough" if start_kind == "peak" else "peak"

    for start, end in zip(pivots, pivots[1:]):
        if start[2] != start_kind or end[2] != end_kind:
            continue
        bars = int(end[0] - start[0])
        if start_kind == "peak":
            move = float((start[1] - end[1]) / start[1])
        else:
            move = float((end[1] - start[1]) / start[1])
        if bars >= int(min_bars) and move >= float(min_move):
            labels[start[0] + 1 : end[0]] = 1.0

    labels[~np.isfinite(prices)] = np.nan
    return pd.Series(labels, index=df.index, dtype="float64")


def bear_body_labels(df: Any) -> pd.Series:
    """Return the offline confirmed peak-to-trough ZigZag body target.

    Future closes confirm the supervised target only. Model features at row t
    are built separately from raw OHLCV available through t. Peak/trough bars
    remain 0, and the unfinished final ZigZag leg is not emitted as a pivot.
    """
    return _confirmed_zigzag_body_labels(
        df,
        tolerance=float(BEAR_ZIGZAG_TOLERANCE),
        min_move=float(BEAR_MIN_DROP),
        min_bars=int(BEAR_MIN_BARS),
        start_kind="peak",
    )


def bull_body_labels(df: Any) -> pd.Series:
    """Return the offline confirmed trough-to-peak ZigZag body target.

    Label 1 is assigned strictly between each adjacent trough and peak whose
    rise and duration meet ``BULL_MIN_RISE`` and ``BULL_MIN_BARS``. The target
    is future-confirmed, while prediction features at t remain causal.
    """
    return _confirmed_zigzag_body_labels(
        df,
        tolerance=float(BULL_ZIGZAG_TOLERANCE),
        min_move=float(BULL_MIN_RISE),
        min_bars=int(BULL_MIN_BARS),
        start_kind="trough",
    )


def _confirmed_zigzag_pivot_zone_labels(
    df: Any,
    *,
    tolerance: float,
    min_move: float,
    min_bars: int,
    pivot_kind: str,
    zone_radius: int,
) -> pd.Series:
    """Label a fixed zone around pivots whose following swing is valid."""
    if pivot_kind not in {"peak", "trough"}:
        raise ValueError("pivot_kind must be 'peak' or 'trough'.")
    if int(zone_radius) < 0:
        raise ValueError("zone_radius must be non-negative.")

    prices, pivots = _confirmed_close_zigzag_pivots(df, float(tolerance))
    labels = np.zeros(len(prices), dtype="float64")
    end_kind = "trough" if pivot_kind == "peak" else "peak"
    radius = int(zone_radius)

    for start, end in zip(pivots, pivots[1:]):
        if start[2] != pivot_kind or end[2] != end_kind:
            continue
        bars = int(end[0] - start[0])
        if pivot_kind == "peak":
            move = float((start[1] - end[1]) / start[1])
        else:
            move = float((end[1] - start[1]) / start[1])
        if bars >= int(min_bars) and move >= float(min_move):
            left = max(0, int(start[0]) - radius)
            right = min(len(labels), int(start[0]) + radius + 1)
            labels[left:right] = 1.0

    labels[~np.isfinite(prices)] = np.nan
    return pd.Series(labels, index=df.index, dtype="float64")


def peak_zone_labels(df: Any) -> pd.Series:
    """Label t-1..t+1 around valid Bear-definition ZigZag peaks."""
    return _confirmed_zigzag_pivot_zone_labels(
        df,
        tolerance=float(BEAR_ZIGZAG_TOLERANCE),
        min_move=float(BEAR_MIN_DROP),
        min_bars=int(BEAR_MIN_BARS),
        pivot_kind="peak",
        zone_radius=int(PEAK_ZONE_RADIUS),
    )


def trough_zone_labels(df: Any) -> pd.Series:
    """Label t-1..t+1 around valid Bull-definition ZigZag troughs."""
    return _confirmed_zigzag_pivot_zone_labels(
        df,
        tolerance=float(BULL_ZIGZAG_TOLERANCE),
        min_move=float(BULL_MIN_RISE),
        min_bars=int(BULL_MIN_BARS),
        pivot_kind="trough",
        zone_radius=int(TROUGH_ZONE_RADIUS),
    )


def bear_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> pd.Series:
    """Registry adapter; bear labels are horizon- and direction-neutral."""
    del horizon, direction
    return bear_body_labels(df)


def bull_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> pd.Series:
    """Registry adapter; bull labels are horizon- and direction-neutral."""
    del horizon, direction
    return bull_body_labels(df)


def peak_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> pd.Series:
    """Registry adapter; peak zones are horizon- and direction-neutral."""
    del horizon, direction
    return peak_zone_labels(df)


def trough_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> pd.Series:
    """Registry adapter; trough zones are horizon- and direction-neutral."""
    del horizon, direction
    return trough_zone_labels(df)


def quantile_trade_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> pd.Series:
    """Registry adapter for the non-binary quantile_trade pipeline.

    The dedicated data branch creates all distributional targets. Returning
    close H here keeps the mode registry's callable contract intact; the
    binary fitness evaluator never consumes this adapter for quantile_trade.
    """
    del direction
    entry_open = df["open"].shift(-1)
    return df["close"].shift(-int(horizon)).div(entry_open).sub(1.0)


def meta_learner_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> pd.Series:
    """Registry adapter for the OOF-only meta data paths.

    Direct label construction is intentionally rejected: this target requires
    predictions from a separately trained base quantile-MFE archive.
    """
    del df, horizon, direction
    raise ValueError(
        "Meta labels require OOF base-MFE predictions and must be constructed "
        "through crypto.main with a meta label mode."
    )


def meta_regime_exit_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> pd.Series:
    """Registry adapter for the OOF Bull/Bear regime-exit data path."""
    del df, horizon, direction
    raise ValueError(
        "meta_regime_exit requires OOF Bull/Bear predictions and lower-timeframe "
        "execution data; construct it through crypto.main."
    )


def meta_regime_entry_future_return(
    df: Any,
    horizon: int,
    direction: str | None = None,
) -> pd.Series:
    """Registry adapter for the OOF Bull/Bear episode-entry data path."""
    del df, horizon, direction
    raise ValueError(
        "meta_regime_entry requires OOF Bull/Bear predictions and executable "
        "episode outcomes; construct it through crypto.main."
    )


LABEL_RETURN_FNS: dict[str, Callable[[Any, int], Any]] = {
    "close_exit": close_exit_future_return,
    "monotonic_close_path": monotonic_close_path_future_return,
    "high_exit": high_exit_future_return,
    "slope_slowdown": slope_slowdown_future_return,
    "slope_slowdown_all": slope_slowdown_all_future_return,
    "ma_slope_reversal": ma_slope_reversal_future_return,
    "close_path_mean": close_path_mean_future_return,
    "mfe": mfe_future_return,
    "mfe_entry_m2": mfe_entry_m2_future_return,
    "mfe_ahead": mfe_ahead_future_return,
    "adverse_floor": adverse_floor_future_return,
    "safe_path_mfe": safe_path_mfe_future_return,
    "payoff": payoff_future_return,
    "two_sided_tp": two_sided_tp_future_return,
    "exit_after_k": exit_after_k_future_return,
    "bear": bear_future_return,
    "bull": bull_future_return,
    "peak": peak_future_return,
    "trough": trough_future_return,
    "quantile_trade": quantile_trade_future_return,
    "quantile_exit": quantile_trade_future_return,
    "meta_learner": meta_learner_future_return,
    "meta_close_exit": meta_learner_future_return,
    "meta_strategy_profit": meta_learner_future_return,
    "meta_regime_exit": meta_regime_exit_future_return,
    "meta_regime_entry": meta_regime_entry_future_return,
}

LABEL_MODE_ALIASES: dict[str, str] = {
    "first_hit_safe_close": "safe_path_mfe",
    "safe_close": "safe_path_mfe",
    "ma3_slope_reversal": "ma_slope_reversal",
    "slope_reversal": "ma_slope_reversal",
}


def canonical_label_mode(mode: str | None = None) -> str:
    selected_mode = str(mode or LABEL_MODE).strip().lower()
    selected_mode = LABEL_MODE_ALIASES.get(selected_mode, selected_mode)
    if selected_mode not in LABEL_RETURN_FNS:
        allowed = ", ".join(sorted([*LABEL_RETURN_FNS, *LABEL_MODE_ALIASES]))
        raise ValueError(f"Unknown LABEL_MODE={selected_mode!r}. Allowed: {allowed}.")
    return selected_mode


def resolve_exit_after_k(
    mode: str | None = None,
    exit_after_k: int | None = None,
) -> int | None:
    """Resolve the decision candle for the generic exit_after_k mode."""
    if canonical_label_mode(mode) != "exit_after_k":
        return None
    resolved = int(EXIT_AFTER_K if exit_after_k is None else exit_after_k)
    if resolved < 1:
        raise ValueError("exit_after_k must be at least 1.")
    return resolved


PRECISION_ONLY_LABEL_MODES: frozenset[str] = frozenset(
    {
        "adverse_floor",
        "bear",
        "bull",
        "peak",
        "trough",
        "high_exit",
        "slope_slowdown",
        "slope_slowdown_all",
        "ma_slope_reversal",
        "monotonic_close_path",
    }
)

DIRECTION_NEUTRAL_LABEL_MODES: frozenset[str] = frozenset(
    {
        "bear",
        "bull",
        "peak",
        "trough",
        "meta_regime_entry",
        "meta_regime_exit",
        "quantile_trade",
        "two_sided_tp",
    }
)


def is_precision_only_label_mode(mode: str | None = None) -> bool:
    return canonical_label_mode(mode) in PRECISION_ONLY_LABEL_MODES


def is_direction_neutral_label_mode(mode: str | None = None) -> bool:
    return canonical_label_mode(mode) in DIRECTION_NEUTRAL_LABEL_MODES


def get_label_return_fn(mode: str | None = None) -> Callable[[Any, int], Any]:
    selected_mode = canonical_label_mode(mode)
    return LABEL_RETURN_FNS[selected_mode]


def default_label_threshold(
    mode: str | None = None, threshold: float | None = None
) -> float:
    if threshold is not None:
        return float(threshold)
    selected_mode = canonical_label_mode(mode)
    if selected_mode == "payoff":
        return float(TRADE_COST)
    if selected_mode == "safe_path_mfe":
        return float(SAFE_ADVERSE_FLOOR)
    if selected_mode in {"slope_slowdown", "slope_slowdown_all"}:
        return float(SLOPE_SLOWDOWN_THRESHOLD)
    if selected_mode in {
        "bear",
        "bull",
        "peak",
        "trough",
        "ma_slope_reversal",
        "monotonic_close_path",
        "quantile_trade",
        "quantile_exit",
        "meta_learner",
        "meta_close_exit",
    }:
        return 0.0
    if selected_mode == "meta_strategy_profit":
        return float(TRADE_COST)
    if selected_mode == "meta_regime_exit":
        return float(META_REGIME_EXIT_THRESHOLD)
    if selected_mode == "meta_regime_entry":
        return 0.0
    return float(LABEL_THRESHOLD)


LABEL_RETURN_FN: Callable[[Any, int], Any] = get_label_return_fn()

# Final split, kept separate from the stock settings.
VAL_START: str = "2024-01-01"
TEST_START: str = "2025-01-01"
TEST_END: str | None = None

# Walk-forward folds used during evolution stop before the final validation
# period. This keeps VAL_START..TEST_START independent from feature evolution.
WF_END: str = VAL_START
WF_MIN_TRAIN_MONTHS: int = 36
WF_VAL_MONTHS: int = 6
WF_STEP_MONTHS: int = 6
WF_PURGE_BARS: int | None = None  # None => max(HOLDING_HORIZONS) + 1

# Safe feature construction. All features are time-series/ratio normalized;
# raw price/volume scale columns are intentionally not selectable.
# WINDOWS: list[int] = [1,2,3,4,5,7,10,14,20,30,40,50,60,80,120,160,240,320,400,480,600,800,960,1200,1440,]
#WINDOWS: list[int] = [1, 2, 3, 4, 5, 7, 9, 10]
WINDOWS: list[int] = [2, 3, 5, 7, 9, 15, 20, 25, 30, 37, 45, 60]
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
FITNESS_HORIZON_MODE: str = (
    "ensemble"  # "mean" keeps old behavior; "ensemble" requires all H signals
)
TRADE_TOP_FRACTION: float = 0.05

MIN_TRADES_PER_SPLIT: int = 20
TRADE_COST: float = 0.0002  # 0.1% Futures round-trip fee plus slippage allowance
RETURN_SCORE_SCALE: float = 0.01
BAD_AUC_THRESHOLD: float = 0.50

FITNESS_WEIGHTS: dict[str, float] = {
    "auc_edge": 0.20,  # old: 0.40
    "precision_excess": 0.50,  # old: 0.30
    "trade_return_score": 0.20,  # old: 0.20
    "auc_std": -0.30,  # old: -0.20
    "overfit_gap": -0.25,  # old: -0.25
    "bad_fold_ratio": -0.30,  # old: -0.30
}

# meta_regime_entry must improve the actual selected-entry strategy, not merely
# rank net winners. A 0.10% scale keeps small per-trade improvements visible.
META_REGIME_ENTRY_RETURN_SCORE_SCALE: float = 0.001
META_REGIME_ENTRY_FITNESS_WEIGHTS: dict[str, float] = {
    "auc_edge": 0.15,
    "precision_excess": 0.25,
    "trade_return_score": 0.30,
    "strategy_delta_score": 0.30,
    "auc_std": -0.20,
    "overfit_gap": -0.20,
    "bad_fold_ratio": -0.30,
}

# quantile_trade is prediction-only. Pinball skill compares the model against
# the constant train-fold quantile baseline; coverage measures calibration and
# Spearman IC measures rank information.
# MAE/RMSE are reported for diagnosis but deliberately excluded from fitness,
# because optimizing them would pull non-median quantiles toward the mean.
QUANTILE_TRADE_FITNESS_WEIGHTS: dict[str, float] = {
    "quantile_pinball_skill": 0.55,
    "quantile_coverage_error": -0.20,
    "quantile_spearman_ic": 0.15,
    "quantile_pinball_skill_std": -0.10,
    "overfit_gap": -0.20,
    "bad_fold_ratio": -0.20,
}

# quantile_exit selects the close horizon with the highest predicted close
# quantile. Unselected rows receive zero return; selected rows pay TRADE_COST.
QUANTILE_EXIT_FITNESS_WEIGHTS: dict[str, float] = {
    "realized_return_score": 1.00,
    "regret_score": -0.25,
    "return_std_score": -0.10,
    "overfit_gap": -0.20,
    "bad_fold_ratio": -0.30,
}


def quantile_exit_min_return(value: float | None = None) -> float:
    selected = QUANTILE_EXIT_MIN_RETURN if value is None else value
    result = float(TRADE_COST if selected is None else selected)
    if not isfinite(result):
        raise ValueError("QUANTILE_EXIT_MIN_RETURN must be finite.")
    return result

def quantile_trade_fitness_weights(target: str | None = None) -> dict[str, float]:
    # Validate the target but deliberately use one shared quantile objective.
    # Direction metrics for CLOSE remain diagnostic and do not affect fitness.
    canonical_quantile_target(target)
    return dict(QUANTILE_TRADE_FITNESS_WEIGHTS)

QUANTILE_TRADE_LGBM_PARAMS: dict = {
    "objective": "quantile",
    "metric": "quantile",
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

# Binary LightGBM. This higher-capacity profile is intended for experiments
# where the depth-7 model still improves out of sample. Sampling and L1/L2
# regularization remain enabled to constrain the larger trees.
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
    if SAMPLE_FILTER not in {"none", "slope_accumulation"}:
        raise ValueError("SAMPLE_FILTER must be 'none' or 'slope_accumulation'.")
    if int(SAMPLE_FILTER_SLOPE_WINDOW) < 2:
        raise ValueError("SAMPLE_FILTER_SLOPE_WINDOW must be at least 2.")
    if int(SAMPLE_FILTER_SLOPE_PREVIOUS_WINDOWS) < 1:
        raise ValueError(
            "SAMPLE_FILTER_SLOPE_PREVIOUS_WINDOWS must be at least 1."
        )
    if (
        not isfinite(float(SAMPLE_FILTER_SLOPE_RATIO))
        or SAMPLE_FILTER_SLOPE_RATIO < 0.0
    ):
        raise ValueError(
            "SAMPLE_FILTER_SLOPE_RATIO must be finite and non-negative."
        )
    get_label_return_fn()
    canonical_label_direction()
    canonical_quantile_target(QUANTILE_TARGET)
    validate_quantile_alpha(QUANTILE_ALPHA)
    quantile_exit_min_return()
    if not 0.0 <= float(QUANTILE_BAD_COVERAGE_ERROR) <= 1.0:
        raise ValueError("QUANTILE_BAD_COVERAGE_ERROR must be in [0, 1].")
    if not -1.0 <= float(QUANTILE_BAD_SPEARMAN_IC) <= 1.0:
        raise ValueError("QUANTILE_BAD_SPEARMAN_IC must be in [-1, 1].")
    if int(META_LEARNER_BASE_RANK) < 1:
        raise ValueError("META_LEARNER_BASE_RANK must be positive.")
    if (
        not isfinite(float(META_LEARNER_MIN_PREDICTION))
        or META_LEARNER_MIN_PREDICTION < 0.0
    ):
        raise ValueError("META_LEARNER_MIN_PREDICTION must be finite and non-negative.")
    if (
        not isfinite(float(META_LEARNER_TP_OFFSET))
        or META_LEARNER_TP_OFFSET < 0.0
    ):
        raise ValueError("META_LEARNER_TP_OFFSET must be finite and non-negative.")
    if (
        not isfinite(float(META_STRATEGY_STOP_LOSS))
        or META_STRATEGY_STOP_LOSS < 0.0
    ):
        raise ValueError("META_STRATEGY_STOP_LOSS must be finite and non-negative.")
    if not 0.0 < float(META_LEARNER_META_VAL_FRACTION) < 0.5:
        raise ValueError("META_LEARNER_META_VAL_FRACTION must be in (0, 0.5).")
    if PAYOFF_TP <= 0:
        raise ValueError("PAYOFF_TP must be positive.")
    if not isfinite(float(PAYOFF_ADVERSE_FLOOR)) or PAYOFF_ADVERSE_FLOOR >= 0:
        raise ValueError("PAYOFF_ADVERSE_FLOOR must be finite and negative.")
    if not isfinite(float(TP_SAFE_PATH)) or TP_SAFE_PATH <= 0:
        raise ValueError("TP_SAFE_PATH must be finite and positive.")
    if not isfinite(float(SAFE_ADVERSE_FLOOR)):
        raise ValueError("SAFE_ADVERSE_FLOOR must be finite.")
    if int(SLOPE_LOOKBACK) < 2:
        raise ValueError("SLOPE_LOOKBACK must be at least 2.")
    if not isfinite(float(SLOPE_SLOWDOWN_THRESHOLD)) or SLOPE_SLOWDOWN_THRESHOLD <= 0:
        raise ValueError("SLOPE_SLOWDOWN_THRESHOLD must be finite and positive.")
    if not isfinite(float(SLOPE_MIN_INITIAL)) or SLOPE_MIN_INITIAL < 0:
        raise ValueError("SLOPE_MIN_INITIAL must be finite and non-negative.")
    if SLOPE_PRICE_COLUMN != "high":
        raise ValueError("SLOPE_PRICE_COLUMN must be 'high'.")
    if int(MA_SLOPE_FAST_WINDOW) < 1:
        raise ValueError("MA slope fast window must be positive.")
    if min(int(MA_SLOPE_FAST_SHIFT), int(MA_SLOPE_FUTURE_SHIFT)) < 1:
        raise ValueError("MA slope shifts must be positive.")
    if (
        not isfinite(float(BEAR_ZIGZAG_TOLERANCE))
        or BEAR_ZIGZAG_TOLERANCE <= 0
        or BEAR_ZIGZAG_TOLERANCE >= 1
    ):
        raise ValueError("BEAR_ZIGZAG_TOLERANCE must be finite and in (0, 1).")
    if not isfinite(float(BEAR_MIN_DROP)) or BEAR_MIN_DROP < 0 or BEAR_MIN_DROP >= 1:
        raise ValueError("BEAR_MIN_DROP must be finite and in [0, 1).")
    if int(BEAR_MIN_BARS) < 1:
        raise ValueError("BEAR_MIN_BARS must be at least 1.")
    if (
        not isfinite(float(BULL_ZIGZAG_TOLERANCE))
        or BULL_ZIGZAG_TOLERANCE <= 0
        or BULL_ZIGZAG_TOLERANCE >= 1
    ):
        raise ValueError("BULL_ZIGZAG_TOLERANCE must be finite and in (0, 1).")
    if not isfinite(float(BULL_MIN_RISE)) or BULL_MIN_RISE < 0 or BULL_MIN_RISE >= 1:
        raise ValueError("BULL_MIN_RISE must be finite and in [0, 1).")
    if int(BULL_MIN_BARS) < 1:
        raise ValueError("BULL_MIN_BARS must be at least 1.")
    if int(PEAK_ZONE_RADIUS) < 0:
        raise ValueError("PEAK_ZONE_RADIUS must be non-negative.")
    if int(TROUGH_ZONE_RADIUS) < 0:
        raise ValueError("TROUGH_ZONE_RADIUS must be non-negative.")
    if int(EXIT_AFTER_K) < 1:
        raise ValueError("EXIT_AFTER_K must be at least 1.")
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
    if (
        not isfinite(float(META_REGIME_ENTRY_STOP_LOSS))
        or META_REGIME_ENTRY_STOP_LOSS < 0
        or META_REGIME_ENTRY_STOP_LOSS >= 1
    ):
        raise ValueError("META_REGIME_ENTRY_STOP_LOSS must be in [0, 1).")
    if META_REGIME_ENTRY_RETURN_SCORE_SCALE <= 0:
        raise ValueError("META_REGIME_ENTRY_RETURN_SCORE_SCALE must be positive.")
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
