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
PAYOFF_TP: float = 0.004  # legacy payoff archives were evolved with TP=0.40%
# Effectively disables the newer adverse-path filter for legacy payoff archives.
PAYOFF_ADVERSE_FLOOR: float = -999.0
TP_SAFE_PATH: float = 0.001  # TP used by LABEL_MODE="safe_path_mfe"
SAFE_ADVERSE_FLOOR: float = -0.0015  # stop-first low/high floor for safe_path_mfe
SAFE_PATH_RULE: str = "adverse_stop_first_v1"
SLOPE_LOOKBACK: int = 2
SLOPE_SLOWDOWN_THRESHOLD: float = 0.0002  # 0.03% log-OLS slope per candle
SLOPE_MIN_INITIAL: float = 0.0001  # 0.02% log-OLS slope per candle
SLOPE_PRICE_COLUMN: str = "high"
SLOPE_SLOWDOWN_RULE: str = "eligible_initial_slope_only_v2"
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


LABEL_RETURN_FNS: dict[str, Callable[[Any, int], Any]] = {
    "close_exit": close_exit_future_return,
    "high_exit": high_exit_future_return,
    "slope_slowdown": slope_slowdown_future_return,
    "close_path_mean": close_path_mean_future_return,
    "mfe": mfe_future_return,
    "mfe_ahead": mfe_ahead_future_return,
    "adverse_floor": adverse_floor_future_return,
    "safe_path_mfe": safe_path_mfe_future_return,
    "payoff": payoff_future_return,
    "two_sided_tp": two_sided_tp_future_return,
    "exit_after_k": exit_after_k_future_return,
    "bear": bear_future_return,
    "bull": bull_future_return,
}

LABEL_MODE_ALIASES: dict[str, str] = {
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
    {"adverse_floor", "bear", "bull", "high_exit", "slope_slowdown"}
)

DIRECTION_NEUTRAL_LABEL_MODES: frozenset[str] = frozenset(
    {"bear", "bull", "two_sided_tp"}
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
    if selected_mode == "slope_slowdown":
        return float(SLOPE_SLOWDOWN_THRESHOLD)
    if selected_mode in {"bear", "bull"}:
        return 0.0
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
# WINDOWS: list[int] = [1,2,3,4,5,7,10,14,20,30,40,50,60,80,120,160,240,320,400,480,600,800,960,1200,1440,]
WINDOWS: list[int] = [1,2,3,4,5,6]
#WINDOWS: list[int] = [2, 5, 7, 15, 25, 30]
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
TRADE_COST: float = 0.0005  # 0.1% Futures round-trip fee plus slippage allowance
RETURN_SCORE_SCALE: float = 0.01
BAD_AUC_THRESHOLD: float = 0.50

FITNESS_WEIGHTS: dict[str, float] = {
    "auc_edge": 0.20, # old: 0.40
    "precision_excess": 0.50,  # old: 0.30
    "trade_return_score": 0.20,  # old: 0.20
    "auc_std": -0.30,  # old: -0.20
    "overfit_gap": -0.25,  # old: -0.25
    "bad_fold_ratio": -0.30,  # old: -0.30
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
    if not isfinite(float(PAYOFF_ADVERSE_FLOOR)) or PAYOFF_ADVERSE_FLOOR >= 0:
        raise ValueError("PAYOFF_ADVERSE_FLOOR must be finite and negative.")
    if not isfinite(float(TP_SAFE_PATH)) or TP_SAFE_PATH <= 0:
        raise ValueError("TP_SAFE_PATH must be finite and positive.")
    if not isfinite(float(SAFE_ADVERSE_FLOOR)):
        raise ValueError("SAFE_ADVERSE_FLOOR must be finite.")
    if int(SLOPE_LOOKBACK) < 2:
        raise ValueError("SLOPE_LOOKBACK must be at least 2.")
    if (
        not isfinite(float(SLOPE_SLOWDOWN_THRESHOLD))
        or SLOPE_SLOWDOWN_THRESHOLD <= 0
    ):
        raise ValueError("SLOPE_SLOWDOWN_THRESHOLD must be finite and positive.")
    if not isfinite(float(SLOPE_MIN_INITIAL)) or SLOPE_MIN_INITIAL < 0:
        raise ValueError("SLOPE_MIN_INITIAL must be finite and non-negative.")
    if SLOPE_PRICE_COLUMN != "high":
        raise ValueError("SLOPE_PRICE_COLUMN must be 'high'.")
    if (
        not isfinite(float(BEAR_ZIGZAG_TOLERANCE))
        or BEAR_ZIGZAG_TOLERANCE <= 0
        or BEAR_ZIGZAG_TOLERANCE >= 1
    ):
        raise ValueError("BEAR_ZIGZAG_TOLERANCE must be finite and in (0, 1).")
    if (
        not isfinite(float(BEAR_MIN_DROP))
        or BEAR_MIN_DROP < 0
        or BEAR_MIN_DROP >= 1
    ):
        raise ValueError("BEAR_MIN_DROP must be finite and in [0, 1).")
    if int(BEAR_MIN_BARS) < 1:
        raise ValueError("BEAR_MIN_BARS must be at least 1.")
    if (
        not isfinite(float(BULL_ZIGZAG_TOLERANCE))
        or BULL_ZIGZAG_TOLERANCE <= 0
        or BULL_ZIGZAG_TOLERANCE >= 1
    ):
        raise ValueError("BULL_ZIGZAG_TOLERANCE must be finite and in (0, 1).")
    if (
        not isfinite(float(BULL_MIN_RISE))
        or BULL_MIN_RISE < 0
        or BULL_MIN_RISE >= 1
    ):
        raise ValueError("BULL_MIN_RISE must be finite and in [0, 1).")
    if int(BULL_MIN_BARS) < 1:
        raise ValueError("BULL_MIN_BARS must be at least 1.")
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
