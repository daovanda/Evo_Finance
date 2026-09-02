"""Data loading, labeling, and walk-forward splits for crypto."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer

from crypto import config


@dataclass(frozen=True)
class CryptoFold:
    name: str
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def load_ohlcv(path: str | Path = config.DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Crypto data file not found: {path}")

    df = pd.read_csv(path)
    if config.DATE_COLUMN not in df.columns:
        raise ValueError(f"Missing date column: {config.DATE_COLUMN!r}")

    df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN])
    df = df.sort_values(config.DATE_COLUMN).drop_duplicates(
        config.DATE_COLUMN, keep="last"
    )
    df = df.set_index(config.DATE_COLUMN)
    df.index.name = "date"

    required = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Crypto data missing columns: {missing}")

    for col in sorted(required):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=sorted(required)).copy()

    price_cols = ["open", "high", "low", "close"]
    if (df[price_cols] <= 0).any().any():
        raise ValueError("OHLC prices must be positive.")
    if (df["volume"] < 0).any():
        raise ValueError("Volume must be non-negative.")
    if (df["trade_count"] < 0).any():
        raise ValueError("trade_count must be non-negative.")
    if (df["high"] < df["low"]).any():
        raise ValueError("OHLC rows must satisfy high >= low.")

    return df


def attach_mfe_entry_m2_inputs(
    target_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    horizons: list[int] | tuple[int, ...],
) -> pd.DataFrame:
    """Attach future targets whose entry is open minute 2 of target H1."""
    if len(target_df.index) < 2 or len(minute_df.index) < 2:
        raise ValueError("mfe_entry_m2 requires at least two target and minute rows.")
    target_interval = pd.Series(target_df.index).diff().dropna().mode().iloc[0]
    minute_interval = pd.Series(minute_df.index).diff().dropna().mode().iloc[0]
    if target_interval != pd.Timedelta(minutes=5):
        raise ValueError(
            f"mfe_entry_m2 currently requires 5m --data, got {target_interval}."
        )
    if minute_interval != pd.Timedelta(minutes=1):
        raise ValueError(
            f"--mfe-entry-data must contain 1m candles, got {minute_interval}."
        )

    result = target_df.copy()
    entry_times = result.index + target_interval + minute_interval
    entry = minute_df["open"].reindex(entry_times)
    entry.index = result.index

    for horizon in sorted({int(value) for value in horizons}):
        if horizon < 1:
            raise ValueError("mfe_entry_m2 horizons must be positive.")
        path_bars = horizon * 5 - 1
        indexer = FixedForwardWindowIndexer(window_size=path_bars)
        path_high = minute_df["high"].rolling(
            window=indexer, min_periods=path_bars
        ).max().reindex(entry_times)
        path_low = minute_df["low"].rolling(
            window=indexer, min_periods=path_bars
        ).min().reindex(entry_times)
        path_high.index = result.index
        path_low.index = result.index
        close_h = result["close"].shift(-horizon)
        complete = entry.notna() & path_high.notna() & path_low.notna() & close_h.notna()

        result[f"mfe_entry_m2_long_h{horizon}"] = (
            path_high.div(entry).sub(1.0).where(complete)
        )
        result[f"mfe_entry_m2_short_h{horizon}"] = (
            1.0 - path_low.div(entry)
        ).where(complete)
        result[f"close_entry_m2_long_h{horizon}"] = (
            close_h.div(entry).sub(1.0).where(complete)
        )
        result[f"close_entry_m2_short_h{horizon}"] = (
            1.0 - close_h.div(entry)
        ).where(complete)
    return result


def add_binary_labels(
    df: pd.DataFrame,
    horizons: list[int] | tuple[int, ...] = tuple(config.HOLDING_HORIZONS),
    threshold: float | None = None,
    return_fn: Callable[[pd.DataFrame, int], pd.Series] | None = None,
    label_mode: str | None = None,
    label_direction: str | None = None,
    exit_after_k: int | None = None,
) -> pd.DataFrame:
    """
    Add future_return_h{h} and label_h{h}.

    The future_return formula is controlled by label_mode/config.LABEL_MODE and
    label_direction/config.LABEL_DIRECTION.
    label = 1 if the mode-specific label return > threshold, else 0.

    For mfe, the label source remains maximum favorable excursion, while
    future_return_h stores the executable strategy payoff used by fitness:
    threshold on a TP hit, otherwise the close return at the final horizon.
    For mfe_ahead, the model row is H1 and may use its completed candle. Entry
    remains open H1; MFE and the TP-or-close payoff span H1..Hh. This mode
    intentionally contains H1 look-ahead relative to the entry price.
    The safe_path_mfe mode uses its explicit first-hit/path-safety label.
    For payoff, label=1 additionally requires the full H1..H adverse path to
    stay strictly above config.PAYOFF_ADVERSE_FLOOR. Its future_return remains
    the executable TP-or-final-close payoff used by fitness.
    For adverse_floor, label=1 means the full future path stays above the
    directional adverse floor, while future_return_h is zero because this
    mode is optimized as a classification filter rather than a payoff model.
    For high_exit, label=1 means the favorable extreme of the exact H candle
    exceeds the threshold. Its future_return_h is zero because this mode is
    also optimized only as a classification filter.
    For slope_slowdown, label=1 means the high-price OLS slope weakens by more
    than the threshold in the selected direction. Rows that do not satisfy
    the observable initial-slope gate are excluded as NaN rather than treated
    as label 0. Its future_return_h is zero because slope change is not an
    executable trading return.
    For ma_slope_reversal, Long labels a positive MA3 close slope at t whose
    slope is negative after config.MA_SLOPE_FUTURE_SHIFT candles; Short uses
    the sign-symmetric reversal. This target is horizon-neutral and
    future_return_h is zero for precision-only fitness.
    For monotonic_close_path, Long requires the strict chain
    open(H1) < close(H1) < ... < close(Horizon); Short uses the symmetric
    decreasing chain. Its future_return_h is zero for precision-only fitness.
    For two_sided_tp, label=1 means both the Long and Short TP are reached;
    future_return_h is the combined gross payoff of the two positions. Label
    direction is ignored for this direction-neutral mode.
    For bear, label=1 marks candles strictly inside an offline confirmed
    close-ZigZag peak-to-trough bear body. Peak and trough candles are 0. The
    target may use later closes for confirmation, but model features at row t
    are built independently from raw OHLCV available through t. Its
    future_return_h is zero because bear is precision-only classification.
    For bull, label=1 symmetrically marks candles strictly inside an offline
    confirmed close-ZigZag trough-to-peak body. Its future_return_h is also
    zero and its direction/horizon arguments do not alter the target.
    For peak, label=1 marks the confirmed valid Bear-definition peak zone
    t-1..t+1. For trough, label=1 marks the symmetric Bull-definition trough
    zone. Both are precision-only, horizon-neutral, and direction-neutral.
    For quantile_trade, no binary label is created. Each horizon receives
    upward-MFE and downward-MFE columns. Its dedicated evaluator selects the
    configured target: MFE is the maximum upward high excursion, MAE is the
    maximum downward low excursion, and close is the signed close-H return
    from open H1. Direction is ignored. It predicts one configured quantile
    and evaluates accuracy only; no trade is simulated.
    For exit_after_k, rows whose TP was already reached in H1..Hk are
    excluded. Label 1 means the remaining H(k+1)..Hh path reaches TP, while
    future_return_h is the executable gross payoff: threshold on a TP hit or
    the directional close Hh return on a miss.
    """
    labeled = df.sort_index().copy()
    selected_mode = config.canonical_label_mode(label_mode)
    selected_direction = config.canonical_label_direction(label_direction)
    label_return_fn = return_fn or config.get_label_return_fn(selected_mode)
    label_threshold = config.default_label_threshold(selected_mode, threshold)
    decision_k = config.resolve_exit_after_k(label_mode, exit_after_k)
    zigzag_labels = None
    if selected_mode == "bear":
        zigzag_labels = config.bear_body_labels(labeled)
    elif selected_mode == "bull":
        zigzag_labels = config.bull_body_labels(labeled)
    elif selected_mode == "peak":
        zigzag_labels = config.peak_zone_labels(labeled)
    elif selected_mode == "trough":
        zigzag_labels = config.trough_zone_labels(labeled)
    if decision_k is not None:
        invalid_horizons = [int(h) for h in horizons if int(h) <= decision_k]
        if invalid_horizons:
            raise ValueError(
                "exit_after_k must be smaller than every holding horizon; "
                f"k={decision_k}, invalid horizons={invalid_horizons}."
            )
    for h in horizons:
        h = int(h)
        if selected_mode in {"quantile_trade", "quantile_exit"}:
            _add_quantile_trade_horizon_targets(labeled, h)
            continue

        if selected_mode in {"bear", "bull", "peak", "trough"}:
            assert zigzag_labels is not None
            explicit_label = zigzag_labels
            complete = explicit_label.notna()
            labeled[f"future_return_h{h}"] = pd.Series(
                0.0,
                index=labeled.index,
                dtype="float64",
            ).where(complete)
            labeled[f"label_h{h}"] = explicit_label
            continue

        if selected_mode == "two_sided_tp":
            future_return, explicit_label = config.two_sided_tp_outcome(
                labeled,
                h,
                threshold=float(label_threshold),
            )
            labeled[f"future_return_h{h}"] = future_return
            labeled[f"label_h{h}"] = explicit_label
            continue

        if selected_mode == "safe_path_mfe":
            future_return, explicit_label = config.safe_path_mfe_outcome(
                labeled,
                h,
                adverse_floor=float(label_threshold),
                direction=selected_direction,
            )
            labeled[f"future_return_h{h}"] = future_return
            labeled[f"label_h{h}"] = explicit_label
            continue

        if decision_k is not None:
            future_return, explicit_label = config.exit_after_k_outcome(
                labeled,
                h,
                threshold=float(label_threshold),
                direction=selected_direction,
                exit_after_k=decision_k,
            )
            entry_open = labeled["open"].shift(decision_k - 1)
            hit_price = (
                labeled["low"] if selected_direction == "short" else labeled["high"]
            )
            prior_hit = pd.Series(False, index=labeled.index, dtype=bool)
            for offset in range(decision_k - 1, -1, -1):
                prior_return = config.directional_price_return(
                    hit_price.shift(offset),
                    entry_open,
                    selected_direction,
                )
                prior_hit |= prior_return >= float(label_threshold)
            future_return = future_return.mask(prior_hit)
            explicit_label = explicit_label.mask(prior_hit)
            labeled[f"future_return_h{h}"] = future_return
            labeled[f"label_h{h}"] = explicit_label
            continue

        future_return = _call_label_return_fn(
            label_return_fn, labeled, h, selected_direction
        )
        if selected_mode == "adverse_floor":
            if float(label_threshold) <= 0.0:
                raise ValueError(
                    "adverse_floor label_threshold must be positive; "
                    "for example 0.003 means the path must stay above -0.3%."
                )
            complete = future_return.notna()
            adverse_floor = -float(label_threshold)
            explicit_label = (
                future_return.gt(adverse_floor).astype("float64").where(complete)
            )
            labeled[f"future_return_h{h}"] = pd.Series(
                0.0,
                index=labeled.index,
                dtype="float64",
            ).where(complete)
            labeled[f"label_h{h}"] = explicit_label
            continue

        if selected_mode in {
            "high_exit",
            "slope_slowdown",
            "slope_slowdown_all",
            "ma_slope_reversal",
            "monotonic_close_path",
        }:
            if (
                selected_mode in {"slope_slowdown", "slope_slowdown_all"}
                and float(label_threshold) <= 0.0
            ):
                raise ValueError(
                    f"{selected_mode} label_threshold must be positive; "
                    "for example 0.0003 means 0.03% slope change per candle."
                )
            complete = future_return.notna()
            if selected_mode in {"ma_slope_reversal", "monotonic_close_path"}:
                explicit_label = future_return.astype("float64").where(complete)
            else:
                explicit_label = (
                    future_return.gt(float(label_threshold))
                    .astype("float64")
                    .where(complete)
                )
            labeled[f"future_return_h{h}"] = pd.Series(
                0.0,
                index=labeled.index,
                dtype="float64",
            ).where(complete)
            labeled[f"label_h{h}"] = explicit_label
            continue

        if selected_mode in {"mfe", "mfe_ahead", "mfe_entry_m2"}:
            if selected_mode == "mfe_ahead":
                close_return = config.mfe_ahead_close_return(
                    labeled,
                    h,
                    direction=selected_direction,
                )
            elif selected_mode == "mfe_entry_m2":
                close_return = config.mfe_entry_m2_close_return(
                    labeled,
                    h,
                    direction=selected_direction,
                )
            else:
                close_return = config.close_exit_future_return(
                    labeled,
                    h,
                    direction=selected_direction,
                )
            complete = future_return.notna() & close_return.notna()
            hit_tp = future_return > float(label_threshold)
            strategy_return = close_return.where(
                ~hit_tp,
                float(label_threshold),
            ).where(complete)
            labeled[f"future_return_h{h}"] = strategy_return
            labeled[f"label_h{h}"] = hit_tp.astype("float").where(complete)
            continue

        if selected_mode == "payoff":
            adverse_return = config.adverse_floor_future_return(
                labeled,
                h,
                direction=selected_direction,
            )
            complete = future_return.notna() & adverse_return.notna()
            explicit_label = future_return.gt(
                float(label_threshold)
            ) & adverse_return.gt(float(config.PAYOFF_ADVERSE_FLOOR))
            labeled[f"future_return_h{h}"] = future_return.where(complete)
            labeled[f"label_h{h}"] = explicit_label.astype("float64").where(complete)
            continue

        labeled[f"future_return_h{h}"] = future_return
        labeled[f"label_h{h}"] = (future_return > float(label_threshold)).astype(
            "float"
        )
        labeled.loc[future_return.isna(), f"label_h{h}"] = np.nan
    return labeled


def _add_quantile_trade_horizon_targets(
    labeled: pd.DataFrame,
    horizon: int,
) -> None:
    """Add complete-path distributional targets for one trading horizon."""
    h = int(horizon)
    if h < 1:
        raise ValueError("quantile_trade horizon must be positive.")

    entry_open = labeled["open"].shift(-1)
    up_steps: list[pd.Series] = []
    down_steps: list[pd.Series] = []
    for step in range(1, h + 1):
        up = labeled["high"].shift(-step).div(entry_open).sub(1.0)
        down = 1.0 - labeled["low"].shift(-step).div(entry_open)
        up_steps.append(up)
        down_steps.append(down)

    close_return = labeled["close"].shift(-h).div(entry_open).sub(1.0)
    complete = (
        entry_open.notna()
        & pd.concat(up_steps, axis=1).notna().all(axis=1)
        & pd.concat(down_steps, axis=1).notna().all(axis=1)
        & close_return.notna()
    )
    for step, (up, down) in enumerate(zip(up_steps, down_steps, strict=True), start=1):
        # Path steps are shared across horizons. This avoids duplicating H1..H5
        # again for H7/H10/etc. when quantile_trade uses an ensemble.
        up_col = f"quantile_up_s{step}"
        down_col = f"quantile_down_s{step}"
        step_complete = entry_open.notna() & up.notna() & down.notna()
        if up_col not in labeled.columns:
            labeled[up_col] = up.where(step_complete)
        if down_col not in labeled.columns:
            labeled[down_col] = down.where(step_complete)

    labeled[f"quantile_up_mfe_h{h}"] = (
        pd.concat(up_steps, axis=1).max(axis=1, skipna=False).where(complete)
    )
    labeled[f"quantile_down_mfe_h{h}"] = (
        pd.concat(down_steps, axis=1).max(axis=1, skipna=False).where(complete)
    )
    labeled[f"quantile_close_return_h{h}"] = close_return.where(complete)


def _call_label_return_fn(
    return_fn: Callable[[pd.DataFrame, int], pd.Series],
    df: pd.DataFrame,
    horizon: int,
    label_direction: str,
) -> pd.Series:
    signature = inspect.signature(return_fn)
    if "direction" in signature.parameters:
        return return_fn(df, horizon, direction=label_direction)
    return return_fn(df, horizon)


def split_labeled_by_dates(
    labeled_df: pd.DataFrame,
    val_start: str = config.VAL_START,
    test_start: str = config.TEST_START,
    test_end: str | None = config.TEST_END,
    purge_bars: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if labeled_df.empty:
        raise ValueError("Cannot split empty labeled dataframe.")

    val_start_ts = pd.Timestamp(val_start)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end) if test_end else None
    if val_start_ts >= test_start_ts:
        raise ValueError("VAL_START must be before TEST_START.")
    if test_end_ts is not None and test_end_ts < test_start_ts:
        raise ValueError("TEST_END must be >= TEST_START.")

    idx = pd.DatetimeIndex(labeled_df.index)
    unique_dates = pd.DatetimeIndex(sorted(pd.unique(idx)))
    train_end = _purged_boundary(unique_dates, val_start_ts, purge_bars)
    val_end = _purged_boundary(unique_dates, test_start_ts, purge_bars)

    train_df = labeled_df[idx < train_end].copy()
    val_df = labeled_df[(idx >= val_start_ts) & (idx < val_end)].copy()
    if test_end_ts is None:
        test_df = labeled_df[idx >= test_start_ts].copy()
    else:
        test_end_exclusive = _purged_inclusive_end(
            unique_dates,
            test_end_ts,
            purge_bars,
        )
        test_df = labeled_df[(idx >= test_start_ts) & (idx < test_end_exclusive)].copy()

    return train_df, val_df, test_df


def make_walk_forward_folds(
    labeled_df: pd.DataFrame,
    wf_end: str = config.WF_END,
    min_train_months: int = config.WF_MIN_TRAIN_MONTHS,
    val_months: int = config.WF_VAL_MONTHS,
    step_months: int = config.WF_STEP_MONTHS,
    purge_bars: int = 0,
) -> list[CryptoFold]:
    """
    Build anchored expanding walk-forward folds before wf_end.

    Purge is counted in bars, not calendar days, because crypto data is intraday.
    """
    if labeled_df.empty:
        raise ValueError("Cannot build folds from empty dataframe.")

    wf_end_ts = pd.Timestamp(wf_end)
    all_dates = pd.DatetimeIndex(labeled_df.index)
    unique_dates = pd.DatetimeIndex(sorted(pd.unique(all_dates[all_dates < wf_end_ts])))
    if len(unique_dates) == 0:
        raise ValueError(f"No data before WF_END={wf_end}.")

    first_val_candidate = unique_dates[0] + pd.DateOffset(months=int(min_train_months))
    val_start = _first_date_at_or_after(unique_dates, first_val_candidate)
    folds: list[CryptoFold] = []

    while val_start is not None and val_start < wf_end_ts:
        val_end_candidate = val_start + pd.DateOffset(months=int(val_months))
        if val_end_candidate > wf_end_ts:
            break
        val_end = _first_date_at_or_after(unique_dates, val_end_candidate)
        if val_end is None:
            val_end = wf_end_ts
        if val_end <= val_start:
            break

        val_pos = int(unique_dates.searchsorted(val_start, side="left"))
        train_end_pos = val_pos - max(int(purge_bars), 0)
        if train_end_pos <= 0:
            val_start = _first_date_at_or_after(
                unique_dates,
                val_start + pd.DateOffset(months=int(step_months)),
            )
            continue
        train_end = unique_dates[train_end_pos]

        train_df = labeled_df[all_dates < train_end].copy()
        val_df = labeled_df[(all_dates >= val_start) & (all_dates < val_end)].copy()
        if not train_df.empty and not val_df.empty:
            folds.append(
                CryptoFold(
                    name=f"wf_{len(folds) + 1:02d}",
                    train_df=train_df,
                    val_df=val_df,
                    train_start=unique_dates[0],
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                )
            )

        val_start = _first_date_at_or_after(
            unique_dates,
            val_start + pd.DateOffset(months=int(step_months)),
        )

    if not folds:
        raise ValueError(
            "No walk-forward folds created. Reduce WF_MIN_TRAIN_MONTHS or check WF_END."
        )
    return folds


def _first_date_at_or_after(
    dates: pd.DatetimeIndex,
    target: pd.Timestamp,
) -> pd.Timestamp | None:
    pos = int(dates.searchsorted(pd.Timestamp(target), side="left"))
    if pos >= len(dates):
        return None
    return pd.Timestamp(dates[pos])


def _purged_boundary(
    dates: pd.DatetimeIndex,
    boundary: pd.Timestamp,
    purge_bars: int,
) -> pd.Timestamp:
    pos = int(dates.searchsorted(pd.Timestamp(boundary), side="left"))
    purged_pos = max(pos - max(int(purge_bars), 0), 0)
    if purged_pos >= len(dates):
        return pd.Timestamp(boundary)
    return pd.Timestamp(dates[purged_pos])


def _purged_inclusive_end(
    dates: pd.DatetimeIndex,
    end: pd.Timestamp,
    purge_bars: int,
) -> pd.Timestamp:
    """
    Return an exclusive upper bound after purging a closed test window tail.

    If labels look forward N bars, the final N bars up to TEST_END cannot be
    scored without using close/open values beyond the declared test window.
    """
    pos_after_end = int(dates.searchsorted(pd.Timestamp(end), side="right"))
    if pos_after_end >= len(dates) and int(purge_bars) <= 0:
        return pd.Timestamp(end) + pd.Timedelta(nanoseconds=1)
    purged_pos = max(pos_after_end - max(int(purge_bars), 0), 0)
    if purged_pos >= len(dates):
        return pd.Timestamp(dates[-1]) + pd.Timedelta(nanoseconds=1)
    return pd.Timestamp(dates[purged_pos])
