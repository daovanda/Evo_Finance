"""
Data utilities for Evo_Finance.

The module keeps the old final split API and adds walk-forward fold creation
for the evolutionary loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import pandas as pd

from config.settings import (
    HOLDING_HORIZON,
    LABEL_FN,
    SECTORS,
    TEST_END,
    TEST_START,
    VAL_START,
    WF_END,
    WF_MIN_TRAIN_MONTHS,
    WF_PURGE_DAYS,
    WF_STEP_MONTHS,
    WF_VAL_MONTHS,
)


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def label_dataframe(
    df: pd.DataFrame,
    label_fn: Callable = LABEL_FN,
    holding_horizon: int = HOLDING_HORIZON,
) -> pd.DataFrame:
    """Return a copy of df with per-ticker forward labels attached."""
    _validate_index(df)
    labeled = df.copy()

    def _label_per_ticker(group: pd.DataFrame) -> pd.Series:
        try:
            return label_fn(group, h=holding_horizon)
        except TypeError:
            return label_fn(group)

    labeled["label"] = (
        labeled.groupby(level="ticker", group_keys=False)
        .apply(_label_per_ticker)
    )
    return labeled


def split_labeled_by_dates(
    labeled_df: pd.DataFrame,
    val_start: str = VAL_START,
    test_start: str = TEST_START,
    test_end: Optional[str] = TEST_END,
    purge_days: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split an already-labeled dataframe into final train/val/test.

    purge_days removes the last N trading dates before each split boundary from
    the earlier split. This prevents labels near VAL_START/TEST_START from using
    future prices from the next period.
    """
    _validate_index(labeled_df)
    if "label" not in labeled_df.columns:
        raise ValueError("labeled_df must contain a 'label' column.")

    val_start_ts = pd.Timestamp(val_start)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end) if test_end else None

    all_dates = labeled_df.index.get_level_values("date")
    data_start = all_dates.min()
    data_end = all_dates.max()

    if val_start_ts <= data_start:
        raise ValueError(
            f"VAL_START ({val_start}) must be after data start ({data_start.date()})."
        )
    if test_start_ts <= val_start_ts:
        raise ValueError(
            f"TEST_START ({test_start}) must be after VAL_START ({val_start})."
        )
    if test_end_ts is not None and test_end_ts < test_start_ts:
        raise ValueError(
            f"TEST_END ({test_end}) must be >= TEST_START ({test_start})."
        )

    unique_dates = pd.DatetimeIndex(sorted(pd.unique(all_dates)))
    purge_days = max(int(purge_days), 0)
    train_end_ts = _purged_boundary(unique_dates, val_start_ts, purge_days)
    val_end_ts = _purged_boundary(unique_dates, test_start_ts, purge_days)

    train_mask = all_dates < train_end_ts
    val_mask = (all_dates >= val_start_ts) & (all_dates < val_end_ts)
    if test_end_ts is not None:
        test_mask = (all_dates >= test_start_ts) & (all_dates <= test_end_ts)
    else:
        test_mask = all_dates >= test_start_ts

    train_df = labeled_df[train_mask].copy().dropna(subset=["label"])
    val_df = labeled_df[val_mask].copy().dropna(subset=["label"])
    test_df = labeled_df[test_mask].copy().dropna(subset=["label"])

    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        "Final split: train [%s -> %s) %d rows | val [%s -> %s) %d rows | "
        "test [%s -> %s] %d rows | purge=%d trading days",
        data_start.date(),
        train_end_ts.date(),
        len(train_df),
        val_start,
        val_end_ts.date(),
        len(val_df),
        test_start,
        test_end or data_end.date(),
        len(test_df),
        purge_days,
    )

    return train_df, val_df, test_df


def split_and_label(
    df: pd.DataFrame,
    label_fn: Callable = LABEL_FN,
    holding_horizon: int = HOLDING_HORIZON,
    val_start: str = VAL_START,
    test_start: str = TEST_START,
    test_end: Optional[str] = TEST_END,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Backward-compatible helper: label once, then split by final dates.
    """
    labeled_df = label_dataframe(
        df,
        label_fn=label_fn,
        holding_horizon=holding_horizon,
    )
    return split_labeled_by_dates(
        labeled_df,
        val_start=val_start,
        test_start=test_start,
        test_end=test_end,
        purge_days=holding_horizon,
    )


def make_walk_forward_folds(
    labeled_df: pd.DataFrame,
    wf_end: str = WF_END,
    min_train_months: int = WF_MIN_TRAIN_MONTHS,
    val_months: int = WF_VAL_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    purge_days: int = WF_PURGE_DAYS,
) -> List[WalkForwardFold]:
    """
    Build anchored expanding walk-forward folds before wf_end.

    Train windows expand from the first available date. Validation windows are
    non-overlapping by default when val_months == step_months. purge_days is
    counted in trading dates between train end and validation start.
    """
    _validate_index(labeled_df)
    if "label" not in labeled_df.columns:
        raise ValueError("labeled_df must contain a 'label' column.")

    wf_end_ts = pd.Timestamp(wf_end)
    all_dates = labeled_df.index.get_level_values("date")
    unique_dates = pd.DatetimeIndex(sorted(pd.unique(all_dates[all_dates < wf_end_ts])))
    if len(unique_dates) == 0:
        raise ValueError(f"No data before WF_END={wf_end}.")

    first_val_candidate = unique_dates[0] + pd.DateOffset(months=min_train_months)
    val_start = _first_date_at_or_after(unique_dates, first_val_candidate)
    folds: List[WalkForwardFold] = []

    while val_start is not None and val_start < wf_end_ts:
        val_end_candidate = val_start + pd.DateOffset(months=val_months)
        val_end = _first_date_at_or_after(unique_dates, val_end_candidate)
        if val_end is None or val_end > wf_end_ts:
            break
        if val_end <= val_start:
            break

        val_pos = int(unique_dates.searchsorted(val_start, side="left"))
        train_end_pos = val_pos - max(int(purge_days), 0)
        if train_end_pos <= 0:
            next_candidate = val_start + pd.DateOffset(months=step_months)
            val_start = _first_date_at_or_after(unique_dates, next_candidate)
            continue
        train_end = unique_dates[train_end_pos]

        fold_dates = labeled_df.index.get_level_values("date")
        train_mask = fold_dates < train_end
        val_mask = (fold_dates >= val_start) & (fold_dates < val_end)
        train_df = labeled_df[train_mask].copy().dropna(subset=["label"])
        val_df = labeled_df[val_mask].copy().dropna(subset=["label"])

        if not train_df.empty and not val_df.empty:
            folds.append(
                WalkForwardFold(
                    name=f"wf_{len(folds) + 1:02d}",
                    train_df=train_df,
                    val_df=val_df,
                    train_start=unique_dates[0],
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                )
            )

        next_candidate = val_start + pd.DateOffset(months=step_months)
        val_start = _first_date_at_or_after(unique_dates, next_candidate)

    if not folds:
        raise ValueError(
            "No walk-forward folds created. Reduce WF_MIN_TRAIN_MONTHS or check WF_END."
        )
    return folds


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Basic sanity checks on the input DataFrame."""
    required = {"open", "high", "close", "low", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    _validate_index(df)


def tickers_missing_sector(df: pd.DataFrame) -> list[str]:
    """Return tickers present in df but absent from the SECTORS mapping."""
    _validate_index(df)
    mapped = {
        str(ticker).upper()
        for tickers in SECTORS.values()
        for ticker in tickers
    }
    present = {
        str(ticker).upper()
        for ticker in df.index.get_level_values("ticker").unique()
    }
    return sorted(present - mapped)


def _validate_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame must have MultiIndex (date, ticker).")
    if df.index.names != ["date", "ticker"]:
        raise ValueError(
            f"MultiIndex names must be ['date', 'ticker'], got {df.index.names}"
        )


def _first_date_at_or_after(
    dates: pd.DatetimeIndex,
    target: pd.Timestamp,
) -> Optional[pd.Timestamp]:
    pos = int(dates.searchsorted(pd.Timestamp(target), side="left"))
    if pos >= len(dates):
        return None
    return dates[pos]


def _purged_boundary(
    dates: pd.DatetimeIndex,
    boundary: pd.Timestamp,
    purge_days: int,
) -> pd.Timestamp:
    """Return the exclusive end date for the previous split after purge."""
    pos = int(dates.searchsorted(boundary, side="left"))
    purged_pos = max(pos - max(int(purge_days), 0), 0)
    if purged_pos >= len(dates):
        return boundary
    return dates[purged_pos]
