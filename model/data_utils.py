"""
Evo_Finance — Data Utilities
──────────────────────────────
Handles train/val/test splitting and label generation.

Expected input DataFrame
  - MultiIndex (date, ticker) — dates must be sorted ascending.
  - Columns: open, high, close, low, volume  (all numeric).

Split is **date-based** using explicit cutoff dates from config:
  train  : [start of data]  →  VAL_START   (exclusive)
  val    : VAL_START        →  TEST_START  (exclusive)
  test   : TEST_START       →  TEST_END    (inclusive, None = end of data)
"""

from __future__ import annotations
from typing import Tuple, Callable, Optional
import pandas as pd

from config.settings import (
    VAL_START, TEST_START, TEST_END,
    LABEL_FN, HOLDING_HORIZON,
)


def split_and_label(
    df: pd.DataFrame,
    label_fn: Callable  = LABEL_FN,
    holding_horizon: int = HOLDING_HORIZON,
    val_start:  str     = VAL_START,
    test_start: str     = TEST_START,
    test_end:   Optional[str] = TEST_END,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    1. Compute labels per-ticker (groupby) để tránh cross-ticker shift leakage.
    2. Split thành train / val / test theo ngày cụ thể.
    3. Drop rows không có label (cuối chuỗi, horizon chưa realised).

    Parameters
    ----------
    val_start  : Ngày đầu tiên của val (= ngày kết thúc train).
                 Ví dụ "2022-01-01"
    test_start : Ngày đầu tiên của test (= ngày kết thúc val).
                 Ví dụ "2023-07-01"
    test_end   : Ngày cuối cùng của test (inclusive).
                 None = lấy hết data còn lại.
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame phải có MultiIndex (date, ticker).")
    if df.index.names != ["date", "ticker"]:
        raise ValueError(
            f"MultiIndex names phải là ['date', 'ticker'], got {df.index.names}"
        )

    val_start_ts  = pd.Timestamp(val_start)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts   = pd.Timestamp(test_end) if test_end else None

    # Validate
    all_dates = df.index.get_level_values("date")
    data_start = all_dates.min()
    data_end   = all_dates.max()

    if val_start_ts <= data_start:
        raise ValueError(
            f"VAL_START ({val_start}) phải sau ngày đầu data ({data_start.date()})"
        )
    if test_start_ts <= val_start_ts:
        raise ValueError(
            f"TEST_START ({test_start}) phải sau VAL_START ({val_start})"
        )
    if test_end_ts is not None and test_end_ts < test_start_ts:
        raise ValueError(
            f"TEST_END ({test_end}) phải >= TEST_START ({test_start})"
        )

    df = df.copy()

    # ── Label per-ticker (tránh cross-ticker shift leakage) ───────────────────
    def _label_per_ticker(group: pd.DataFrame) -> pd.Series:
        try:
            return label_fn(group, h=holding_horizon)
        except TypeError:
            return label_fn(group)

    label_series = (
        df.groupby(level="ticker", group_keys=False)
        .apply(_label_per_ticker)
    )
    df["label"] = label_series

    # ── Split theo ngày ───────────────────────────────────────────────────────
    date_level = df.index.get_level_values("date")

    train_mask = date_level < val_start_ts
    val_mask   = (date_level >= val_start_ts) & (date_level < test_start_ts)

    if test_end_ts is not None:
        test_mask = (date_level >= test_start_ts) & (date_level <= test_end_ts)
    else:
        test_mask = date_level >= test_start_ts

    train_df = df[train_mask].copy()
    val_df   = df[val_mask].copy()
    test_df  = df[test_mask].copy()

    # Drop dòng chưa có label
    train_df = train_df.dropna(subset=["label"])
    val_df   = val_df.dropna(subset=["label"])
    test_df  = test_df.dropna(subset=["label"])

    # Log thông tin split
    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        "Split: train [%s → %s) %d rows | val [%s → %s) %d rows | test [%s → %s] %d rows",
        data_start.date(), val_start,   len(train_df),
        val_start,  test_start, len(val_df),
        test_start, test_end or data_end.date(), len(test_df),
    )

    return train_df, val_df, test_df


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Basic sanity checks on the input DataFrame."""
    required = {"open", "high", "close", "low", "volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame thiếu cột: {missing}")
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame phải có MultiIndex (date, ticker).")
    if df.index.names != ["date", "ticker"]:
        raise ValueError(
            f"MultiIndex names phải là ['date', 'ticker'], got {df.index.names}"
        )