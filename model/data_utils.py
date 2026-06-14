"""
Evo_Finance — Data Utilities
──────────────────────────────
Handles train/val/test splitting and label generation.

Expected input DataFrame
  - MultiIndex (date, ticker) — dates must be sorted ascending.
  - Columns: open, high, close, low, volume  (all numeric, no NaNs preferred).

All splits are **time-based** (no shuffling) to avoid look-ahead bias:
  train  = first TRAIN_RATIO of dates
  val    = next  VAL_RATIO  of dates
  test   = remainder
"""

from __future__ import annotations
from typing import Tuple, Callable

import numpy as np
import pandas as pd

from config.settings import TRAIN_RATIO, VAL_RATIO, LABEL_FN, HOLDING_HORIZON


def split_and_label(
    df: pd.DataFrame,
    label_fn: Callable = LABEL_FN,
    holding_horizon: int = HOLDING_HORIZON,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    1. Compute labels using label_fn — PER TICKER via groupby để tránh
       cross-ticker shift leakage.
    2. Split into train / val / test by date (time-based, no shuffle).
    3. Drop rows where label is NaN (future not yet available).

    Returns
    -------
    train_df, val_df, test_df — each with a 'label' column added.
    """
    df = df.copy()

    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            "DataFrame must have a MultiIndex (date, ticker). "
            "Flat DataFrames are not supported."
        )

    dates   = df.index.get_level_values("date").unique().sort_values()
    n       = len(dates)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_dates = dates[:n_train]
    val_dates   = dates[n_train : n_train + n_val]
    test_dates  = dates[n_train + n_val :]

    # ── Label: tính riêng cho từng ticker để tránh cross-ticker leakage ───────
    # shift(-h) trên full MultiIndex sẽ nhảy sang ticker khác ở cuối mỗi group.
    # Groupby ticker đảm bảo shift chỉ chạy trong chuỗi thời gian của 1 ticker.
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

    # ── Split theo date ────────────────────────────────────────────────────────
    date_level   = df.index.get_level_values("date")
    train_df = df.loc[date_level.isin(train_dates)].copy()
    val_df   = df.loc[date_level.isin(val_dates)].copy()
    test_df  = df.loc[date_level.isin(test_dates)].copy()

    # Bỏ các dòng chưa có label (cuối chuỗi, horizon chưa realised)
    train_df = train_df.dropna(subset=["label"])
    val_df   = val_df.dropna(subset=["label"])
    test_df  = test_df.dropna(subset=["label"])

    return train_df, val_df, test_df


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Basic sanity checks on the input DataFrame."""
    required = {"open", "high", "close", "low", "volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame must have MultiIndex (date, ticker).")
    if df.index.names != ["date", "ticker"]:
        raise ValueError(
            f"MultiIndex names must be ['date', 'ticker'], got {df.index.names}"
        )