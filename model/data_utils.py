"""
Data utilities for Evo_Finance.

The module keeps the old final split API and adds walk-forward fold creation
for the evolutionary loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import pandas as pd

from config import settings
from config.settings import (
    HOLDING_HORIZON,
    LABEL_FN,
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
    labeled = df.sort_index() if not df.index.is_monotonic_increasing else df
    labeled = labeled.copy()

    def _label_per_ticker(group: pd.DataFrame) -> pd.Series:
        try:
            return label_fn(group, h=holding_horizon)
        except TypeError:
            return label_fn(group)

    label_parts = [
        _label_per_ticker(group)
        for _, group in labeled.groupby(level="ticker", group_keys=False, sort=False)
    ]
    labels = (
        pd.concat(label_parts).reindex(labeled.index)
        if label_parts
        else pd.Series(dtype=float, index=labeled.index)
    )
    labeled["label"] = labels
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
        test_end_exclusive = _purged_inclusive_end(
            unique_dates,
            test_end_ts,
            purge_days,
        )
        test_mask = (all_dates >= test_start_ts) & (all_dates < test_end_exclusive)
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


def validate_temporal_splits(
    val_start: str = VAL_START,
    test_start: str = TEST_START,
    test_end: Optional[str] = TEST_END,
    wf_end: str = WF_END,
    wf_min_train_months: int = WF_MIN_TRAIN_MONTHS,
    wf_val_months: int = WF_VAL_MONTHS,
    wf_step_months: int = WF_STEP_MONTHS,
    wf_purge_days: int = WF_PURGE_DAYS,
    holding_horizon: int = HOLDING_HORIZON,
) -> None:
    """
    Validate runtime date/WF parameters, including CLI overrides.

    settings.validate_config() protects defaults at import time, but callers can
    still pass custom values. This guard keeps evolution out of final test and
    ensures split purging covers the forward label horizon.
    """
    val_start_ts = _parse_timestamp("VAL_START", val_start)
    test_start_ts = _parse_timestamp("TEST_START", test_start)
    wf_end_ts = _parse_timestamp("WF_END", wf_end)
    test_end_ts = _parse_optional_timestamp("TEST_END", test_end)

    if val_start_ts >= test_start_ts:
        raise ValueError("VAL_START must be before TEST_START.")
    if test_end_ts is not None and test_end_ts < test_start_ts:
        raise ValueError("TEST_END must be >= TEST_START.")
    if wf_end_ts > test_start_ts:
        raise ValueError("WF_END must be <= TEST_START to keep evolution out of final test.")

    if int(holding_horizon) < 1:
        raise ValueError("HOLDING_HORIZON must be positive.")
    if (
        int(wf_min_train_months) < 1
        or int(wf_val_months) < 1
        or int(wf_step_months) < 1
    ):
        raise ValueError("WF month settings must be positive.")
    if int(wf_purge_days) < 0:
        raise ValueError("WF_PURGE_DAYS must be non-negative.")
    if int(wf_purge_days) < int(holding_horizon):
        raise ValueError("WF_PURGE_DAYS must be >= HOLDING_HORIZON to avoid split leakage.")


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
        if val_end_candidate > wf_end_ts:
            break
        val_end = _first_date_at_or_after(unique_dates, val_end_candidate)
        if val_end is None:
            boundary_gap_days = (wf_end_ts - unique_dates[-1]).days
            if val_end_candidate <= wf_end_ts and 0 <= boundary_gap_days <= 14:
                val_end = wf_end_ts
            else:
                break
        elif val_end > wf_end_ts:
            val_end = wf_end_ts
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
    validate_balanced_panel(df)

    required_cols = sorted(required)
    numeric = df[required_cols].apply(pd.to_numeric, errors="coerce")
    bad_numeric = sorted(col for col in required_cols if numeric[col].isna().any())
    if bad_numeric:
        raise ValueError(f"OHLCV columns contain missing or non-numeric values: {bad_numeric}")

    price_cols = ["open", "high", "low", "close"]
    non_positive_prices = sorted(col for col in price_cols if (numeric[col] <= 0).any())
    if non_positive_prices:
        raise ValueError(f"OHLC price columns must be positive: {non_positive_prices}")
    if (numeric["volume"] <= 0).any():
        raise ValueError("Volume must be positive for every loaded row.")
    if (numeric["high"] < numeric["low"]).any():
        raise ValueError("OHLC rows must satisfy high >= low.")


def validate_balanced_panel(df: pd.DataFrame) -> None:
    """
    Ensure each trading date contains the same loaded ticker universe.

    Evolution compares stocks cross-sectionally each day. If direct --data
    inputs have a changing universe, hit-rate baselines, ranks, breadth and
    sector features become harder to compare across time.
    """
    _validate_index(df)
    dates = df.index.get_level_values("date")
    tickers = df.index.get_level_values("ticker")
    n_tickers = tickers.nunique()
    if n_tickers == 0:
        raise ValueError("Panel contains no tickers.")

    per_date_counts = tickers.to_series(index=df.index).groupby(dates).nunique()
    bad_dates = per_date_counts[per_date_counts != n_tickers]
    if not bad_dates.empty:
        sample = {
            str(pd.Timestamp(date).date()): int(count)
            for date, count in bad_dates.head(5).items()
        }
        raise ValueError(
            "Input panel must contain the full ticker universe on every date; "
            f"expected {n_tickers} tickers per date, bad date counts: {sample}."
        )


def validate_full_universe_panel(
    df: pd.DataFrame,
    expected_tickers,
    name: str = "panel",
) -> None:
    """
    Ensure every date in a split/fold still contains the expected ticker universe.

    Raw OHLCV validation checks the loaded panel before labels. This guard is
    for the post-label, post-dropna frames used by LightGBM/fitness, where a
    custom label function or bad row could otherwise remove only some tickers
    from a date and distort ranks, hit-rate baselines, breadth, and sector stats.
    """
    _validate_index(df)
    expected = {str(ticker).upper() for ticker in expected_tickers}
    if not expected:
        raise ValueError(f"{name} expected ticker universe is empty.")
    if df.empty:
        raise ValueError(f"{name} is empty after labeling/splitting.")

    dates = df.index.get_level_values("date")
    tickers = df.index.get_level_values("ticker")
    bad: dict[str, dict[str, list[str]]] = {}
    for date, group_tickers in tickers.to_series(index=df.index).groupby(dates):
        present = {str(ticker).upper() for ticker in group_tickers}
        if present != expected:
            bad[str(pd.Timestamp(date).date())] = {
                "missing": sorted(expected - present),
                "extra": sorted(present - expected),
            }
            if len(bad) >= 5:
                break

    if bad:
        raise ValueError(
            f"{name} must contain the full ticker universe on every date; "
            f"sample bad dates: {bad}."
        )


def validate_market_ohlcv(df: pd.DataFrame) -> None:
    """
    Ensure broad-market OHLCV columns exist for market/index primitives.

    Domain.seed() currently includes market features unconditionally, so direct
    --data inputs must already contain the columns normally created by
    data.loader.load_from_dir().
    """
    required = {
        "market_open", "market_high", "market_close",
        "market_low", "market_volume",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "DataFrame missing market columns required by market/index "
            f"primitives: {missing}. Use --data-dir with the configured "
            "market index CSV, or add market_* columns to direct --data input."
        )
    _validate_index(df)

    required_cols = sorted(required)
    numeric = df[required_cols].apply(pd.to_numeric, errors="coerce")
    bad_numeric = sorted(col for col in required_cols if numeric[col].isna().any())
    if bad_numeric:
        raise ValueError(
            "Market columns contain missing or non-numeric values: "
            f"{bad_numeric}. Market/index primitives require a complete "
            "benchmark series for every loaded trading date."
        )

    price_cols = ["market_open", "market_high", "market_low", "market_close"]
    non_positive_prices = sorted(col for col in price_cols if (numeric[col] <= 0).any())
    if non_positive_prices:
        raise ValueError(f"Market price columns must be positive: {non_positive_prices}")
    if (numeric["market_volume"] <= 0).any():
        raise ValueError("Market volume must be positive for every loaded row.")
    if (numeric["market_high"] < numeric["market_low"]).any():
        raise ValueError("Market OHLC rows must satisfy market_high >= market_low.")

    inconsistent_cols: list[str] = []
    for col in required_cols:
        per_date_unique = numeric[col].groupby(level="date").nunique(dropna=False)
        bad_dates = per_date_unique[per_date_unique > 1]
        if not bad_dates.empty:
            inconsistent_cols.append(f"{col} ({len(bad_dates)} dates)")

    if inconsistent_cols:
        raise ValueError(
            "Market columns must be identical within each date across all "
            f"tickers; inconsistent columns: {inconsistent_cols}."
        )


def tickers_missing_sector(df: pd.DataFrame) -> list[str]:
    """Return tickers present in df but absent from the SECTORS mapping."""
    _validate_index(df)
    mapped = {
        str(ticker).upper()
        for tickers in settings.SECTORS.values()
        for ticker in tickers
    }
    present = {
        str(ticker).upper()
        for ticker in df.index.get_level_values("ticker").unique()
    }
    return sorted(present - mapped)


def sector_member_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return loaded ticker counts per configured sector."""
    _validate_index(df)
    ticker_to_sector = {
        str(ticker).upper(): str(sector)
        for sector, tickers in settings.SECTORS.items()
        for ticker in tickers
    }
    present = {
        str(ticker).upper()
        for ticker in df.index.get_level_values("ticker").unique()
    }

    counts: dict[str, int] = {}
    for ticker in sorted(present):
        sector = ticker_to_sector.get(ticker)
        if sector is None:
            continue
        counts[sector] = counts.get(sector, 0) + 1
    return dict(sorted(counts.items()))


def small_sectors_in_universe(
    df: pd.DataFrame,
    min_members: int,
) -> dict[str, int]:
    """Return sectors represented by fewer than min_members loaded tickers."""
    min_members = int(min_members)
    if min_members <= 1:
        return {}
    return {
        sector: count
        for sector, count in sector_member_counts(df).items()
        if count < min_members
    }


def _validate_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame must have MultiIndex (date, ticker).")
    if df.index.names != ["date", "ticker"]:
        raise ValueError(
            f"MultiIndex names must be ['date', 'ticker'], got {df.index.names}"
        )
    if df.index.has_duplicates:
        raise ValueError("MultiIndex (date, ticker) must not contain duplicates.")
    dates = df.index.get_level_values("date")
    tickers = df.index.get_level_values("ticker")
    if dates.hasnans or tickers.hasnans:
        raise ValueError("MultiIndex date/ticker levels must not contain missing values.")
    try:
        pd.DatetimeIndex(pd.to_datetime(dates))
    except Exception as exc:
        raise ValueError("MultiIndex date level must be datetime-like.") from exc


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


def _purged_inclusive_end(
    dates: pd.DatetimeIndex,
    end: pd.Timestamp,
    purge_days: int,
) -> pd.Timestamp:
    """
    Return an exclusive upper bound for a closed test window after tail purge.

    If labels look forward N trading dates, the final N dates up to TEST_END
    cannot be scored without peeking past the declared test window.
    """
    pos_after_end = int(dates.searchsorted(end, side="right"))
    if pos_after_end >= len(dates) and int(purge_days) <= 0:
        return end + pd.Timedelta(nanoseconds=1)
    purged_pos = max(pos_after_end - max(int(purge_days), 0), 0)
    if purged_pos >= len(dates):
        return dates[-1] + pd.Timedelta(nanoseconds=1)
    return dates[purged_pos]


def _parse_timestamp(name: str, value: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a valid date string.") from exc


def _parse_optional_timestamp(name: str, value: str | None) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    return _parse_timestamp(name, value)
