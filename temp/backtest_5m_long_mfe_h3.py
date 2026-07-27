"""Backtest the 5-minute Long MFE H3 archive with slope-based Long exits.

Strategy:

1. Train the selected archive rank using its metadata Horizon and label.
2. Learn the top-fraction prediction cutoff on Final Val.
3. Apply that unchanged cutoff to Test.
4. Before entry, require initial_slope(t) < -minimum and a Short slowdown
   signal at t.
5. Enter Long at open H1 for every signal that passes the entry filter.
6. During H1 only, resolve TP versus the dedicated H1 stop chronologically
   across its five one-minute candles; a tie inside one minute uses SL-first.
7. After H1 and H2 close, recompute the two-candle high slope:
   - slope > +minimum and Long slowdown signal: arm the slowdown stop;
   - slope < -minimum and no Short slowdown signal: arm the slowdown stop;
   - neutral slope: hold.
8. TP is active only during H1. From H2 onward, only an armed slowdown stop
   can close the position early; an open below SL fills at that worse open,
   and any remaining position exits at close H3.

PowerShell:
    python temp/backtest_5m_long_mfe_h3.py `
      --archive crypto/results/crypto_btc_5m_long_mfe_h3_tp01_top40_seed1_8h.json `
      --long-slowdown-archive crypto/results/crypto_btc_5m_long_slope_slowdown_lb2_h1_thr003_top40_seed1_1h.json `
      --short-slowdown-archive crypto/results/crypto_btc_5m_Short_slope_slowdown_lb2_h1_thr003_top20_seed1_1h.json `
      --rank 1 `
      --top-fraction 0.40 `
      --long-slowdown-top-fraction 0.40 `
      --short-slowdown-top-fraction 0.40 `
      --entry-filter `
      --take-profit 0.001 `
      --h1-stop-loss 0.0004 `
      --slowdown-stop-loss 0.0004 `
      --trade-cost 0.00016 `
      --data data/crypto/BTCUSDT_5m.csv `
      --skip-1m-analysis
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto import config
from crypto.analyze import _required_windows_for_entries
from crypto.backtest import (
    ModelSpec,
    _archive_horizons,
    _cached_feature_space,
    _load_rank_entry,
    _quality_train_index,
    _train_spec_bundle,
)
from crypto.data import load_ohlcv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("temp.backtest_5m_long_mfe_h3")


DEFAULT_ARCHIVE = Path(
    "crypto/results/crypto_btc_5m_long_mfe_h3_tp01_top40_seed1_8h.json"
)
DEFAULT_LONG_SLOWDOWN_ARCHIVE = Path(
    "crypto/results/"
    "crypto_btc_5m_long_slope_slowdown_lb2_h1_thr003_top40_seed1_1h.json"
)
DEFAULT_SHORT_SLOWDOWN_ARCHIVE = Path(
    "crypto/results/"
    "crypto_btc_5m_Short_slope_slowdown_lb2_h1_thr003_top20_seed1_1h.json"
)
DEFAULT_DATA = Path("data/crypto/BTCUSDT_5m.csv")
DEFAULT_DATA_1M = Path("data/crypto/BTCUSDT_1m.csv")
DEFAULT_OUT_DIR = Path("temp/output")
DEFAULT_RANK = 1
DEFAULT_TOP_FRACTION = 0.40
DEFAULT_SLOWDOWN_RANK = 1
DEFAULT_LONG_SLOWDOWN_TOP_FRACTION = 0.40
DEFAULT_SHORT_SLOWDOWN_TOP_FRACTION = 0.40
DEFAULT_TAKE_PROFIT = 0.001
DEFAULT_H1_STOP_LOSS = 0.0004
DEFAULT_SLOWDOWN_STOP_LOSS = 0.0004
DEFAULT_TRADE_COST = 0.00016
DEFAULT_STOP_LOSS = 0.001
DEFAULT_SAME_CANDLE_POLICY = "stop_first"
MINUTE_CHECKPOINTS = (1, 2, 3, 5, 10)
EXIT_RETURN_THRESHOLDS = tuple(
    float(value) for value in np.arange(-0.0020, 0.00101, 0.00025)
)
NO_H1_MFE_THRESHOLDS = tuple(
    float(value) for value in np.arange(-0.0010, 0.00201, 0.00025)
)


def load_archive_metadata(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if not isinstance(metadata, dict):
        raise ValueError(f"Archive metadata must be an object: {path}")
    return dict(metadata)


def make_price_path(raw_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Build future OHLC returns relative to the next-candle entry open."""
    entry = pd.to_numeric(raw_df["open"], errors="coerce").shift(-1)
    result = pd.DataFrame(index=raw_df.index)
    result["entry_open"] = entry
    for step in range(1, int(horizon) + 1):
        result[f"open_h{step}"] = (
            pd.to_numeric(raw_df["open"], errors="coerce").shift(-step).div(entry)
            - 1.0
        )
        result[f"high_h{step}"] = (
            pd.to_numeric(raw_df["high"], errors="coerce").shift(-step).div(entry)
            - 1.0
        )
        result[f"low_h{step}"] = (
            pd.to_numeric(raw_df["low"], errors="coerce").shift(-step).div(entry)
            - 1.0
        )
    result[f"close_h{horizon}"] = (
        pd.to_numeric(raw_df["close"], errors="coerce").shift(-horizon).div(entry)
        - 1.0
    )
    return result.replace([np.inf, -np.inf], np.nan)


def simulate_long_tp_stop(
    path: pd.DataFrame,
    horizon: int,
    take_profit: float,
    stop_loss: float,
    trade_cost: float,
    same_candle_policy: str = DEFAULT_SAME_CANDLE_POLICY,
) -> pd.DataFrame:
    """Simulate one Long position per row using conservative OHLC ordering."""
    horizon = int(horizon)
    take_profit = float(take_profit)
    stop_loss = float(stop_loss)
    trade_cost = float(trade_cost)
    policy = str(same_candle_policy).strip().lower()
    if horizon < 1:
        raise ValueError("horizon must be positive.")
    if take_profit <= 0.0 or stop_loss <= 0.0:
        raise ValueError("take_profit and stop_loss must be positive distances.")
    if policy not in {"stop_first", "tp_first"}:
        raise ValueError("same_candle_policy must be stop_first or tp_first.")

    required = [
        *[f"high_h{step}" for step in range(1, horizon + 1)],
        *[f"low_h{step}" for step in range(1, horizon + 1)],
        f"close_h{horizon}",
    ]
    missing = [column for column in required if column not in path.columns]
    if missing:
        raise ValueError(f"Price path is missing columns: {missing}")

    result = path.dropna(subset=required).copy()
    n_rows = len(result)
    active = np.ones(n_rows, dtype=bool)
    gross_return = np.full(n_rows, np.nan, dtype=float)
    exit_h = np.full(n_rows, horizon, dtype=int)
    outcome = np.full(n_rows, f"close_h{horizon}", dtype=object)

    for step in range(1, horizon + 1):
        high_hit = (
            pd.to_numeric(result[f"high_h{step}"], errors="coerce").to_numpy()
            >= take_profit
        )
        stop_hit = (
            pd.to_numeric(result[f"low_h{step}"], errors="coerce").to_numpy()
            <= -stop_loss
        )
        if policy == "stop_first":
            stop_exit = active & stop_hit
            tp_exit = active & ~stop_hit & high_hit
        else:
            tp_exit = active & high_hit
            stop_exit = active & ~high_hit & stop_hit

        gross_return[stop_exit] = -stop_loss
        exit_h[stop_exit] = step
        outcome[stop_exit] = f"stop_h{step}"
        active[stop_exit] = False

        gross_return[tp_exit] = take_profit
        exit_h[tp_exit] = step
        outcome[tp_exit] = f"tp_h{step}"
        active[tp_exit] = False

    close_return = pd.to_numeric(
        result[f"close_h{horizon}"], errors="coerce"
    ).to_numpy()
    gross_return[active] = close_return[active]

    result["outcome"] = outcome
    result["exit_h"] = exit_h
    result["gross_return"] = gross_return
    result["net_return"] = result["gross_return"] - trade_cost
    result["cumulative_net_return"] = result["net_return"].cumsum()
    return result


def simulate_long_slowdown_strategy(
    selected_path: pd.DataFrame,
    h1_barrier_outcomes: pd.DataFrame,
    raw_index: pd.DatetimeIndex,
    initial_slope: pd.Series,
    long_slowdown_signals: Any,
    short_slowdown_signals: Any,
    slope_min_initial: float,
    take_profit: float,
    h1_stop_loss: float,
    slowdown_stop_loss: float,
    trade_cost: float,
) -> pd.DataFrame:
    """Run the Long position state machine through H3."""
    required = [
        "open_h2",
        "open_h3",
        "low_h2",
        "low_h3",
        "close_h3",
    ]
    result = selected_path.dropna(subset=required).copy()
    if result.empty:
        return result.assign(
            outcome=pd.Series(dtype=str),
            exit_h=pd.Series(dtype=int),
            gross_return=pd.Series(dtype=float),
            net_return=pd.Series(dtype=float),
            cumulative_net_return=pd.Series(dtype=float),
        )

    base_positions = raw_index.get_indexer(pd.DatetimeIndex(result.index))
    if bool((base_positions < 0).any()):
        raise ValueError("Selected base signal is missing from the raw data index.")

    decision_h1_index = raw_index.take(base_positions + 1)
    decision_h2_index = raw_index.take(base_positions + 2)
    slope_h1 = initial_slope.reindex(decision_h1_index).to_numpy(dtype=float)
    slope_h2 = initial_slope.reindex(decision_h2_index).to_numpy(dtype=float)
    long_selected = pd.Index(long_slowdown_signals.selected_index)
    short_selected = pd.Index(short_slowdown_signals.selected_index)
    long_signal_h1 = decision_h1_index.isin(long_selected)
    long_signal_h2 = decision_h2_index.isin(long_selected)
    short_signal_h1 = decision_h1_index.isin(short_selected)
    short_signal_h2 = decision_h2_index.isin(short_selected)

    open_h2 = pd.to_numeric(result["open_h2"], errors="coerce").to_numpy()
    open_h3 = pd.to_numeric(result["open_h3"], errors="coerce").to_numpy()
    low_h2 = pd.to_numeric(result["low_h2"], errors="coerce").to_numpy()
    low_h3 = pd.to_numeric(result["low_h3"], errors="coerce").to_numpy()
    close_h3 = pd.to_numeric(result["close_h3"], errors="coerce").to_numpy()
    tp = float(take_profit)
    first_candle_stop = float(h1_stop_loss)
    stop_loss = float(slowdown_stop_loss)
    min_slope = float(slope_min_initial)
    if first_candle_stop <= 0.0 or stop_loss <= 0.0:
        raise ValueError(
            "h1_stop_loss and slowdown_stop_loss must be positive distances."
        )

    n = len(result)
    active = np.ones(n, dtype=bool)
    gross_return = np.full(n, np.nan, dtype=float)
    exit_h = np.full(n, 3, dtype=int)
    outcome = np.full(n, "close_h3", dtype=object)
    decision_h1 = np.full(n, "not_reached", dtype=object)
    decision_h2 = np.full(n, "not_reached", dtype=object)

    h1_resolution = h1_barrier_outcomes.reindex(result.index)
    if h1_resolution["h1_outcome"].isna().any():
        missing = result.index[h1_resolution["h1_outcome"].isna()][:5]
        raise ValueError(
            "Missing 1m H1 TP/SL resolution for signals: "
            f"{list(missing)}"
        )
    h1_outcome = h1_resolution["h1_outcome"].astype(str).to_numpy()
    h1_first_minute = pd.to_numeric(
        h1_resolution["h1_first_minute"],
        errors="coerce",
    ).to_numpy(dtype=float)
    h1_fill_return = pd.to_numeric(
        h1_resolution["h1_fill_return"],
        errors="coerce",
    ).to_numpy(dtype=float)
    h1_same_minute_tie = (
        h1_resolution["h1_same_minute_tie"].fillna(False).to_numpy(dtype=bool)
    )

    # H1 barrier ordering is resolved from exactly five one-minute candles.
    stop_h1 = active & (h1_outcome == "h1_stop")
    gross_return[stop_h1] = h1_fill_return[stop_h1]
    exit_h[stop_h1] = 1
    outcome[stop_h1] = "h1_stop"
    decision_h1[stop_h1] = "h1_stop_before_decision"
    active[stop_h1] = False

    hit_h1 = active & (h1_outcome == "tp_h1")
    gross_return[hit_h1] = h1_fill_return[hit_h1]
    exit_h[hit_h1] = 1
    outcome[hit_h1] = "tp_h1"
    decision_h1[hit_h1] = "tp_before_decision"
    active[hit_h1] = False

    positive_h1 = active & (slope_h1 > min_slope)
    negative_h1 = active & (slope_h1 < -min_slope)
    neutral_h1 = active & ~(positive_h1 | negative_h1)
    exit_after_h1 = (positive_h1 & long_signal_h1) | (
        negative_h1 & ~short_signal_h1
    )
    decision_h1[positive_h1 & long_signal_h1] = "long_slowdown_exit"
    decision_h1[positive_h1 & ~long_signal_h1] = "long_no_signal_hold"
    decision_h1[negative_h1 & short_signal_h1] = "short_slowdown_hold"
    decision_h1[negative_h1 & ~short_signal_h1] = "short_no_signal_exit"
    decision_h1[neutral_h1] = "neutral_hold"
    stop_armed_h1 = exit_after_h1.copy()

    # The H1-close decision can only arm a stop from H2 onward. TP is disabled
    # after H1, so a gap through SL exits at open and a low touch fills at SL.
    stop_gap_h2 = active & stop_armed_h1 & (open_h2 <= -stop_loss)
    gross_return[stop_gap_h2] = open_h2[stop_gap_h2]
    exit_h[stop_gap_h2] = 2
    outcome[stop_gap_h2] = "slowdown_stop_open_h2"
    active[stop_gap_h2] = False

    stop_h2 = active & stop_armed_h1 & (low_h2 <= -stop_loss)
    gross_return[stop_h2] = -stop_loss
    exit_h[stop_h2] = 2
    outcome[stop_h2] = "slowdown_stop_h2"
    active[stop_h2] = False

    decision_h2_eligible = active & ~stop_armed_h1
    decision_h2[active & stop_armed_h1] = "stop_already_armed"
    positive_h2 = decision_h2_eligible & (slope_h2 > min_slope)
    negative_h2 = decision_h2_eligible & (slope_h2 < -min_slope)
    neutral_h2 = decision_h2_eligible & ~(positive_h2 | negative_h2)
    exit_after_h2 = (positive_h2 & long_signal_h2) | (
        negative_h2 & ~short_signal_h2
    )
    decision_h2[positive_h2 & long_signal_h2] = "long_slowdown_exit"
    decision_h2[positive_h2 & ~long_signal_h2] = "long_no_signal_hold"
    decision_h2[negative_h2 & short_signal_h2] = "short_slowdown_hold"
    decision_h2[negative_h2 & ~short_signal_h2] = "short_no_signal_exit"
    decision_h2[neutral_h2] = "neutral_hold"
    stop_armed_h2 = active & (stop_armed_h1 | exit_after_h2)

    stop_gap_h3 = active & stop_armed_h2 & (open_h3 <= -stop_loss)
    gross_return[stop_gap_h3] = open_h3[stop_gap_h3]
    exit_h[stop_gap_h3] = 3
    outcome[stop_gap_h3] = "slowdown_stop_open_h3"
    active[stop_gap_h3] = False

    stop_h3 = active & stop_armed_h2 & (low_h3 <= -stop_loss)
    gross_return[stop_h3] = -stop_loss
    exit_h[stop_h3] = 3
    outcome[stop_h3] = "slowdown_stop_h3"
    active[stop_h3] = False

    gross_return[active] = close_h3[active]
    result["initial_slope_h1"] = slope_h1
    result["initial_slope_h2"] = slope_h2
    result["h1_first_minute"] = h1_first_minute
    result["h1_same_minute_tie"] = h1_same_minute_tie
    result["decision_h1"] = decision_h1
    result["decision_h2"] = decision_h2
    result["outcome"] = outcome
    result["exit_h"] = exit_h
    result["gross_return"] = gross_return
    result["net_return"] = gross_return - float(trade_cost)
    result["cumulative_net_return"] = result["net_return"].cumsum()
    return result


def load_one_minute_ohlc(
    path: str | Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Load only the 1-minute OHLC rows needed by selected signals."""
    minute = pd.read_csv(
        Path(path),
        usecols=["date", "open", "high", "low", "close"],
    )
    minute["date"] = pd.to_datetime(minute["date"], errors="coerce")
    minute = minute.dropna(subset=["date"]).set_index("date").sort_index()
    minute = minute.loc[
        (minute.index >= pd.Timestamp(start))
        & (minute.index <= pd.Timestamp(end))
    ].copy()
    for column in ("open", "high", "low", "close"):
        minute[column] = pd.to_numeric(minute[column], errors="coerce")
    minute = minute.dropna(subset=["open", "high", "low", "close"])
    if not minute.index.is_unique:
        duplicates = minute.index[minute.index.duplicated()].unique()[:5]
        raise ValueError(
            "1m OHLC contains duplicate timestamps; examples: "
            f"{list(duplicates)}"
        )
    return minute


def resolve_h1_tp_sl_from_1m(
    signal_index: pd.Index,
    expected_entry_open: pd.Series,
    minute: pd.DataFrame,
    take_profit: float,
    h1_stop_loss: float,
) -> pd.DataFrame:
    """Resolve H1 TP/SL order from the five causal 1-minute candles.

    A 5-minute signal indexed at t enters at t+5 minutes. H1 therefore
    contains one-minute bars with open timestamps t+5 through t+9. Within
    each minute SL wins a TP/SL tie because lower-frequency OHLC cannot reveal
    the ordering inside that minute.
    """
    signals = pd.DatetimeIndex(pd.to_datetime(signal_index, errors="coerce"))
    if signals.isna().any():
        raise ValueError("Signal index contains invalid timestamps.")
    if signals.empty:
        return pd.DataFrame(
            columns=[
                "h1_outcome",
                "h1_first_minute",
                "h1_fill_return",
                "h1_same_minute_tie",
                "entry_time",
            ],
            index=signals,
        )
    misaligned = (
        (signals.second != 0)
        | (signals.microsecond != 0)
        | ((signals.minute % 5) != 0)
    )
    if bool(misaligned.any()):
        raise ValueError(
            "5m signal timestamps must align to five-minute open times; "
            f"examples: {list(signals[misaligned][:5])}"
        )

    entry_times = signals + pd.Timedelta(minutes=5)
    offsets = np.arange(5, dtype="timedelta64[m]")
    lookup_values = (
        entry_times.to_numpy(dtype="datetime64[ns]")[:, None] + offsets[None, :]
    )
    lookup_index = pd.DatetimeIndex(lookup_values.reshape(-1))
    aligned = minute.reindex(lookup_index)
    shape = (len(signals), 5)
    open_values = aligned["open"].to_numpy(dtype=float).reshape(shape)
    high_values = aligned["high"].to_numpy(dtype=float).reshape(shape)
    low_values = aligned["low"].to_numpy(dtype=float).reshape(shape)
    complete = (
        np.isfinite(open_values).all(axis=1)
        & np.isfinite(high_values).all(axis=1)
        & np.isfinite(low_values).all(axis=1)
        & (open_values[:, 0] > 0.0)
    )
    if not bool(complete.all()):
        missing_signals = signals[~complete][:5]
        missing_entries = entry_times[~complete][:5]
        raise ValueError(
            "Missing one or more of the five 1m H1 candles. "
            f"signal examples={list(missing_signals)}, "
            f"entry examples={list(missing_entries)}"
        )

    expected = pd.to_numeric(
        expected_entry_open.reindex(signals),
        errors="coerce",
    ).to_numpy(dtype=float)
    entry = open_values[:, 0]
    entry_matches = np.isfinite(expected) & np.isclose(
        entry,
        expected,
        rtol=1e-10,
        atol=1e-8,
    )
    if not bool(entry_matches.all()):
        bad = np.flatnonzero(~entry_matches)[:5]
        details = [
            {
                "signal": str(signals[position]),
                "entry_time": str(entry_times[position]),
                "open_5m": float(expected[position]),
                "open_1m": float(entry[position]),
            }
            for position in bad
        ]
        raise ValueError(
            "1m H1 entry open does not match the 5m entry open: "
            f"{details}"
        )

    high_return = high_values / entry[:, None] - 1.0
    low_return = low_values / entry[:, None] - 1.0
    open_return = open_values / entry[:, None] - 1.0
    tp_hits = high_return >= float(take_profit)
    sl_hits = low_return <= -float(h1_stop_loss)
    active = np.ones(len(signals), dtype=bool)
    outcomes = np.full(len(signals), "h1_none", dtype=object)
    first_minute = np.zeros(len(signals), dtype=np.int8)
    fill_return = np.full(len(signals), np.nan, dtype=float)
    same_minute_tie = np.zeros(len(signals), dtype=bool)
    for minute_offset in range(5):
        gap_stop = active & (open_return[:, minute_offset] <= -float(h1_stop_loss))
        outcomes[gap_stop] = "h1_stop"
        first_minute[gap_stop] = minute_offset + 1
        fill_return[gap_stop] = open_return[gap_stop, minute_offset]
        active[gap_stop] = False

        # Same 1m candle touches both barriers: SL wins.
        stop_now = active & sl_hits[:, minute_offset]
        outcomes[stop_now] = "h1_stop"
        first_minute[stop_now] = minute_offset + 1
        fill_return[stop_now] = -float(h1_stop_loss)
        same_minute_tie[stop_now] = tp_hits[stop_now, minute_offset]
        active[stop_now] = False

        tp_now = active & tp_hits[:, minute_offset]
        outcomes[tp_now] = "tp_h1"
        first_minute[tp_now] = minute_offset + 1
        fill_return[tp_now] = float(take_profit)
        active[tp_now] = False

    result = pd.DataFrame(index=signals)
    result.index.name = "signal_time"
    result["h1_outcome"] = outcomes
    result["h1_first_minute"] = first_minute
    result["h1_fill_return"] = fill_return
    result["h1_same_minute_tie"] = same_minute_tie
    result["entry_time"] = entry_times
    return result


def build_one_minute_paths(
    signal_index: pd.Index,
    minute: pd.DataFrame,
    take_profit: float,
    minutes: int = 15,
) -> pd.DataFrame:
    """Align each 5-minute signal with H1-H3 represented by 15 one-minute bars."""
    signals = pd.DatetimeIndex(pd.to_datetime(signal_index, errors="coerce"))
    signals = signals[~signals.isna()]
    if signals.empty:
        return pd.DataFrame()

    entry_times = signals + pd.Timedelta(minutes=5)
    offsets = np.arange(int(minutes), dtype="timedelta64[m]")
    lookup_values = (
        entry_times.to_numpy(dtype="datetime64[ns]")[:, None] + offsets[None, :]
    )
    lookup_index = pd.DatetimeIndex(lookup_values.reshape(-1))
    aligned = minute.reindex(lookup_index)
    shape = (len(signals), int(minutes))
    open_values = aligned["open"].to_numpy(dtype=float).reshape(shape)
    high_values = aligned["high"].to_numpy(dtype=float).reshape(shape)
    low_values = aligned["low"].to_numpy(dtype=float).reshape(shape)
    close_values = aligned["close"].to_numpy(dtype=float).reshape(shape)
    valid = (
        np.isfinite(open_values).all(axis=1)
        & np.isfinite(high_values).all(axis=1)
        & np.isfinite(low_values).all(axis=1)
        & np.isfinite(close_values).all(axis=1)
        & (open_values[:, 0] > 0.0)
    )
    if not valid.any():
        return pd.DataFrame()

    signals = signals[valid]
    entry_times = entry_times[valid]
    entry = open_values[valid, 0]
    high_return = high_values[valid] / entry[:, None] - 1.0
    low_return = low_values[valid] / entry[:, None] - 1.0
    close_return = close_values[valid] / entry[:, None] - 1.0
    hit_matrix = high_return >= float(take_profit)
    tp_hit = hit_matrix.any(axis=1)
    first_hit = np.where(tp_hit, hit_matrix.argmax(axis=1) + 1, 0)

    result = pd.DataFrame(index=signals)
    result.index.name = "signal_time"
    result["entry_time"] = entry_times
    result["entry_open"] = entry
    result["tp_hit"] = tp_hit
    result["tp_first_minute"] = first_hit
    for minute_number in range(1, int(minutes) + 1):
        end_index = minute_number - 1
        result[f"close_m{minute_number}"] = close_return[:, end_index]
        result[f"mfe_m{minute_number}"] = np.max(
            high_return[:, :minute_number],
            axis=1,
        )
        result[f"mae_m{minute_number}"] = np.min(
            low_return[:, :minute_number],
            axis=1,
        )
    return result


def minute_checkpoint_rows(
    split: str,
    paths: pd.DataFrame,
    checkpoints: tuple[int, ...] = MINUTE_CHECKPOINTS,
) -> list[dict[str, Any]]:
    """Compare eventual TP hits and misses among positions still open."""
    rows: list[dict[str, Any]] = []
    for minute_number in checkpoints:
        already_hit = paths["tp_hit"] & paths["tp_first_minute"].le(minute_number)
        active = paths.loc[~already_hit]
        future_hit = active["tp_hit"]
        hit_group = active.loc[future_hit]
        miss_group = active.loc[~future_hit]
        close_column = f"close_m{minute_number}"
        mfe_column = f"mfe_m{minute_number}"
        mae_column = f"mae_m{minute_number}"
        rows.append(
            {
                "split": split,
                "minute": int(minute_number),
                "active": int(len(active)),
                "eventual_hit_rate": (
                    float(future_hit.mean()) if len(active) else float("nan")
                ),
                "hit_close_now": (
                    float(hit_group[close_column].mean())
                    if len(hit_group)
                    else float("nan")
                ),
                "miss_close_now": (
                    float(miss_group[close_column].mean())
                    if len(miss_group)
                    else float("nan")
                ),
                "hit_mfe_now": (
                    float(hit_group[mfe_column].mean())
                    if len(hit_group)
                    else float("nan")
                ),
                "miss_mfe_now": (
                    float(miss_group[mfe_column].mean())
                    if len(miss_group)
                    else float("nan")
                ),
                "hit_mae_now": (
                    float(hit_group[mae_column].mean())
                    if len(hit_group)
                    else float("nan")
                ),
                "miss_mae_now": (
                    float(miss_group[mae_column].mean())
                    if len(miss_group)
                    else float("nan")
                ),
                "miss_close_h3": (
                    float(miss_group["close_m15"].mean())
                    if len(miss_group)
                    else float("nan")
                ),
            }
        )
    return rows


def early_exit_grid(
    split: str,
    paths: pd.DataFrame,
    take_profit: float,
    trade_cost: float,
) -> pd.DataFrame:
    """Evaluate causal close-return exit rules on positions still open."""
    baseline_gross = np.where(
        paths["tp_hit"].to_numpy(dtype=bool),
        float(take_profit),
        pd.to_numeric(paths["close_m15"], errors="coerce").to_numpy(),
    )
    rows: list[dict[str, Any]] = []
    for minute_number in MINUTE_CHECKPOINTS:
        already_hit = (
            paths["tp_hit"] & paths["tp_first_minute"].le(minute_number)
        ).to_numpy(dtype=bool)
        close_now = pd.to_numeric(
            paths[f"close_m{minute_number}"],
            errors="coerce",
        ).to_numpy()
        for threshold in EXIT_RETURN_THRESHOLDS:
            exit_mask = ~already_hit & (close_now <= float(threshold))
            strategy_gross = baseline_gross.copy()
            strategy_gross[exit_mask] = close_now[exit_mask]
            exited = int(exit_mask.sum())
            exited_miss = int(
                (exit_mask & ~paths["tp_hit"].to_numpy(dtype=bool)).sum()
            )
            rows.append(
                {
                    "split": split,
                    "minute": int(minute_number),
                    "exit_threshold": float(threshold),
                    "exit_count": exited,
                    "exit_rate": float(exited / len(paths)) if len(paths) else 0.0,
                    "exit_miss_precision": (
                        float(exited_miss / exited) if exited else float("nan")
                    ),
                    "false_exit_count": exited - exited_miss,
                    "gross_mean": float(np.nanmean(strategy_gross)),
                    "net_mean": float(np.nanmean(strategy_gross) - trade_cost),
                    "baseline_net_mean": float(
                        np.nanmean(baseline_gross) - trade_cost
                    ),
                }
            )
    return pd.DataFrame(rows)


def _format_minute_diagnostics(rows: pd.DataFrame) -> pd.DataFrame:
    display = rows.copy()
    for column in (
        "eventual_hit_rate",
        "hit_close_now",
        "miss_close_now",
        "hit_mfe_now",
        "miss_mfe_now",
        "hit_mae_now",
        "miss_mae_now",
        "miss_close_h3",
    ):
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):+.3%}"
        )
    return display


def draw_one_minute_miss_report(
    paths_by_split: dict[str, pd.DataFrame],
    checkpoint_rows: pd.DataFrame,
    exit_summary: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(18.0, 15.0),
        gridspec_kw={"height_ratios": [2.1, 3.0, 1.5]},
        constrained_layout=True,
    )
    ax_table, ax_path, ax_exit = axes
    ax_table.axis("off")
    display = _format_minute_diagnostics(checkpoint_rows)
    table = ax_table.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.35)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#9ca3af")
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")

    colors = {
        ("val", True): "#2563eb",
        ("val", False): "#dc2626",
        ("test", True): "#16a34a",
        ("test", False): "#f59e0b",
    }
    minute_axis = np.arange(1, 16)
    for split, paths in paths_by_split.items():
        for hit_value, group_name in ((True, "TP_HIT"), (False, "TP_MISS")):
            group = paths.loc[paths["tp_hit"] == hit_value]
            mean_path = np.array(
                [group[f"close_m{minute}"].mean() for minute in minute_axis]
            )
            ax_path.plot(
                minute_axis,
                mean_path * 100.0,
                color=colors[(split, hit_value)],
                linewidth=1.8,
                label=f"{split.upper()} {group_name} n={len(group):,}",
            )
    ax_path.axhline(0.0, color="#4b5563", linestyle="--", linewidth=0.8)
    ax_path.set_xticks(minute_axis)
    ax_path.set_title("Mean 1-minute close path from open H1")
    ax_path.set_xlabel("Elapsed minute")
    ax_path.set_ylabel("Return from entry (%)")
    ax_path.grid(True, color="#d1d5db", alpha=0.65, linewidth=0.6)
    ax_path.legend(frameon=False, ncol=2)

    ax_exit.axis("off")
    exit_display = exit_summary.copy()
    for column in (
        "threshold",
        "val_net",
        "val_delta",
        "test_net",
        "test_delta",
        "val_exit_rate",
        "test_exit_rate",
        "val_miss_precision",
        "test_miss_precision",
    ):
        exit_display[column] = exit_display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):+.3%}"
        )
    exit_table = ax_exit.table(
        cellText=exit_display.values,
        colLabels=exit_display.columns,
        cellLoc="center",
        loc="center",
    )
    exit_table.auto_set_font_size(False)
    exit_table.set_fontsize(8)
    exit_table.scale(1.0, 1.5)
    for (row, _), cell in exit_table.get_celld().items():
        cell.set_edgecolor("#9ca3af")
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")

    fig.suptitle(title, fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def summarize_split(
    split: str,
    simulations: pd.DataFrame,
    available_rows: int,
    prediction_threshold: float,
) -> dict[str, Any]:
    n = len(simulations)
    outcomes = simulations.get("outcome", pd.Series(dtype=str)).astype(str)
    tp_count = int(outcomes.str.startswith("tp_").sum())
    exit_h = pd.to_numeric(
        simulations.get("exit_h", pd.Series(index=simulations.index, dtype=float)),
        errors="coerce",
    )
    tp_mask = outcomes.str.startswith("tp_")
    tp_h_counts = {
        step: int((tp_mask & exit_h.le(step)).sum())
        for step in range(1, 4)
    }
    h1_stop_count = int(outcomes.eq("h1_stop").sum())
    h1_tie = (
        simulations.get(
            "h1_same_minute_tie",
            pd.Series(False, index=simulations.index),
        )
        .fillna(False)
        .astype(bool)
    )
    h1_tie_count = int(h1_tie.sum())
    exit_open_h2_mask = outcomes.eq("slowdown_stop_open_h2")
    exit_open_h3_mask = outcomes.eq("slowdown_stop_open_h3")
    slowdown_stop_h2_mask = outcomes.eq("slowdown_stop_h2")
    slowdown_stop_h3_mask = outcomes.eq("slowdown_stop_h3")
    exit_open_h2_count = int(exit_open_h2_mask.sum())
    exit_open_h3_count = int(exit_open_h3_mask.sum())
    slowdown_stop_h2_count = int(slowdown_stop_h2_mask.sum())
    slowdown_stop_h3_count = int(slowdown_stop_h3_mask.sum())
    close_count = int(outcomes.str.startswith("close_").sum())
    gross_return = pd.to_numeric(
        simulations.get(
            "gross_return",
            pd.Series(index=simulations.index, dtype=float),
        ),
        errors="coerce",
    )
    h1_stop_mask = outcomes.eq("h1_stop")
    close_mask = outcomes.str.startswith("close_")
    close_h3_return = pd.to_numeric(
        simulations.get(
            "close_h3",
            pd.Series(index=simulations.index, dtype=float),
        ),
        errors="coerce",
    )
    slowdown_stop_h2_mean = (
        float(gross_return.loc[slowdown_stop_h2_mask].mean())
        if slowdown_stop_h2_count
        else float("nan")
    )
    slowdown_stop_h3_mean = (
        float(gross_return.loc[slowdown_stop_h3_mask].mean())
        if slowdown_stop_h3_count
        else float("nan")
    )
    slowdown_stop_h2_close_h3_mean = (
        float(close_h3_return.loc[slowdown_stop_h2_mask].mean())
        if slowdown_stop_h2_count
        else float("nan")
    )
    slowdown_stop_h3_close_h3_mean = (
        float(close_h3_return.loc[slowdown_stop_h3_mask].mean())
        if slowdown_stop_h3_count
        else float("nan")
    )
    exit_open_h2_close_h3_mean = (
        float(close_h3_return.loc[exit_open_h2_mask].mean())
        if exit_open_h2_count
        else float("nan")
    )
    exit_open_h3_close_h3_mean = (
        float(close_h3_return.loc[exit_open_h3_mask].mean())
        if exit_open_h3_count
        else float("nan")
    )
    index = pd.DatetimeIndex(simulations.index)
    day_count = max((index.max() - index.min()).total_seconds() / 86400.0, 1.0) if n else 0.0
    return {
        "split": split,
        "available_rows": int(available_rows),
        "signals": int(n),
        "selected_rate": float(n / available_rows) if available_rows else 0.0,
        "prediction_threshold": float(prediction_threshold),
        "trades_per_day": float(n / day_count) if day_count else 0.0,
        "tp_count": tp_count,
        "tp_rate": float(tp_count / n) if n else 0.0,
        **{
            f"tp_h{step}_count": tp_h_counts[step]
            for step in range(1, 4)
        },
        **{
            f"tp_h{step}_rate": float(tp_h_counts[step] / n) if n else 0.0
            for step in range(1, 4)
        },
        "h1_stop_count": h1_stop_count,
        "h1_stop_rate": float(h1_stop_count / n) if n else 0.0,
        "h1_stop_mean": (
            float(gross_return.loc[h1_stop_mask].mean())
            if h1_stop_count
            else float("nan")
        ),
        "h1_tie_count": h1_tie_count,
        "h1_tie_rate": float(h1_tie_count / n) if n else 0.0,
        "slowdown_stop_h2_count": slowdown_stop_h2_count,
        "slowdown_stop_h2_rate": (
            float(slowdown_stop_h2_count / n) if n else 0.0
        ),
        "slowdown_stop_h2_mean": slowdown_stop_h2_mean,
        "exit_open_h2_count": exit_open_h2_count,
        "exit_open_h2_rate": float(exit_open_h2_count / n) if n else 0.0,
        "exit_open_h2_mean": (
            float(gross_return.loc[exit_open_h2_mask].mean())
            if exit_open_h2_count
            else float("nan")
        ),
        "exit_open_h2_close_h3_mean": exit_open_h2_close_h3_mean,
        "exit_open_h2_benefit": (
            float(gross_return.loc[exit_open_h2_mask].mean())
            - exit_open_h2_close_h3_mean
            if exit_open_h2_count
            else float("nan")
        ),
        "slowdown_stop_h2_close_h3_mean": slowdown_stop_h2_close_h3_mean,
        "slowdown_stop_h2_benefit": (
            slowdown_stop_h2_mean - slowdown_stop_h2_close_h3_mean
        ),
        "slowdown_stop_h3_count": slowdown_stop_h3_count,
        "slowdown_stop_h3_rate": (
            float(slowdown_stop_h3_count / n) if n else 0.0
        ),
        "slowdown_stop_h3_mean": slowdown_stop_h3_mean,
        "exit_open_h3_count": exit_open_h3_count,
        "exit_open_h3_rate": float(exit_open_h3_count / n) if n else 0.0,
        "exit_open_h3_mean": (
            float(gross_return.loc[exit_open_h3_mask].mean())
            if exit_open_h3_count
            else float("nan")
        ),
        "exit_open_h3_close_h3_mean": exit_open_h3_close_h3_mean,
        "exit_open_h3_benefit": (
            float(gross_return.loc[exit_open_h3_mask].mean())
            - exit_open_h3_close_h3_mean
            if exit_open_h3_count
            else float("nan")
        ),
        "slowdown_stop_h3_close_h3_mean": slowdown_stop_h3_close_h3_mean,
        "slowdown_stop_h3_benefit": (
            slowdown_stop_h3_mean - slowdown_stop_h3_close_h3_mean
        ),
        "close_count": close_count,
        "close_rate": float(close_count / n) if n else 0.0,
        "close_mean": (
            float(gross_return.loc[close_mask].mean())
            if close_count
            else float("nan")
        ),
        "gross_mean": float(simulations["gross_return"].mean()) if n else 0.0,
        "net_mean": float(simulations["net_return"].mean()) if n else 0.0,
        "total_net": float(simulations["net_return"].sum()) if n else 0.0,
        "win_rate": float((simulations["net_return"] > 0.0).mean()) if n else 0.0,
    }


def no_h1_mfe_sweep(
    split: str,
    selected_path: pd.DataFrame,
    take_profit: float,
) -> pd.DataFrame:
    """Analyze H2/H3 recovery after the original TP was not reached in H1."""
    required = ["high_h1", "high_h2", "high_h3", "close_h3"]
    cohort = selected_path.dropna(subset=required).copy()
    if "outcome" in cohort.columns:
        cohort = cohort[
            ~cohort["outcome"].astype(str).isin({"tp_h1", "h1_stop"})
        ]
    else:
        cohort = cohort[
            pd.to_numeric(cohort["high_h1"], errors="coerce")
            < float(take_profit)
        ]
    high_h2 = pd.to_numeric(cohort["high_h2"], errors="coerce")
    mfe_h2_h3 = cohort[["high_h2", "high_h3"]].apply(
        pd.to_numeric, errors="coerce"
    ).max(axis=1)
    close_h3 = pd.to_numeric(cohort["close_h3"], errors="coerce")
    n = len(cohort)

    rows: list[dict[str, Any]] = []
    for threshold in NO_H1_MFE_THRESHOLDS:
        hit_h2 = high_h2.ge(threshold)
        hit_h2_h3 = mfe_h2_h3.ge(threshold)
        miss_h2_h3 = ~hit_h2_h3
        rows.append(
            {
                "split": split,
                "threshold": float(threshold),
                "no_h1_count": int(n),
                "mean_mfe_h2": float(high_h2.mean()) if n else np.nan,
                "mean_mfe_h2_h3": float(mfe_h2_h3.mean()) if n else np.nan,
                "hit_h2_count": int(hit_h2.sum()),
                "hit_h2_rate": float(hit_h2.mean()) if n else np.nan,
                "hit_h2_h3_count": int(hit_h2_h3.sum()),
                "hit_h2_h3_rate": float(hit_h2_h3.mean()) if n else np.nan,
                "miss_h2_h3_count": int(miss_h2_h3.sum()),
                "miss_h2_h3_rate": float(miss_h2_h3.mean()) if n else np.nan,
                "miss_close_h3_mean": (
                    float(close_h3[miss_h2_h3].mean())
                    if bool(miss_h2_h3.any())
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def no_h1_mfe_display(sweep: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Format the no-H1 MFE sweep as a compact threshold-column table."""
    threshold_columns = [
        f"{threshold * 100.0:+.3f}%"
        for threshold in NO_H1_MFE_THRESHOLDS
    ]
    display_rows: list[dict[str, str]] = []
    title_parts: list[str] = []
    for split in ("val", "test"):
        frame = sweep[sweep["split"] == split].sort_values("threshold")
        if frame.empty:
            continue
        first = frame.iloc[0]
        title_parts.append(
            f"{split.upper()} n={int(first['no_h1_count']):,}, "
            f"mean MFE H2={float(first['mean_mfe_h2']):+.3%}, "
            f"H2-H3={float(first['mean_mfe_h2_h3']):+.3%}"
        )
        metrics = (
            ("H2 hit / no-H1", "hit_h2_rate", "percent"),
            ("H2-H3 hit / no-H1", "hit_h2_h3_rate", "percent"),
            ("H2-H3 miss / no-H1", "miss_h2_h3_rate", "percent"),
            ("miss close H3 mean", "miss_close_h3_mean", "signed"),
        )
        for label, column, style in metrics:
            row = {"split / metric": f"{split} {label}"}
            for threshold_label, value in zip(
                threshold_columns,
                frame[column].to_numpy(),
            ):
                if pd.isna(value):
                    row[threshold_label] = ""
                elif style == "signed":
                    row[threshold_label] = f"{float(value):+.3%}"
                else:
                    row[threshold_label] = f"{float(value):.2%}"
            display_rows.append(row)
    return pd.DataFrame(display_rows), " | ".join(title_parts)


def summary_display(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "split": str(row["split"]),
                "signals": f"{int(row['signals']):,}",
                "selected": f"{float(row['selected_rate']):.2%}",
                "trades/day": f"{float(row['trades_per_day']):.2f}",
                "TP": f"{int(row['tp_count']):,} ({float(row['tp_rate']):.2%})",
                "gross mean": f"{float(row['gross_mean']):+.3%}",
                "E[net]": f"{float(row['net_mean']):+.3%}",
                "win rate": f"{float(row['win_rate']):.2%}",
            }
        )
    return pd.DataFrame(rows)


def exit_detail_display(summary: pd.DataFrame) -> pd.DataFrame:
    def signed_percent(value: Any) -> str:
        return "n/a" if pd.isna(value) else f"{float(value):+.3%}"

    rows: list[dict[str, str]] = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "split": str(row["split"]),
                "H1 SL": (
                    f"{int(row['h1_stop_count']):,} "
                    f"({float(row['h1_stop_rate']):.2%}) "
                    f"mean={signed_percent(row['h1_stop_mean'])}"
                ),
                "H1 same-1m TP+SL -> SL": (
                    f"{int(row['h1_tie_count']):,} "
                    f"({float(row['h1_tie_rate']):.2%})"
                ),
                "slowdown SL H2": (
                    f"{int(row['slowdown_stop_h2_count']):,} "
                    f"({float(row['slowdown_stop_h2_rate']):.2%}) "
                    f"mean={signed_percent(row['slowdown_stop_h2_mean'])}"
                ),
                "exit open H2": (
                    f"{int(row['exit_open_h2_count']):,} "
                    f"({float(row['exit_open_h2_rate']):.2%}) "
                    f"mean={signed_percent(row['exit_open_h2_mean'])}"
                ),
                "open H2 benefit": signed_percent(
                    row["exit_open_h2_benefit"]
                ),
                "SL H2 benefit": (
                    signed_percent(row["slowdown_stop_h2_benefit"])
                ),
                "slowdown SL H3": (
                    f"{int(row['slowdown_stop_h3_count']):,} "
                    f"({float(row['slowdown_stop_h3_rate']):.2%}) "
                    f"mean={signed_percent(row['slowdown_stop_h3_mean'])}"
                ),
                "exit open H3": (
                    f"{int(row['exit_open_h3_count']):,} "
                    f"({float(row['exit_open_h3_rate']):.2%}) "
                    f"mean={signed_percent(row['exit_open_h3_mean'])}"
                ),
                "open H3 benefit": signed_percent(
                    row["exit_open_h3_benefit"]
                ),
                "SL H3 benefit": (
                    signed_percent(row["slowdown_stop_h3_benefit"])
                ),
                "close H3": (
                    f"{int(row['close_count']):,} "
                    f"({float(row['close_rate']):.2%}) "
                    f"mean={signed_percent(row['close_mean'])}"
                ),
            }
        )
    return pd.DataFrame(rows)


def draw_report(
    summary: pd.DataFrame,
    simulations: dict[str, pd.DataFrame],
    no_h1_sweep: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(27.0, 20.0),
        gridspec_kw={"height_ratios": [0.9, 1.0, 2.0, 3.2, 2.4]},
        constrained_layout=True,
    )
    ax_table, ax_exit, ax_mfe, ax_equity, ax_daily = axes

    ax_table.axis("off")
    table_df = summary_display(summary)
    table = ax_table.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#9ca3af")
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
    ax_table.set_title(title, fontsize=11, pad=10)

    ax_exit.axis("off")
    exit_df = exit_detail_display(summary)
    exit_table = ax_exit.table(
        cellText=exit_df.values,
        colLabels=exit_df.columns,
        cellLoc="center",
        loc="center",
    )
    exit_table.auto_set_font_size(False)
    exit_table.set_fontsize(6.8)
    exit_table.scale(1.0, 1.5)
    for (row, _), cell in exit_table.get_celld().items():
        cell.set_edgecolor("#9ca3af")
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
    ax_exit.set_title(
        "Slowdown-triggered stop-losses | benefit = SL return - same trades' "
        "close-H3 return",
        fontsize=10,
        pad=8,
    )

    ax_mfe.axis("off")
    mfe_display, mfe_title = no_h1_mfe_display(no_h1_sweep)
    mfe_table = ax_mfe.table(
        cellText=mfe_display.values,
        colLabels=mfe_display.columns,
        cellLoc="center",
        loc="center",
    )
    mfe_table.auto_set_font_size(False)
    mfe_table.set_fontsize(7.2)
    mfe_table.scale(1.0, 1.45)
    for (row, _), cell in mfe_table.get_celld().items():
        cell.set_edgecolor("#9ca3af")
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
    ax_mfe.set_title(
        "No-H1-TP cohort: H2/H3 high-path reach by threshold "
        "(relative to open H1; negative levels are recovery levels)\n"
        + mfe_title,
        fontsize=10,
        pad=8,
    )

    colors = {"val": "#2563eb", "test": "#dc2626"}
    for split in ("val", "test"):
        frame = simulations.get(split, pd.DataFrame()).sort_index()
        if frame.empty:
            continue
        cumulative = pd.to_numeric(frame["net_return"], errors="coerce").cumsum()
        ax_equity.plot(
            frame.index,
            cumulative * 100.0,
            color=colors[split],
            linewidth=1.2,
            label=(
                f"{split.upper()} n={len(frame):,} | "
                f"end={float(cumulative.iloc[-1]) * 100.0:+.2f}%"
            ),
        )
    ax_equity.axhline(0.0, color="#4b5563", linestyle="--", linewidth=0.8)
    ax_equity.set_title("Cumulative net return by signal time")
    ax_equity.set_ylabel("Cumulative net return (percentage points)")
    ax_equity.grid(True, color="#d1d5db", alpha=0.65, linewidth=0.6)
    ax_equity.legend(frameon=False)

    combined = pd.concat(
        [
            frame.assign(split=split)
            for split, frame in simulations.items()
            if not frame.empty
        ],
        axis=0,
    ).sort_index()
    daily = pd.Series(1, index=pd.DatetimeIndex(combined.index)).resample("D").sum()
    ax_daily.bar(
        daily.index,
        daily.to_numpy(),
        width=0.9,
        color="#f59e0b",
        alpha=0.55,
    )
    ax_daily.set_title("Selected trades per day")
    ax_daily.set_xlabel("Signal time")
    ax_daily.set_ylabel("Trades/day")
    ax_daily.grid(True, axis="y", color="#d1d5db", alpha=0.6, linewidth=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def run(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, Path, Path | None, pd.DataFrame]:
    archive_path = Path(args.archive)
    long_slowdown_path = Path(args.long_slowdown_archive)
    short_slowdown_path = Path(args.short_slowdown_archive)
    data_path = Path(args.data)
    metadata = load_archive_metadata(archive_path)
    long_slowdown_metadata = load_archive_metadata(long_slowdown_path)
    short_slowdown_metadata = load_archive_metadata(short_slowdown_path)
    label_mode = config.canonical_label_mode(metadata.get("label_mode"))
    label_direction = config.canonical_label_direction(
        metadata.get("label_direction")
    )
    label_threshold = float(metadata.get("label_threshold"))
    horizons = _archive_horizons(
        archive_path,
        fallback=[3],
        label="Long MFE backtest",
    )
    long_slowdown_horizons = _archive_horizons(
        long_slowdown_path,
        fallback=[1],
        label="Long slowdown backtest",
    )
    short_slowdown_horizons = _archive_horizons(
        short_slowdown_path,
        fallback=[1],
        label="Short slowdown backtest",
    )
    if label_mode != "mfe" or label_direction != "long":
        raise ValueError(
            "This script requires an MFE Long archive; got "
            f"mode={label_mode}, direction={label_direction}."
        )
    if horizons != [3]:
        raise ValueError(
            f"This script requires archive metadata horizons=[3], got {horizons}."
        )
    for name, slowdown_metadata, expected_direction, slowdown_horizons in (
        (
            "Long",
            long_slowdown_metadata,
            "long",
            long_slowdown_horizons,
        ),
        (
            "Short",
            short_slowdown_metadata,
            "short",
            short_slowdown_horizons,
        ),
    ):
        mode = config.canonical_label_mode(slowdown_metadata.get("label_mode"))
        direction = config.canonical_label_direction(
            slowdown_metadata.get("label_direction")
        )
        if (
            mode != "slope_slowdown"
            or direction != expected_direction
            or slowdown_horizons != [1]
        ):
            raise ValueError(
                f"{name} slowdown archive must use slope_slowdown "
                f"direction={expected_direction}, horizons=[1]; got "
                f"mode={mode}, direction={direction}, "
                f"horizons={slowdown_horizons}."
            )

    slope_lookback = int(long_slowdown_metadata.get("slope_lookback"))
    slope_min_initial = float(
        long_slowdown_metadata.get("slope_min_initial")
    )
    if (
        slope_lookback != 2
        or int(short_slowdown_metadata.get("slope_lookback")) != slope_lookback
        or not np.isclose(
            float(short_slowdown_metadata.get("slope_min_initial")),
            slope_min_initial,
        )
    ):
        raise ValueError(
            "Both slowdown archives must use slope_lookback=2 and the same "
            "slope_min_initial."
        )
    if (
        int(config.SLOPE_LOOKBACK) != slope_lookback
        or not np.isclose(float(config.SLOPE_MIN_INITIAL), slope_min_initial)
    ):
        raise ValueError(
            "Current config slope settings do not match the slowdown archives: "
            f"config=({config.SLOPE_LOOKBACK}, {config.SLOPE_MIN_INITIAL}), "
            f"archive=({slope_lookback}, {slope_min_initial})."
        )

    spec = ModelSpec(
        archive_path=archive_path,
        rank=int(args.rank),
        label_mode=label_mode,
        label_threshold=label_threshold,
        top_fraction=float(args.top_fraction),
        label_direction=label_direction,
    )
    long_slowdown_spec = ModelSpec(
        archive_path=long_slowdown_path,
        rank=int(args.long_slowdown_rank),
        label_mode="slope_slowdown",
        label_threshold=float(long_slowdown_metadata["label_threshold"]),
        top_fraction=float(args.long_slowdown_top_fraction),
        label_direction="long",
    )
    short_slowdown_spec = ModelSpec(
        archive_path=short_slowdown_path,
        rank=int(args.short_slowdown_rank),
        label_mode="slope_slowdown",
        label_threshold=float(short_slowdown_metadata["label_threshold"]),
        top_fraction=float(args.short_slowdown_top_fraction),
        label_direction="short",
    )
    raw_df = load_ohlcv(data_path)
    purge_bars = config.purge_bars_for_horizons([3, 1])
    entry = _load_rank_entry(archive_path, spec.rank)
    long_slowdown_entry = _load_rank_entry(
        long_slowdown_path,
        long_slowdown_spec.rank,
    )
    short_slowdown_entry = _load_rank_entry(
        short_slowdown_path,
        short_slowdown_spec.rank,
    )
    quality_indices = []
    for quality_spec, quality_horizons in (
        (spec, horizons),
        (long_slowdown_spec, [1]),
        (short_slowdown_spec, [1]),
    ):
        quality_indices.append(
            _quality_train_index(
                raw_df=raw_df,
                spec=quality_spec,
                horizons=quality_horizons,
                val_start=args.val_start,
                test_start=args.test_start,
                test_end=args.test_end,
                purge_bars=purge_bars,
            )
        )
    quality_train = quality_indices[0].union(
        quality_indices[1]
    ).union(quality_indices[2])
    feature_space = _cached_feature_space(
        raw_df=raw_df,
        data_path=data_path,
        required_windows=_required_windows_for_entries(
            [entry, long_slowdown_entry, short_slowdown_entry]
        ),
        quality_index=quality_train,
    )
    bundle = _train_spec_bundle(
        spec=spec,
        entry=entry,
        raw_df=raw_df,
        feature_space=feature_space,
        horizons=horizons,
        val_start=args.val_start,
        test_start=args.test_start,
        test_end=args.test_end,
        purge_bars=purge_bars,
    )
    long_slowdown_bundle = _train_spec_bundle(
        spec=long_slowdown_spec,
        entry=long_slowdown_entry,
        raw_df=raw_df,
        feature_space=feature_space,
        horizons=[1],
        val_start=args.val_start,
        test_start=args.test_start,
        test_end=args.test_end,
        purge_bars=purge_bars,
    )
    short_slowdown_bundle = _train_spec_bundle(
        spec=short_slowdown_spec,
        entry=short_slowdown_entry,
        raw_df=raw_df,
        feature_space=feature_space,
        horizons=[1],
        val_start=args.val_start,
        test_start=args.test_start,
        test_end=args.test_end,
        purge_bars=purge_bars,
    )

    path = make_price_path(raw_df, horizon=3)
    initial_slope = (
        pd.to_numeric(raw_df["high"], errors="coerce")
        .div(pd.to_numeric(raw_df["high"], errors="coerce").shift(1))
        .sub(1.0)
    )
    simulations: dict[str, pd.DataFrame] = {}
    no_h1_sweeps: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    prepared_splits: list[tuple[str, Any, Any, Any, pd.DataFrame]] = []
    split_inputs = (
        (
            "val",
            bundle.val,
            long_slowdown_bundle.val,
            short_slowdown_bundle.val,
        ),
        (
            "test",
            bundle.test,
            long_slowdown_bundle.test,
            short_slowdown_bundle.test,
        ),
    )
    for (
        split_name,
        signals,
        long_slowdown_signals,
        short_slowdown_signals,
    ) in split_inputs:
        base_selected_index = pd.Index(signals.selected_index)
        if args.entry_filter:
            short_selected_index = pd.Index(
                short_slowdown_signals.selected_index
            )
            negative_slope_index = pd.Index(
                initial_slope.index[
                    pd.to_numeric(initial_slope, errors="coerce")
                    .lt(-slope_min_initial)
                    .fillna(False)
                ]
            )
            entry_filter_index = base_selected_index[
                base_selected_index.isin(short_selected_index)
                & base_selected_index.isin(negative_slope_index)
            ]
        else:
            entry_filter_index = base_selected_index
        logger.info(
            "%s entry filter=%s: base=%d | selected=%d (%.2f%% of base)",
            split_name.upper(),
            "ON" if args.entry_filter else "OFF",
            len(base_selected_index),
            len(entry_filter_index),
            (
                100.0 * len(entry_filter_index) / len(base_selected_index)
                if len(base_selected_index)
                else 0.0
            ),
        )
        selected_path = path.reindex(entry_filter_index)
        prepared_splits.append(
            (
                split_name,
                signals,
                long_slowdown_signals,
                short_slowdown_signals,
                selected_path,
            )
        )

    selected_indexes = [
        pd.DatetimeIndex(selected_path.index)
        for _, _, _, _, selected_path in prepared_splits
        if not selected_path.empty
    ]
    if selected_indexes:
        all_selected_times = selected_indexes[0]
        for selected_index in selected_indexes[1:]:
            all_selected_times = all_selected_times.union(selected_index)
        minute_start = all_selected_times.min() + pd.Timedelta(minutes=5)
        minute_end_offset = 19 if not args.skip_1m_analysis else 9
        minute_end = all_selected_times.max() + pd.Timedelta(
            minutes=minute_end_offset
        )
        logger.info(
            "Loading 1m OHLC for causal H1 ordering: %s -> %s",
            minute_start,
            minute_end,
        )
        minute_df = load_one_minute_ohlc(
            args.data_1m,
            start=minute_start,
            end=minute_end,
        )
    else:
        minute_df = pd.DataFrame(columns=["open", "high", "low", "close"])
        minute_df.index = pd.DatetimeIndex([], name="date")

    for (
        split_name,
        signals,
        long_slowdown_signals,
        short_slowdown_signals,
        selected_path,
    ) in prepared_splits:
        h1_barrier_outcomes = resolve_h1_tp_sl_from_1m(
            signal_index=selected_path.index,
            expected_entry_open=selected_path["entry_open"],
            minute=minute_df,
            take_profit=float(args.take_profit),
            h1_stop_loss=float(args.h1_stop_loss),
        )
        simulated = simulate_long_slowdown_strategy(
            selected_path=selected_path,
            h1_barrier_outcomes=h1_barrier_outcomes,
            raw_index=pd.DatetimeIndex(raw_df.index),
            initial_slope=initial_slope,
            long_slowdown_signals=long_slowdown_signals,
            short_slowdown_signals=short_slowdown_signals,
            slope_min_initial=slope_min_initial,
            take_profit=float(args.take_profit),
            h1_stop_loss=float(args.h1_stop_loss),
            slowdown_stop_loss=float(args.slowdown_stop_loss),
            trade_cost=float(args.trade_cost),
        )
        no_h1_sweeps.append(
            no_h1_mfe_sweep(
                split=split_name,
                selected_path=simulated,
                take_profit=float(args.take_profit),
            )
        )
        simulations[split_name] = simulated
        rows.append(
            summarize_split(
                split=split_name,
                simulations=simulated,
                available_rows=len(signals.data),
                prediction_threshold=signals.pred_threshold,
            )
        )

    summary = pd.DataFrame(rows)
    no_h1_sweep = pd.concat(no_h1_sweeps, ignore_index=True)
    run_name = (
        f"{archive_path.stem}_r{spec.rank:02d}_top"
        f"{float(args.top_fraction) * 100.0:.0f}_slowL"
        f"{long_slowdown_spec.top_fraction * 100.0:.0f}_slowS"
        f"{short_slowdown_spec.top_fraction * 100.0:.0f}_"
        f"{'entryNegSlowS' if args.entry_filter else 'entryFilterOff'}_tp"
        f"{float(args.take_profit) * 100.0:.3f}pct_h1SL"
        f"{float(args.h1_stop_loss) * 100.0:.3f}pct_h1Order1m_slowSL"
        f"{float(args.slowdown_stop_loss) * 100.0:.3f}pct"
    ).replace(".", "p")
    output_path = Path(args.out_dir) / f"{run_name}.png"
    entry_filter_title = (
        f"entry filter: slope<-{slope_min_initial:.2%} + Short signal"
        if args.entry_filter
        else "entry filter: OFF"
    )
    title = (
        "5m Long MFE H3 + per-candle slope slowdown exits | "
        f"base r{spec.rank} top={spec.top_fraction:.0%} | "
        f"Long slowdown top={long_slowdown_spec.top_fraction:.0%} | "
        f"Short slowdown top={short_slowdown_spec.top_fraction:.0%} | "
        f"{entry_filter_title} | "
        f"H1-only TP=+{float(args.take_profit):.2%}, H1-only SL="
        f"-{float(args.h1_stop_loss):.2%}, slowdown SL="
        f"-{float(args.slowdown_stop_loss):.2%}, H1 order=1m/SL-first tie, "
        "neutral=HOLD, "
        f"max exit=close H3 | cost={float(args.trade_cost):.2%}"
    )
    draw_report(summary, simulations, no_h1_sweep, output_path, title)
    logger.info("Saved report: %s", output_path)
    print("\n=== No-H1-TP MFE sweep (H2 and cumulative H2-H3) ===")
    print(no_h1_sweep.to_string(index=False))

    miss_report_path: Path | None = None
    exit_summary = pd.DataFrame()
    if not args.skip_1m_analysis:
        all_signal_times = pd.DatetimeIndex(
            pd.Index(
                np.concatenate(
                    [
                        frame.index.to_numpy()
                        for frame in simulations.values()
                        if not frame.empty
                    ]
                )
            )
        )
        if len(all_signal_times):
            paths_by_split: dict[str, pd.DataFrame] = {}
            checkpoint_records: list[dict[str, Any]] = []
            exit_grids: list[pd.DataFrame] = []
            for split_name, frame in simulations.items():
                minute_paths = build_one_minute_paths(
                    frame.index,
                    minute=minute_df,
                    take_profit=float(args.take_profit),
                    minutes=15,
                )
                paths_by_split[split_name] = minute_paths
                checkpoint_records.extend(
                    minute_checkpoint_rows(split_name, minute_paths)
                )
                exit_grids.append(
                    early_exit_grid(
                        split=split_name,
                        paths=minute_paths,
                        take_profit=float(args.take_profit),
                        trade_cost=float(args.trade_cost),
                    )
                )
                logger.info(
                    "%s 1m coverage: %d/%d signals | TP hit=%.2f%%",
                    split_name.upper(),
                    len(minute_paths),
                    len(frame),
                    (
                        float(minute_paths["tp_hit"].mean()) * 100.0
                        if len(minute_paths)
                        else float("nan")
                    ),
                )

            checkpoint_df = pd.DataFrame(checkpoint_records)
            exit_grid = pd.concat(exit_grids, ignore_index=True)
            val_candidates = exit_grid[
                (exit_grid["split"] == "val") & (exit_grid["exit_count"] > 0)
            ]
            if not val_candidates.empty:
                best_val = val_candidates.sort_values(
                    ["net_mean", "exit_miss_precision"],
                    ascending=[False, False],
                ).iloc[0]
                matching_test = exit_grid[
                    (exit_grid["split"] == "test")
                    & (exit_grid["minute"] == int(best_val["minute"]))
                    & np.isclose(
                        exit_grid["exit_threshold"],
                        float(best_val["exit_threshold"]),
                    )
                ]
                test_row = matching_test.iloc[0]
                exit_summary = pd.DataFrame(
                    [
                        {
                            "rule": "Val-best close-return exit",
                            "minute": int(best_val["minute"]),
                            "threshold": float(best_val["exit_threshold"]),
                            "val_net": float(best_val["net_mean"]),
                            "val_delta": float(
                                best_val["net_mean"]
                                - best_val["baseline_net_mean"]
                            ),
                            "test_net": float(test_row["net_mean"]),
                            "test_delta": float(
                                test_row["net_mean"]
                                - test_row["baseline_net_mean"]
                            ),
                            "val_exit_rate": float(best_val["exit_rate"]),
                            "test_exit_rate": float(test_row["exit_rate"]),
                            "val_miss_precision": float(
                                best_val["exit_miss_precision"]
                            ),
                            "test_miss_precision": float(
                                test_row["exit_miss_precision"]
                            ),
                        }
                    ]
                )
                miss_report_path = (
                    Path(args.out_dir) / f"{run_name}_tp_miss_1m.png"
                )
                draw_one_minute_miss_report(
                    paths_by_split=paths_by_split,
                    checkpoint_rows=checkpoint_df,
                    exit_summary=exit_summary,
                    output_path=miss_report_path,
                    title=(
                        "1m TP_HIT vs TP_MISS diagnostics | "
                        f"TP=+{float(args.take_profit):.2%} | "
                        "exit rule optimized on Val, applied unchanged to Test"
                    ),
                )
                logger.info("Saved 1m miss report: %s", miss_report_path)
                print("\n=== 1m TP miss diagnostics ===")
                print(_format_minute_diagnostics(checkpoint_df).to_string(index=False))
                print("\n=== Val-selected early-exit candidate ===")
                print(exit_summary.to_string(index=False))

    return summary, output_path, miss_report_path, exit_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    parser.add_argument(
        "--long-slowdown-archive",
        default=str(DEFAULT_LONG_SLOWDOWN_ARCHIVE),
    )
    parser.add_argument(
        "--short-slowdown-archive",
        default=str(DEFAULT_SHORT_SLOWDOWN_ARCHIVE),
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--data-1m", default=str(DEFAULT_DATA_1M))
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument(
        "--long-slowdown-rank",
        type=int,
        default=DEFAULT_SLOWDOWN_RANK,
    )
    parser.add_argument(
        "--short-slowdown-rank",
        type=int,
        default=DEFAULT_SLOWDOWN_RANK,
    )
    parser.add_argument("--top-fraction", type=float, default=DEFAULT_TOP_FRACTION)
    parser.add_argument(
        "--long-slowdown-top-fraction",
        type=float,
        default=DEFAULT_LONG_SLOWDOWN_TOP_FRACTION,
    )
    parser.add_argument(
        "--short-slowdown-top-fraction",
        type=float,
        default=DEFAULT_SHORT_SLOWDOWN_TOP_FRACTION,
    )
    parser.add_argument(
        "--entry-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require initial_slope(t) < -SLOPE_MIN_INITIAL and a Short "
            "slowdown signal before entering Long. Use --no-entry-filter "
            "to disable."
        ),
    )
    parser.add_argument("--take-profit", type=float, default=DEFAULT_TAKE_PROFIT)
    parser.add_argument(
        "--h1-stop-loss",
        type=float,
        default=DEFAULT_H1_STOP_LOSS,
        help=(
            "Positive stop-loss distance active only during H1 "
            "(default: 0.0004 = 0.04%%)."
        ),
    )
    parser.add_argument(
        "--slowdown-stop-loss",
        type=float,
        default=DEFAULT_SLOWDOWN_STOP_LOSS,
        help=(
            "Positive stop-loss distance armed by a slowdown exit decision "
            "(default: 0.0004 = 0.04%%)."
        ),
    )
    parser.add_argument(
        "--trade-cost",
        type=float,
        default=DEFAULT_TRADE_COST,
        help="Round-trip trading cost (default: 0.00016 = 0.016%%).",
    )
    parser.add_argument("--val-start", default=config.VAL_START)
    parser.add_argument("--test-start", default=config.TEST_START)
    parser.add_argument("--test-end", default=config.TEST_END)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--skip-1m-analysis",
        action="store_true",
        help="Skip TP_HIT/TP_MISS diagnostics using 1-minute candles.",
    )
    return parser.parse_args()


def main() -> None:
    summary, output_path, miss_report_path, _ = run(parse_args())
    print(summary_display(summary).to_string(index=False))
    print(f"\nSaved: {output_path}")
    if miss_report_path is not None:
        print(f"Saved 1m miss analysis: {miss_report_path}")


if __name__ == "__main__":
    main()
