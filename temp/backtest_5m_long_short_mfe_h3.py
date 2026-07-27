"""Backtest a two-leg H15 strategy from Long and Short 5-minute MFE models.

Both models are retrained from their archive rank. Each model learns its own
prediction cutoff on Final Val, and that cutoff is applied unchanged to Test.
By default, a trade is opened when either model votes for the timestamp.

At every selected signal:

1. Open one Long and one Short at open H1.
2. Long and Short TP orders are active during H1 only.
3. After H1 and H2, use the current high-price slope and the Long/Short
   slowdown models to decide whether each remaining leg receives a dynamic
   TP in the next candle or continues without an exit order.
4. Solve the next high that would move the expanded log-OLS slope by the
   slowdown threshold, and use that projected high directly as the dynamic
   exit price for either leg.
5. Repeat the causal slowdown decision through H14, placing the final dynamic
   TP in H15.
6. Any position still active exits at close H15.
7. The pair return is the sum of both legs minus ``trade_cost`` once.

PowerShell:
    python -m temp.backtest_5m_long_short_mfe_h3 `
      --long-archive crypto/results/crypto_btc_5m_long_mfe_h3_tp01_top40_seed1_8h.json `
      --short-archive crypto/results/crypto_btc_short_mfe_h3_tp01_top40_seed1_8h.json `
      --long-slowdown-archive crypto/results/crypto_btc_5m_long_slope_slowdown_lb2_h1_thr003_top40_seed1_1h.json `
      --short-slowdown-archive crypto/results/crypto_btc_5m_Short_slope_slowdown_lb2_h1_thr003_top20_seed1_1h.json `
      --long-rank 1 `
      --short-rank 1 `
      --long-slowdown-rank 1 `
      --short-slowdown-rank 1 `
      --long-top-fraction 0.40 `
      --short-top-fraction 0.40 `
      --long-slowdown-top-fraction 0.40 `
      --short-slowdown-top-fraction 0.20 `
      --voting or `
      --long-tp 0.001 `
      --short-tp 0.001 `
      --trade-cost 0.00016 `
      --data data/crypto/BTCUSDT_5m.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
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
logger = logging.getLogger("temp.backtest_5m_long_short_mfe_h3")


DEFAULT_LONG_ARCHIVE = Path(
    "crypto/results/crypto_btc_5m_long_mfe_h3_tp01_top40_seed1_8h.json"
)
DEFAULT_SHORT_ARCHIVE = Path(
    "crypto/results/crypto_btc_short_mfe_h3_tp01_top40_seed1_8h.json"
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
DEFAULT_OUT_DIR = Path("temp/output")
DEFAULT_TP = 0.001
DEFAULT_TRADE_COST = 0.00016
BASE_MODEL_HORIZON = 3
MAX_HOLDING_HORIZON = 15


@dataclass(frozen=True)
class ArchiveSettings:
    path: Path
    rank: int
    direction: str
    label_mode: str
    label_threshold: float
    top_fraction: float
    horizons: list[int]
    metadata: dict[str, Any]


def _load_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if not isinstance(metadata, dict):
        raise ValueError(f"Archive metadata must be an object: {path}")
    return dict(metadata)


def _archive_settings(
    path: Path,
    rank: int,
    expected_direction: str,
    top_fraction: float | None,
) -> ArchiveSettings:
    metadata = _load_metadata(path)
    mode = config.canonical_label_mode(metadata.get("label_mode", "mfe"))
    direction = config.canonical_label_direction(
        metadata.get("label_direction", expected_direction)
    )
    horizons = _archive_horizons(
        path,
        [BASE_MODEL_HORIZON],
        expected_direction,
    )
    if mode != "mfe":
        raise ValueError(f"{path} must use label_mode=mfe, got {mode!r}.")
    if direction != expected_direction:
        raise ValueError(
            f"{path} must use direction={expected_direction}, got {direction!r}."
        )
    if horizons != [BASE_MODEL_HORIZON]:
        raise ValueError(
            f"{path} must contain only horizon H{BASE_MODEL_HORIZON}, "
            f"got {horizons}."
        )

    resolved_top = (
        float(top_fraction)
        if top_fraction is not None
        else float(metadata.get("trade_top_fraction", 0.40))
    )
    if not 0.0 < resolved_top <= 1.0:
        raise ValueError("Top fraction must be in (0, 1].")
    return ArchiveSettings(
        path=path,
        rank=int(rank),
        direction=direction,
        label_mode=mode,
        label_threshold=float(metadata.get("label_threshold", 0.001)),
        top_fraction=resolved_top,
        horizons=horizons,
        metadata=metadata,
    )


def _slowdown_settings(
    path: Path,
    rank: int,
    expected_direction: str,
    top_fraction: float | None,
) -> ArchiveSettings:
    metadata = _load_metadata(path)
    mode = config.canonical_label_mode(metadata.get("label_mode"))
    direction = config.canonical_label_direction(metadata.get("label_direction"))
    horizons = _archive_horizons(path, [1], f"{expected_direction} slowdown")
    if mode != "slope_slowdown":
        raise ValueError(
            f"{path} must use label_mode=slope_slowdown, got {mode!r}."
        )
    if direction != expected_direction:
        raise ValueError(
            f"{path} must use direction={expected_direction}, got {direction!r}."
        )
    if horizons != [1]:
        raise ValueError(f"{path} must contain only H1, got {horizons}.")
    resolved_top = (
        float(top_fraction)
        if top_fraction is not None
        else float(metadata.get("trade_top_fraction", 0.40))
    )
    if not 0.0 < resolved_top <= 1.0:
        raise ValueError("Slowdown top fraction must be in (0, 1].")
    return ArchiveSettings(
        path=path,
        rank=int(rank),
        direction=direction,
        label_mode=mode,
        label_threshold=float(metadata["label_threshold"]),
        top_fraction=resolved_top,
        horizons=horizons,
        metadata=metadata,
    )


def make_h15_path(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return H1-H15 OHLC paths relative to open H1."""
    entry = pd.to_numeric(raw_df["open"], errors="coerce").shift(-1)
    result = pd.DataFrame(index=raw_df.index)
    result["entry_open"] = entry
    for step in range(1, MAX_HOLDING_HORIZON + 1):
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
    result["close_h15"] = (
        pd.to_numeric(raw_df["close"], errors="coerce")
        .shift(-MAX_HOLDING_HORIZON)
        .div(entry)
        - 1.0
    )
    return result.replace([np.inf, -np.inf], np.nan)


def project_next_high_for_slowdown(
    high: pd.Series,
    initial_slope: pd.Series,
    lookback: int,
    slowdown_threshold: float,
    direction: str,
) -> pd.Series:
    """Solve the next high needed to reach the slowdown label boundary."""
    width = int(lookback)
    if width < 2:
        raise ValueError("Slope lookback must be at least 2.")
    threshold = float(slowdown_threshold)
    if threshold <= 0.0:
        raise ValueError("Slowdown threshold must be positive.")
    direction = config.canonical_label_direction(direction)

    numeric = pd.to_numeric(high, errors="coerce").to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_high = np.log(numeric)
    result = np.full(len(log_high), np.nan, dtype="float64")
    if len(log_high) < width:
        return pd.Series(result, index=high.index, dtype="float64")

    full_x = np.arange(width + 1, dtype="float64")
    centered_x = full_x - full_x.mean()
    denominator = float(np.square(centered_x).sum())
    unknown_coefficient = float(centered_x[-1])
    known_coefficients = centered_x[:-1]
    windows = np.lib.stride_tricks.sliding_window_view(log_high, width)
    known_contribution = windows @ known_coefficients

    initial = pd.to_numeric(initial_slope, errors="coerce").to_numpy(
        dtype="float64"
    )[width - 1 :]
    target_slope = (
        initial - threshold if direction == "long" else initial + threshold
    )
    valid = (
        np.isfinite(windows).all(axis=1)
        & np.isfinite(target_slope)
        & (target_slope > -1.0)
    )
    target_beta = np.full(len(target_slope), np.nan, dtype="float64")
    target_beta[valid] = np.log1p(target_slope[valid])
    target_log_high = (
        target_beta * denominator - known_contribution
    ) / unknown_coefficient
    with np.errstate(over="ignore", invalid="ignore"):
        projected = np.exp(target_log_high)
    projected[~np.isfinite(projected)] = np.nan
    result[width - 1 :] = projected
    return pd.Series(result, index=high.index, dtype="float64")


def simulate_two_leg_h3(
    path: pd.DataFrame,
    raw_index: pd.DatetimeIndex,
    initial_slope: pd.Series,
    long_projected_high: pd.Series,
    short_projected_high: pd.Series,
    long_slowdown_signals: Any,
    short_slowdown_signals: Any,
    slope_min_initial: float,
    long_tp: float,
    short_tp: float,
    trade_cost: float,
) -> pd.DataFrame:
    """Simulate H1 TPs and causal slowdown exits for both position legs."""
    long_tp = float(long_tp)
    short_tp = float(short_tp)
    trade_cost = float(trade_cost)
    if long_tp <= 0.0 or short_tp <= 0.0:
        raise ValueError("Long TP and Short TP must be positive distances.")
    if trade_cost < 0.0:
        raise ValueError("Trade cost must be non-negative.")
    required = [
        "entry_open",
        "open_h2",
        "high_h2",
        "low_h2",
        "open_h3",
        "high_h3",
        "low_h3",
        "high_h1",
        "low_h1",
        "close_h3",
    ]
    result = path.dropna(subset=required).sort_index().copy()

    base_positions = raw_index.get_indexer(pd.DatetimeIndex(result.index))
    if bool((base_positions < 0).any()):
        raise ValueError("Selected signal is missing from the raw data index.")
    if bool((base_positions + 2 >= len(raw_index)).any()):
        raise ValueError("Selected signal does not have a complete H3 path.")

    decision_h1_index = raw_index.take(base_positions + 1)
    decision_h2_index = raw_index.take(base_positions + 2)
    slope_h1 = initial_slope.reindex(decision_h1_index).to_numpy(dtype=float)
    slope_h2 = initial_slope.reindex(decision_h2_index).to_numpy(dtype=float)
    long_target_h2 = long_projected_high.reindex(
        decision_h1_index
    ).to_numpy(dtype=float)
    long_target_h3 = long_projected_high.reindex(
        decision_h2_index
    ).to_numpy(dtype=float)
    short_target_h2 = short_projected_high.reindex(
        decision_h1_index
    ).to_numpy(dtype=float)
    short_target_h3 = short_projected_high.reindex(
        decision_h2_index
    ).to_numpy(dtype=float)
    long_selected = pd.Index(long_slowdown_signals.selected_index)
    short_selected = pd.Index(short_slowdown_signals.selected_index)
    long_signal_h1 = decision_h1_index.isin(long_selected)
    long_signal_h2 = decision_h2_index.isin(long_selected)
    short_signal_h1 = decision_h1_index.isin(short_selected)
    short_signal_h2 = decision_h2_index.isin(short_selected)

    high_h1 = pd.to_numeric(result["high_h1"], errors="coerce").to_numpy()
    low_h1 = pd.to_numeric(result["low_h1"], errors="coerce").to_numpy()
    entry_open = pd.to_numeric(
        result["entry_open"], errors="coerce"
    ).to_numpy()
    open_h2 = pd.to_numeric(result["open_h2"], errors="coerce").to_numpy()
    high_h2 = pd.to_numeric(result["high_h2"], errors="coerce").to_numpy()
    low_h2 = pd.to_numeric(result["low_h2"], errors="coerce").to_numpy()
    open_h3 = pd.to_numeric(result["open_h3"], errors="coerce").to_numpy()
    high_h3 = pd.to_numeric(result["high_h3"], errors="coerce").to_numpy()
    low_h3 = pd.to_numeric(result["low_h3"], errors="coerce").to_numpy()
    close_return = pd.to_numeric(result["close_h3"], errors="coerce")
    close_h3 = close_return.to_numpy()
    min_slope = float(slope_min_initial)
    n_rows = len(result)

    long_hit = high_h1 >= long_tp
    short_hit = low_h1 <= -short_tp
    long_active = ~long_hit
    short_active = ~short_hit
    long_return = np.full(n_rows, np.nan, dtype=float)
    short_return = np.full(n_rows, np.nan, dtype=float)
    long_outcome = np.full(n_rows, "close_h3", dtype=object)
    short_outcome = np.full(n_rows, "close_h3", dtype=object)
    long_return[long_hit] = long_tp
    short_return[short_hit] = short_tp
    long_outcome[long_hit] = "tp_h1"
    short_outcome[short_hit] = "tp_h1"

    # Decisions made after H1 close place causal limit orders for H2.
    positive_h1 = slope_h1 > min_slope
    negative_h1 = slope_h1 < -min_slope
    long_order_h2 = long_active & (
        (positive_h1 & long_signal_h1)
        | (negative_h1 & ~short_signal_h1)
    )
    short_order_h2 = short_active & (
        (positive_h1 & ~long_signal_h1)
        | (negative_h1 & short_signal_h1)
    )
    long_limit_h2 = long_target_h2 / entry_open - 1.0
    short_limit_h2 = short_target_h2 / entry_open - 1.0
    long_fill_open_h2 = (
        long_order_h2 & np.isfinite(long_limit_h2) & (open_h2 >= long_limit_h2)
    )
    long_fill_high_h2 = (
        long_order_h2
        & ~long_fill_open_h2
        & np.isfinite(long_limit_h2)
        & (high_h2 >= long_limit_h2)
    )
    short_fill_open_h2 = (
        short_order_h2
        & np.isfinite(short_limit_h2)
        & (open_h2 <= short_limit_h2)
    )
    short_fill_low_h2 = (
        short_order_h2
        & ~short_fill_open_h2
        & np.isfinite(short_limit_h2)
        & (low_h2 <= short_limit_h2)
    )
    long_return[long_fill_open_h2] = open_h2[long_fill_open_h2]
    long_return[long_fill_high_h2] = long_limit_h2[long_fill_high_h2]
    short_return[short_fill_open_h2] = -open_h2[short_fill_open_h2]
    short_return[short_fill_low_h2] = -short_limit_h2[short_fill_low_h2]
    long_outcome[long_fill_open_h2] = "dynamic_tp_open_h2"
    long_outcome[long_fill_high_h2] = "dynamic_tp_high_h2"
    short_outcome[short_fill_open_h2] = "dynamic_tp_open_h2"
    short_outcome[short_fill_low_h2] = "dynamic_tp_low_h2"
    long_active[long_fill_open_h2 | long_fill_high_h2] = False
    short_active[short_fill_open_h2 | short_fill_low_h2] = False

    # Decisions made after H2 close place causal limit orders for H3.
    positive_h2 = slope_h2 > min_slope
    negative_h2 = slope_h2 < -min_slope
    long_order_h3 = long_active & (
        (positive_h2 & long_signal_h2)
        | (negative_h2 & ~short_signal_h2)
    )
    short_order_h3 = short_active & (
        (positive_h2 & ~long_signal_h2)
        | (negative_h2 & short_signal_h2)
    )
    long_limit_h3 = long_target_h3 / entry_open - 1.0
    short_limit_h3 = short_target_h3 / entry_open - 1.0
    long_fill_open_h3 = (
        long_order_h3 & np.isfinite(long_limit_h3) & (open_h3 >= long_limit_h3)
    )
    long_fill_high_h3 = (
        long_order_h3
        & ~long_fill_open_h3
        & np.isfinite(long_limit_h3)
        & (high_h3 >= long_limit_h3)
    )
    short_fill_open_h3 = (
        short_order_h3
        & np.isfinite(short_limit_h3)
        & (open_h3 <= short_limit_h3)
    )
    short_fill_low_h3 = (
        short_order_h3
        & ~short_fill_open_h3
        & np.isfinite(short_limit_h3)
        & (low_h3 <= short_limit_h3)
    )
    long_return[long_fill_open_h3] = open_h3[long_fill_open_h3]
    long_return[long_fill_high_h3] = long_limit_h3[long_fill_high_h3]
    short_return[short_fill_open_h3] = -open_h3[short_fill_open_h3]
    short_return[short_fill_low_h3] = -short_limit_h3[short_fill_low_h3]
    long_outcome[long_fill_open_h3] = "dynamic_tp_open_h3"
    long_outcome[long_fill_high_h3] = "dynamic_tp_high_h3"
    short_outcome[short_fill_open_h3] = "dynamic_tp_open_h3"
    short_outcome[short_fill_low_h3] = "dynamic_tp_low_h3"
    long_active[long_fill_open_h3 | long_fill_high_h3] = False
    short_active[short_fill_open_h3 | short_fill_low_h3] = False

    long_return[long_active] = close_h3[long_active]
    short_return[short_active] = -close_h3[short_active]
    result["initial_slope_h1"] = slope_h1
    result["initial_slope_h2"] = slope_h2
    result["long_dynamic_tp_h2"] = long_limit_h2
    result["long_dynamic_tp_h3"] = long_limit_h3
    result["short_dynamic_tp_h2"] = short_limit_h2
    result["short_dynamic_tp_h3"] = short_limit_h3
    result["long_hit"] = long_hit
    result["short_hit"] = short_hit
    result["long_outcome"] = long_outcome
    result["short_outcome"] = short_outcome
    result["long_return"] = long_return
    result["short_return"] = short_return
    result["gross_return"] = result["long_return"] + result["short_return"]
    result["net_return"] = result["gross_return"] - trade_cost
    result["cumulative_net_return"] = result["net_return"].cumsum()
    return result


def simulate_two_leg_h15(
    path: pd.DataFrame,
    raw_index: pd.DatetimeIndex,
    initial_slope: pd.Series,
    long_projected_high: pd.Series,
    short_projected_high: pd.Series,
    long_slowdown_signals: Any,
    short_slowdown_signals: Any,
    slope_min_initial: float,
    long_tp: float,
    short_tp: float,
    trade_cost: float,
) -> pd.DataFrame:
    """Run H1-only fixed TPs and dynamic slowdown exits through H15."""
    long_tp = float(long_tp)
    short_tp = float(short_tp)
    trade_cost = float(trade_cost)
    if long_tp <= 0.0 or short_tp <= 0.0:
        raise ValueError("Long TP and Short TP must be positive distances.")
    if trade_cost < 0.0:
        raise ValueError("Trade cost must be non-negative.")

    required = ["entry_open", "high_h1", "low_h1", "close_h15"]
    for step in range(2, MAX_HOLDING_HORIZON + 1):
        required.extend(
            [f"open_h{step}", f"high_h{step}", f"low_h{step}"]
        )
    result = path.dropna(subset=required).sort_index().copy()
    if result.empty:
        return result.assign(
            long_hit=pd.Series(dtype=bool),
            short_hit=pd.Series(dtype=bool),
            long_outcome=pd.Series(dtype=str),
            short_outcome=pd.Series(dtype=str),
            long_return=pd.Series(dtype=float),
            short_return=pd.Series(dtype=float),
            gross_return=pd.Series(dtype=float),
            net_return=pd.Series(dtype=float),
            cumulative_net_return=pd.Series(dtype=float),
        )

    base_positions = raw_index.get_indexer(pd.DatetimeIndex(result.index))
    if bool((base_positions < 0).any()):
        raise ValueError("Selected signal is missing from the raw data index.")
    if bool(
        (base_positions + MAX_HOLDING_HORIZON >= len(raw_index)).any()
    ):
        raise ValueError("Selected signal does not have a complete H15 path.")

    entry_open = pd.to_numeric(
        result["entry_open"], errors="coerce"
    ).to_numpy()
    high_h1 = pd.to_numeric(result["high_h1"], errors="coerce").to_numpy()
    low_h1 = pd.to_numeric(result["low_h1"], errors="coerce").to_numpy()
    close_h15 = pd.to_numeric(
        result["close_h15"], errors="coerce"
    ).to_numpy()
    long_selected = pd.Index(long_slowdown_signals.selected_index)
    short_selected = pd.Index(short_slowdown_signals.selected_index)
    min_slope = float(slope_min_initial)
    n_rows = len(result)

    long_hit = high_h1 >= long_tp
    short_hit = low_h1 <= -short_tp
    long_active = ~long_hit
    short_active = ~short_hit
    long_return = np.full(n_rows, np.nan, dtype=float)
    short_return = np.full(n_rows, np.nan, dtype=float)
    long_outcome = np.full(n_rows, "close_h15", dtype=object)
    short_outcome = np.full(n_rows, "close_h15", dtype=object)
    long_return[long_hit] = long_tp
    short_return[short_hit] = short_tp
    long_outcome[long_hit] = "tp_h1"
    short_outcome[short_hit] = "tp_h1"

    for decision_h in range(1, MAX_HOLDING_HORIZON):
        execution_h = decision_h + 1
        decision_index = raw_index.take(base_positions + decision_h)
        slope = initial_slope.reindex(decision_index).to_numpy(dtype=float)
        long_signal = decision_index.isin(long_selected)
        short_signal = decision_index.isin(short_selected)
        positive = slope > min_slope
        negative = slope < -min_slope

        long_order = long_active & (
            (positive & long_signal) | (negative & ~short_signal)
        )
        short_order = short_active & (
            (positive & ~long_signal) | (negative & short_signal)
        )
        long_target = long_projected_high.reindex(
            decision_index
        ).to_numpy(dtype=float)
        short_target = short_projected_high.reindex(
            decision_index
        ).to_numpy(dtype=float)
        long_limit = long_target / entry_open - 1.0
        short_limit = short_target / entry_open - 1.0
        open_return = pd.to_numeric(
            result[f"open_h{execution_h}"], errors="coerce"
        ).to_numpy()
        high_return = pd.to_numeric(
            result[f"high_h{execution_h}"], errors="coerce"
        ).to_numpy()
        low_return = pd.to_numeric(
            result[f"low_h{execution_h}"], errors="coerce"
        ).to_numpy()

        long_fill_open = (
            long_order
            & np.isfinite(long_limit)
            & (open_return >= long_limit)
        )
        long_fill_high = (
            long_order
            & ~long_fill_open
            & np.isfinite(long_limit)
            & (high_return >= long_limit)
        )
        short_fill_open = (
            short_order
            & np.isfinite(short_limit)
            & (open_return <= short_limit)
        )
        short_fill_low = (
            short_order
            & ~short_fill_open
            & np.isfinite(short_limit)
            & (low_return <= short_limit)
        )

        long_return[long_fill_open] = open_return[long_fill_open]
        long_return[long_fill_high] = long_limit[long_fill_high]
        short_return[short_fill_open] = -open_return[short_fill_open]
        short_return[short_fill_low] = -short_limit[short_fill_low]
        long_outcome[long_fill_open] = f"dynamic_tp_open_h{execution_h}"
        long_outcome[long_fill_high] = f"dynamic_tp_high_h{execution_h}"
        short_outcome[short_fill_open] = f"dynamic_tp_open_h{execution_h}"
        short_outcome[short_fill_low] = f"dynamic_tp_low_h{execution_h}"
        long_active[long_fill_open | long_fill_high] = False
        short_active[short_fill_open | short_fill_low] = False
        result[f"long_dynamic_tp_h{execution_h}"] = long_limit
        result[f"short_dynamic_tp_h{execution_h}"] = short_limit

    long_return[long_active] = close_h15[long_active]
    short_return[short_active] = -close_h15[short_active]
    result["long_hit"] = long_hit
    result["short_hit"] = short_hit
    result["long_outcome"] = long_outcome
    result["short_outcome"] = short_outcome
    result["long_return"] = long_return
    result["short_return"] = short_return
    result["gross_return"] = result["long_return"] + result["short_return"]
    result["net_return"] = result["gross_return"] - trade_cost
    result["cumulative_net_return"] = result["net_return"].cumsum()
    return result


def _voted_index(
    long_index: pd.Index,
    short_index: pd.Index,
    voting: str,
) -> pd.Index:
    long_dt = pd.DatetimeIndex(long_index)
    short_dt = pd.DatetimeIndex(short_index)
    if voting == "and":
        return long_dt.intersection(short_dt)
    return long_dt.union(short_dt)


def _summary_row(split: str, simulation: pd.DataFrame) -> dict[str, Any]:
    def rate_mean(
        mask: pd.Series,
        return_column: str,
    ) -> tuple[float, float]:
        selected = simulation.loc[mask, return_column]
        return (
            float(mask.mean()) if len(mask) else 0.0,
            float(selected.mean()) if not selected.empty else float("nan"),
        )

    n = len(simulation)
    if not n:
        return {
            "split": split,
            "n": 0,
            "trades/day": 0.0,
            "both TP H1": (0.0, float("nan")),
            "Long TP H1": (0.0, float("nan")),
            "Short TP H1": (0.0, float("nan")),
            "L dyn TP H2": (0.0, float("nan")),
            "L dyn TP H3": (0.0, float("nan")),
            "S dyn TP H2": (0.0, float("nan")),
            "S dyn TP H3": (0.0, float("nan")),
            "close H15": (0.0, float("nan")),
            "gross mean": 0.0,
            "E[net]": 0.0,
            "net > 0": 0.0,
        }

    span_days = max(
        1.0,
        (
            pd.DatetimeIndex(simulation.index).max()
            - pd.DatetimeIndex(simulation.index).min()
        ).total_seconds()
        / 86400.0,
    )
    long_outcome = simulation["long_outcome"]
    short_outcome = simulation["short_outcome"]
    both_tp = simulation["long_hit"].astype(bool) & simulation[
        "short_hit"
    ].astype(bool)
    long_dynamic_h2 = long_outcome.str.startswith(
        "dynamic_tp_"
    ) & long_outcome.str.endswith("_h2")
    long_dynamic_h3 = long_outcome.str.startswith(
        "dynamic_tp_"
    ) & long_outcome.str.endswith("_h3")
    short_dynamic_h2 = short_outcome.str.startswith(
        "dynamic_tp_"
    ) & short_outcome.str.endswith("_h2")
    short_dynamic_h3 = short_outcome.str.startswith(
        "dynamic_tp_"
    ) & short_outcome.str.endswith("_h3")
    close_h15 = long_outcome.eq("close_h15") | short_outcome.eq("close_h15")
    return {
        "split": split,
        "n": n,
        "trades/day": n / span_days,
        "both TP H1": rate_mean(both_tp, "gross_return"),
        "Long TP H1": rate_mean(
            simulation["long_hit"].astype(bool),
            "long_return",
        ),
        "Short TP H1": rate_mean(
            simulation["short_hit"].astype(bool),
            "short_return",
        ),
        "L dyn TP H2": rate_mean(long_dynamic_h2, "long_return"),
        "L dyn TP H3": rate_mean(long_dynamic_h3, "long_return"),
        "S dyn TP H2": rate_mean(short_dynamic_h2, "short_return"),
        "S dyn TP H3": rate_mean(short_dynamic_h3, "short_return"),
        "close H15": rate_mean(close_h15, "gross_return"),
        "gross mean": float(simulation["gross_return"].mean()),
        "E[net]": float(simulation["net_return"].mean()),
        "net > 0": float(simulation["net_return"].gt(0.0).mean()),
    }


def _format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    def format_rate_mean(value: Any) -> str:
        if not isinstance(value, tuple) or len(value) != 2:
            return "n/a"
        rate, mean_return = value
        mean_text = (
            "n/a"
            if not np.isfinite(float(mean_return))
            else f"{float(mean_return):+.3%}"
        )
        return f"{float(rate):.2%} ({mean_text})"

    formatted = summary.copy()
    formatted["trades/day"] = formatted["trades/day"].map(lambda x: f"{x:.2f}")
    for column in (
        "both TP H1",
        "Long TP H1",
        "Short TP H1",
        "L dyn TP H2",
        "L dyn TP H3",
        "S dyn TP H2",
        "S dyn TP H3",
        "close H15",
    ):
        formatted[column] = formatted[column].map(format_rate_mean)
    formatted["net > 0"] = formatted["net > 0"].map(lambda x: f"{x:.2%}")
    for column in ("gross mean", "E[net]"):
        formatted[column] = formatted[column].map(lambda x: f"{x:+.4%}")
    return formatted


def _dynamic_tp_detail(
    simulations: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Format H4-H15 dynamic TP rates and leg returns in a compact table."""
    rows: list[dict[str, str]] = []
    for split, simulation in simulations.items():
        for leg, outcome_column, return_column in (
            ("Long", "long_outcome", "long_return"),
            ("Short", "short_outcome", "short_return"),
        ):
            row: dict[str, str] = {"split / leg": f"{split} {leg}"}
            outcomes = simulation[outcome_column].astype(str)
            for horizon in range(4, MAX_HOLDING_HORIZON + 1):
                mask = outcomes.str.startswith(
                    "dynamic_tp_"
                ) & outcomes.str.endswith(f"_h{horizon}")
                mean_return = simulation.loc[mask, return_column].mean()
                mean_text = (
                    "n/a"
                    if not np.isfinite(float(mean_return))
                    else f"{float(mean_return):+.3%}"
                )
                row[f"H{horizon}"] = f"{mask.mean():.2%} ({mean_text})"
            rows.append(row)
    return pd.DataFrame(rows)


def _plot_result(
    summary: pd.DataFrame,
    dynamic_detail: pd.DataFrame,
    simulations: dict[str, pd.DataFrame],
    output_path: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(24, 14), constrained_layout=True)
    grid = fig.add_gridspec(
        4,
        1,
        height_ratios=[1.05, 1.35, 2.2, 1.2],
    )

    table_ax = fig.add_subplot(grid[0])
    table_ax.axis("off")
    formatted = _format_summary(summary)
    table = table_ax.table(
        cellText=formatted.values,
        colLabels=formatted.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.7)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2:
            cell.set_facecolor("#f3f4f6")
    table_ax.set_title(title, fontsize=12, weight="bold", pad=12)

    detail_ax = fig.add_subplot(grid[1])
    detail_ax.axis("off")
    detail_table = detail_ax.table(
        cellText=dynamic_detail.values,
        colLabels=dynamic_detail.columns,
        cellLoc="center",
        loc="center",
    )
    detail_table.auto_set_font_size(False)
    detail_table.set_fontsize(7.6)
    detail_table.scale(1.0, 1.45)
    for (row, _), cell in detail_table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2:
            cell.set_facecolor("#f3f4f6")
    detail_ax.set_title(
        "Dynamic TP H4-H15 | share of all paired signals (mean leg return)",
        fontsize=10,
        weight="bold",
        pad=8,
    )

    equity_ax = fig.add_subplot(grid[2])
    colors = {"val": "#2563eb", "test": "#dc2626"}
    for split, simulation in simulations.items():
        if simulation.empty:
            continue
        equity_ax.plot(
            simulation.index,
            simulation["cumulative_net_return"] * 100.0,
            label=split.upper(),
            color=colors[split],
            linewidth=1.2,
        )
    equity_ax.axhline(0.0, color="#111827", linewidth=0.8)
    equity_ax.set_ylabel("Cumulative net return (%)")
    equity_ax.grid(alpha=0.22)
    equity_ax.legend(loc="best")

    count_ax = fig.add_subplot(grid[3])
    for split, simulation in simulations.items():
        if simulation.empty:
            continue
        daily = simulation["net_return"].resample("1D").size()
        count_ax.plot(
            daily.index,
            daily.values,
            label=split.upper(),
            color=colors[split],
            linewidth=1.0,
        )
    count_ax.set_ylabel("Trades/day")
    count_ax.set_xlabel("Signal time")
    count_ax.grid(alpha=0.22)
    count_ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    long_settings = _archive_settings(
        Path(args.long_archive),
        args.long_rank,
        "long",
        args.long_top_fraction,
    )
    short_settings = _archive_settings(
        Path(args.short_archive),
        args.short_rank,
        "short",
        args.short_top_fraction,
    )
    long_slowdown_settings = _slowdown_settings(
        Path(args.long_slowdown_archive),
        args.long_slowdown_rank,
        "long",
        args.long_slowdown_top_fraction,
    )
    short_slowdown_settings = _slowdown_settings(
        Path(args.short_slowdown_archive),
        args.short_slowdown_rank,
        "short",
        args.short_slowdown_top_fraction,
    )
    slope_lookback = int(long_slowdown_settings.metadata["slope_lookback"])
    slope_min_initial = float(
        long_slowdown_settings.metadata["slope_min_initial"]
    )
    if (
        int(short_slowdown_settings.metadata["slope_lookback"])
        != slope_lookback
        or not np.isclose(
            float(short_slowdown_settings.metadata["slope_min_initial"]),
            slope_min_initial,
        )
    ):
        raise ValueError(
            "Long and Short slowdown archives must use the same slope "
            "lookback and minimum initial slope."
        )
    if (
        int(config.SLOPE_LOOKBACK) != slope_lookback
        or not np.isclose(float(config.SLOPE_MIN_INITIAL), slope_min_initial)
    ):
        raise ValueError(
            "Current config slope settings must match the slowdown archives: "
            f"config=({config.SLOPE_LOOKBACK}, {config.SLOPE_MIN_INITIAL}), "
            f"archive=({slope_lookback}, {slope_min_initial})."
        )

    data_path = Path(args.data)
    raw_df = load_ohlcv(data_path)
    purge_bars = config.purge_bars_for_horizons(
        [MAX_HOLDING_HORIZON, BASE_MODEL_HORIZON, 1]
    )

    long_spec = ModelSpec(
        archive_path=long_settings.path,
        rank=long_settings.rank,
        label_mode=long_settings.label_mode,
        label_threshold=long_settings.label_threshold,
        top_fraction=long_settings.top_fraction,
        label_direction="long",
    )
    short_spec = ModelSpec(
        archive_path=short_settings.path,
        rank=short_settings.rank,
        label_mode=short_settings.label_mode,
        label_threshold=short_settings.label_threshold,
        top_fraction=short_settings.top_fraction,
        label_direction="short",
    )
    long_slowdown_spec = ModelSpec(
        archive_path=long_slowdown_settings.path,
        rank=long_slowdown_settings.rank,
        label_mode="slope_slowdown",
        label_threshold=long_slowdown_settings.label_threshold,
        top_fraction=long_slowdown_settings.top_fraction,
        label_direction="long",
    )
    short_slowdown_spec = ModelSpec(
        archive_path=short_slowdown_settings.path,
        rank=short_slowdown_settings.rank,
        label_mode="slope_slowdown",
        label_threshold=short_slowdown_settings.label_threshold,
        top_fraction=short_slowdown_settings.top_fraction,
        label_direction="short",
    )
    long_entry = _load_rank_entry(long_settings.path, long_settings.rank)
    short_entry = _load_rank_entry(short_settings.path, short_settings.rank)
    long_slowdown_entry = _load_rank_entry(
        long_slowdown_settings.path,
        long_slowdown_settings.rank,
    )
    short_slowdown_entry = _load_rank_entry(
        short_slowdown_settings.path,
        short_slowdown_settings.rank,
    )
    long_quality = _quality_train_index(
        raw_df,
        long_spec,
        long_settings.horizons,
        args.val_start,
        args.test_start,
        args.test_end,
        purge_bars,
    )
    short_quality = _quality_train_index(
        raw_df,
        short_spec,
        short_settings.horizons,
        args.val_start,
        args.test_start,
        args.test_end,
        purge_bars,
    )
    long_slowdown_quality = _quality_train_index(
        raw_df,
        long_slowdown_spec,
        [1],
        args.val_start,
        args.test_start,
        args.test_end,
        purge_bars,
    )
    short_slowdown_quality = _quality_train_index(
        raw_df,
        short_slowdown_spec,
        [1],
        args.val_start,
        args.test_start,
        args.test_end,
        purge_bars,
    )
    quality_index = (
        long_quality.union(short_quality)
        .union(long_slowdown_quality)
        .union(short_slowdown_quality)
    )
    feature_space = _cached_feature_space(
        raw_df=raw_df,
        data_path=data_path,
        required_windows=_required_windows_for_entries(
            [
                long_entry,
                short_entry,
                long_slowdown_entry,
                short_slowdown_entry,
            ]
        ),
        quality_index=quality_index,
    )
    long_bundle = _train_spec_bundle(
        spec=long_spec,
        entry=long_entry,
        raw_df=raw_df,
        feature_space=feature_space,
        horizons=long_settings.horizons,
        val_start=args.val_start,
        test_start=args.test_start,
        test_end=args.test_end,
        purge_bars=purge_bars,
    )
    short_bundle = _train_spec_bundle(
        spec=short_spec,
        entry=short_entry,
        raw_df=raw_df,
        feature_space=feature_space,
        horizons=short_settings.horizons,
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

    h15_path = make_h15_path(raw_df)
    initial_slope = config._rolling_log_ols_slope(
        pd.to_numeric(raw_df["high"], errors="coerce"),
        slope_lookback,
    )
    long_projected_high = project_next_high_for_slowdown(
        high=raw_df["high"],
        initial_slope=initial_slope,
        lookback=slope_lookback,
        slowdown_threshold=long_slowdown_settings.label_threshold,
        direction="long",
    )
    short_projected_high = project_next_high_for_slowdown(
        high=raw_df["high"],
        initial_slope=initial_slope,
        lookback=slope_lookback,
        slowdown_threshold=short_slowdown_settings.label_threshold,
        direction="short",
    )
    simulations: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for (
        split,
        long_signals,
        short_signals,
        long_slowdown_signals,
        short_slowdown_signals,
    ) in (
        (
            "val",
            long_bundle.val,
            short_bundle.val,
            long_slowdown_bundle.val,
            short_slowdown_bundle.val,
        ),
        (
            "test",
            long_bundle.test,
            short_bundle.test,
            long_slowdown_bundle.test,
            short_slowdown_bundle.test,
        ),
    ):
        selected_index = _voted_index(
            long_signals.selected_index,
            short_signals.selected_index,
            args.voting,
        )
        selected_path = h15_path.reindex(selected_index)
        simulation = simulate_two_leg_h15(
            selected_path,
            raw_index=pd.DatetimeIndex(raw_df.index),
            initial_slope=initial_slope,
            long_projected_high=long_projected_high,
            short_projected_high=short_projected_high,
            long_slowdown_signals=long_slowdown_signals,
            short_slowdown_signals=short_slowdown_signals,
            slope_min_initial=slope_min_initial,
            long_tp=args.long_tp,
            short_tp=args.short_tp,
            trade_cost=args.trade_cost,
        )
        simulations[split] = simulation
        rows.append(_summary_row(split, simulation))
        logger.info(
            "%s voting=%s | Long=%d Short=%d pair=%d | gross=%+.4f%% "
            "net=%+.4f%%",
            split.upper(),
            args.voting.upper(),
            len(long_signals.selected_index),
            len(short_signals.selected_index),
            len(simulation),
            100.0 * rows[-1]["gross mean"],
            100.0 * rows[-1]["E[net]"],
        )

    summary = pd.DataFrame(rows)
    dynamic_detail = _dynamic_tp_detail(simulations)
    output_name = (
        f"two_leg_5m_h15_{args.voting}_"
        f"L{long_settings.rank:02d}_top{long_settings.top_fraction:.0%}_"
        f"S{short_settings.rank:02d}_top{short_settings.top_fraction:.0%}_"
        f"slowL{long_slowdown_settings.top_fraction:.0%}_"
        f"slowS{short_slowdown_settings.top_fraction:.0%}_"
        f"LTP{args.long_tp * 100:.3f}pct_"
        f"STP{args.short_tp * 100:.3f}pct.png"
    ).replace(".", "p")
    output_name = output_name.replace("ppng", ".png")
    output_path = Path(args.out_dir) / output_name
    title = (
        f"Two-leg 5m H15 | voting={args.voting.upper()} | "
        f"H1-only Long TP={args.long_tp:.3%} | "
        f"Short TP={args.short_tp:.3%} | "
        "unbuffered dynamic exits | "
        f"cost once={args.trade_cost:.3%}"
    )
    _plot_result(summary, dynamic_detail, simulations, output_path, title)
    print("\n=== Two-leg H15 summary ===")
    print(_format_summary(summary).to_string(index=False))
    print("\n=== Dynamic TP H4-H15 ===")
    print(dynamic_detail.to_string(index=False))
    print(f"\nSaved: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest paired Long/Short positions through H15 from two "
            "H3 MFE archives."
        )
    )
    parser.add_argument("--long-archive", default=str(DEFAULT_LONG_ARCHIVE))
    parser.add_argument("--short-archive", default=str(DEFAULT_SHORT_ARCHIVE))
    parser.add_argument(
        "--long-slowdown-archive",
        default=str(DEFAULT_LONG_SLOWDOWN_ARCHIVE),
    )
    parser.add_argument(
        "--short-slowdown-archive",
        default=str(DEFAULT_SHORT_SLOWDOWN_ARCHIVE),
    )
    parser.add_argument("--long-rank", type=int, default=1)
    parser.add_argument("--short-rank", type=int, default=1)
    parser.add_argument("--long-slowdown-rank", type=int, default=1)
    parser.add_argument("--short-slowdown-rank", type=int, default=1)
    parser.add_argument("--long-top-fraction", type=float)
    parser.add_argument("--short-top-fraction", type=float)
    parser.add_argument("--long-slowdown-top-fraction", type=float)
    parser.add_argument("--short-slowdown-top-fraction", type=float)
    parser.add_argument(
        "--voting",
        choices=("and", "or"),
        default="or",
        help="and: both models must vote; or: at least one model votes.",
    )
    parser.add_argument("--long-tp", type=float, default=DEFAULT_TP)
    parser.add_argument("--short-tp", type=float, default=DEFAULT_TP)
    parser.add_argument("--trade-cost", type=float, default=DEFAULT_TRADE_COST)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--val-start", default=config.VAL_START)
    parser.add_argument("--test-start", default=config.TEST_START)
    parser.add_argument("--test-end", default=config.TEST_END)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
