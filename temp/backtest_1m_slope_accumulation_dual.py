"""Backtest a direction-neutral trade after simple 1m slope accumulation.

Signal at the close of candle t:

    abs(current slope) <= ratio * abs(any previous slope)

The current slope uses a rolling log-close window. Previous windows have the
same length and are shifted by one candle each.

Execution:
- Enter one Long and one Short at open(t+1).
- ATR is calculated through signal candle t only. Each trade receives TP/SL
  distances derived from ATR14/close(t), clipped to configured bounds.
- The dynamic TP and SL are fixed after entry; they are not trailed.
- Hold each leg until TP or SL; there is no time exit.
- If TP and SL are both touched by one leg in the same 1m candle, the
  ``--same-candle-policy`` argument decides which one wins.
- Multiple signal pairs may overlap.

PowerShell:
    python -m temp.backtest_1m_slope_accumulation_dual
"""

from __future__ import annotations

import argparse
import heapq
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto import config


DEFAULT_DATA = Path("data/crypto/BTCUSDT_1m.csv")
DEFAULT_OUT_DIR = Path("temp/output")
DEFAULT_WINDOW = 3
DEFAULT_PREVIOUS_WINDOWS = 1
DEFAULT_SLOPE_RATIO = 0.15
ATR_LOOKBACK = 14
TP_ATR_MULTIPLIER = 1.0
SL_ATR_MULTIPLIER = 0.5
TP_MIN = 0.0004
TP_MAX = 0.0020
SL_MIN = 0.0002
SL_MAX = 0.0010
MIN_TP_SL_GAP = 0.0002
TP_ATR_GRID = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
SL_ATR_GRID = (0.25, 0.50, 0.75, 1.00)
DEFAULT_TRADE_COST = 0.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("temp.backtest_1m_slope_accumulation_dual")


def load_prices(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["date", "open", "high", "low", "close"],
        parse_dates=["date"],
    )
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame = frame.loc[
        (frame[["open", "high", "low", "close"]] > 0.0).all(axis=1)
    ]
    return frame.reset_index(drop=True)


def rolling_log_slopes(close: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(close, dtype=float)
    if int(window) < 2:
        raise ValueError("window must be at least 2.")
    slopes = np.full(len(values), np.nan, dtype=float)
    if len(values) < int(window):
        return slopes
    x = np.arange(int(window), dtype=float)
    x -= x.mean()
    weights = x / np.dot(x, x)
    windows = np.lib.stride_tricks.sliding_window_view(
        np.log(values),
        int(window),
    )
    slopes[int(window) - 1 :] = windows @ weights
    return slopes


def accumulation_signals(
    close: np.ndarray,
    *,
    window: int,
    previous_windows: int,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    if int(previous_windows) < 1:
        raise ValueError("previous_windows must be at least 1.")
    if not 0.0 <= float(ratio):
        raise ValueError("ratio must be non-negative.")
    slopes = rolling_log_slopes(close, int(window))
    signals = np.zeros(len(slopes), dtype=bool)
    first_end = int(window) + int(previous_windows) - 1
    if first_end >= len(slopes):
        return signals, slopes

    ends = np.arange(first_end, len(slopes))
    previous = np.vstack(
        [np.abs(slopes[ends - offset]) for offset in range(1, previous_windows + 1)]
    )
    previous_max = np.nanmax(previous, axis=0)
    current_abs = np.abs(slopes[ends])
    signals[ends] = (
        np.isfinite(current_abs)
        & np.isfinite(previous_max)
        & (current_abs <= float(ratio) * previous_max)
    )
    return signals, slopes


def atr_percent(frame: pd.DataFrame, lookback: int) -> np.ndarray:
    """Return causal Wilder ATR divided by close for every candle."""
    period = int(lookback)
    if period < 1:
        raise ValueError("atr_lookback must be at least 1.")
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()
    return (atr / close).to_numpy(dtype=float)


def _drop_inactive(heap: list[tuple[float, int]], active: np.ndarray) -> None:
    while heap and not bool(active[heap[0][1]]):
        heapq.heappop(heap)


def simulate_dual_tp_sl(
    frame: pd.DataFrame,
    signal_mask: np.ndarray,
    *,
    atr_pct: np.ndarray,
    tp_atr_multiplier: float,
    sl_atr_multiplier: float,
    tp_min: float,
    tp_max: float,
    sl_min: float,
    sl_max: float,
    min_tp_sl_gap: float = MIN_TP_SL_GAP,
    same_candle_policy: str = "sl_first",
) -> pd.DataFrame:
    atr_values = np.asarray(atr_pct, dtype=float)
    if len(atr_values) != len(frame):
        raise ValueError("atr_pct must have the same length as frame.")
    if float(tp_atr_multiplier) <= 0.0 or float(sl_atr_multiplier) <= 0.0:
        raise ValueError("ATR multipliers must be positive.")
    if not 0.0 < float(tp_min) <= float(tp_max):
        raise ValueError("TP bounds must satisfy 0 < tp_min <= tp_max.")
    if not 0.0 < float(sl_min) <= float(sl_max):
        raise ValueError("SL bounds must satisfy 0 < sl_min <= sl_max.")
    if float(min_tp_sl_gap) < 0.0:
        raise ValueError("min_tp_sl_gap must be non-negative.")
    if float(tp_max) < float(sl_max) + float(min_tp_sl_gap):
        raise ValueError("tp_max must be at least sl_max + min_tp_sl_gap.")
    policy = str(same_candle_policy).lower()
    if policy not in {"sl_first", "tp_first"}:
        raise ValueError("same_candle_policy must be 'sl_first' or 'tp_first'.")

    signal_ends = np.flatnonzero(signal_mask)
    entry_indices = signal_ends + 1
    valid = (entry_indices < len(frame)) & np.isfinite(atr_values[signal_ends])
    signal_ends = signal_ends[valid]
    entry_indices = entry_indices[valid]
    count = len(entry_indices)
    if count == 0:
        return pd.DataFrame()

    opens = frame["open"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    entries = opens[entry_indices]

    trade_atr_pct = atr_values[signal_ends]
    tp_values = np.clip(
        float(tp_atr_multiplier) * trade_atr_pct,
        float(tp_min),
        float(tp_max),
    )
    sl_values = np.clip(
        float(sl_atr_multiplier) * trade_atr_pct,
        float(sl_min),
        float(sl_max),
    )
    tp_values = np.maximum(tp_values, sl_values + float(min_tp_sl_gap))

    long_tp_level = entries * (1.0 + tp_values)
    long_sl_level = entries * (1.0 - sl_values)
    short_tp_level = entries * (1.0 - tp_values)
    short_sl_level = entries * (1.0 + sl_values)

    long_active = np.zeros(count, dtype=bool)
    short_active = np.zeros(count, dtype=bool)
    long_return = np.full(count, np.nan)
    short_return = np.full(count, np.nan)
    long_exit = np.full(count, -1, dtype=np.int64)
    short_exit = np.full(count, -1, dtype=np.int64)
    long_outcome = np.full(count, "unresolved", dtype=object)
    short_outcome = np.full(count, "unresolved", dtype=object)
    long_same_candle = np.zeros(count, dtype=bool)
    short_same_candle = np.zeros(count, dtype=bool)

    long_sl_heap: list[tuple[float, int]] = []
    long_tp_heap: list[tuple[float, int]] = []
    short_sl_heap: list[tuple[float, int]] = []
    short_tp_heap: list[tuple[float, int]] = []
    trade_at_entry = np.full(len(frame), -1, dtype=np.int64)
    trade_at_entry[entry_indices] = np.arange(count, dtype=np.int64)

    for candle in range(int(entry_indices[0]), len(frame)):
        trade = int(trade_at_entry[candle])
        if trade >= 0:
            long_active[trade] = True
            short_active[trade] = True
            heapq.heappush(long_sl_heap, (-long_sl_level[trade], trade))
            heapq.heappush(long_tp_heap, (long_tp_level[trade], trade))
            heapq.heappush(short_sl_heap, (short_sl_level[trade], trade))
            heapq.heappush(short_tp_heap, (-short_tp_level[trade], trade))

        candle_low = float(lows[candle])
        candle_high = float(highs[candle])

        if policy == "tp_first":
            _drop_inactive(long_tp_heap, long_active)
            while long_tp_heap and long_tp_heap[0][0] <= candle_high:
                _, trade_id = heapq.heappop(long_tp_heap)
                if not long_active[trade_id]:
                    continue
                long_same_candle[trade_id] = candle_low <= long_sl_level[trade_id]
                long_return[trade_id] = tp_values[trade_id]
                long_exit[trade_id] = candle
                long_outcome[trade_id] = "tp"
                long_active[trade_id] = False
                _drop_inactive(long_tp_heap, long_active)

            _drop_inactive(long_sl_heap, long_active)
            while long_sl_heap and -long_sl_heap[0][0] >= candle_low:
                _, trade_id = heapq.heappop(long_sl_heap)
                if not long_active[trade_id]:
                    continue
                long_same_candle[trade_id] = candle_high >= long_tp_level[trade_id]
                long_return[trade_id] = -sl_values[trade_id]
                long_exit[trade_id] = candle
                long_outcome[trade_id] = "sl"
                long_active[trade_id] = False
                _drop_inactive(long_sl_heap, long_active)

            _drop_inactive(short_tp_heap, short_active)
            while short_tp_heap and -short_tp_heap[0][0] >= candle_low:
                _, trade_id = heapq.heappop(short_tp_heap)
                if not short_active[trade_id]:
                    continue
                short_same_candle[trade_id] = candle_high >= short_sl_level[trade_id]
                short_return[trade_id] = tp_values[trade_id]
                short_exit[trade_id] = candle
                short_outcome[trade_id] = "tp"
                short_active[trade_id] = False
                _drop_inactive(short_tp_heap, short_active)

            _drop_inactive(short_sl_heap, short_active)
            while short_sl_heap and short_sl_heap[0][0] <= candle_high:
                _, trade_id = heapq.heappop(short_sl_heap)
                if not short_active[trade_id]:
                    continue
                short_same_candle[trade_id] = candle_low <= short_tp_level[trade_id]
                short_return[trade_id] = -sl_values[trade_id]
                short_exit[trade_id] = candle
                short_outcome[trade_id] = "sl"
                short_active[trade_id] = False
                _drop_inactive(short_sl_heap, short_active)
        else:
            _drop_inactive(long_sl_heap, long_active)
            while long_sl_heap and -long_sl_heap[0][0] >= candle_low:
                _, trade_id = heapq.heappop(long_sl_heap)
                if not long_active[trade_id]:
                    continue
                long_same_candle[trade_id] = candle_high >= long_tp_level[trade_id]
                long_return[trade_id] = -sl_values[trade_id]
                long_exit[trade_id] = candle
                long_outcome[trade_id] = "sl"
                long_active[trade_id] = False
                _drop_inactive(long_sl_heap, long_active)

            _drop_inactive(long_tp_heap, long_active)
            while long_tp_heap and long_tp_heap[0][0] <= candle_high:
                _, trade_id = heapq.heappop(long_tp_heap)
                if not long_active[trade_id]:
                    continue
                long_same_candle[trade_id] = candle_low <= long_sl_level[trade_id]
                long_return[trade_id] = tp_values[trade_id]
                long_exit[trade_id] = candle
                long_outcome[trade_id] = "tp"
                long_active[trade_id] = False
                _drop_inactive(long_tp_heap, long_active)

            _drop_inactive(short_sl_heap, short_active)
            while short_sl_heap and short_sl_heap[0][0] <= candle_high:
                _, trade_id = heapq.heappop(short_sl_heap)
                if not short_active[trade_id]:
                    continue
                short_same_candle[trade_id] = candle_low <= short_tp_level[trade_id]
                short_return[trade_id] = -sl_values[trade_id]
                short_exit[trade_id] = candle
                short_outcome[trade_id] = "sl"
                short_active[trade_id] = False
                _drop_inactive(short_sl_heap, short_active)

            _drop_inactive(short_tp_heap, short_active)
            while short_tp_heap and -short_tp_heap[0][0] >= candle_low:
                _, trade_id = heapq.heappop(short_tp_heap)
                if not short_active[trade_id]:
                    continue
                short_same_candle[trade_id] = candle_high >= short_sl_level[trade_id]
                short_return[trade_id] = tp_values[trade_id]
                short_exit[trade_id] = candle
                short_outcome[trade_id] = "tp"
                short_active[trade_id] = False
                _drop_inactive(short_tp_heap, short_active)

    result = pd.DataFrame(
        {
            "signal_time": frame["date"].iloc[signal_ends].to_numpy(),
            "entry_time": frame["date"].iloc[entry_indices].to_numpy(),
            "entry_index": entry_indices,
            "entry": entries,
            "atr_pct": trade_atr_pct,
            "take_profit": tp_values,
            "stop_loss": sl_values,
            "long_outcome": long_outcome,
            "long_return": long_return,
            "long_exit_index": long_exit,
            "long_same_candle_tp_sl": long_same_candle,
            "short_outcome": short_outcome,
            "short_return": short_return,
            "short_exit_index": short_exit,
            "short_same_candle_tp_sl": short_same_candle,
        }
    )
    result["resolved"] = (
        result["long_return"].notna() & result["short_return"].notna()
    )
    result["pair_gross_return"] = (
        result["long_return"] + result["short_return"]
    )
    result["long_hold_bars"] = result["long_exit_index"] - result["entry_index"] + 1
    result["short_hold_bars"] = (
        result["short_exit_index"] - result["entry_index"] + 1
    )
    return result


def summarize(
    name: str,
    trades: pd.DataFrame,
    *,
    trade_cost: float,
) -> dict[str, object]:
    resolved = trades.loc[trades["resolved"]].copy()
    gross = resolved["pair_gross_return"]
    net = gross - float(trade_cost)
    long_tp = resolved["long_outcome"].eq("tp")
    short_tp = resolved["short_outcome"].eq("tp")
    both_tp = long_tp & short_tp
    mixed = long_tp ^ short_tp
    both_sl = ~long_tp & ~short_tp
    n = len(trades)
    nr = len(resolved)
    return {
        "split": name,
        "signals": n,
        "resolved": nr,
        "unresolved": n - nr,
        "long TP": float(long_tp.mean()) if nr else np.nan,
        "short TP": float(short_tp.mean()) if nr else np.nan,
        "both TP": float(both_tp.mean()) if nr else np.nan,
        "one TP + one SL": float(mixed.mean()) if nr else np.nan,
        "both SL": float(both_sl.mean()) if nr else np.nan,
        "L same candle": (
            float(resolved["long_same_candle_tp_sl"].mean()) if nr else np.nan
        ),
        "S same candle": (
            float(resolved["short_same_candle_tp_sl"].mean()) if nr else np.nan
        ),
        "ATR mean": float(resolved["atr_pct"].mean()) if nr else np.nan,
        "TP mean": float(resolved["take_profit"].mean()) if nr else np.nan,
        "SL mean": float(resolved["stop_loss"].mean()) if nr else np.nan,
        "gross mean": float(gross.mean()) if nr else np.nan,
        "E[net]": float(net.mean()) if nr else np.nan,
        "win rate": float((net > 0.0).mean()) if nr else np.nan,
        "L hold bars": (
            float(resolved["long_hold_bars"].mean()) if nr else np.nan
        ),
        "S hold bars": (
            float(resolved["short_hold_bars"].mean()) if nr else np.nan
        ),
    }


def format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    for column in ("signals", "resolved", "unresolved"):
        result[column] = result[column].map(lambda value: f"{int(value):,}")
    for column in (
        "long TP",
        "short TP",
        "both TP",
        "one TP + one SL",
        "both SL",
        "L same candle",
        "S same candle",
        "win rate",
        "ATR mean",
        "TP mean",
        "SL mean",
    ):
        result[column] = result[column].map(lambda value: f"{value:.2%}")
    for column in ("gross mean", "E[net]"):
        result[column] = result[column].map(lambda value: f"{value:+.4%}")
    for column in ("L hold bars", "S hold bars"):
        result[column] = result[column].map(lambda value: f"{value:.2f}")
    return result


def summarize_splits(trades: pd.DataFrame, trade_cost: float) -> pd.DataFrame:
    entry_time = pd.to_datetime(trades["entry_time"])
    val_start = pd.Timestamp(config.VAL_START)
    test_start = pd.Timestamp(config.TEST_START)
    test_end = pd.Timestamp(config.TEST_END) if config.TEST_END is not None else None
    masks = {
        "all": np.ones(len(trades), dtype=bool),
        "val": (entry_time >= val_start) & (entry_time < test_start),
        "test": (entry_time >= test_start)
        & (
            (entry_time <= test_end)
            if test_end is not None
            else np.ones(len(trades), dtype=bool)
        ),
    }
    return pd.DataFrame(
        [
            summarize(
                name,
                trades.loc[np.asarray(mask)],
                trade_cost=float(trade_cost),
            )
            for name, mask in masks.items()
        ]
    )


def parse_multiplier_grid(value: str) -> tuple[float, ...]:
    try:
        grid = tuple(sorted({float(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Grid must be comma-separated numbers.") from exc
    if not grid or any(item <= 0.0 for item in grid):
        raise argparse.ArgumentTypeError("Grid values must be positive.")
    return grid


def optimize_atr_multipliers(
    frame: pd.DataFrame,
    signal_mask: np.ndarray,
    atr_pct: np.ndarray,
    *,
    tp_grid: tuple[float, ...],
    sl_grid: tuple[float, ...],
    tp_min: float,
    tp_max: float,
    sl_min: float,
    sl_max: float,
    min_tp_sl_gap: float,
    same_candle_policy: str,
    trade_cost: float,
) -> tuple[float, float, pd.DataFrame]:
    dates = pd.to_datetime(frame["date"])
    val_positions = np.flatnonzero(
        ((dates >= pd.Timestamp(config.VAL_START)) &
         (dates < pd.Timestamp(config.TEST_START))).to_numpy()
    )
    if len(val_positions) < 2:
        raise ValueError("Validation candle range is empty or too short.")

    start = int(val_positions[0])
    stop = int(val_positions[-1]) + 1
    val_frame = frame.iloc[start:stop].reset_index(drop=True)
    val_signals = np.asarray(signal_mask[start:stop], dtype=bool)
    val_atr = np.asarray(atr_pct[start:stop], dtype=float)
    rows: list[dict[str, float]] = []

    for tp_multiplier in tp_grid:
        for sl_multiplier in sl_grid:
            trades = simulate_dual_tp_sl(
                val_frame,
                val_signals,
                atr_pct=val_atr,
                tp_atr_multiplier=float(tp_multiplier),
                sl_atr_multiplier=float(sl_multiplier),
                tp_min=float(tp_min),
                tp_max=float(tp_max),
                sl_min=float(sl_min),
                sl_max=float(sl_max),
                min_tp_sl_gap=float(min_tp_sl_gap),
                same_candle_policy=str(same_candle_policy),
            )
            resolved = trades.loc[trades["resolved"]]
            # Mark unresolved boundary trades at zero gross while still charging cost.
            net_sum = float(resolved["pair_gross_return"].sum()) - (
                float(trade_cost) * len(trades)
            )
            objective = net_sum / len(trades) if len(trades) else -np.inf
            rows.append(
                {
                    "tp_atr_multiplier": float(tp_multiplier),
                    "sl_atr_multiplier": float(sl_multiplier),
                    "val_E[net]": float(objective),
                    "signals": float(len(trades)),
                    "resolved": float(len(resolved)),
                }
            )

    candidates = pd.DataFrame(rows).sort_values(
        ["val_E[net]", "resolved"], ascending=[False, False]
    ).reset_index(drop=True)
    if candidates.empty or not np.isfinite(candidates.iloc[0]["val_E[net]"]):
        raise ValueError("ATR multiplier optimization produced no valid candidate.")
    best = candidates.iloc[0]
    return (
        float(best["tp_atr_multiplier"]),
        float(best["sl_atr_multiplier"]),
        candidates,
    )


def plot_report(
    trades: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    optimized_trades: pd.DataFrame | None,
    optimized_summary: pd.DataFrame | None,
    atr_lookback: int,
    tp_atr_multiplier: float,
    sl_atr_multiplier: float,
    optimized_tp_atr_multiplier: float | None,
    optimized_sl_atr_multiplier: float | None,
    tp_min: float,
    tp_max: float,
    sl_min: float,
    sl_max: float,
    min_tp_sl_gap: float,
    slope_ratio: float,
    trade_cost: float,
    same_candle_policy: str,
    output: Path,
) -> None:
    formatted = format_summary(summary)
    resolved = trades.loc[trades["resolved"]].copy()
    resolved["net_return"] = (
        resolved["pair_gross_return"] - float(trade_cost)
    )
    resolved = resolved.sort_values("entry_time")
    resolved["cumulative_net"] = resolved["net_return"].cumsum()

    has_optimized = optimized_trades is not None and optimized_summary is not None
    row_count = 3 if has_optimized else 2
    height_ratios = [1.2, 1.2, 3.2] if has_optimized else [1.2, 3.2]
    fig, axes = plt.subplots(
        row_count,
        1,
        figsize=(20, 10 if has_optimized else 8),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.38},
    )
    fig.patch.set_facecolor("#f4f6f8")

    def draw_table(axis: plt.Axes, table: pd.DataFrame, title: str) -> None:
        axis.axis("off")
        artist = axis.table(
            cellText=table.values,
            colLabels=table.columns,
            loc="center",
            cellLoc="center",
        )
        artist.auto_set_font_size(False)
        artist.set_fontsize(7.0)
        artist.scale(1.0, 1.48)
        for (row, _), cell in artist.get_celld().items():
            cell.set_edgecolor("#9aa3ad")
            cell.set_facecolor("#222831" if row == 0 else "#ffffff")
            cell.set_text_props(color="#ffffff" if row == 0 else "#111111")
        axis.set_title(title, fontsize=11, pad=12)

    draw_table(
        axes[0],
        formatted,
        "Configured ATR strategy | "
        f"ATR{atr_lookback} dynamic TP={tp_atr_multiplier:g}x "
        f"[{tp_min:.3%},{tp_max:.3%}] SL={sl_atr_multiplier:g}x "
        f"[{sl_min:.3%},{sl_max:.3%}] | "
        f"TP-SL gap >= {min_tp_sl_gap:.3%}",
    )

    equity_ax = axes[-1]
    if has_optimized:
        assert optimized_trades is not None
        assert optimized_summary is not None
        assert optimized_tp_atr_multiplier is not None
        assert optimized_sl_atr_multiplier is not None
        draw_table(
            axes[1],
            format_summary(optimized_summary),
            "Val-optimized ATR strategy, applied unchanged to Test | "
            f"TP={optimized_tp_atr_multiplier:g}x ATR | "
            f"SL={optimized_sl_atr_multiplier:g}x ATR | "
            f"same candle={same_candle_policy} | slope ratio={slope_ratio:.0%}",
        )
        optimized_resolved = optimized_trades.loc[
            optimized_trades["resolved"]
        ].copy()
        optimized_resolved["net_return"] = (
            optimized_resolved["pair_gross_return"] - float(trade_cost)
        )
        optimized_resolved = optimized_resolved.sort_values("entry_time")
        optimized_resolved["cumulative_net"] = optimized_resolved[
            "net_return"
        ].cumsum()
        equity_ax.plot(
            pd.to_datetime(optimized_resolved["entry_time"]),
            optimized_resolved["cumulative_net"],
            color="#c0392b",
            linewidth=1.0,
            label="Val-optimized multipliers",
        )

    equity_ax.plot(
        pd.to_datetime(resolved["entry_time"]),
        resolved["cumulative_net"],
        color="#1769aa",
        linewidth=1.0,
        label="Configured multipliers",
    )
    equity_ax.axhline(0.0, color="#444444", linewidth=0.8)
    equity_ax.grid(True, alpha=0.3)
    equity_ax.set_ylabel("Cumulative pair net return")
    equity_ax.set_xlabel("Entry time")
    equity_ax.set_title(
        "Cumulative arithmetic return (overlapping trades allowed)"
    )
    equity_ax.legend(loc="best")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument(
        "--previous-windows",
        type=int,
        default=DEFAULT_PREVIOUS_WINDOWS,
    )
    parser.add_argument("--slope-ratio", type=float, default=DEFAULT_SLOPE_RATIO)
    parser.add_argument("--atr-lookback", type=int, default=ATR_LOOKBACK)
    parser.add_argument(
        "--tp-atr-multiplier", type=float, default=TP_ATR_MULTIPLIER
    )
    parser.add_argument(
        "--sl-atr-multiplier", type=float, default=SL_ATR_MULTIPLIER
    )
    parser.add_argument("--tp-min", type=float, default=TP_MIN)
    parser.add_argument("--tp-max", type=float, default=TP_MAX)
    parser.add_argument("--sl-min", type=float, default=SL_MIN)
    parser.add_argument("--sl-max", type=float, default=SL_MAX)
    parser.add_argument("--min-tp-sl-gap", type=float, default=MIN_TP_SL_GAP)
    parser.add_argument(
        "--optimize-atr-multipliers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Optimize TP/SL ATR multipliers on Val and apply them to Test.",
    )
    parser.add_argument(
        "--tp-atr-grid",
        type=parse_multiplier_grid,
        default=TP_ATR_GRID,
        help="Comma-separated TP multiplier candidates.",
    )
    parser.add_argument(
        "--sl-atr-grid",
        type=parse_multiplier_grid,
        default=SL_ATR_GRID,
        help="Comma-separated SL multiplier candidates.",
    )
    parser.add_argument("--trade-cost", type=float, default=DEFAULT_TRADE_COST)
    parser.add_argument(
        "--same-candle-policy",
        choices=("sl_first", "tp_first"),
        default="sl_first",
        help="Outcome when one leg touches both TP and SL in the same 1m candle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Loading %s", args.data)
    frame = load_prices(args.data)
    logger.info("Loaded %s one-minute candles.", f"{len(frame):,}")
    signal_mask, _ = accumulation_signals(
        frame["close"].to_numpy(dtype=float),
        window=int(args.window),
        previous_windows=int(args.previous_windows),
        ratio=float(args.slope_ratio),
    )
    logger.info(
        "Accumulation signals: %s",
        f"{int(signal_mask.sum()):,}",
    )
    atr_pct = atr_percent(frame, int(args.atr_lookback))
    trades = simulate_dual_tp_sl(
        frame,
        signal_mask,
        atr_pct=atr_pct,
        tp_atr_multiplier=float(args.tp_atr_multiplier),
        sl_atr_multiplier=float(args.sl_atr_multiplier),
        tp_min=float(args.tp_min),
        tp_max=float(args.tp_max),
        sl_min=float(args.sl_min),
        sl_max=float(args.sl_max),
        min_tp_sl_gap=float(args.min_tp_sl_gap),
        same_candle_policy=str(args.same_candle_policy),
    )
    summary = summarize_splits(trades, float(args.trade_cost))

    optimized_trades: pd.DataFrame | None = None
    optimized_summary: pd.DataFrame | None = None
    optimized_tp_multiplier: float | None = None
    optimized_sl_multiplier: float | None = None
    candidates: pd.DataFrame | None = None
    if bool(args.optimize_atr_multipliers):
        logger.info(
            "Optimizing ATR multipliers on Val | TP grid=%s | SL grid=%s",
            args.tp_atr_grid,
            args.sl_atr_grid,
        )
        (
            optimized_tp_multiplier,
            optimized_sl_multiplier,
            candidates,
        ) = optimize_atr_multipliers(
            frame,
            signal_mask,
            atr_pct,
            tp_grid=tuple(args.tp_atr_grid),
            sl_grid=tuple(args.sl_atr_grid),
            tp_min=float(args.tp_min),
            tp_max=float(args.tp_max),
            sl_min=float(args.sl_min),
            sl_max=float(args.sl_max),
            min_tp_sl_gap=float(args.min_tp_sl_gap),
            same_candle_policy=str(args.same_candle_policy),
            trade_cost=float(args.trade_cost),
        )
        logger.info(
            "Val-optimal ATR multipliers | TP=%gx | SL=%gx | E[net]=%+.4f%%",
            optimized_tp_multiplier,
            optimized_sl_multiplier,
            float(candidates.iloc[0]["val_E[net]"]) * 100.0,
        )
        if (
            optimized_tp_multiplier == float(args.tp_atr_multiplier)
            and optimized_sl_multiplier == float(args.sl_atr_multiplier)
        ):
            optimized_trades = trades.copy()
        else:
            optimized_trades = simulate_dual_tp_sl(
                frame,
                signal_mask,
                atr_pct=atr_pct,
                tp_atr_multiplier=optimized_tp_multiplier,
                sl_atr_multiplier=optimized_sl_multiplier,
                tp_min=float(args.tp_min),
                tp_max=float(args.tp_max),
                sl_min=float(args.sl_min),
                sl_max=float(args.sl_max),
                min_tp_sl_gap=float(args.min_tp_sl_gap),
                same_candle_policy=str(args.same_candle_policy),
            )
        optimized_summary = summarize_splits(
            optimized_trades, float(args.trade_cost)
        )
    output = args.out_dir / (
        "BTCUSDT_1m_slope_accumulation_dual_"
        f"atr{args.atr_lookback}_"
        f"tpx{args.tp_atr_multiplier:g}_slx{args.sl_atr_multiplier:g}_"
        f"{'optimized_' if args.optimize_atr_multipliers else ''}"
        f"{args.same_candle_policy}.png"
    )
    plot_report(
        trades,
        summary,
        optimized_trades=optimized_trades,
        optimized_summary=optimized_summary,
        atr_lookback=int(args.atr_lookback),
        tp_atr_multiplier=float(args.tp_atr_multiplier),
        sl_atr_multiplier=float(args.sl_atr_multiplier),
        optimized_tp_atr_multiplier=optimized_tp_multiplier,
        optimized_sl_atr_multiplier=optimized_sl_multiplier,
        tp_min=float(args.tp_min),
        tp_max=float(args.tp_max),
        sl_min=float(args.sl_min),
        sl_max=float(args.sl_max),
        min_tp_sl_gap=float(args.min_tp_sl_gap),
        slope_ratio=float(args.slope_ratio),
        trade_cost=float(args.trade_cost),
        same_candle_policy=str(args.same_candle_policy),
        output=output,
    )
    print("\n=== 1m slope accumulation dual strategy ===")
    print(format_summary(summary).to_string(index=False))
    if optimized_summary is not None and candidates is not None:
        print("\n=== Val-optimized ATR multiplier strategy ===")
        print(
            f"TP multiplier={optimized_tp_multiplier:g} | "
            f"SL multiplier={optimized_sl_multiplier:g}"
        )
        print(format_summary(optimized_summary).to_string(index=False))
        printable = candidates.head(10).copy()
        printable["val_E[net]"] = printable["val_E[net]"].map(
            lambda value: f"{value:+.4%}"
        )
        printable[["signals", "resolved"]] = printable[
            ["signals", "resolved"]
        ].astype(int)
        print("\nTop Val candidates:")
        print(printable.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
