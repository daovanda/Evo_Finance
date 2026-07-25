"""Backtest the slope-slowdown Long/Short reversal strategy.

The archive direction names describe the slope regime being predicted:

* Long archive: rising ``high`` slope slows down -> open a Short position.
* Short archive: falling ``high`` slope recovers -> open a Long position.

Signals are evaluated at candle ``t`` and executed at ``open(t+1)``. After the
archive horizon has completed, an unconfirmed slowdown exits at the next open.
A confirmed slowdown remains open until the opposite slope gate is reached;
the opposite model is not consulted for exits. Only one position may be open
at a time.

PowerShell:
    python -m crypto.backtest_long_short `
      --long-archive crypto/results/crypto_btc_long_slope_slowdown_h3_seed1_1h.json `
      --short-archive crypto/results/crypto_btc_short_slope_slowdown_h3_seed1_1h.json `
      --long-rank 1 `
      --short-rank 1 `
      --data data/crypto/BTCUSDT_15m.csv `
      --out-dir crypto/results/backtest_long_short

Bash/VM:
    python -m crypto.backtest_long_short \
      --long-archive crypto/results/crypto_btc_long_slope_slowdown_h3_seed1_1h.json \
      --short-archive crypto/results/crypto_btc_short_slope_slowdown_h3_seed1_1h.json \
      --long-rank 1 \
      --short-rank 1 \
      --data data/crypto/BTCUSDT_15m.csv \
      --out-dir crypto/results/backtest_long_short
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto import config
from crypto.analyze import _load_rank_entry, _required_windows_for_entries
from crypto.backtest import (
    BundleSignals,
    ModelSpec,
    _archive_horizons,
    _cached_feature_space,
    _quality_train_index,
    _train_spec_bundle,
)
from crypto.data import load_ohlcv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crypto.backtest_long_short")


DEFAULT_OUT_DIR = config.RESULTS_DIR / "backtest_long_short"


@dataclass(frozen=True)
class SlopeArchiveSettings:
    path: Path
    rank: int
    direction: str
    horizon: int
    label_threshold: float
    top_fraction: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LongShortBacktestResult:
    trades: pd.DataFrame
    summary: pd.DataFrame
    trades_csv_path: Path
    summary_csv_path: Path
    chart_path: Path


def run_backtest_long_short(
    long_archive: str | Path,
    short_archive: str | Path,
    long_rank: int = 1,
    short_rank: int = 1,
    long_top_fraction: float | None = None,
    short_top_fraction: float | None = None,
    data_path: str | Path = config.DATA_PATH,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    val_start: str = config.VAL_START,
    test_start: str = config.TEST_START,
    test_end: str | None = config.TEST_END,
) -> LongShortBacktestResult:
    """Train the two archive individuals and backtest their reversal flow."""
    long_settings = _read_slope_archive(
        long_archive,
        rank=long_rank,
        expected_direction="long",
        top_fraction=long_top_fraction,
    )
    short_settings = _read_slope_archive(
        short_archive,
        rank=short_rank,
        expected_direction="short",
        top_fraction=short_top_fraction,
    )
    if long_settings.horizon != short_settings.horizon:
        raise ValueError(
            "Long and Short slope archives must use the same single horizon; "
            f"got H{long_settings.horizon} and H{short_settings.horizon}."
        )
    horizon = int(long_settings.horizon)
    purge_bars = config.purge_bars_for_horizons([horizon])

    long_spec = _model_spec(long_settings)
    short_spec = _model_spec(short_settings)
    logger.info("Loading crypto data from %s", data_path)
    raw_df = load_ohlcv(data_path)
    entries = [
        _load_rank_entry(long_settings.path, long_settings.rank),
        _load_rank_entry(short_settings.path, short_settings.rank),
    ]
    quality_index = _quality_train_index(
        raw_df,
        long_spec,
        [horizon],
        val_start,
        test_start,
        test_end,
        purge_bars,
    ).union(
        _quality_train_index(
            raw_df,
            short_spec,
            [horizon],
            val_start,
            test_start,
            test_end,
            purge_bars,
        )
    )
    feature_space = _cached_feature_space(
        raw_df=raw_df,
        data_path=data_path,
        required_windows=_required_windows_for_entries(entries),
        quality_index=quality_index,
    )
    long_bundle = _train_spec_bundle(
        long_spec,
        entries[0],
        raw_df,
        feature_space,
        [horizon],
        val_start,
        test_start,
        test_end,
        purge_bars,
    )
    short_bundle = _train_spec_bundle(
        short_spec,
        entries[1],
        raw_df,
        feature_space,
        [horizon],
        val_start,
        test_start,
        test_end,
        purge_bars,
    )

    high = pd.to_numeric(raw_df["high"], errors="coerce")
    initial_slope = config._rolling_log_ols_slope(
        high,
        int(config.SLOPE_LOOKBACK),
    )
    long_strength = config.slope_slowdown_future_return(
        raw_df,
        horizon,
        direction="long",
    )
    short_strength = config.slope_slowdown_future_return(
        raw_df,
        horizon,
        direction="short",
    )

    frames: list[pd.DataFrame] = []
    for split, long_signals, short_signals in (
        ("val", long_bundle.val, short_bundle.val),
        ("test", long_bundle.test, short_bundle.test),
    ):
        split_index = _raw_split_index(
            raw_df.index,
            split=split,
            val_start=val_start,
            test_start=test_start,
            test_end=test_end,
        )
        frames.append(
            _simulate_split(
                raw_df=raw_df,
                split=split,
                split_index=split_index,
                long_bundle=long_signals,
                short_bundle=short_signals,
                initial_slope=initial_slope,
                long_strength=long_strength,
                short_strength=short_strength,
                horizon=horizon,
                long_threshold=long_settings.label_threshold,
                short_threshold=short_settings.label_threshold,
            )
        )
    trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = _summarize_trades(trades)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = _run_name(long_settings, short_settings)
    trades_csv_path = out_path / f"{stem}_trades.csv"
    summary_csv_path = out_path / f"{stem}_summary.csv"
    chart_path = out_path / f"{stem}.png"
    trades.to_csv(trades_csv_path, index=False)
    summary.to_csv(summary_csv_path, index=False)
    _plot_result(trades, summary, chart_path, long_settings, short_settings)

    logger.info("Trades: %s", trades_csv_path)
    logger.info("Summary: %s", summary_csv_path)
    logger.info("Chart: %s", chart_path)
    return LongShortBacktestResult(
        trades=trades,
        summary=summary,
        trades_csv_path=trades_csv_path,
        summary_csv_path=summary_csv_path,
        chart_path=chart_path,
    )


def _simulate_split(
    raw_df: pd.DataFrame,
    split: str,
    split_index: pd.Index,
    long_bundle: Any,
    short_bundle: Any,
    initial_slope: pd.Series,
    long_strength: pd.Series,
    short_strength: pd.Series,
    horizon: int,
    long_threshold: float,
    short_threshold: float,
) -> pd.DataFrame:
    """Simulate one-position-at-a-time trades without future execution."""
    columns = [
        "split",
        "entry_model",
        "position_side",
        "signal_time",
        "entry_time",
        "entry_price",
        "entry_prediction",
        "initial_slope",
        "strength_at_h",
        "confirmed_at_h",
        "review_time",
        "exit_reason",
        "exit_signal_time",
        "exit_time",
        "exit_price",
        "holding_bars",
        "gross_return",
        "trade_cost",
        "net_return",
    ]
    if len(split_index) == 0:
        return pd.DataFrame(columns=columns)

    raw = raw_df.sort_index()
    raw_index = pd.DatetimeIndex(raw.index)
    positions = raw_index.get_indexer(pd.DatetimeIndex(split_index))
    positions = positions[positions >= 0]
    if len(positions) == 0:
        return pd.DataFrame(columns=columns)
    split_start = int(positions.min())
    split_end = int(positions.max())

    long_selected = set(pd.DatetimeIndex(long_bundle.selected_index))
    short_selected = set(pd.DatetimeIndex(short_bundle.selected_index))
    long_pred = pd.to_numeric(long_bundle.data.get("pred"), errors="coerce")
    short_pred = pd.to_numeric(short_bundle.data.get("pred"), errors="coerce")
    initial = pd.to_numeric(initial_slope, errors="coerce")
    long_future = pd.to_numeric(long_strength, errors="coerce")
    short_future = pd.to_numeric(short_strength, errors="coerce")
    open_price = pd.to_numeric(raw["open"], errors="coerce")
    close_price = pd.to_numeric(raw["close"], errors="coerce")
    min_initial = float(config.SLOPE_MIN_INITIAL)

    rows: list[dict[str, Any]] = []
    decision_pos = split_start
    while decision_pos < split_end:
        timestamp = raw_index[decision_pos]
        slope_value = initial.get(timestamp, np.nan)
        entry_model: str | None = None
        position_side: str | None = None
        selected_strength: pd.Series | None = None
        selected_threshold = float("nan")
        prediction = float("nan")

        if (
            timestamp in long_selected
            and np.isfinite(slope_value)
            and slope_value > min_initial
        ):
            entry_model = "long_slowdown"
            position_side = "short"
            selected_strength = long_future
            selected_threshold = float(long_threshold)
            prediction = _series_value(long_pred, timestamp)
        elif (
            timestamp in short_selected
            and np.isfinite(slope_value)
            and slope_value < -min_initial
        ):
            entry_model = "short_recovery"
            position_side = "long"
            selected_strength = short_future
            selected_threshold = float(short_threshold)
            prediction = _series_value(short_pred, timestamp)

        if entry_model is None or selected_strength is None:
            decision_pos += 1
            continue

        entry_pos = decision_pos + 1
        review_pos = decision_pos + int(horizon)
        if review_pos >= split_end or not np.isfinite(open_price.iloc[entry_pos]):
            decision_pos += 1
            continue

        strength_value = _series_value(selected_strength, timestamp)
        confirmed = bool(
            np.isfinite(strength_value) and strength_value > selected_threshold
        )
        exit_signal_pos: int | None = None
        exit_reason: str

        if not confirmed:
            exit_signal_pos = review_pos
            exit_reason = f"not_confirmed_h{horizon}"
        else:
            for opposite_pos in range(review_pos, split_end):
                opposite_time = raw_index[opposite_pos]
                opposite_slope = initial.get(opposite_time, np.nan)
                if position_side == "short":
                    opposite = (
                        np.isfinite(opposite_slope)
                        and opposite_slope < -min_initial
                    )
                else:
                    opposite = (
                        np.isfinite(opposite_slope)
                        and opposite_slope > min_initial
                    )
                if opposite:
                    exit_signal_pos = opposite_pos
                    exit_reason = (
                        "opposite_short_slope"
                        if position_side == "short"
                        else "opposite_long_slope"
                    )
                    break
            else:
                exit_reason = "split_end"

        if exit_signal_pos is None:
            exit_pos = split_end
            exit_price = float(close_price.iloc[exit_pos])
            exit_signal_time = pd.NaT
        else:
            exit_pos = exit_signal_pos + 1
            if exit_pos <= split_end and np.isfinite(open_price.iloc[exit_pos]):
                exit_price = float(open_price.iloc[exit_pos])
            else:
                exit_pos = split_end
                exit_price = float(close_price.iloc[exit_pos])
                exit_reason = f"{exit_reason}_split_close"
            exit_signal_time = raw_index[exit_signal_pos]

        entry_value = float(open_price.iloc[entry_pos])
        gross_return = _position_return(position_side, entry_value, exit_price)
        rows.append(
            {
                "split": split,
                "entry_model": entry_model,
                "position_side": position_side,
                "signal_time": timestamp,
                "entry_time": raw_index[entry_pos],
                "entry_price": entry_value,
                "entry_prediction": prediction,
                "initial_slope": float(slope_value),
                "strength_at_h": strength_value,
                "confirmed_at_h": confirmed,
                "review_time": raw_index[review_pos],
                "exit_reason": exit_reason,
                "exit_signal_time": exit_signal_time,
                "exit_time": raw_index[exit_pos],
                "exit_price": exit_price,
                "holding_bars": int(exit_pos - entry_pos),
                "gross_return": gross_return,
                "trade_cost": float(config.TRADE_COST),
                "net_return": gross_return - float(config.TRADE_COST),
            }
        )
        decision_pos = max(decision_pos + 1, exit_pos)

    return pd.DataFrame(rows, columns=columns)


def _position_return(side: str, entry_price: float, exit_price: float) -> float:
    if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(exit_price):
        return float("nan")
    if side == "short":
        return float(1.0 - exit_price / entry_price)
    return float(exit_price / entry_price - 1.0)


def _summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "split",
        "position_side",
        "trades",
        "win_rate",
        "gross_mean",
        "net_mean",
        "net_compound",
        "median_holding_bars",
        "confirmed_at_h_rate",
        "not_confirmed_exits",
        "opposite_slope_exits",
        "split_end_exits",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for split in ("val", "test"):
        split_trades = trades[trades["split"] == split]
        for side in ("short", "long", "all"):
            frame = (
                split_trades
                if side == "all"
                else split_trades[split_trades["position_side"] == side]
            )
            n = int(len(frame))
            net = pd.to_numeric(frame.get("net_return"), errors="coerce").dropna()
            reason = frame.get("exit_reason", pd.Series(dtype=str)).astype(str)
            rows.append(
                {
                    "split": split,
                    "position_side": side,
                    "trades": n,
                    "win_rate": float((net > 0).mean()) if len(net) else float("nan"),
                    "gross_mean": _mean_column(frame, "gross_return"),
                    "net_mean": float(net.mean()) if len(net) else float("nan"),
                    "net_compound": (
                        float((1.0 + net).prod() - 1.0)
                        if len(net)
                        else float("nan")
                    ),
                    "median_holding_bars": _median_column(frame, "holding_bars"),
                    "confirmed_at_h_rate": _mean_column(frame, "confirmed_at_h"),
                    "not_confirmed_exits": int(reason.str.startswith("not_confirmed").sum()),
                    "opposite_slope_exits": int(
                        reason.str.startswith("opposite_").sum()
                    ),
                    "split_end_exits": int((reason == "split_end").sum()),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _plot_result(
    trades: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
    long_settings: SlopeArchiveSettings,
    short_settings: SlopeArchiveSettings,
) -> None:
    fig, (ax_equity, ax_table) = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        gridspec_kw={"height_ratios": [1.5, 1.0]},
    )
    for split, color in (("val", "#2f6f9f"), ("test", "#c45a36")):
        frame = trades[trades["split"] == split].sort_values("exit_time")
        net = pd.to_numeric(frame.get("net_return"), errors="coerce").fillna(0.0)
        equity = (1.0 + net).cumprod() - 1.0
        ax_equity.plot(
            pd.to_datetime(frame.get("exit_time")),
            equity,
            label=split,
            color=color,
            linewidth=1.4,
        )
    ax_equity.axhline(0.0, color="#333333", linewidth=0.8)
    ax_equity.set_ylabel("Compounded net return")
    ax_equity.grid(alpha=0.2)
    ax_equity.legend(loc="best")
    ax_equity.set_title(
        "Slope slowdown reversal | "
        f"H{long_settings.horizon} | Long-model -> Short, "
        "Short-model -> Long | exit=opposite slope only"
    )

    ax_table.axis("off")
    display = summary.copy()
    for column in (
        "win_rate",
        "gross_mean",
        "net_mean",
        "net_compound",
        "confirmed_at_h_rate",
    ):
        if column in display:
            display[column] = display[column].map(_format_percent)
    if "median_holding_bars" in display:
        display["median_holding_bars"] = display["median_holding_bars"].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.1f}"
        )
    table = ax_table.table(
        cellText=display.astype(str).values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.45)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _read_slope_archive(
    path: str | Path,
    rank: int,
    expected_direction: str,
    top_fraction: float | None,
) -> SlopeArchiveSettings:
    archive_path = Path(path)
    payload = json.loads(archive_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if config.canonical_label_mode(metadata.get("label_mode")) != "slope_slowdown":
        raise ValueError(f"{archive_path} is not a slope_slowdown archive.")
    direction = config.canonical_label_direction(metadata.get("label_direction"))
    if direction != expected_direction:
        raise ValueError(
            f"{archive_path} direction is {direction!r}; expected "
            f"{expected_direction!r}."
        )
    horizons = _archive_horizons(
        archive_path,
        fallback=list(config.HOLDING_HORIZONS),
        label=expected_direction,
    )
    if len(horizons) != 1:
        raise ValueError(
            f"{archive_path} must contain exactly one slope horizon, got {horizons}."
        )
    _validate_slope_metadata(archive_path, metadata)
    selected_top = (
        float(top_fraction)
        if top_fraction is not None
        else float(metadata.get("trade_top_fraction", config.TRADE_TOP_FRACTION))
    )
    if not 0.0 < selected_top <= 1.0:
        raise ValueError("top fraction must be in (0, 1].")
    return SlopeArchiveSettings(
        path=archive_path,
        rank=int(rank),
        direction=direction,
        horizon=int(horizons[0]),
        label_threshold=float(metadata["label_threshold"]),
        top_fraction=selected_top,
        metadata=metadata,
    )


def _validate_slope_metadata(path: Path, metadata: dict[str, Any]) -> None:
    expected = {
        "slope_lookback": int(config.SLOPE_LOOKBACK),
        "slope_min_initial": float(config.SLOPE_MIN_INITIAL),
        "slope_price_column": str(config.SLOPE_PRICE_COLUMN),
        "slope_slowdown_rule": str(config.SLOPE_SLOWDOWN_RULE),
    }
    for key, current in expected.items():
        archived = metadata.get(key)
        if archived != current:
            raise ValueError(
                f"{path} metadata {key}={archived!r} does not match current "
                f"config value {current!r}."
            )


def _model_spec(settings: SlopeArchiveSettings) -> ModelSpec:
    return ModelSpec(
        archive_path=settings.path,
        rank=settings.rank,
        label_mode="slope_slowdown",
        label_threshold=settings.label_threshold,
        top_fraction=settings.top_fraction,
        label_direction=settings.direction,
    )


def _raw_split_index(
    index: pd.Index,
    split: str,
    val_start: str,
    test_start: str,
    test_end: str | None,
) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if split == "val":
        return idx[(idx >= pd.Timestamp(val_start)) & (idx < pd.Timestamp(test_start))]
    mask = idx >= pd.Timestamp(test_start)
    if test_end:
        mask &= idx <= pd.Timestamp(test_end)
    return idx[mask]


def _run_name(
    long_settings: SlopeArchiveSettings,
    short_settings: SlopeArchiveSettings,
) -> str:
    def safe(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")

    return safe(
        "slope_reversal_"
        f"long_{long_settings.path.stem}_r{long_settings.rank:02d}_"
        f"top{long_settings.top_fraction:.0%}_"
        f"short_{short_settings.path.stem}_r{short_settings.rank:02d}_"
        f"top{short_settings.top_fraction:.0%}_exit_slope_only"
    )


def _series_value(series: pd.Series, index: Any) -> float:
    try:
        value = float(series.get(index, np.nan))
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _mean_column(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def _median_column(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if len(values) else float("nan")


def _format_percent(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:+.2%}" if np.isfinite(number) else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-archive", required=True)
    parser.add_argument("--short-archive", required=True)
    parser.add_argument("--long-rank", type=int, default=1)
    parser.add_argument("--short-rank", type=int, default=1)
    parser.add_argument("--long-top-fraction", type=float, default=None)
    parser.add_argument("--short-top-fraction", type=float, default=None)
    parser.add_argument("--data", default=str(config.DATA_PATH))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--val-start", default=config.VAL_START)
    parser.add_argument("--test-start", default=config.TEST_START)
    parser.add_argument("--test-end", default=config.TEST_END)
    args = parser.parse_args()
    result = run_backtest_long_short(
        long_archive=args.long_archive,
        short_archive=args.short_archive,
        long_rank=args.long_rank,
        short_rank=args.short_rank,
        long_top_fraction=args.long_top_fraction,
        short_top_fraction=args.short_top_fraction,
        data_path=args.data,
        out_dir=args.out_dir,
        val_start=args.val_start,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    print(result.summary.to_string(index=False))
    print(f"Trades: {result.trades_csv_path}")
    print(f"Summary: {result.summary_csv_path}")
    print(f"Chart: {result.chart_path}")


if __name__ == "__main__":
    main()
