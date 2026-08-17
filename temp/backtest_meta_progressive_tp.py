"""Backtest progressive TP decisions from cached MFE Q20 and 1m meta models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto.data import load_ohlcv


DEFAULT_MODEL_DIR = Path("temp/model/meta_mfe_q20_after_1m_2m_3m")
DEFAULT_DATA_5M = Path("data/crypto/BTCUSDT_5m.csv")
DEFAULT_DATA_1M = Path("data/crypto/BTCUSDT_1m.csv")
DEFAULT_OUT_DIR = Path("temp/output")


def _threshold(prediction: pd.Series, fraction: float) -> float:
    clean = prediction.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError("Cannot derive a score threshold from empty predictions.")
    count = min(len(clean), max(1, int(np.ceil(len(clean) * float(fraction)))))
    return float(clean.nlargest(count).iloc[-1])


def _model_record(manifest: dict, lookahead: int) -> dict:
    record = next(
        (
            item
            for item in manifest.get("models", [])
            if item.get("kind") == "meta_rank1_analysis_replay"
            and int(item.get("lookahead_bars", -1)) == int(lookahead)
        ),
        None,
    )
    if record is None:
        raise ValueError(f"Manifest has no after-{lookahead}m meta model.")
    return record


def _load_predictions(
    model_dir: Path,
    manifest: dict,
    meta: pd.DataFrame,
    lookahead: int,
) -> pd.Series:
    record = _model_record(manifest, lookahead)
    model_path = model_dir / str(record["model_path"])
    feature_path = model_dir / str(
        record.get("feature_cache_path", f"features_after_{lookahead}m.pkl")
    )
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Missing {feature_path}; rebuild the shared feature cache first."
        )
    features = pd.read_pickle(feature_path)
    expected = [str(value) for value in record["features"]]
    missing = [name for name in expected if name not in features.columns]
    if missing:
        raise ValueError(f"Feature cache {feature_path} is missing: {missing}")
    matrix = features.reindex(meta.index)[expected]
    if not matrix.index.equals(meta.index):
        raise ValueError(f"Feature cache {feature_path} could not align to meta rows.")
    booster = lgb.Booster(model_file=str(model_path))
    prediction = pd.Series(
        booster.predict(matrix.to_numpy(float)), index=meta.index, dtype=float
    )
    booster.free_dataset()
    return prediction


def _price_series(
    minute: pd.DataFrame,
    entry_index: pd.DatetimeIndex,
    column: str,
    minute_number: int,
) -> np.ndarray:
    target = entry_index + pd.Timedelta(minutes=int(minute_number) - 1)
    return (
        pd.to_numeric(minute[column], errors="coerce")
        .reindex(target)
        .to_numpy(float)
    )


def _band_take_profit(
    score: np.ndarray,
    thresholds: dict[float, float],
    *,
    tp_top10: float,
    tp_top20: float,
    tp_top40: float,
    tp_top70: float,
    tp_outside: float,
) -> tuple[np.ndarray, np.ndarray]:
    tp = np.full(len(score), float(tp_outside), dtype=float)
    band = np.full(len(score), "outside_top70", dtype=object)
    top70 = score >= thresholds[0.70]
    top40 = score >= thresholds[0.40]
    top20 = score >= thresholds[0.20]
    top10 = score >= thresholds[0.10]
    tp[top70] = float(tp_top70)
    band[top70] = "top40_70"
    tp[top40] = float(tp_top40)
    band[top40] = "top20_40"
    tp[top20] = float(tp_top20)
    band[top20] = "top10_20"
    tp[top10] = float(tp_top10)
    band[top10] = "top0_10"
    return tp, band


def _simulate_split(
    *,
    name: str,
    meta: pd.DataFrame,
    raw_5m: pd.DataFrame,
    minute: pd.DataFrame,
    predictions: dict[int, pd.Series],
    thresholds: dict[int, dict[float, float]],
    initial_tp_offset: float,
    tp_top10: float,
    tp_top20: float,
    tp_top40: float,
    tp_top70: float,
    tp_outside: float,
    trade_cost: float,
) -> pd.DataFrame:
    frame = meta.copy()
    frame["score_m1"] = predictions[1].reindex(frame.index)
    frame["score_m2"] = predictions[2].reindex(frame.index)
    frame["score_m3"] = predictions[3].reindex(frame.index)
    required = [
        "meta_dynamic_tp_h3",
        "score_m1",
        "score_m2",
        "score_m3",
    ]
    frame = frame.dropna(subset=required)
    signal_index = pd.DatetimeIndex(frame.index)
    entry_index = signal_index + pd.Timedelta(minutes=5)
    entry = pd.to_numeric(raw_5m["open"], errors="coerce").reindex(entry_index).to_numpy(float)
    close_h3 = (
        pd.to_numeric(raw_5m["close"], errors="coerce")
        .reindex(signal_index + pd.Timedelta(minutes=15))
        .to_numpy(float)
    )
    minute_open = {
        minute_number: _price_series(minute, entry_index, "open", minute_number)
        for minute_number in (2, 3, 4)
    }
    minute_high = {
        minute_number: _price_series(minute, entry_index, "high", minute_number)
        for minute_number in range(1, 16)
    }
    base_tp = pd.to_numeric(frame["meta_dynamic_tp_h3"], errors="coerce").to_numpy(float)
    initial_tp = base_tp + float(initial_tp_offset)
    score1 = frame["score_m1"].to_numpy(float)
    score2 = frame["score_m2"].to_numpy(float)
    score3 = frame["score_m3"].to_numpy(float)
    tp_m2, band_m1 = _band_take_profit(
        score1,
        thresholds[1],
        tp_top10=tp_top10,
        tp_top20=tp_top20,
        tp_top40=tp_top40,
        tp_top70=tp_top70,
        tp_outside=tp_outside,
    )
    tp_m3, band_m2 = _band_take_profit(
        score2,
        thresholds[2],
        tp_top10=tp_top10,
        tp_top20=tp_top20,
        tp_top40=tp_top40,
        tp_top70=tp_top70,
        tp_outside=tp_outside,
    )
    tp_m4_m15, band_m3 = _band_take_profit(
        score3,
        thresholds[3],
        tp_top10=tp_top10,
        tp_top20=tp_top20,
        tp_top40=tp_top40,
        tp_top70=tp_top70,
        tp_outside=tp_outside,
    )

    n = len(frame)
    gross = np.full(n, np.nan)
    exit_reason = np.full(n, "", dtype=object)
    exit_minute = np.full(n, np.nan)
    applied_tp = initial_tp.copy()
    valid_price = np.isfinite(entry) & (entry > 0.0) & np.isfinite(close_h3)
    active = valid_price.copy()

    hit_m1 = active & ((minute_high[1] / entry - 1.0) >= initial_tp)
    gross[hit_m1] = initial_tp[hit_m1]
    exit_reason[hit_m1] = "tp_m1"
    exit_minute[hit_m1] = 1
    active &= ~hit_m1

    applied_tp[active] = tp_m2[active]
    open_m2_return = minute_open[2] / entry - 1.0
    open_hit_m2 = active & (open_m2_return >= tp_m2)
    high_hit_m2 = (
        active
        & ~open_hit_m2
        & ((minute_high[2] / entry - 1.0) >= tp_m2)
    )
    hit_m2 = open_hit_m2 | high_hit_m2
    gross[open_hit_m2] = open_m2_return[open_hit_m2]
    gross[high_hit_m2] = tp_m2[high_hit_m2]
    exit_reason[hit_m2] = "tp_m2"
    exit_minute[hit_m2] = 2
    active &= ~hit_m2

    applied_tp[active] = tp_m3[active]
    open_m3_return = minute_open[3] / entry - 1.0
    open_hit_m3 = active & (open_m3_return >= tp_m3)
    high_hit_m3 = (
        active
        & ~open_hit_m3
        & ((minute_high[3] / entry - 1.0) >= tp_m3)
    )
    hit_m3 = open_hit_m3 | high_hit_m3
    gross[open_hit_m3] = open_m3_return[open_hit_m3]
    gross[high_hit_m3] = tp_m3[high_hit_m3]
    exit_reason[hit_m3] = "tp_m3"
    exit_minute[hit_m3] = 3
    active &= ~hit_m3

    applied_tp[active] = tp_m4_m15[active]
    future_hit = np.zeros(n, dtype=bool)
    first_hit_minute = np.full(n, np.nan)
    for minute_number in range(4, 16):
        if minute_number == 4:
            open_return = minute_open[4] / entry - 1.0
            open_hit = active & ~future_hit & (open_return >= tp_m4_m15)
            gross[open_hit] = open_return[open_hit]
            future_hit |= open_hit
            first_hit_minute[open_hit] = minute_number
        newly_hit = (
            active
            & ~future_hit
            & ((minute_high[minute_number] / entry - 1.0) >= tp_m4_m15)
        )
        future_hit |= newly_hit
        first_hit_minute[newly_hit] = minute_number
        gross[newly_hit] = tp_m4_m15[newly_hit]
    exit_reason[future_hit] = "tp_m4_m15"
    exit_minute[future_hit] = first_hit_minute[future_hit]
    active &= ~future_hit

    gross[active] = close_h3[active] / entry[active] - 1.0
    exit_reason[active] = "close_h3"
    exit_minute[active] = 15

    result = pd.DataFrame(
        {
            "split": name,
            "entry_time": entry_index,
            "entry": entry,
            "base_tp": base_tp,
            "initial_tp": initial_tp,
            "tp_after_m1": tp_m2,
            "tp_after_m2": tp_m3,
            "tp_after_m3": tp_m4_m15,
            "band_m1": band_m1,
            "band_m2": band_m2,
            "band_m3": band_m3,
            "final_tp": applied_tp,
            "score_m1": score1,
            "score_m2": score2,
            "score_m3": score3,
            "exit_reason": exit_reason,
            "exit_minute": exit_minute,
            "gross_return": gross,
            "net_return": gross - float(trade_cost),
        },
        index=frame.index,
    )
    return result.replace([np.inf, -np.inf], np.nan).dropna(subset=["gross_return"])


def _summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for split, group in trades.groupby("split", sort=False):
        counts = group["exit_reason"].value_counts()
        tp_mask = group["exit_reason"].str.startswith("tp_")
        close_h3_returns = group.loc[
            group["exit_reason"].eq("close_h3"), "gross_return"
        ]
        close_h3_loss = group.loc[
            group["exit_reason"].eq("close_h3") & group["gross_return"].lt(0.0),
            "gross_return",
        ]
        rows.append(
            {
                "split": split,
                "trades": len(group),
                "tp_m1": int(counts.get("tp_m1", 0)),
                "tp_m2": int(counts.get("tp_m2", 0)),
                "tp_m3": int(counts.get("tp_m3", 0)),
                "tp_m4_m15": int(counts.get("tp_m4_m15", 0)),
                "tp_hit": int(tp_mask.sum()),
                "tp_hit_rate": tp_mask.mean(),
                "tp_mean_hit": group.loc[tp_mask, "gross_return"].mean(),
                "tp_after_m1_mean": group["tp_after_m1"].mean(),
                "tp_after_m2_mean": group["tp_after_m2"].mean(),
                "tp_after_m3_mean": group["tp_after_m3"].mean(),
                "close_h3": int(counts.get("close_h3", 0)),
                "close_h3_mean": close_h3_returns.mean(),
                "close_h3_loss": int(len(close_h3_loss)),
                "close_h3_loss_mean": close_h3_loss.mean(),
                "gross_mean": group["gross_return"].mean(),
                "net_mean": group["net_return"].mean(),
                "net_win_rate": group["net_return"].gt(0.0).mean(),
            }
        )
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, trades: pd.DataFrame, output: Path) -> None:
    fig = plt.figure(figsize=(15, 10), facecolor="#f4f6f8")
    grid = fig.add_gridspec(
        4, 1, height_ratios=[0.50, 0.50, 0.55, 2.2], hspace=0.10
    )
    display = summary.copy()
    for column in (
        "tp_hit_rate",
        "tp_mean_hit",
        "tp_after_m1_mean",
        "tp_after_m2_mean",
        "tp_after_m3_mean",
        "close_h3_mean",
        "close_h3_loss_mean",
        "gross_mean",
        "net_mean",
        "net_win_rate",
    ):
        display[column] = display[column].map(lambda value: f"{value:.3%}")

    hit_columns = [
        "split", "trades", "tp_m1", "tp_m2", "tp_m3", "tp_m4_m15",
        "tp_hit", "tp_hit_rate", "tp_mean_hit",
    ]
    target_columns = [
        "split", "tp_after_m1_mean", "tp_after_m2_mean", "tp_after_m3_mean",
    ]
    performance_columns = [
        "split", "close_h3", "close_h3_mean", "close_h3_loss", "close_h3_loss_mean",
        "gross_mean", "net_mean", "net_win_rate",
    ]
    labels = {
        "tp_m4_m15": "tp_m4-15",
        "tp_hit_rate": "tp_rate",
        "tp_mean_hit": "mean_hit_tp",
        "tp_after_m1_mean": "mean_tp_after_m1",
        "tp_after_m2_mean": "mean_tp_after_m2",
        "tp_after_m3_mean": "mean_tp_after_m3",
        "close_h3_mean": "h3_mean",
        "close_h3_loss": "h3_loss_n",
        "close_h3_loss_mean": "h3_loss_mean",
        "net_win_rate": "net_win",
    }
    for row, (columns, title) in enumerate(
        (
            (hit_columns, "TP hits by execution minute"),
            (target_columns, "Mean dynamic TP after each meta decision"),
            (performance_columns, "Close H3 and aggregate returns"),
        )
    ):
        table_ax = fig.add_subplot(grid[row])
        table_ax.axis("off")
        table = table_ax.table(
            cellText=display[columns].values,
            colLabels=[labels.get(column, column) for column in columns],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.45)
        table_ax.set_title(title, fontsize=11, weight="bold", pad=3)
    fig.suptitle("Progressive TP meta strategy", fontsize=15, weight="bold", y=0.995)

    equity_ax = fig.add_subplot(grid[3])
    for split, color in (("val", "#136f63"), ("test", "#c44536")):
        group = trades.loc[trades["split"].eq(split)].sort_index()
        if not group.empty:
            equity_ax.plot(
                group.index,
                group["net_return"].cumsum(),
                label=split,
                color=color,
                linewidth=1.2,
            )
    equity_ax.axhline(0.0, color="#222222", linewidth=0.8)
    equity_ax.set_ylabel("Cumulative arithmetic net return")
    equity_ax.grid(alpha=0.2)
    equity_ax.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    model_dir = args.model_dir
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    meta_data = pd.read_pickle(model_dir / "common_meta_oof.pkl")
    val_meta = meta_data.val_df.dropna(subset=["label_h3", "meta_dynamic_tp_h3"])
    test_meta = meta_data.test_df.dropna(subset=["label_h3", "meta_dynamic_tp_h3"])
    all_meta = pd.concat([val_meta, test_meta]).sort_index()
    predictions = {
        lookahead: _load_predictions(model_dir, manifest, all_meta, lookahead)
        for lookahead in (1, 2, 3)
    }
    val_predictions = {
        lookahead: prediction.reindex(val_meta.index)
        for lookahead, prediction in predictions.items()
    }
    thresholds = {
        lookahead: {
            fraction: _threshold(val_predictions[lookahead], fraction)
            for fraction in (0.10, 0.20, 0.40, 0.70)
        }
        for lookahead in (1, 2, 3)
    }
    raw_5m = load_ohlcv(args.data)
    minute = load_ohlcv(args.data_1m)
    val = _simulate_split(
        name="val",
        meta=val_meta,
        raw_5m=raw_5m,
        minute=minute,
        predictions=predictions,
        thresholds=thresholds,
        initial_tp_offset=args.initial_tp_offset,
        tp_top10=args.tp_top10,
        tp_top20=args.tp_top20,
        tp_top40=args.tp_top40,
        tp_top70=args.tp_top70,
        tp_outside=args.tp_outside,
        trade_cost=args.trade_cost,
    )
    test = _simulate_split(
        name="test",
        meta=test_meta,
        raw_5m=raw_5m,
        minute=minute,
        predictions=predictions,
        thresholds=thresholds,
        initial_tp_offset=args.initial_tp_offset,
        tp_top10=args.tp_top10,
        tp_top20=args.tp_top20,
        tp_top40=args.tp_top40,
        tp_top70=args.tp_top70,
        tp_outside=args.tp_outside,
        trade_cost=args.trade_cost,
    )
    trades = pd.concat([val, test]).sort_index()
    summary = _summary(trades)
    stem = "meta_dynamic_band_tp_m1_m2_m3"
    output = args.out_dir / f"{stem}.png"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(args.out_dir / f"{stem}_trades.csv", index_label="signal_time")
    summary.to_csv(args.out_dir / f"{stem}_summary.csv", index=False)
    _plot(summary, trades, output)
    print(
        "Thresholds:",
        {
            lookahead: {
                fraction: f"{value:.8f}"
                for fraction, value in stage_thresholds.items()
            }
            for lookahead, stage_thresholds in thresholds.items()
        },
    )
    print(summary.to_string(index=False))
    print(f"Saved: {output}")
    return summary, trades, output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_5M)
    parser.add_argument("--data-1m", type=Path, default=DEFAULT_DATA_1M)
    parser.add_argument("--initial-tp-offset", type=float, default=0.0020)
    parser.add_argument("--tp-top10", type=float, default=0.0035)
    parser.add_argument("--tp-top20", type=float, default=0.0025)
    parser.add_argument("--tp-top40", type=float, default=0.0020)
    parser.add_argument("--tp-top70", type=float, default=0.0010)
    parser.add_argument("--tp-outside", type=float, default=0.0)
    parser.add_argument("--trade-cost", type=float, default=0.0002)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    run(parse_args())
