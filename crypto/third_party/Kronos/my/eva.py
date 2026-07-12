"""Evaluate Kronos forecasts on Evo_Finance BTCUSDT 15m val+test data.

Run from the Kronos virtualenv:

    cd crypto/third_party/Kronos
    python my/eva.py

The script runs a non-overlapping sliding forecast. Each window predicts
PRED_LEN bars, then moves forward by PRED_LEN bars. This means every future
candle receives exactly one prediction path.

For each signal point, it forecasts the next PRED_LEN bars, classifies the
forecasted path as up/down, then compares that with the realized next-PRED_LEN
path direction. No CSV is written; the script prints metrics and saves a chart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config as cfg


KRONOS_ROOT = Path(__file__).resolve().parents[1]
if str(KRONOS_ROOT) not in sys.path:
    sys.path.insert(0, str(KRONOS_ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402


REQUIRED_COLS = ["open", "high", "low", "close", "volume", "amount"]


def main() -> None:
    args = parse_args()
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_btc_data(args.data)
    windows = make_eval_windows(
        data=data,
        lookback=args.lookback,
        pred_len=args.pred_len,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
    )
    if not windows:
        raise ValueError("No evaluation windows were created. Check dates/lookback/pred_len.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data rows: {len(data)}")
    print(f"Windows: {len(windows)} | step={args.pred_len} bars | pred_len={args.pred_len}")

    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(
        model,
        tokenizer,
        device=device,
        max_context=args.max_context,
    )

    eval_rows: list[dict[str, object]] = []

    for window_id, end_pos in enumerate(tqdm(windows, desc="Kronos eval"), start=1):
        row = predict_one_window(
            predictor=predictor,
            data=data,
            end_pos=end_pos,
            lookback=args.lookback,
            pred_len=args.pred_len,
            temperature=args.temperature,
            top_p=args.top_p,
            sample_count=args.sample_count,
        )
        eval_rows.append(row)

    eval_df = pd.DataFrame(eval_rows).sort_values("signal_time")

    stem = (
        f"kronos_{args.model.split('/')[-1]}_btc15m"
        f"_lb{args.lookback}_h{args.pred_len}_step1_path_direction"
    )
    chart_path = cfg.OUTPUT_DIR / f"{stem}.png"

    plot_results(
        eval_df=eval_df,
        chart_path=chart_path,
    )

    print_metrics(eval_df=eval_df)
    print(f"Saved chart: {chart_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=cfg.DATA_PATH)
    parser.add_argument("--model", default=cfg.MODEL_ID)
    parser.add_argument("--tokenizer", default=cfg.TOKENIZER_ID)
    parser.add_argument("--lookback", type=int, default=cfg.LOOKBACK)
    parser.add_argument("--max-context", type=int, default=cfg.MAX_CONTEXT)
    parser.add_argument("--pred-len", type=int, default=cfg.PRED_LEN)
    parser.add_argument("--eval-start", default=cfg.EVAL_START)
    parser.add_argument("--eval-end", default=cfg.EVAL_END)
    parser.add_argument("--temperature", type=float, default=cfg.TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=cfg.TOP_P)
    parser.add_argument("--sample-count", type=int, default=cfg.SAMPLE_COUNT)
    parser.add_argument("--take-profit-pct", type=float, default=cfg.TAKE_PROFIT_PCT)
    parser.add_argument("--device", default=None, help="cuda, cpu, or blank for auto.")
    return parser.parse_args()


def load_btc_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    if "date" not in data.columns:
        raise ValueError(f"Missing date column in {path}")
    data["timestamps"] = pd.to_datetime(data["date"])
    data = data.sort_values("timestamps").drop_duplicates("timestamps")

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if "amount" not in data.columns:
        data["amount"] = data["close"] * data["volume"]
    else:
        data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
        data["amount"] = data["amount"].fillna(data["close"] * data["volume"])

    data = data.dropna(subset=["timestamps", *REQUIRED_COLS]).reset_index(drop=True)
    return data


def make_eval_windows(
    data: pd.DataFrame,
    lookback: int,
    pred_len: int,
    eval_start: str,
    eval_end: str | None,
) -> list[int]:
    timestamps = pd.to_datetime(data["timestamps"])
    eval_start_ts = pd.Timestamp(eval_start)
    eval_end_ts = pd.Timestamp(eval_end) if eval_end else None

    first_future_pos = int(timestamps.searchsorted(eval_start_ts, side="left"))
    first_end_pos = max(first_future_pos, int(lookback))
    last_end_pos = len(data) - int(pred_len)
    if eval_end_ts is not None:
        last_end_pos = min(
            last_end_pos,
            int(timestamps.searchsorted(eval_end_ts, side="left")) - int(pred_len),
        )
    if last_end_pos <= first_end_pos:
        return []

    return list(range(first_end_pos, last_end_pos + 1, 1))


def predict_one_window(
    predictor: KronosPredictor,
    data: pd.DataFrame,
    end_pos: int,
    lookback: int,
    pred_len: int,
    temperature: float,
    top_p: float,
    sample_count: int,
) -> dict[str, object]:
    start_pos = int(end_pos) - int(lookback)
    future_end = int(end_pos) + int(pred_len)
    context = data.iloc[start_pos:end_pos].copy()
    actual = data.iloc[end_pos:future_end].copy()

    x_df = context[REQUIRED_COLS].reset_index(drop=True)
    x_timestamp = context["timestamps"].reset_index(drop=True)
    y_timestamp = actual["timestamps"].reset_index(drop=True)

    pred = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=temperature,
        top_p=top_p,
        sample_count=sample_count,
    ).reset_index(drop=False)

    entry_open = float(actual["open"].iloc[0])
    actual_mid_path = ((actual["high"].astype(float) + actual["low"].astype(float)) / 2.0)
    pred_mid_path = ((pred["high"].astype(float) + pred["low"].astype(float)) / 2.0)
    actual_mid_ret = float(actual_mid_path.mean() / entry_open - 1.0)
    pred_mid_ret = float(pred_mid_path.mean() / entry_open - 1.0)
    actual_close_ret = float(actual["close"].iloc[-1] / entry_open - 1.0)
    pred_close_ret = float(pred["close"].iloc[-1] / entry_open - 1.0)
    actual_mfe = float(actual["high"].max() / entry_open - 1.0)
    pred_mfe = float(pred["high"].max() / entry_open - 1.0)
    actual_mae = float(actual["low"].min() / entry_open - 1.0)
    pred_mae = float(pred["low"].min() / entry_open - 1.0)

    return {
        "signal_time": pd.Timestamp(context["timestamps"].iloc[-1]),
        "entry_time": pd.Timestamp(actual["timestamps"].iloc[0]),
        "exit_time": pd.Timestamp(actual["timestamps"].iloc[-1]),
        "entry_open": entry_open,
        "actual_mid_ret": actual_mid_ret,
        "pred_mid_ret": pred_mid_ret,
        "actual_close_ret": actual_close_ret,
        "pred_close_ret": pred_close_ret,
        "actual_mfe": actual_mfe,
        "pred_mfe": pred_mfe,
        "actual_mae": actual_mae,
        "pred_mae": pred_mae,
        "actual_mid_up": bool(actual_mid_ret > 0.0),
        "pred_mid_up": bool(pred_mid_ret > 0.0),
        "actual_close_up": bool(actual_close_ret > 0.0),
        "pred_close_up": bool(pred_close_ret > 0.0),
    }


def plot_results(eval_df: pd.DataFrame, chart_path: Path) -> None:
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    data = eval_df.copy()
    data["signal_time"] = pd.to_datetime(data["signal_time"])

    fig, axes = plt.subplots(3, 1, figsize=(17, 13), sharex=True, constrained_layout=True)
    fig.suptitle("Kronos BTCUSDT 15m H-Path Direction Evaluation", fontsize=15, fontweight="bold")

    ax = axes[0]
    ax.plot(data["signal_time"], data["actual_mid_ret"] * 100.0, label="Actual path mid ret", linewidth=1.1)
    ax.plot(data["signal_time"], data["pred_mid_ret"] * 100.0, label="Pred path mid ret", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Path mid ret (%)")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(data["signal_time"], data["actual_close_ret"] * 100.0, label="Actual close H ret", linewidth=1.1)
    ax.plot(data["signal_time"], data["pred_close_ret"] * 100.0, label="Pred close H ret", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Close ret (%)")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.scatter(data["actual_mid_ret"] * 100.0, data["pred_mid_ret"] * 100.0, s=10, alpha=0.45)
    lim_min = float(np.nanmin([data["actual_mid_ret"].min(), data["pred_mid_ret"].min()]) * 100.0)
    lim_max = float(np.nanmax([data["actual_mid_ret"].max(), data["pred_mid_ret"].max()]) * 100.0)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="tab:gray", linewidth=0.8)
    ax.axvline(0, color="tab:gray", linewidth=0.8)
    ax.set_xlabel("Actual path mid ret (%)")
    ax.set_ylabel("Pred path mid ret (%)")
    ax.set_title("Predicted vs Actual Path Direction Strength")
    ax.grid(alpha=0.25)

    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


def print_metrics(eval_df: pd.DataFrame) -> None:
    if eval_df.empty:
        return
    data = eval_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["actual_mid_ret", "pred_mid_ret", "actual_close_ret", "pred_close_ret"]
    )
    print("\n=== Kronos Eval Metrics ===")
    print(f"windows: {len(data)}")
    print_direction_metrics(data, actual_col="actual_mid_up", pred_col="pred_mid_up", label="path_mid")
    print_direction_metrics(data, actual_col="actual_close_up", pred_col="pred_close_up", label="close_h")
    print(f"path_mid_ret corr={data[['actual_mid_ret', 'pred_mid_ret']].corr().iloc[0, 1]:.4f}")
    print(f"close_h_ret corr={data[['actual_close_ret', 'pred_close_ret']].corr().iloc[0, 1]:.4f}")
    print(f"mean actual_mid_ret={data['actual_mid_ret'].mean():.4%}")
    print(f"mean pred_mid_ret={data['pred_mid_ret'].mean():.4%}")
    print(f"mean actual_close_ret={data['actual_close_ret'].mean():.4%}")
    print(f"mean pred_close_ret={data['pred_close_ret'].mean():.4%}")


def print_direction_metrics(data: pd.DataFrame, actual_col: str, pred_col: str, label: str) -> None:
    actual = data[actual_col].astype(bool)
    pred = data[pred_col].astype(bool)
    acc = (actual == pred).mean()
    pred_up = pred
    pred_down = ~pred
    precision_up = actual[pred_up].mean() if pred_up.any() else np.nan
    precision_down = (~actual[pred_down]).mean() if pred_down.any() else np.nan
    print(f"{label} direction_acc={acc:.2%}")
    print(f"{label} actual_up_rate={actual.mean():.2%}")
    print(f"{label} pred_up_rate={pred.mean():.2%}")
    print(f"{label} precision_when_pred_up={precision_up:.2%}" if np.isfinite(precision_up) else f"{label} precision_when_pred_up=NA")
    print(f"{label} precision_when_pred_down={precision_down:.2%}" if np.isfinite(precision_down) else f"{label} precision_when_pred_down=NA")


if __name__ == "__main__":
    main()
