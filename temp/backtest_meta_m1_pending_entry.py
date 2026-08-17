"""Backtest an M1-only pending-entry Long strategy from cached meta features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto.data import load_ohlcv
from temp.backtest_meta_progressive_tp import _load_predictions


DEFAULT_MODEL_DIR = Path("temp/model/meta_mfe_q20_after_1m_2m_3m")
DEFAULT_DATA_5M = Path("data/crypto/BTCUSDT_5m.csv")
DEFAULT_DATA_1M = Path("data/crypto/BTCUSDT_1m.csv")
DEFAULT_OUT_DIR = Path("temp/output")


def _minute_values(
    minute: pd.DataFrame,
    entry_index: pd.DatetimeIndex,
    column: str,
    minute_number: int,
) -> np.ndarray:
    index = entry_index + pd.Timedelta(minutes=int(minute_number) - 1)
    return (
        pd.to_numeric(minute[column], errors="coerce")
        .reindex(index)
        .to_numpy(float)
    )


def _simulate(
    *,
    split: str,
    frame: pd.DataFrame,
    prediction: pd.Series,
    top_fraction: float,
    raw_5m: pd.DataFrame,
    minute: pd.DataFrame,
    side: str,
    entry_offset: float,
    take_profit: float,
    pending_end_minute: int,
    exclude_fill_minute_tp: bool,
    trade_cost: float,
) -> tuple[pd.DataFrame, dict]:
    score = prediction.reindex(frame.index)
    clean_score = score.replace([np.inf, -np.inf], np.nan).dropna()
    selected_count = min(
        len(clean_score),
        max(1, int(np.ceil(len(clean_score) * float(top_fraction)))),
    )
    selected_index = clean_score.sort_values(
        ascending=False, kind="stable"
    ).index[:selected_count]
    selected = pd.Series(frame.index.isin(selected_index), index=frame.index)
    signal_index = pd.DatetimeIndex(frame.index[selected])
    selected_score = score.loc[selected].to_numpy(float)
    entry_h1_index = signal_index + pd.Timedelta(minutes=5)
    open_h1 = (
        pd.to_numeric(raw_5m["open"], errors="coerce")
        .reindex(entry_h1_index)
        .to_numpy(float)
    )
    close_h3 = (
        pd.to_numeric(raw_5m["close"], errors="coerce")
        .reindex(signal_index + pd.Timedelta(minutes=15))
        .to_numpy(float)
    )
    direction = 1.0 if side == "long" else -1.0
    pending_entry = open_h1 * (1.0 + direction * float(entry_offset))
    # This experiment defines entry as a low touch: Long behaves like a buy
    # limit, while Short behaves like a sell stop.
    fill_column = "low"
    pending_extreme = {
        minute_number: _minute_values(
            minute, entry_h1_index, fill_column, minute_number
        )
        for minute_number in range(2, int(pending_end_minute) + 1)
    }
    tp_column = "high" if side == "long" else "low"
    future_tp_extreme = {
        minute_number: _minute_values(
            minute, entry_h1_index, tp_column, minute_number
        )
        for minute_number in range(2, 16)
    }

    valid = np.isfinite(open_h1) & (open_h1 > 0.0) & np.isfinite(close_h3)
    filled = np.zeros(len(signal_index), dtype=bool)
    fill_minute = np.full(len(signal_index), np.nan)
    for minute_number in range(2, int(pending_end_minute) + 1):
        new_fill = (
            valid
            & ~filled
            & np.isfinite(pending_extreme[minute_number])
            & (pending_extreme[minute_number] <= pending_entry)
        )
        filled |= new_fill
        fill_minute[new_fill] = minute_number

    target_price = pending_entry * (1.0 + direction * float(take_profit))
    tp_hit = np.zeros(len(signal_index), dtype=bool)
    tp_minute = np.full(len(signal_index), np.nan)
    for minute_number in range(2, 16):
        first_tp_minute = fill_minute + (1 if exclude_fill_minute_tp else 0)
        eligible = filled & (first_tp_minute <= minute_number) & ~tp_hit
        new_hit = (
            eligible
            & np.isfinite(future_tp_extreme[minute_number])
            & (
                (future_tp_extreme[minute_number] >= target_price)
                if side == "long"
                else (future_tp_extreme[minute_number] <= target_price)
            )
        )
        tp_hit |= new_hit
        tp_minute[new_hit] = minute_number

    gross = np.full(len(signal_index), np.nan)
    gross[tp_hit] = float(take_profit)
    close_exit = filled & ~tp_hit
    gross[close_exit] = direction * (
        close_h3[close_exit] / pending_entry[close_exit] - 1.0
    )
    exit_reason = np.where(tp_hit, "tp", np.where(close_exit, "close_h3", "not_filled"))

    result = pd.DataFrame(
        {
            "split": split,
            "side": side,
            "entry_h1_time": entry_h1_index,
            "m1_score": selected_score,
            "open_h1": open_h1,
            "pending_entry": pending_entry,
            "filled": filled,
            "fill_minute": fill_minute,
            "tp_price": target_price,
            "tp_hit": tp_hit,
            "tp_minute": tp_minute,
            "exit_reason": exit_reason,
            "gross_return": gross,
            "net_return": gross - float(trade_cost),
        },
        index=signal_index,
    )
    trades = result.loc[result["filled"]].copy()
    close_returns = trades.loc[trades["exit_reason"].eq("close_h3"), "gross_return"]
    close_losses = close_returns.loc[close_returns.lt(0.0)]
    stats = {
        "split": split,
        "universe": int(len(frame)),
        "m1_top50": int(selected.sum()),
        "m1_cutoff": float(clean_score.reindex(selected_index).min()),
        "filled": int(filled.sum()),
        "fill_rate": float(filled.mean()) if len(filled) else np.nan,
        "fill_m2": int(np.sum(fill_minute == 2)),
        "fill_m3": int(np.sum(fill_minute == 3)),
        "fill_m4": int(np.sum(fill_minute == 4)),
        "fill_m5": int(np.sum(fill_minute == 5)),
        "tp_hit": int(tp_hit.sum()),
        "tp_hit_rate": float(tp_hit[filled].mean()) if filled.any() else np.nan,
        "close_h3": int(close_exit.sum()),
        "close_h3_mean": float(close_returns.mean()),
        "close_h3_loss": int(len(close_losses)),
        "close_h3_loss_mean": float(close_losses.mean()),
        "gross_mean": float(trades["gross_return"].mean()),
        "net_mean": float(trades["net_return"].mean()),
        "net_win_rate": float(trades["net_return"].gt(0.0).mean()),
    }
    return trades, stats


def _plot(summary: pd.DataFrame, trades: pd.DataFrame, output: Path) -> None:
    fig = plt.figure(figsize=(15, 9), facecolor="#f4f6f8")
    grid = fig.add_gridspec(3, 1, height_ratios=[0.65, 0.65, 2.1], hspace=0.14)
    display = summary.copy()
    for column in (
        "fill_rate",
        "tp_hit_rate",
        "close_h3_mean",
        "close_h3_loss_mean",
        "gross_mean",
        "net_mean",
        "net_win_rate",
    ):
        display[column] = display[column].map(lambda value: f"{value:.3%}")
    tables = (
        (
            [
                "split", "universe", "m1_top50", "filled", "fill_rate",
                "fill_m2",
            ],
            "M1 selection and pending-entry fills",
        ),
        (
            [
                "split", "tp_hit", "tp_hit_rate", "close_h3",
                "close_h3_mean", "close_h3_loss", "close_h3_loss_mean",
                "gross_mean", "net_mean", "net_win_rate",
            ],
            "Trade outcomes",
        ),
    )
    for row, (columns, title) in enumerate(tables):
        axis = fig.add_subplot(grid[row])
        axis.axis("off")
        table = axis.table(
            cellText=display[columns].values,
            colLabels=columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.45)
        axis.set_title(title, fontsize=11, weight="bold", pad=3)

    equity_axis = fig.add_subplot(grid[2])
    for split, color in (("val", "#136f63"), ("test", "#c44536")):
        selected = trades.loc[trades["split"].eq(split)].sort_index()
        equity_axis.plot(
            selected.index,
            selected["net_return"].cumsum(),
            label=split,
            color=color,
            linewidth=1.2,
        )
    equity_axis.axhline(0.0, color="#222222", linewidth=0.8)
    equity_axis.set_ylabel("Cumulative arithmetic net return")
    equity_axis.grid(alpha=0.2)
    equity_axis.legend()
    side = str(trades["side"].iloc[0]).upper() if not trades.empty else ""
    fig.suptitle(
        f"MFE Q20 + M1 Top50 {side} pending-entry strategy",
        fontsize=15,
        weight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    manifest = json.loads(
        (args.model_dir / "manifest.json").read_text(encoding="utf-8")
    )
    meta_data = pd.read_pickle(args.model_dir / "common_meta_oof.pkl")
    required = ["label_h3", "meta_dynamic_tp_h3"]
    val_frame = meta_data.val_df.dropna(subset=required).copy()
    test_frame = meta_data.test_df.dropna(subset=required).copy()
    all_frame = pd.concat([val_frame, test_frame]).sort_index()
    prediction = _load_predictions(args.model_dir, manifest, all_frame, 1)
    raw_5m = load_ohlcv(args.data)
    minute = load_ohlcv(args.data_1m)
    val_trades, val_stats = _simulate(
        split="val",
        frame=val_frame,
        prediction=prediction,
        top_fraction=args.top_fraction,
        raw_5m=raw_5m,
        minute=minute,
        side=args.side,
        entry_offset=args.entry_offset,
        take_profit=args.take_profit,
        pending_end_minute=args.pending_end_minute,
        exclude_fill_minute_tp=args.exclude_fill_minute_tp,
        trade_cost=args.trade_cost,
    )
    test_trades, test_stats = _simulate(
        split="test",
        frame=test_frame,
        prediction=prediction,
        top_fraction=args.top_fraction,
        raw_5m=raw_5m,
        minute=minute,
        side=args.side,
        entry_offset=args.entry_offset,
        take_profit=args.take_profit,
        pending_end_minute=args.pending_end_minute,
        exclude_fill_minute_tp=args.exclude_fill_minute_tp,
        trade_cost=args.trade_cost,
    )
    trades = pd.concat([val_trades, test_trades]).sort_index()
    summary = pd.DataFrame([val_stats, test_stats])
    stem = (
        f"mfe_q20_meta_m1_top50_{args.side}_pending_m2_lowfill_"
        f"entry{args.entry_offset * 100:.3f}pct_tp{args.take_profit * 100:.3f}pct_h3"
    ).replace(".", "p")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / f"{stem}.png"
    trades.to_csv(args.out_dir / f"{stem}_trades.csv", index_label="signal_time")
    summary.to_csv(args.out_dir / f"{stem}_summary.csv", index=False)
    _plot(summary, trades, output)
    print(f"M1 selection: exact Top {args.top_fraction:.0%} per split")
    print(summary.to_string(index=False))
    print(f"Saved: {output}")
    return summary, trades, output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_5M)
    parser.add_argument("--data-1m", type=Path, default=DEFAULT_DATA_1M)
    parser.add_argument("--top-fraction", type=float, default=0.50)
    parser.add_argument("--side", choices=("long", "short"), default="long")
    parser.add_argument("--entry-offset", type=float, default=0.0005)
    parser.add_argument("--take-profit", type=float, default=0.0015)
    parser.add_argument("--pending-end-minute", type=int, default=2)
    parser.add_argument("--exclude-fill-minute-tp", action="store_true")
    parser.add_argument("--trade-cost", type=float, default=0.0002)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if not 2 <= int(args.pending_end_minute) <= 5:
        parser.error("--pending-end-minute must be between 2 and 5.")
    return args


if __name__ == "__main__":
    run(parse_args())
