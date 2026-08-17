"""Backtest baseline MFE Q20 with a minimum entry-to-entry candle gap."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from crypto.data import load_ohlcv
from temp.analyze_mfe_q20_loss_clusters import _extract_runs, _summary
from temp.analyze_mfe_q20_prediction_bands import _trade_frame


DEFAULT_CACHE = Path("temp/model/meta_mfe_q20_after_1m_2m_3m/common_meta_oof.pkl")
DEFAULT_DATA = Path("data/crypto/BTCUSDT_5m.csv")
DEFAULT_OUTPUT = Path("temp/output/mfe_q20_min_gap_6_bars_summary.csv")


def _apply_min_gap(
    trades: pd.DataFrame,
    *,
    gap_bars: int,
    candle_minutes: int,
) -> pd.DataFrame:
    ordered = trades.sort_index()
    minimum_delta = pd.Timedelta(minutes=int(gap_bars) * int(candle_minutes))
    selected_positions = []
    last_entry: pd.Timestamp | None = None
    for position, timestamp in enumerate(pd.DatetimeIndex(ordered.index)):
        entry_time = pd.Timestamp(timestamp) + pd.Timedelta(minutes=candle_minutes)
        if last_entry is None or entry_time - last_entry >= minimum_delta:
            selected_positions.append(position)
            last_entry = entry_time
    return ordered.iloc[selected_positions].copy()


def _strategy_summary(
    split: str,
    original: pd.DataFrame,
    selected: pd.DataFrame,
    trade_cost: float,
) -> dict:
    net = selected["gross_return"] - float(trade_cost)
    return {
        "split": split,
        "baseline_trades": len(original),
        "selected_trades": len(selected),
        "retained_rate": len(selected) / len(original),
        "tp_hit_rate": selected["tp_hit"].mean(),
        "gross_mean": selected["gross_return"].mean(),
        "net_mean": net.mean(),
        "net_win_rate": net.gt(0.0).mean(),
        "net_sum": net.sum(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--gap-bars", type=int, default=6)
    parser.add_argument("--candle-minutes", type=int, default=5)
    parser.add_argument("--trade-cost", type=float, default=0.0002)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cached = pd.read_pickle(args.cache)
    raw = load_ohlcv(args.data)
    required = ["meta_dynamic_tp_h3"]
    frames = {
        "val": _trade_frame(cached.val_df.dropna(subset=required), raw, args.horizon),
        "test": _trade_frame(cached.test_df.dropna(subset=required), raw, args.horizon),
    }
    strategy_rows = []
    cluster_rows = []
    selected_parts = []
    for split, original in frames.items():
        selected = _apply_min_gap(
            original,
            gap_bars=args.gap_bars,
            candle_minutes=args.candle_minutes,
        )
        selected_parts.append(selected.assign(split=split))
        strategy_rows.append(
            _strategy_summary(split, original, selected, args.trade_cost)
        )
        runs = _extract_runs(split, selected, args.trade_cost)
        cluster_rows.append(_summary(split, selected, runs, args.trade_cost))

    strategy = pd.DataFrame(strategy_rows)
    clusters = pd.DataFrame(cluster_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    strategy.to_csv(args.out, index=False)
    cluster_out = args.out.with_name(args.out.stem + "_clusters.csv")
    clusters.to_csv(cluster_out, index=False)
    trades_out = args.out.with_name(args.out.stem + "_trades.csv")
    pd.concat(selected_parts).sort_index().to_csv(trades_out)

    strategy_display = strategy.copy()
    for column in (
        "retained_rate", "tp_hit_rate", "gross_mean", "net_mean",
        "net_win_rate", "net_sum",
    ):
        strategy_display[column] = strategy_display[column].map(
            lambda value: f"{100.0 * value:+.4f}%"
        )
    cluster_display = clusters.copy()
    for column in (
        "loss_rate", "P(next_loss_given_loss)", "share_losses_in_run_ge_2",
        "share_losses_in_run_ge_3", "share_losses_in_run_ge_5",
    ):
        cluster_display[column] = cluster_display[column].map(
            lambda value: f"{100.0 * value:.3f}%"
        )
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print("=== Minimum entry-gap strategy ===")
        print(strategy_display.to_string(index=False))
        print("\n=== Loss runs after minimum entry gap ===")
        print(cluster_display.to_string(index=False))
    print(f"\nSaved: {args.out}")
    print(f"Saved: {cluster_out}")
    print(f"Saved: {trades_out}")


if __name__ == "__main__":
    main()
