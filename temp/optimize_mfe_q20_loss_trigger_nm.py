"""Optimize paper-loss-triggered skip/trade signal counts on Final Val."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from crypto.data import load_ohlcv
from temp.analyze_mfe_q20_prediction_bands import _trade_frame


DEFAULT_CACHE = Path("temp/model/meta_mfe_q20_after_1m_2m_3m/common_meta_oof.pkl")
DEFAULT_DATA = Path("data/crypto/BTCUSDT_5m.csv")
DEFAULT_OUTPUT = Path("temp/output/mfe_q20_loss_trigger_nm_grid.csv")


def _trigger_positions(
    trades: pd.DataFrame,
    *,
    horizon: int,
    loss_threshold: float,
) -> np.ndarray:
    """Map each paper loss to the first signal where its H-close is known."""
    index = pd.DatetimeIndex(trades.index)
    loss_positions = np.flatnonzero(
        trades["gross_return"].to_numpy(float) < -float(loss_threshold)
    )
    trigger = np.full(len(trades), -1, dtype=np.int64)
    if len(loss_positions) == 0:
        return trigger
    known_times = index[loss_positions] + pd.Timedelta(minutes=5 * int(horizon))
    trigger[loss_positions] = index.searchsorted(known_times, side="left")
    return trigger


def _simulate(
    trades: pd.DataFrame,
    trigger_positions: np.ndarray,
    loss_positions: np.ndarray,
    *,
    skip_signals: int,
    trade_signals: int,
    trade_cost: float,
) -> dict[str, float | int]:
    returns = trades["gross_return"].to_numpy(float) - float(trade_cost)
    selected_count = 0
    selected_sum = 0.0
    selected_sum_sq = 0.0
    selected_wins = 0
    observer_start = 0
    n_rows = len(trades)
    while observer_start < n_rows:
        loss_cursor = int(np.searchsorted(loss_positions, observer_start, side="left"))
        if loss_cursor >= len(loss_positions):
            break
        paper_loss_position = int(loss_positions[loss_cursor])
        trigger_position = int(trigger_positions[paper_loss_position])
        if trigger_position >= n_rows:
            break
        trade_start = trigger_position + int(skip_signals)
        trade_end = min(trade_start + int(trade_signals), n_rows)
        if trade_start < n_rows:
            chunk = returns[trade_start:trade_end]
            selected_count += len(chunk)
            selected_sum += float(chunk.sum())
            selected_sum_sq += float(np.square(chunk).sum())
            selected_wins += int(np.count_nonzero(chunk > 0.0))
        observer_start = trade_end

    mean = selected_sum / selected_count if selected_count else np.nan
    if selected_count > 1:
        variance = max(
            (selected_sum_sq - selected_count * mean * mean) / (selected_count - 1),
            0.0,
        )
        std = float(np.sqrt(variance))
    else:
        std = np.nan
    return {
        "trades": selected_count,
        "net_mean": mean,
        "net_std": std,
        "net_win_rate": selected_wins / selected_count if selected_count else np.nan,
        "net_sum": selected_sum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--loss-threshold", type=float, default=0.001)
    parser.add_argument("--trade-cost", type=float, default=0.0002)
    parser.add_argument("--max-n", type=int, default=100)
    parser.add_argument("--max-m", type=int, default=100)
    parser.add_argument("--min-val-trades", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cached = pd.read_pickle(args.cache)
    raw = load_ohlcv(args.data)
    required = ["meta_dynamic_tp_h3"]
    val = _trade_frame(cached.val_df.dropna(subset=required), raw, args.horizon)
    test = _trade_frame(cached.test_df.dropna(subset=required), raw, args.horizon)
    val_trigger = _trigger_positions(
        val, horizon=args.horizon, loss_threshold=args.loss_threshold
    )
    test_trigger = _trigger_positions(
        test, horizon=args.horizon, loss_threshold=args.loss_threshold
    )
    val_loss_positions = np.flatnonzero(val_trigger >= 0)
    test_loss_positions = np.flatnonzero(test_trigger >= 0)

    rows = []
    for n_value in range(args.max_n + 1):
        for m_value in range(1, args.max_m + 1):
            val_stats = _simulate(
                val,
                val_trigger,
                val_loss_positions,
                skip_signals=n_value,
                trade_signals=m_value,
                trade_cost=args.trade_cost,
            )
            rows.append({"N": n_value, "M": m_value, **val_stats})
    grid = pd.DataFrame(rows)
    eligible = grid.loc[grid["trades"].ge(args.min_val_trades)].copy()
    if eligible.empty:
        raise ValueError("No N/M pair satisfies --min-val-trades.")
    eligible = eligible.sort_values(
        ["net_mean", "net_sum", "trades"], ascending=[False, False, False]
    )
    best = eligible.iloc[0]
    test_stats = _simulate(
        test,
        test_trigger,
        test_loss_positions,
        skip_signals=int(best["N"]),
        trade_signals=int(best["M"]),
        trade_cost=args.trade_cost,
    )
    grid["eligible"] = grid["trades"].ge(args.min_val_trades)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.out, index=False)

    top = eligible.head(20).copy()
    for column in ("net_mean", "net_std", "net_win_rate", "net_sum"):
        top[column] = top[column].map(lambda value: f"{100.0 * value:+.4f}%")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print("=== Top 20 Val N/M pairs by mean net return ===")
        print(top.to_string(index=False))
    print("\n=== Frozen best pair on Test ===")
    print(f"N={int(best['N'])} M={int(best['M'])}")
    for key, value in test_stats.items():
        if key == "trades":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {100.0 * float(value):+.4f}%")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
