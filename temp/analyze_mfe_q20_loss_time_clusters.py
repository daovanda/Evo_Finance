"""Analyze MFE Q20 loss runs with a maximum candle gap between losses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from crypto.data import load_ohlcv
from temp.analyze_mfe_q20_prediction_bands import _trade_frame


DEFAULT_CACHE = Path("temp/model/meta_mfe_q20_after_1m_2m_3m/common_meta_oof.pkl")
DEFAULT_DATA = Path("data/crypto/BTCUSDT_5m.csv")
DEFAULT_OUTPUT = Path("temp/output/mfe_q20_loss_time_cluster_lengths.csv")


def _runs(
    split: str,
    trades: pd.DataFrame,
    *,
    trade_cost: float,
    candle_minutes: int,
    allowed_empty_candles: int,
) -> pd.DataFrame:
    data = trades.sort_index().copy()
    data["net_return"] = data["gross_return"] - float(trade_cost)
    max_delta = pd.Timedelta(
        minutes=int(candle_minutes) * (int(allowed_empty_candles) + 1)
    )
    rows = []
    cluster: list[int] = []
    previous_position: int | None = None
    previous_was_loss = False

    def finish() -> None:
        nonlocal cluster
        if not cluster:
            return
        selected = data.iloc[cluster]
        rows.append(
            {
                "split": split,
                "cluster_id": len(rows) + 1,
                "start_signal": selected.index[0],
                "end_signal": selected.index[-1],
                "run_length": len(selected),
                "net_mean": selected["net_return"].mean(),
                "net_sum": selected["net_return"].sum(),
                "worst_net": selected["net_return"].min(),
            }
        )
        cluster = []

    for position, (timestamp, row) in enumerate(data.iterrows()):
        is_loss = float(row["net_return"]) < 0.0
        if not is_loss:
            finish()
            previous_position = position
            previous_was_loss = False
            continue
        can_continue = (
            previous_was_loss
            and previous_position is not None
            and pd.Timestamp(timestamp) - pd.Timestamp(data.index[previous_position])
            <= max_delta
        )
        if not can_continue:
            finish()
        cluster.append(position)
        previous_position = position
        previous_was_loss = True
    finish()
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--trade-cost", type=float, default=0.0002)
    parser.add_argument("--candle-minutes", type=int, default=5)
    parser.add_argument("--allowed-empty-candles", type=int, default=1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cached = pd.read_pickle(args.cache)
    raw = load_ohlcv(args.data)
    required = ["meta_dynamic_tp_h3"]
    split_trades = {
        "val": _trade_frame(cached.val_df.dropna(subset=required), raw, args.horizon),
        "test": _trade_frame(cached.test_df.dropna(subset=required), raw, args.horizon),
    }
    run_frames = [
        _runs(
            split,
            trades,
            trade_cost=args.trade_cost,
            candle_minutes=args.candle_minutes,
            allowed_empty_candles=args.allowed_empty_candles,
        )
        for split, trades in split_trades.items()
    ]
    runs = pd.concat(run_frames, ignore_index=True)
    rows = []
    for split, selected in runs.groupby("split", sort=False):
        loss_count = int(selected["run_length"].sum())
        for length in range(1, int(selected["run_length"].max()) + 1):
            count = int(selected["run_length"].eq(length).sum())
            rows.append(
                {
                    "split": split,
                    "run_length": length,
                    "clusters": count,
                    "contained_losses": count * length,
                    "share_of_losses": count * length / loss_count,
                }
            )
    output = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    runs_out = args.out.with_name(args.out.stem + "_clusters.csv")
    runs.to_csv(runs_out, index=False)

    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(output.to_string(index=False))
    for split, selected in runs.groupby("split", sort=False):
        lengths = selected["run_length"]
        losses = int(lengths.sum())
        print(
            f"{split}: clusters={len(selected):,} mean={lengths.mean():.4f} "
            f"max={int(lengths.max())} "
            f"losses_in_ge2={lengths[lengths.ge(2)].sum() / losses:.3%} "
            f"losses_in_ge3={lengths[lengths.ge(3)].sum() / losses:.3%}"
        )
    print(f"Saved: {args.out}")
    print(f"Saved: {runs_out}")


if __name__ == "__main__":
    main()
