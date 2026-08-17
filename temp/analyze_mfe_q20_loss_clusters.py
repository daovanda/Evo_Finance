"""Analyze consecutive net-loss runs in the baseline MFE Q20 strategy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from crypto.data import load_ohlcv
from temp.analyze_mfe_q20_prediction_bands import _trade_frame


DEFAULT_CACHE = Path("temp/model/meta_mfe_q20_after_1m_2m_3m/common_meta_oof.pkl")
DEFAULT_DATA = Path("data/crypto/BTCUSDT_5m.csv")
DEFAULT_SUMMARY = Path("temp/output/mfe_q20_loss_cluster_summary.csv")
DEFAULT_RUNS = Path("temp/output/mfe_q20_loss_clusters.csv")
DEFAULT_DISTRIBUTION = Path("temp/output/mfe_q20_loss_cluster_lengths.csv")


def _extract_runs(split: str, trades: pd.DataFrame, trade_cost: float) -> pd.DataFrame:
    data = trades.sort_index().copy()
    data["net_return"] = data["gross_return"] - float(trade_cost)
    loss = data["net_return"].lt(0.0).to_numpy(bool)
    starts = np.flatnonzero(loss & ~np.r_[False, loss[:-1]])
    ends = np.flatnonzero(loss & ~np.r_[loss[1:], False])
    rows = []
    for cluster_id, (start, end) in enumerate(zip(starts, ends, strict=True), start=1):
        selected = data.iloc[start : end + 1]
        rows.append(
            {
                "split": split,
                "cluster_id": cluster_id,
                "start_signal": selected.index[0],
                "end_signal": selected.index[-1],
                "run_length": len(selected),
                "net_mean": selected["net_return"].mean(),
                "net_sum": selected["net_return"].sum(),
                "worst_net": selected["net_return"].min(),
            }
        )
    return pd.DataFrame(rows)


def _summary(split: str, trades: pd.DataFrame, runs: pd.DataFrame, trade_cost: float) -> dict:
    net = trades.sort_index()["gross_return"] - float(trade_cost)
    loss = net.lt(0.0).to_numpy(bool)
    loss_count = int(loss.sum())
    loss_rate = float(loss.mean())
    after_loss = loss[:-1]
    next_loss_rate = (
        float(loss[1:][after_loss].mean()) if after_loss.any() else np.nan
    )
    lengths = runs["run_length"] if not runs.empty else pd.Series(dtype=int)
    return {
        "split": split,
        "trades": len(trades),
        "losses": loss_count,
        "loss_rate": loss_rate,
        "loss_clusters": len(runs),
        "mean_run_length": lengths.mean(),
        "median_run_length": lengths.median(),
        "max_run_length": lengths.max(),
        "P(next_loss_given_loss)": next_loss_rate,
        "loss_transition_lift": next_loss_rate / loss_rate if loss_rate else np.nan,
        "losses_in_run_ge_2": int(
            (lengths.loc[lengths.ge(2)]).sum() if len(lengths) else 0
        ),
        "share_losses_in_run_ge_2": (
            lengths.loc[lengths.ge(2)].sum() / loss_count if loss_count else np.nan
        ),
        "losses_in_run_ge_3": int(
            (lengths.loc[lengths.ge(3)]).sum() if len(lengths) else 0
        ),
        "share_losses_in_run_ge_3": (
            lengths.loc[lengths.ge(3)].sum() / loss_count if loss_count else np.nan
        ),
        "losses_in_run_ge_5": int(
            (lengths.loc[lengths.ge(5)]).sum() if len(lengths) else 0
        ),
        "share_losses_in_run_ge_5": (
            lengths.loc[lengths.ge(5)].sum() / loss_count if loss_count else np.nan
        ),
    }


def _distribution(split: str, runs: pd.DataFrame, loss_count: int) -> list[dict]:
    lengths = runs["run_length"]
    rows = []
    max_length = int(lengths.max()) if len(lengths) else 0
    for run_length in range(1, max_length + 1):
        cluster_count = int(lengths.eq(run_length).sum())
        contained_losses = cluster_count * run_length
        rows.append(
            {
                "split": split,
                "run_length": run_length,
                "clusters": cluster_count,
                "contained_losses": contained_losses,
                "share_of_clusters": cluster_count / len(runs) if len(runs) else np.nan,
                "share_of_losses": contained_losses / loss_count if loss_count else np.nan,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--trade-cost", type=float, default=0.0002)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--runs-out", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--distribution-out", type=Path, default=DEFAULT_DISTRIBUTION)
    args = parser.parse_args()

    cached = pd.read_pickle(args.cache)
    raw = load_ohlcv(args.data)
    required = ["meta_dynamic_tp_h3"]
    split_trades = {
        "val": _trade_frame(cached.val_df.dropna(subset=required), raw, args.horizon),
        "test": _trade_frame(cached.test_df.dropna(subset=required), raw, args.horizon),
    }
    all_runs = []
    summaries = []
    distributions = []
    for split, trades in split_trades.items():
        runs = _extract_runs(split, trades, args.trade_cost)
        summary = _summary(split, trades, runs, args.trade_cost)
        all_runs.append(runs)
        summaries.append(summary)
        distributions.extend(_distribution(split, runs, int(summary["losses"])))

    run_frame = pd.concat(all_runs, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    distribution_frame = pd.DataFrame(distributions)
    for output in (args.summary_out, args.runs_out, args.distribution_out):
        output.parent.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(args.summary_out, index=False)
    run_frame.to_csv(args.runs_out, index=False)
    distribution_frame.to_csv(args.distribution_out, index=False)

    display = summary_frame.copy()
    for column in (
        "loss_rate", "P(next_loss_given_loss)", "share_losses_in_run_ge_2",
        "share_losses_in_run_ge_3", "share_losses_in_run_ge_5",
    ):
        display[column] = display[column].map(lambda value: f"{100.0 * value:.3f}%")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print("=== Loss cluster summary ===")
        print(display.to_string(index=False))
        print("\n=== Exact loss-run length distribution ===")
        print(distribution_frame.to_string(index=False))
    print(f"\nSaved: {args.summary_out}")
    print(f"Saved: {args.runs_out}")
    print(f"Saved: {args.distribution_out}")


if __name__ == "__main__":
    main()
