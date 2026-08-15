"""Backtest a quantile MAE model as a dynamic Short take-profit signal.

The model is retrained on the archive split policy. A Short trade enters at
open H1 when predicted MAE is at least ``--min-prediction``. Its TP distance
equals the prediction plus ``--tp-offset``. TP is checked through the archive
horizon and a miss exits at the final close. No stop loss is used.

PowerShell example:
    python -m temp.backtest_quantile_mae_dynamic_tp_h1 `
      --archive crypto/results/crypto_btc_5m_quantile_mae_q20_h1_seed1_1h.json `
      --rank 1 `
      --min-prediction 0.0002 `
      --tp-offset 0.0005 `
      --loss-cooldown-threshold 0.001 `
      --cooldown-bars 5 `
      --trade-cost 0.0002 `
      --data data/crypto/BTCUSDT_5m.csv `
      --out-dir temp/output
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto import config
from crypto.analyze import _required_windows_for_entries
from crypto.data import (
    add_binary_labels,
    load_ohlcv,
    make_walk_forward_folds,
    split_labeled_by_dates,
)
from crypto.evolution import CryptoIndividual
from crypto.expression import CryptoFeatureSpace
from crypto.features import build_feature_frame, selectable_features
from crypto.quantile_fitness import QuantileFitnessEvaluator
from temp.backtest_quantile_mfe_dynamic_tp import _load_archive, _split_policy


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("temp.backtest_quantile_mae_dynamic_tp_h1")

DEFAULT_ARCHIVE = Path(
    "crypto/results/crypto_btc_5m_quantile_mae_q20_h1_seed1_1h.json"
)
DEFAULT_DATA = Path("data/crypto/BTCUSDT_5m.csv")
DEFAULT_OUT_DIR = Path("temp/output")


def _make_trades(
    split: str,
    frame: pd.DataFrame,
    prediction: pd.Series,
    raw: pd.DataFrame,
    *,
    horizon: int,
    min_prediction: float,
    tp_offset: float,
    loss_cooldown_threshold: float,
    cooldown_bars: int,
    bar_delta: pd.Timedelta,
    trade_cost: float,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    entry = pd.to_numeric(raw["open"], errors="coerce").shift(-1)
    adverse_paths = [
        1.0 - pd.to_numeric(raw["low"], errors="coerce").shift(-step).div(entry)
        for step in range(1, int(horizon) + 1)
    ]
    actual_mae = pd.concat(adverse_paths, axis=1).max(axis=1)
    close_return = 1.0 - pd.to_numeric(raw["close"], errors="coerce").shift(
        -int(horizon)
    ).div(entry)

    aligned = pd.DataFrame(index=frame.index)
    aligned["prediction"] = prediction.reindex(frame.index)
    aligned["entry_open"] = entry.reindex(frame.index)
    aligned["actual_mae"] = actual_mae.reindex(frame.index)
    aligned["short_close_return"] = close_return.reindex(frame.index)
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna()

    candidate_rows = len(aligned)
    trades = aligned.loc[
        aligned["prediction"].ge(float(min_prediction))
    ].copy()
    raw_signals = len(trades)
    trades["trade_tp"] = trades["prediction"] + float(tp_offset)
    trades["tp_price"] = trades["entry_open"] * (1.0 - trades["trade_tp"])
    trades["tp_hit"] = trades["actual_mae"].ge(trades["trade_tp"])
    trades["exit_type"] = np.where(
        trades["tp_hit"], f"tp_h1_h{int(horizon)}", f"close_h{int(horizon)}"
    )
    trades["gross_return"] = np.where(
        trades["tp_hit"],
        trades["trade_tp"],
        trades["short_close_return"],
    )
    cooldown_skipped = 0
    if cooldown_bars > 0 and loss_cooldown_threshold > 0.0 and len(trades):
        keep = pd.Series(False, index=trades.index)
        cooldown_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        cooldown_cursor = 0
        active_cooldown_end: pd.Timestamp | None = None
        for timestamp, row in trades.iterrows():
            timestamp = pd.Timestamp(timestamp)
            while (
                cooldown_cursor < len(cooldown_intervals)
                and cooldown_intervals[cooldown_cursor][0] <= timestamp
            ):
                _, interval_end = cooldown_intervals[cooldown_cursor]
                active_cooldown_end = (
                    interval_end
                    if active_cooldown_end is None
                    else max(active_cooldown_end, interval_end)
                )
                cooldown_cursor += 1
            in_cooldown = (
                active_cooldown_end is not None and timestamp <= active_cooldown_end
            )
            if in_cooldown:
                cooldown_skipped += 1
                continue
            keep.loc[timestamp] = True
            severe_close_loss = (
                row["exit_type"] == f"close_h{int(horizon)}"
                and float(row["gross_return"]) <= -float(loss_cooldown_threshold)
            )
            if severe_close_loss:
                # The outcome is unavailable until the final holding candle
                # closes. Signals before that point must remain executable.
                cooldown_start = timestamp + int(horizon) * bar_delta
                cooldown_end = cooldown_start + (int(cooldown_bars) - 1) * bar_delta
                cooldown_intervals.append((cooldown_start, cooldown_end))
        trades = trades.loc[keep].copy()

    trades["net_return"] = trades["gross_return"] - float(trade_cost)
    trades["cum_net"] = trades["net_return"].cumsum()

    n = len(trades)
    elapsed_days = (
        max((trades.index.max() - trades.index.min()).total_seconds() / 86400.0, 1.0)
        if n
        else 1.0
    )
    misses = trades.loc[~trades["tp_hit"]]
    summary: dict[str, float | int | str] = {
        "split": split,
        "candidate rows": candidate_rows,
        "raw signals": raw_signals,
        "cooldown skipped": cooldown_skipped,
        "trades": n,
        "selection rate": n / max(candidate_rows, 1),
        "trades/day": n / elapsed_days,
        "prediction mean": float(trades["prediction"].mean()) if n else np.nan,
        "trade TP mean": float(trades["trade_tp"].mean()) if n else np.nan,
        "actual MAE mean": float(trades["actual_mae"].mean()) if n else np.nan,
        "TP hit rate": float(trades["tp_hit"].mean()) if n else np.nan,
        "close exit rate": float((~trades["tp_hit"]).mean()) if n else np.nan,
        "miss close mean": (
            float(misses["short_close_return"].mean())
            if len(misses)
            else np.nan
        ),
        "gross mean": float(trades["gross_return"].mean()) if n else np.nan,
        "E[net]": float(trades["net_return"].mean()) if n else np.nan,
        "net win rate": float(trades["net_return"].gt(0.0).mean()) if n else np.nan,
    }
    return trades, summary


def _render_report(
    summaries: list[dict[str, float | int | str]],
    trades_by_split: dict[str, pd.DataFrame],
    output: Path,
    *,
    title: str,
) -> None:
    display = pd.DataFrame(summaries).copy()
    percent_columns = [
        "selection rate",
        "prediction mean",
        "trade TP mean",
        "actual MAE mean",
        "TP hit rate",
        "close exit rate",
        "miss close mean",
        "gross mean",
        "E[net]",
        "net win rate",
    ]
    for column in percent_columns:
        display[column] = display[column].map(
            lambda value: "n/a" if pd.isna(value) else f"{100.0 * float(value):+.3f}%"
        )
    display["selection rate"] = pd.DataFrame(summaries)["selection rate"].map(
        lambda value: f"{100.0 * float(value):.2f}%"
    )
    for column in ("TP hit rate", "close exit rate", "net win rate"):
        display[column] = pd.DataFrame(summaries)[column].map(
            lambda value: "n/a" if pd.isna(value) else f"{100.0 * float(value):.2f}%"
        )
    display["trades/day"] = pd.DataFrame(summaries)["trades/day"].map(
        lambda value: f"{float(value):.1f}"
    )

    fig, (ax_table, ax_curve) = plt.subplots(
        2,
        1,
        figsize=(18, 9),
        gridspec_kw={"height_ratios": [1.0, 2.4]},
    )
    ax_table.axis("off")
    table = ax_table.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.45)
    ax_table.set_title(title, fontsize=13, pad=12)

    for split, trades in trades_by_split.items():
        if len(trades):
            ax_curve.plot(trades.index, trades["cum_net"] * 100.0, label=split)
    ax_curve.axhline(0.0, color="black", linewidth=0.8)
    ax_curve.set_title("Cumulative net return (overlapping signals allowed)")
    ax_curve.set_ylabel("Cumulative net return (%)")
    ax_curve.set_xlabel("Time")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    archive_path = Path(args.archive)
    metadata, entry = _load_archive(archive_path, args.rank)
    if config.canonical_label_mode(metadata.get("label_mode")) != "quantile_trade":
        raise ValueError("Archive label_mode must be quantile_trade.")
    if config.canonical_quantile_target(metadata.get("quantile_target")) != "mae":
        raise ValueError("This backtest requires quantile_target=mae.")
    horizons = [int(value) for value in metadata.get("horizons", [])]
    if len(horizons) != 1 or horizons[0] <= 0:
        raise ValueError(f"This backtest requires one positive horizon, got {horizons}.")
    horizon = horizons[0]

    quantile = config.validate_quantile_alpha(metadata.get("quantile_alpha"))
    policy = _split_policy(metadata)
    purge = config.purge_bars_for_horizons(horizons)
    raw = load_ohlcv(Path(args.data))
    bar_delta = raw.index.to_series().diff().dropna().median()
    if not isinstance(bar_delta, pd.Timedelta) or bar_delta <= pd.Timedelta(0):
        raise ValueError("Unable to infer a positive candle interval.")
    labeled = add_binary_labels(raw, horizons=horizons, label_mode="quantile_trade")
    train, val, test = split_labeled_by_dates(
        labeled,
        val_start=policy["val_start"],
        test_start=policy["test_start"],
        test_end=policy["test_end"],
        purge_bars=purge,
    )
    wf_source = labeled.loc[labeled.index < pd.Timestamp(policy["wf_end"])]
    folds = make_walk_forward_folds(
        wf_source,
        wf_end=policy["wf_end"],
        min_train_months=policy["wf_min_train_months"],
        val_months=policy["wf_val_months"],
        step_months=policy["wf_step_months"],
        purge_bars=purge,
    )
    if not folds:
        raise ValueError("Archive split policy produced no walk-forward folds.")

    windows = _required_windows_for_entries([entry])
    logger.info("Building required feature windows: %s", windows)
    feature_frame = build_feature_frame(
        raw,
        windows=windows,
        quality_index=folds[0].train_df.index,
    )
    feature_space = CryptoFeatureSpace(feature_frame, selectable_features(feature_frame))
    individual = CryptoIndividual(
        features=[str(value) for value in entry["features"]],
        generation=int(entry.get("generation", 0) or 0),
        score=float(entry.get("score", np.nan)),
    )
    evaluator = QuantileFitnessEvaluator(
        horizons=horizons,
        target="mae",
        quantile=quantile,
    )
    valid_train = evaluator._valid_frame(train, horizon)
    valid_val = evaluator._valid_frame(val, horizon)
    valid_test = evaluator._valid_frame(test, horizon)
    _, val_prediction, test_prediction = evaluator._fit_predict(
        individual,
        feature_space,
        horizon,
        valid_train,
        valid_val,
        valid_test,
    )
    if test_prediction is None:
        raise RuntimeError("Quantile evaluator did not return Test predictions.")

    trades_by_split: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, float | int | str]] = []
    for split, frame, prediction in (
        ("val", valid_val, val_prediction),
        ("test", valid_test, test_prediction),
    ):
        trades, summary = _make_trades(
            split,
            frame,
            prediction,
            raw,
            horizon=horizon,
            min_prediction=args.min_prediction,
            tp_offset=args.tp_offset,
            loss_cooldown_threshold=args.loss_cooldown_threshold,
            cooldown_bars=args.cooldown_bars,
            bar_delta=bar_delta,
            trade_cost=args.trade_cost,
        )
        trades_by_split[split] = trades
        summaries.append(summary)

    print(f"\n=== Quantile MAE dynamic Short TP H{horizon} backtest ===")
    print(pd.DataFrame(summaries).to_string(index=False))
    threshold_name = f"{100.0 * args.min_prediction:.3f}".replace(".", "p")
    offset_name = f"{100.0 * args.tp_offset:.3f}".replace(".", "p")
    cooldown_name = f"{100.0 * args.loss_cooldown_threshold:.3f}".replace(
        ".", "p"
    )
    output = Path(args.out_dir) / (
        f"{archive_path.stem}_r{int(args.rank):02d}_pred_ge_"
        f"{threshold_name}pct_tp_plus_{offset_name}pct_"
        f"loss{cooldown_name}pct_cd{int(args.cooldown_bars)}_close_h{horizon}.png"
    )
    _render_report(
        summaries,
        trades_by_split,
        output,
        title=(
            f"Rank {args.rank} MAE Q{100.0 * quantile:.0f} H{horizon} | Short at open H1 | "
            f"prediction >= {100.0 * args.min_prediction:.3f}% | "
            f"TP=prediction+{100.0 * args.tp_offset:.3f}% | miss -> close H{horizon} | "
            f"loss <= -{100.0 * args.loss_cooldown_threshold:.3f}% -> "
            f"skip {int(args.cooldown_bars)} bars"
        ),
    )
    logger.info("Saved report: %s", output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--min-prediction", type=float, default=0.0002)
    parser.add_argument("--tp-offset", type=float, default=0.0005)
    parser.add_argument("--loss-cooldown-threshold", type=float, default=0.001)
    parser.add_argument("--cooldown-bars", type=int, default=5)
    parser.add_argument("--trade-cost", type=float, default=config.TRADE_COST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if not 0.0 <= args.min_prediction < 1.0:
        parser.error("--min-prediction must be in [0, 1).")
    if not 0.0 <= args.tp_offset < 1.0:
        parser.error("--tp-offset must be in [0, 1).")
    if not 0.0 <= args.loss_cooldown_threshold < 1.0:
        parser.error("--loss-cooldown-threshold must be in [0, 1).")
    if args.cooldown_bars < 0:
        parser.error("--cooldown-bars must be non-negative.")
    if args.trade_cost < 0.0:
        parser.error("--trade-cost must be non-negative.")
    return args


def main() -> None:
    output = run(parse_args())
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
