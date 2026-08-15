"""Backtest a quantile MFE archive as a dynamic take-profit model.

For every final Val/Test row, the selected MFE rank predicts a dynamic TP. A
Long trade is opened at open H1 only when MFE is strictly above
``--min-prediction``. A fixed SL relative to open H1 is active over the same
future candles. When TP and SL both occur in one candle,
``--same-candle-policy`` chooses which happened first. If neither barrier is
reached, the trade exits at final close.

PowerShell example:
    python -m temp.backtest_quantile_mfe_dynamic_tp `
      --archive crypto/results/crypto_btc_5m_quantile_mfe_q20_h3_seed1_1h.json `
      --rank 1 `
      --min-prediction 0.0002 `
      --stop-loss 0.001 `
      --same-candle-policy stop_first `
      --trade-cost 0.0002 `
      --data data/crypto/BTCUSDT_5m.csv `
      --out-dir temp/output
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Patch, Rectangle

from crypto import config
from crypto.analyze import _required_windows_for_entries
from crypto.backtest import ModelSpec, _archive_horizons, _train_spec_bundle
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("temp.backtest_quantile_mfe_dynamic_tp")


DEFAULT_ARCHIVE = Path(
    "crypto/results/crypto_btc_5m_quantile_mfe_q20_h3_seed1_1h.json"
)
DEFAULT_DATA = Path("data/crypto/BTCUSDT_5m.csv")
DEFAULT_OUT_DIR = Path("temp/output")
DEFAULT_BEAR_ARCHIVE = Path(
    "crypto/results/crypto_btc_5m_bear_top40_seed1_8h.json"
)
DEFAULT_FILTER_ARCHIVE = Path(
    "crypto/results/crypto_btc_5m_long_adverse_floor_h1_floor015_top80_seed1_1h.json"
)
DEFAULT_SLOPE_REVERSAL_ARCHIVE = Path(
    "crypto/results/crypto_btc_5m_long_ma_slope_reversal_fs2_top20_seed1_1h.json"
)


@dataclass(frozen=True)
class SplitResult:
    split: str
    trades: pd.DataFrame
    candidate_rows: int


def _load_archive(path: Path, rank: int) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    entries = payload.get("entries", [])
    if not isinstance(metadata, dict) or not isinstance(entries, list):
        raise ValueError(f"Malformed archive: {path}")
    if not 1 <= int(rank) <= len(entries):
        raise ValueError(f"Rank {rank} is outside archive range 1..{len(entries)}.")
    entry = entries[int(rank) - 1]
    if not isinstance(entry, dict) or not entry.get("features"):
        raise ValueError(f"Archive rank {rank} has no features: {path}")
    return dict(metadata), dict(entry)


def _split_policy(metadata: dict[str, Any]) -> dict[str, Any]:
    policy = metadata.get("split_policy", {})
    if not isinstance(policy, dict):
        policy = {}
    return {
        "val_start": policy.get("val_start") or config.VAL_START,
        "test_start": policy.get("test_start") or config.TEST_START,
        "test_end": policy.get("test_end", config.TEST_END),
        "wf_end": policy.get("wf_end") or config.WF_END,
        "wf_min_train_months": int(
            policy.get("wf_min_train_months", config.WF_MIN_TRAIN_MONTHS)
        ),
        "wf_val_months": int(policy.get("wf_val_months", config.WF_VAL_MONTHS)),
        "wf_step_months": int(
            policy.get("wf_step_months", config.WF_STEP_MONTHS)
        ),
    }


def _price_path(raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
    entry = pd.to_numeric(raw["open"], errors="coerce").shift(-1)
    result = pd.DataFrame(index=raw.index)
    result["entry_open"] = entry
    highs: list[pd.Series] = []
    for step in range(1, int(horizon) + 1):
        high_return = (
            pd.to_numeric(raw["high"], errors="coerce")
            .shift(-step)
            .div(entry)
            .sub(1.0)
        )
        result[f"high_h{step}"] = high_return
        result[f"low_h{step}"] = (
            pd.to_numeric(raw["low"], errors="coerce")
            .shift(-step)
            .div(entry)
            .sub(1.0)
        )
        result[f"close_h{step}"] = (
            pd.to_numeric(raw["close"], errors="coerce")
            .shift(-step)
            .div(entry)
            .sub(1.0)
        )
        highs.append(high_return)
    result["actual_mfe"] = pd.concat(highs, axis=1).max(axis=1, skipna=False)
    result["close_return"] = (
        pd.to_numeric(raw["close"], errors="coerce")
        .shift(-int(horizon))
        .div(entry)
        .sub(1.0)
    )
    return result.replace([np.inf, -np.inf], np.nan)


def _make_result(
    split: str,
    frame: pd.DataFrame,
    prediction: pd.Series,
    path: pd.DataFrame,
    min_prediction: float,
    stop_loss: float,
    same_candle_policy: str,
    trade_cost: float,
    allowed_index: pd.Index | None = None,
) -> SplitResult:
    aligned = pd.DataFrame(index=frame.index)
    aligned["prediction"] = prediction.reindex(frame.index)
    aligned = aligned.join(path, how="left")
    aligned = aligned.dropna(
        subset=[
            "prediction",
            "entry_open",
            "actual_mfe",
            "close_return",
        ]
    )
    candidate_rows = len(aligned)
    trades = aligned[aligned["prediction"] > float(min_prediction)].copy()
    if allowed_index is not None:
        trades = trades.loc[trades.index.isin(pd.Index(allowed_index))].copy()
    gross = pd.Series(np.nan, index=trades.index, dtype=float)
    exit_type = pd.Series("", index=trades.index, dtype=object)
    horizon = len([column for column in path.columns if column.startswith("high_h")])
    for step in range(1, horizon + 1):
        active = gross.isna()
        if not active.any():
            break
        tp_hit = trades[f"high_h{step}"] >= trades["prediction"]
        sl_hit = trades[f"low_h{step}"] <= -float(stop_loss)
        both = active & tp_hit & sl_hit
        tp_only = active & tp_hit & ~sl_hit
        sl_only = active & sl_hit & ~tp_hit
        if same_candle_policy == "tp_first":
            tp_exit = tp_only | both
            sl_exit = sl_only
        else:
            tp_exit = tp_only
            sl_exit = sl_only | both
        gross.loc[tp_exit] = trades.loc[tp_exit, "prediction"]
        exit_type.loc[tp_exit] = f"tp_h{step}"
        gross.loc[sl_exit] = -float(stop_loss)
        exit_type.loc[sl_exit] = f"sl_h{step}"

    close_exit = gross.isna()
    gross.loc[close_exit] = trades.loc[close_exit, "close_return"]
    exit_type.loc[close_exit] = "close"
    trades["exit_type"] = exit_type
    trades["tp_hit"] = exit_type.str.startswith("tp_")
    trades["sl_hit"] = exit_type.str.startswith("sl_")
    trades["close_exit"] = exit_type.eq("close")
    trades["gross_return"] = gross
    trades["net_return"] = trades["gross_return"] - float(trade_cost)
    trades["cum_net"] = trades["net_return"].cumsum()
    return SplitResult(split=split, trades=trades, candidate_rows=candidate_rows)


def _and_filter_impact_summary(
    baseline_results: list[SplitResult],
    filtered_results: list[SplitResult],
) -> list[dict[str, float | int | str]]:
    """Compare the MFE strategy before and after the auxiliary AND filter."""
    rows: list[dict[str, float | int | str]] = []
    filtered_by_split = {result.split: result for result in filtered_results}
    for baseline in baseline_results:
        filtered = filtered_by_split[baseline.split]
        removed = baseline.trades.loc[
            ~baseline.trades.index.isin(filtered.trades.index)
        ]
        baseline_wins = baseline.trades["net_return"].gt(0.0)
        baseline_losses = baseline.trades["net_return"].lt(0.0)
        removed_wins = removed["net_return"].gt(0.0)
        removed_losses = removed["net_return"].lt(0.0)
        rows.append(
            {
                "split": baseline.split,
                "MFE trades": len(baseline.trades),
                "AND trades": len(filtered.trades),
                "retained": len(filtered.trades) / max(len(baseline.trades), 1),
                "removed": len(removed),
                "removed wins": int(removed_wins.sum()),
                "removed losses": int(removed_losses.sum()),
                "win removal rate": (
                    float(removed_wins.sum() / baseline_wins.sum())
                    if baseline_wins.any()
                    else np.nan
                ),
                "loss removal rate": (
                    float(removed_losses.sum() / baseline_losses.sum())
                    if baseline_losses.any()
                    else np.nan
                ),
                "baseline E[net]": float(baseline.trades["net_return"].mean()),
                "AND E[net]": float(filtered.trades["net_return"].mean()),
                "E[net] delta": (
                    float(filtered.trades["net_return"].mean())
                    - float(baseline.trades["net_return"].mean())
                ),
            }
        )
    return rows


def _ma3_cooldown_allowed_index(
    candidate_index: pd.Index,
    trigger_index: pd.Index,
    slope_positive: pd.Series,
    *,
    cooldown_bars: int,
) -> pd.DatetimeIndex:
    """Build the stateful MA3-reversal risk gate without future information.

    A valid reversal trigger blocks its own signal candle and starts a
    two-future-bar waiting period. At the close of the final waiting candle,
    trading may resume immediately when the observable MA3 slope is positive;
    otherwise the gate remains closed until that condition is met. A new
    trigger while blocked restarts the waiting period.
    """
    index = pd.DatetimeIndex(candidate_index).sort_values().unique()
    triggers = set(pd.DatetimeIndex(trigger_index))
    slope = slope_positive.reindex(index).fillna(False).astype(bool)
    allowed = np.zeros(len(index), dtype=bool)
    blocked = False
    release_position = -1

    for position, timestamp in enumerate(index):
        if timestamp in triggers:
            blocked = True
            release_position = position + int(cooldown_bars)
            continue
        if not blocked:
            allowed[position] = True
            continue
        if position >= release_position and bool(slope.iloc[position]):
            blocked = False
            allowed[position] = True

    return index[allowed]


def _ma3_cooldown_impact_summary(
    before: list[SplitResult],
    after: list[SplitResult],
) -> list[dict[str, float | int | str]]:
    """Summarize trades removed by the stateful MA3 cooldown gate."""
    after_by_split = {result.split: result for result in after}
    rows: list[dict[str, float | int | str]] = []
    for baseline in before:
        filtered = after_by_split[baseline.split]
        removed = baseline.trades.loc[
            ~baseline.trades.index.isin(filtered.trades.index)
        ]
        rows.append(
            {
                "split": baseline.split,
                "MFE trades before gate": len(baseline.trades),
                "trades after gate": len(filtered.trades),
                "retained": len(filtered.trades) / max(len(baseline.trades), 1),
                "blocked trades": len(removed),
                "blocked win rate": (
                    float(removed["net_return"].gt(0.0).mean())
                    if len(removed)
                    else np.nan
                ),
                "before E[net]": float(baseline.trades["net_return"].mean()),
                "after E[net]": float(filtered.trades["net_return"].mean()),
                "E[net] delta": (
                    float(filtered.trades["net_return"].mean())
                    - float(baseline.trades["net_return"].mean())
                ),
            }
        )
    return rows


def _summary(result: SplitResult) -> dict[str, float | int | str]:
    trades = result.trades
    n = len(trades)
    if n == 0:
        return {
            "split": result.split,
            "candidate rows": result.candidate_rows,
            "trades": 0,
            "selection rate": 0.0,
            "avg trades/day": 0.0,
            "pred TP mean": np.nan,
            "TP hit rate": np.nan,
            "SL hit rate": np.nan,
            "close exit rate": np.nan,
            "close exit mean": np.nan,
            "gross mean": np.nan,
            "E[net]": np.nan,
            "net win rate": np.nan,
        }
    days = max((trades.index.max() - trades.index.min()).total_seconds() / 86400.0, 1.0)
    close_returns = trades.loc[trades["close_exit"], "close_return"]
    summary: dict[str, float | int | str] = {
        "split": result.split,
        "candidate rows": result.candidate_rows,
        "trades": n,
        "selection rate": n / max(result.candidate_rows, 1),
        "avg trades/day": n / days,
        "pred TP mean": float(trades["prediction"].mean()),
        "TP hit rate": float(trades["tp_hit"].mean()),
        "SL hit rate": float(trades["sl_hit"].mean()),
        "close exit rate": float(trades["close_exit"].mean()),
        "close exit mean": (
            float(close_returns.mean()) if len(close_returns) else np.nan
        ),
        "gross mean": float(trades["gross_return"].mean()),
        "E[net]": float(trades["net_return"].mean()),
        "net win rate": float((trades["net_return"] > 0.0).mean()),
        "gross loss >0.10%": int(trades["gross_return"].lt(-0.001).sum()),
        "gross loss >0.10% rate": float(
            trades["gross_return"].lt(-0.001).mean()
        ),
        "gross loss >0.15%": int(trades["gross_return"].lt(-0.0015).sum()),
        "gross loss >0.15% rate": float(
            trades["gross_return"].lt(-0.0015).mean()
        ),
    }
    tp_cumulative = pd.Series(False, index=trades.index)
    horizon = len(
        [column for column in trades.columns if column.startswith("high_h")]
    )
    for step in range(1, horizon + 1):
        first_hit = trades["exit_type"].eq(f"tp_h{step}")
        tp_cumulative = tp_cumulative | first_hit
        summary[f"TP first H{step}"] = float(first_hit.mean())
        summary[f"TP by H{step}"] = float(tp_cumulative.mean())
    return summary


def _two_sided_excursion_summary(
    results: list[SplitResult],
    threshold: float,
) -> list[dict[str, float | int | str]]:
    """Count selected trades whose H1..H path crosses both return thresholds."""
    rows: list[dict[str, float | int | str]] = []
    combined: list[pd.DataFrame] = []
    for result in results:
        combined.append(result.trades)
        rows.append(
            _summarize_two_sided_excursion(
                result.trades,
                split=result.split,
                threshold=threshold,
            )
        )
    all_trades = pd.concat(combined, axis=0) if combined else pd.DataFrame()
    rows.append(
        _summarize_two_sided_excursion(
            all_trades,
            split="val+test",
            threshold=threshold,
        )
    )
    return rows


def _summarize_two_sided_excursion(
    trades: pd.DataFrame,
    *,
    split: str,
    threshold: float,
) -> dict[str, float | int | str]:
    low_columns = sorted(
        (column for column in trades.columns if column.startswith("low_h")),
        key=lambda column: int(column.removeprefix("low_h")),
    )
    if trades.empty:
        count = 0
    else:
        actual_mae = -trades[low_columns].min(axis=1)
        count = int(
            (trades["actual_mfe"].gt(threshold) & actual_mae.gt(threshold)).sum()
        )
    return {
        "split": split,
        "selected trades": len(trades),
        "MFE and MAE > threshold": count,
        "rate": count / len(trades) if len(trades) else np.nan,
    }


def _loss_clustering_summary(
    result: SplitResult,
    *,
    bar_delta: pd.Timedelta,
    cooldown_bars: int = 5,
) -> dict[str, float | int | str]:
    """Measure whether losing trades are followed by weak nearby signals."""
    trades = result.trades.sort_index()
    n = len(trades)
    if n == 0:
        return {
            "split": result.split,
            "loss rate": np.nan,
            "P(next loss | loss)": np.nan,
            "post-loss signals": 0,
            "post-loss signal coverage": np.nan,
            "post-loss loss rate": np.nan,
            "loss-rate delta": np.nan,
            "baseline E[net]": np.nan,
            "post-loss E[net]": np.nan,
            "E[net] delta": np.nan,
            "losses in runs>=2": np.nan,
            "max loss run": 0,
        }

    loss = trades["net_return"].lt(0.0).to_numpy(bool)
    loss_rate = float(loss.mean())
    next_loss_rate = (
        float(loss[1:][loss[:-1]].mean())
        if n > 1 and bool(loss[:-1].any())
        else np.nan
    )

    run_lengths: list[int] = []
    current_run = 0
    for is_loss in loss:
        if is_loss:
            current_run += 1
        elif current_run:
            run_lengths.append(current_run)
            current_run = 0
    if current_run:
        run_lengths.append(current_run)
    clustered_losses = sum(length for length in run_lengths if length >= 2)
    loss_count = int(loss.sum())

    horizon = len(
        [column for column in trades.columns if column.startswith("high_h")]
    )
    exit_steps = (
        trades["exit_type"]
        .str.extract(r"_h(\d+)$", expand=False)
        .astype(float)
        .fillna(float(horizon))
        .astype(int)
        .to_numpy()
    )
    signal_ns = trades.index.asi8
    bar_ns = int(bar_delta.value)
    cooldown_ns = int(cooldown_bars) * bar_ns
    loss_exit_ns = signal_ns[loss] + exit_steps[loss] * bar_ns
    starts = np.searchsorted(signal_ns, loss_exit_ns, side="right")
    ends = np.searchsorted(signal_ns, loss_exit_ns + cooldown_ns, side="right")
    coverage_diff = np.zeros(n + 1, dtype=np.int64)
    valid_intervals = starts < ends
    np.add.at(coverage_diff, starts[valid_intervals], 1)
    np.add.at(coverage_diff, ends[valid_intervals], -1)
    post_loss_mask = np.cumsum(coverage_diff[:-1]) > 0
    post_loss = trades.iloc[np.flatnonzero(post_loss_mask)]
    post_loss_rate = (
        float(post_loss["net_return"].lt(0.0).mean())
        if len(post_loss)
        else np.nan
    )
    baseline_net = float(trades["net_return"].mean())
    post_loss_net = (
        float(post_loss["net_return"].mean()) if len(post_loss) else np.nan
    )
    return {
        "split": result.split,
        "loss rate": loss_rate,
        "P(next loss | loss)": next_loss_rate,
        "post-loss signals": len(post_loss),
        "post-loss signal coverage": len(post_loss) / n,
        "post-loss loss rate": post_loss_rate,
        "loss-rate delta": post_loss_rate - loss_rate,
        "baseline E[net]": baseline_net,
        "post-loss E[net]": post_loss_net,
        "E[net] delta": post_loss_net - baseline_net,
        "losses in runs>=2": clustered_losses / max(loss_count, 1),
        "max loss run": max(run_lengths, default=0),
    }


def _loss_run_distribution(result: SplitResult) -> dict[str, str]:
    """Summarize exact loss-run counts and their share of all losing trades."""
    loss = result.trades.sort_index()["net_return"].lt(0.0).to_numpy(bool)
    run_lengths: list[int] = []
    current_run = 0
    for is_loss in loss:
        if is_loss:
            current_run += 1
        elif current_run:
            run_lengths.append(current_run)
            current_run = 0
    if current_run:
        run_lengths.append(current_run)

    total_losses = max(int(loss.sum()), 1)
    summary = {"split": result.split}
    for length in range(2, 15):
        run_count = sum(run_length == length for run_length in run_lengths)
        losses_in_runs = run_count * length
        summary[f"run {length}"] = (
            f"{run_count:,} | {100.0 * losses_in_runs / total_losses:.2f}%"
        )
    long_runs = [run_length for run_length in run_lengths if run_length >= 15]
    summary["run 15+"] = (
        f"{len(long_runs):,} | {100.0 * sum(long_runs) / total_losses:.2f}%"
    )
    return summary


def _h1_close_outcome_summary(
    result: SplitResult,
    low_threshold: float,
) -> list[dict[str, float | int | str]]:
    """Compare close-H1 behavior between profitable and losing trades."""
    trades = result.trades
    rows: list[dict[str, float | int | str]] = []
    populations = {
        "all trades": pd.Series(True, index=trades.index),
        "still open after H1": ~trades["exit_type"].str.endswith("_h1"),
    }
    for population_name, population_mask in populations.items():
        groups = {
            "win": population_mask & trades["net_return"].gt(0.0),
            "loss": population_mask & trades["net_return"].lt(0.0),
        }
        classified = max(int(groups["win"].sum() + groups["loss"].sum()), 1)
        for group_name, mask in groups.items():
            values = trades.loc[mask, "close_h1"].dropna()
            rows.append(
                {
                    "split": result.split,
                    "population": population_name,
                    "outcome": group_name,
                    "n": len(values),
                    "share": len(values) / classified,
                    "close H1 mean": (
                        float(values.mean()) if len(values) else np.nan
                    ),
                    "close H1 Q10": (
                        float(values.quantile(0.10)) if len(values) else np.nan
                    ),
                    "close H1 Q25": (
                        float(values.quantile(0.25)) if len(values) else np.nan
                    ),
                    "close H1 median": (
                        float(values.median()) if len(values) else np.nan
                    ),
                    "close H1 Q75": (
                        float(values.quantile(0.75)) if len(values) else np.nan
                    ),
                    "close H1 Q90": (
                        float(values.quantile(0.90)) if len(values) else np.nan
                    ),
                    "P(close H1 > 0)": (
                        float(values.gt(0.0).mean()) if len(values) else np.nan
                    ),
                    f"P(low H1 < {100.0 * low_threshold:+.3f}%)": (
                        float(trades.loc[mask, "low_h1"].lt(low_threshold).mean())
                        if int(mask.sum())
                        else np.nan
                    ),
                }
            )
    return rows


def _losing_trade_low_h1_summary(
    results: list[SplitResult],
    low_threshold: float,
) -> list[dict[str, float | int | str]]:
    """Split final losing trades by their H1 adverse excursion."""
    rows: list[dict[str, float | int | str]] = []
    combined_parts: list[pd.DataFrame] = []
    for result in results:
        losses = result.trades.loc[result.trades["net_return"].lt(0.0)]
        combined_parts.append(losses)
        rows.extend(
            _summarize_losing_trade_low_h1_groups(
                losses,
                split=result.split,
                low_threshold=low_threshold,
            )
        )
    combined = pd.concat(combined_parts, axis=0) if combined_parts else pd.DataFrame()
    rows.extend(
        _summarize_losing_trade_low_h1_groups(
            combined,
            split="val+test",
            low_threshold=low_threshold,
        )
    )
    return rows


def _summarize_losing_trade_low_h1_groups(
    losses: pd.DataFrame,
    *,
    split: str,
    low_threshold: float,
) -> list[dict[str, float | int | str]]:
    loss_count = max(len(losses), 1)
    conditions = {
        f"low H1 < {100.0 * low_threshold:+.3f}%": losses["low_h1"].lt(
            low_threshold
        ),
        f"low H1 >= {100.0 * low_threshold:+.3f}%": losses["low_h1"].ge(
            low_threshold
        ),
    }
    rows: list[dict[str, float | int | str]] = []
    for condition, mask in conditions.items():
        selected = losses.loc[mask]
        rows.append(
            {
                "split": split,
                "losing-trade group": condition,
                "n": len(selected),
                "share of losses": len(selected) / loss_count,
                "gross return mean": (
                    float(selected["gross_return"].mean())
                    if len(selected)
                    else np.nan
                ),
                "net return mean": (
                    float(selected["net_return"].mean())
                    if len(selected)
                    else np.nan
                ),
            }
        )
    return rows


def _close_threshold_by_outcome_summary(
    results: list[SplitResult],
    threshold: float,
) -> list[dict[str, float | int | str]]:
    """Count close H1/H3 threshold breaches inside final wins and losses."""
    parts: list[pd.DataFrame] = []
    rows: list[dict[str, float | int | str]] = []
    for result in results:
        trades = result.trades.copy()
        parts.append(trades)
        rows.extend(
            _summarize_close_threshold_by_outcome(
                trades,
                split=result.split,
                threshold=threshold,
            )
        )
    combined = pd.concat(parts, axis=0) if parts else pd.DataFrame()
    rows.extend(
        _summarize_close_threshold_by_outcome(
            combined,
            split="val+test",
            threshold=threshold,
        )
    )
    return rows


def _summarize_close_threshold_by_outcome(
    trades: pd.DataFrame,
    *,
    split: str,
    threshold: float,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for outcome, outcome_mask in (
        ("win", trades["net_return"].gt(0.0)),
        ("loss", trades["net_return"].lt(0.0)),
    ):
        group = trades.loc[outcome_mask]
        h1_hit = group["close_h1"].lt(threshold)
        h3_hit = group["close_return"].lt(threshold)
        h1_breach_group = group.loc[h1_hit]
        n = len(group)
        rows.append(
            {
                "split": split,
                "outcome": outcome,
                "n": n,
                "close H1 < threshold": int(h1_hit.sum()),
                "share close H1 < threshold": (
                    float(h1_hit.mean()) if n else np.nan
                ),
                "close H3 < threshold": int(h3_hit.sum()),
                "share close H3 < threshold": (
                    float(h3_hit.mean()) if n else np.nan
                ),
                "close H3 mean | close H1 < threshold": (
                    float(h1_breach_group["close_return"].mean())
                    if len(h1_breach_group)
                    else np.nan
                ),
                "realized gross mean | close H1 < threshold": (
                    float(h1_breach_group["gross_return"].mean())
                    if len(h1_breach_group)
                    else np.nan
                ),
                "predicted TP mean | close H1 < threshold": (
                    float(h1_breach_group["prediction"].mean())
                    if len(h1_breach_group)
                    else np.nan
                ),
            }
        )
    return rows


def _h1_close_threshold_summary(
    result: SplitResult,
    threshold: float,
) -> list[dict[str, float | int | str]]:
    """Evaluate final strategy outcome conditional on close H1 threshold."""
    trades = result.trades
    still_open = ~trades["exit_type"].str.endswith("_h1")
    branches = {
        f"> {100.0 * threshold:+.3f}%": (
            still_open & trades["close_h1"].gt(float(threshold))
        ),
        f"<= {100.0 * threshold:+.3f}%": (
            still_open & trades["close_h1"].le(float(threshold))
        ),
    }
    unresolved_count = max(int(still_open.sum()), 1)
    rows: list[dict[str, float | int | str]] = []
    for branch_name, mask in branches.items():
        selected = trades.loc[mask]
        rows.append(
            {
                "split": result.split,
                "close H1 condition": branch_name,
                "n": len(selected),
                "share of still-open H1": len(selected) / unresolved_count,
                "P(win)": (
                    float(selected["net_return"].gt(0.0).mean())
                    if len(selected)
                    else np.nan
                ),
                "P(loss)": (
                    float(selected["net_return"].lt(0.0).mean())
                    if len(selected)
                    else np.nan
                ),
                "held E[net]": (
                    float(selected["net_return"].mean())
                    if len(selected)
                    else np.nan
                ),
                "close H1 mean": (
                    float(selected["close_h1"].mean())
                    if len(selected)
                    else np.nan
                ),
            }
        )
    return rows


def _loss_close_h1_vs_h3_summary(
    result: SplitResult,
    *,
    trade_cost: float,
    large_loss_threshold: float,
) -> dict[str, float | int | str]:
    """Compare early H1 exit with actual H3 exit for no-SL losing trades."""
    trades = result.trades
    losses = trades.loc[
        trades["close_exit"] & trades["net_return"].lt(0.0)
    ].copy()
    if losses.empty:
        return {
            "split": result.split,
            "no-SL losses": 0,
            "share of trades": np.nan,
            "close H1 gross mean": np.nan,
            "close H3 gross mean": np.nan,
            "H1 improvement mean": np.nan,
            "H1 improvement median": np.nan,
            "P(H1 better than H3)": np.nan,
            "H1 E[net]": np.nan,
            "H3 E[net]": np.nan,
            "H1 large-loss rate": np.nan,
            "H3 large-loss rate": np.nan,
        }

    close_h1 = losses["close_h1"]
    close_h3 = losses["close_return"]
    improvement = close_h1 - close_h3
    h1_net = close_h1 - float(trade_cost)
    h3_net = close_h3 - float(trade_cost)
    return {
        "split": result.split,
        "no-SL losses": len(losses),
        "share of trades": len(losses) / max(len(trades), 1),
        "close H1 gross mean": float(close_h1.mean()),
        "close H3 gross mean": float(close_h3.mean()),
        "H1 improvement mean": float(improvement.mean()),
        "H1 improvement median": float(improvement.median()),
        "P(H1 better than H3)": float(improvement.gt(0.0).mean()),
        "H1 E[net]": float(h1_net.mean()),
        "H3 E[net]": float(h3_net.mean()),
        "H1 large-loss rate": float(h1_net.le(large_loss_threshold).mean()),
        "H3 large-loss rate": float(h3_net.le(large_loss_threshold).mean()),
    }


def _close_h2_vs_hold_h3_summary(
    result: SplitResult,
    *,
    trade_cost: float,
) -> dict[str, float | int | str]:
    """Compare close-H2 exit with the existing TP-H3/close-H3 policy."""
    trades = result.trades
    unresolved = trades.loc[trades["exit_type"].isin(["tp_h3", "close"])].copy()
    if unresolved.empty:
        return {
            "split": result.split,
            "unresolved after H2": 0,
            "share of trades": np.nan,
            "H3 TP rate": np.nan,
            "close H2 gross mean": np.nan,
            "hold H3 gross mean": np.nan,
            "H2 minus H3 mean": np.nan,
            "P(H2 better than H3)": np.nan,
            "close H2 E[net]": np.nan,
            "hold H3 E[net]": np.nan,
            "close H2 win rate": np.nan,
            "hold H3 win rate": np.nan,
            "whole-strategy E[net] delta": np.nan,
        }

    close_h2_gross = unresolved["close_h2"]
    hold_h3_gross = unresolved["gross_return"]
    close_h2_net = close_h2_gross - float(trade_cost)
    hold_h3_net = hold_h3_gross - float(trade_cost)
    improvement = close_h2_gross - hold_h3_gross
    return {
        "split": result.split,
        "unresolved after H2": len(unresolved),
        "share of trades": len(unresolved) / max(len(trades), 1),
        "H3 TP rate": float(unresolved["exit_type"].eq("tp_h3").mean()),
        "close H2 gross mean": float(close_h2_gross.mean()),
        "hold H3 gross mean": float(hold_h3_gross.mean()),
        "H2 minus H3 mean": float(improvement.mean()),
        "P(H2 better than H3)": float(improvement.gt(0.0).mean()),
        "close H2 E[net]": float(close_h2_net.mean()),
        "hold H3 E[net]": float(hold_h3_net.mean()),
        "close H2 win rate": float(close_h2_net.gt(0.0).mean()),
        "hold H3 win rate": float(hold_h3_net.gt(0.0).mean()),
        "whole-strategy E[net] delta": (
            float(improvement.sum()) / max(len(trades), 1)
        ),
    }


def _first_1m_low_summary(
    results: list[SplitResult],
    *,
    data_1m_path: Path,
    entry_delay: pd.Timedelta,
    threshold: float,
    simulated_stop_loss: float,
    trade_cost: float,
    large_loss_threshold: float,
) -> list[dict[str, float | int | str]]:
    """Measure the first entry-minute low relative to the 5m H1 open."""
    minute = pd.read_csv(
        data_1m_path,
        usecols=["date", "high", "low", "close"],
        parse_dates=["date"],
    )
    minute["high"] = pd.to_numeric(minute["high"], errors="coerce")
    minute["low"] = pd.to_numeric(minute["low"], errors="coerce")
    minute["close"] = pd.to_numeric(minute["close"], errors="coerce")
    minute_by_time = (
        minute.dropna(subset=["date", "high", "low", "close"])
        .drop_duplicates(subset="date", keep="last")
        .set_index("date")[["high", "low", "close"]]
        .sort_index()
    )
    rows: list[dict[str, float | int | str]] = []
    for result in results:
        trades = result.trades
        entry_times = pd.DatetimeIndex(trades.index) + entry_delay
        first_minute = minute_by_time.reindex(entry_times)
        first_low = pd.Series(first_minute["low"].to_numpy(float), index=trades.index)
        first_high = pd.Series(
            first_minute["high"].to_numpy(float),
            index=trades.index,
        )
        first_close = pd.Series(
            first_minute["close"].to_numpy(float),
            index=trades.index,
        )
        low_return = first_low.div(trades["entry_open"]).sub(1.0)
        high_return = first_high.div(trades["entry_open"]).sub(1.0)
        close_1m_return = first_close.div(trades["entry_open"]).sub(1.0)
        valid = low_return.notna() & high_return.notna() & close_1m_return.notna()
        first_minute_tp = valid & high_return.ge(trades["prediction"])

        horizon_columns = [
            column
            for column in trades.columns
            if column.startswith("close_h") and column[7:].isdigit()
        ]
        horizon = max(int(column[7:]) for column in horizon_columns)
        minutes_per_bar = int(entry_delay / pd.Timedelta(minutes=1))
        subsequent_offsets = np.arange(1, horizon * minutes_per_bar, dtype=int)
        if len(subsequent_offsets):
            subsequent_times = pd.DatetimeIndex(
                np.repeat(entry_times.to_numpy(), len(subsequent_offsets))
                + np.tile(
                    subsequent_offsets.astype("timedelta64[m]"),
                    len(entry_times),
                )
            )
            subsequent_lows = minute_by_time["low"].reindex(subsequent_times)
            subsequent_low_matrix = subsequent_lows.to_numpy(float).reshape(
                len(trades),
                len(subsequent_offsets),
            )
            complete_subsequent_path = pd.Series(
                np.isfinite(subsequent_low_matrix).all(axis=1),
                index=trades.index,
            )
            subsequent_min_low = pd.Series(
                np.nanmin(subsequent_low_matrix, axis=1),
                index=trades.index,
            )
            subsequent_min_low_return = subsequent_min_low.div(
                trades["entry_open"]
            ).sub(1.0)
        else:
            complete_subsequent_path = pd.Series(False, index=trades.index)
            subsequent_min_low_return = pd.Series(np.nan, index=trades.index)
        first_minute_tp_with_path = first_minute_tp & complete_subsequent_path
        returned_to_open = (
            first_minute_tp_with_path & subsequent_min_low_return.le(0.0)
        )
        hit = valid & low_return.lt(float(threshold))
        same_minute_tp = hit & high_return.ge(trades["prediction"])
        selected = trades.loc[hit]
        recovered = selected.loc[selected["tp_hit"]]
        no_tp = selected.loc[~selected["tp_hit"]]
        no_tp_close_h3 = no_tp["close_h3"]
        large_loss = valid & trades["net_return"].le(float(large_loss_threshold))
        large_loss_low = low_return.loc[large_loss]
        large_loss_close = close_1m_return.loc[large_loss]
        win = valid & trades["net_return"].gt(0.0)
        win_low = low_return.loc[win]
        first_minute_stop = valid & low_return.le(-float(simulated_stop_loss))
        stopped_wins = first_minute_stop & trades["net_return"].gt(0.0)
        stopped_losses = first_minute_stop & trades["net_return"].lt(0.0)
        simulated_stop_net = -float(simulated_stop_loss) - float(trade_cost)
        stopped_original_net = trades.loc[first_minute_stop, "net_return"]
        stopped_delta = simulated_stop_net - stopped_original_net
        row: dict[str, float | int | str] = {
            "split": result.split,
            "trades": len(trades),
            "matched first 1m": int(valid.sum()),
            "match rate": float(valid.mean()) if len(valid) else np.nan,
            "first 1m TP hit": int(first_minute_tp.sum()),
            "first 1m TP hit rate": (
                float(first_minute_tp.sum() / valid.sum())
                if int(valid.sum())
                else np.nan
            ),
            "first 1m TP with complete minute 2-H path": int(
                first_minute_tp_with_path.sum()
            ),
            "returned to open after first 1m TP": int(returned_to_open.sum()),
            "return-to-open rate | first 1m TP": (
                float(
                    returned_to_open.sum()
                    / first_minute_tp_with_path.sum()
                )
                if first_minute_tp_with_path.any()
                else np.nan
            ),
            "post-first-1m min low mean | first 1m TP": (
                float(
                    subsequent_min_low_return.loc[
                        first_minute_tp_with_path
                    ].mean()
                )
                if first_minute_tp_with_path.any()
                else np.nan
            ),
            f"first 1m low < {100.0 * threshold:+.3f}%": int(hit.sum()),
            "rate among matched": (
                float(hit.sum() / valid.sum()) if int(valid.sum()) else np.nan
            ),
            "first 1m low mean": (
                float(low_return.loc[valid].mean()) if valid.any() else np.nan
            ),
            "selected first 1m low mean": (
                float(low_return.loc[hit].mean()) if hit.any() else np.nan
            ),
            "same 1m low+TP count": int(same_minute_tp.sum()),
            "same 1m low+TP / low group": (
                float(same_minute_tp.sum() / hit.sum()) if hit.any() else np.nan
            ),
            "same 1m low+TP / all trades": (
                float(same_minute_tp.mean()) if len(same_minute_tp) else np.nan
            ),
            "final TP hit rate": (
                float(selected["tp_hit"].mean()) if len(selected) else np.nan
            ),
            "TP hit count": len(recovered),
            "hit TP prediction mean": (
                float(recovered["prediction"].mean())
                if len(recovered)
                else np.nan
            ),
            "TP first H1 count": int(selected["exit_type"].eq("tp_h1").sum()),
            "TP first H1 rate": (
                float(selected["exit_type"].eq("tp_h1").mean())
                if len(selected)
                else np.nan
            ),
            "TP first H2 count": int(selected["exit_type"].eq("tp_h2").sum()),
            "TP first H2 rate": (
                float(selected["exit_type"].eq("tp_h2").mean())
                if len(selected)
                else np.nan
            ),
            "TP first H3 count": int(selected["exit_type"].eq("tp_h3").sum()),
            "TP first H3 rate": (
                float(selected["exit_type"].eq("tp_h3").mean())
                if len(selected)
                else np.nan
            ),
            "no TP count": len(no_tp),
            "no TP close H3 mean": (
                float(no_tp_close_h3.mean()) if len(no_tp) else np.nan
            ),
            "no TP close H3 median": (
                float(no_tp_close_h3.median()) if len(no_tp) else np.nan
            ),
            "no TP close H3 Q10": (
                float(no_tp_close_h3.quantile(0.10)) if len(no_tp) else np.nan
            ),
            "no TP close H3 Q90": (
                float(no_tp_close_h3.quantile(0.90)) if len(no_tp) else np.nan
            ),
            "no TP close H3 E[net]": (
                float(no_tp_close_h3.mean() - trade_cost)
                if len(no_tp)
                else np.nan
            ),
        }
        for step in (1, 2, 3):
            close_return = selected[f"close_h{step}"]
            row[f"close H{step} mean"] = (
                float(close_return.mean()) if len(selected) else np.nan
            )
            row[f"close H{step} median"] = (
                float(close_return.median()) if len(selected) else np.nan
            )
            row[f"P(close H{step} > 0)"] = (
                float(close_return.gt(0.0).mean()) if len(selected) else np.nan
            )
        row["large loss count"] = int(large_loss.sum())
        row["large loss rate"] = (
            float(large_loss.mean()) if len(large_loss) else np.nan
        )
        row["win count"] = int(win.sum())
        row["win rate among matched"] = (
            float(win.sum() / valid.sum()) if int(valid.sum()) else np.nan
        )
        row["1m SL"] = float(simulated_stop_loss)
        row["1m SL triggered"] = int(first_minute_stop.sum())
        row["1m SL trigger rate"] = (
            float(first_minute_stop.sum() / valid.sum())
            if int(valid.sum())
            else np.nan
        )
        row["losers stopped"] = int(stopped_losses.sum())
        row["P(stop | loss)"] = (
            float(stopped_losses.sum() / trades["net_return"].lt(0.0).sum())
            if trades["net_return"].lt(0.0).any()
            else np.nan
        )
        row["winners stopped"] = int(stopped_wins.sum())
        row["P(stop | win)"] = (
            float(stopped_wins.sum() / trades["net_return"].gt(0.0).sum())
            if trades["net_return"].gt(0.0).any()
            else np.nan
        )
        row["loss saved sum"] = float(stopped_delta.loc[stopped_losses].sum())
        row["win profit forfeited sum"] = float(
            -stopped_delta.loc[stopped_wins].sum()
        )
        row["net delta sum"] = float(stopped_delta.sum())
        row["E[net] delta all trades"] = (
            float(stopped_delta.sum() / len(trades)) if len(trades) else np.nan
        )
        row["original E[net]"] = (
            float(trades["net_return"].mean()) if len(trades) else np.nan
        )
        row["with 1m SL E[net]"] = (
            float(
                (trades["net_return"].sum() + stopped_delta.sum()) / len(trades)
            )
            if len(trades)
            else np.nan
        )
        for name, values in (
            ("win first 1m low", win_low),
            ("large loss first 1m low", large_loss_low),
            ("large loss first 1m close", large_loss_close),
        ):
            row[f"{name} mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{name} min"] = float(values.min()) if len(values) else np.nan
            for quantile_level in (0.10, 0.25, 0.50, 0.75, 0.90):
                quantile_name = int(100 * quantile_level)
                row[f"{name} Q{quantile_name}"] = (
                    float(values.quantile(quantile_level))
                    if len(values)
                    else np.nan
                )
            row[f"{name} max"] = float(values.max()) if len(values) else np.nan
        rows.append(
            row
        )
    return rows


def _high_h1_loss_severity_summary(
    result: SplitResult,
    large_loss_threshold: float,
) -> list[dict[str, float | int | str]]:
    """Compare H1 favorable excursion across later outcome severity groups."""
    trades = result.trades
    still_open = ~trades["exit_type"].str.endswith("_h1")
    groups = {
        "later win": still_open & trades["net_return"].gt(0.0),
        "small loss": (
            still_open
            & trades["net_return"].lt(0.0)
            & trades["net_return"].gt(float(large_loss_threshold))
        ),
        f"large loss <= {100.0 * large_loss_threshold:+.2f}%": (
            still_open & trades["net_return"].le(float(large_loss_threshold))
        ),
    }
    unresolved_count = max(int(still_open.sum()), 1)
    rows: list[dict[str, float | int | str]] = []
    for group_name, mask in groups.items():
        selected = trades.loc[mask].copy()
        high_h1 = selected["high_h1"]
        tp_progress = high_h1.div(selected["prediction"])
        rows.append(
            {
                "split": result.split,
                "outcome group": group_name,
                "n": len(selected),
                "share of still-open H1": len(selected) / unresolved_count,
                "high H1 mean": float(high_h1.mean()) if len(selected) else np.nan,
                "high H1 Q25": (
                    float(high_h1.quantile(0.25)) if len(selected) else np.nan
                ),
                "high H1 median": (
                    float(high_h1.median()) if len(selected) else np.nan
                ),
                "high H1 Q75": (
                    float(high_h1.quantile(0.75)) if len(selected) else np.nan
                ),
                "high H1 Q90": (
                    float(high_h1.quantile(0.90)) if len(selected) else np.nan
                ),
                "TP progress median": (
                    float(tp_progress.median()) if len(selected) else np.nan
                ),
                "P(H1 < 25% TP)": (
                    float(tp_progress.lt(0.25).mean()) if len(selected) else np.nan
                ),
                "P(H1 < 50% TP)": (
                    float(tp_progress.lt(0.50).mean()) if len(selected) else np.nan
                ),
            }
        )
    return rows


def _tp_level_summary(
    results: list[SplitResult],
    large_loss_threshold: float,
) -> list[dict[str, float | int | str]]:
    """Apply Val-derived predicted-TP quintile bands to Val and Test."""
    val_result = next((result for result in results if result.split == "val"), None)
    if val_result is None or val_result.trades.empty:
        return []
    cut_points = np.quantile(
        val_result.trades["prediction"].to_numpy(float),
        [0.20, 0.40, 0.60, 0.80],
    )
    rows: list[dict[str, float | int | str]] = []
    for result in results:
        trades = result.trades
        band_ids = np.searchsorted(
            cut_points,
            trades["prediction"].to_numpy(float),
            side="right",
        )
        for band_id in range(5):
            selected = trades.iloc[np.flatnonzero(band_ids == band_id)]
            rows.append(
                {
                    "split": result.split,
                    "Val TP band": f"Q{20 * band_id}-Q{20 * (band_id + 1)}",
                    "n": len(selected),
                    "share": len(selected) / max(len(trades), 1),
                    "pred TP mean": (
                        float(selected["prediction"].mean())
                        if len(selected)
                        else np.nan
                    ),
                    "TP hit rate": (
                        float(selected["tp_hit"].mean()) if len(selected) else np.nan
                    ),
                    "loss rate": (
                        float(selected["net_return"].lt(0.0).mean())
                        if len(selected)
                        else np.nan
                    ),
                    "large-loss rate": (
                        float(
                            selected["net_return"]
                            .le(float(large_loss_threshold))
                            .mean()
                        )
                        if len(selected)
                        else np.nan
                    ),
                    "E[net]": (
                        float(selected["net_return"].mean())
                        if len(selected)
                        else np.nan
                    ),
                }
            )
    return rows


def _render(
    summaries: list[dict[str, float | int | str]],
    clustering_summaries: list[dict[str, float | int | str]],
    loss_run_summaries: list[dict[str, str]],
    h1_close_summaries: list[dict[str, float | int | str]],
    h1_threshold_summaries: list[dict[str, float | int | str]],
    loss_h1_vs_h3_summaries: list[dict[str, float | int | str]],
    h2_vs_h3_summaries: list[dict[str, float | int | str]],
    high_h1_severity_summaries: list[dict[str, float | int | str]],
    tp_level_summaries: list[dict[str, float | int | str]],
    results: list[SplitResult],
    output: Path,
    title: str,
) -> None:
    display = pd.DataFrame(summaries)
    for column in (
        "selection rate",
        "pred TP mean",
        "TP hit rate",
        "SL hit rate",
        "close exit rate",
        "close exit mean",
        "gross mean",
        "E[net]",
        "net win rate",
    ):
        display[column] = display[column].map(
            lambda value: "n/a" if pd.isna(value) else f"{100.0 * float(value):+.3f}%"
        )
    for column in display.columns:
        if column.startswith("TP first H") or column.startswith("TP by H"):
            display[column] = display[column].map(
                lambda value: "n/a"
                if pd.isna(value)
                else f"{100.0 * float(value):+.3f}%"
            )
    display["avg trades/day"] = display["avg trades/day"].map(
        lambda value: f"{float(value):.2f}"
    )

    clustering_display = pd.DataFrame(clustering_summaries)
    for column in (
        "loss rate",
        "P(next loss | loss)",
        "post-loss signal coverage",
        "post-loss loss rate",
        "loss-rate delta",
        "baseline E[net]",
        "post-loss E[net]",
        "E[net] delta",
        "losses in runs>=2",
    ):
        clustering_display[column] = clustering_display[column].map(
            lambda value: "n/a"
            if pd.isna(value)
            else f"{100.0 * float(value):+.3f}%"
        )
    loss_run_display = pd.DataFrame(loss_run_summaries)
    h1_close_display = pd.DataFrame(h1_close_summaries)
    for column in (
        "share",
        "close H1 mean",
        "close H1 Q10",
        "close H1 Q25",
        "close H1 median",
        "close H1 Q75",
        "close H1 Q90",
        "P(close H1 > 0)",
    ):
        h1_close_display[column] = h1_close_display[column].map(
            lambda value: "n/a"
            if pd.isna(value)
            else f"{100.0 * float(value):+.3f}%"
        )
    for column in h1_close_display.columns:
        if column.startswith("P(low H1 < "):
            h1_close_display[column] = h1_close_display[column].map(
                lambda value: "n/a"
                if pd.isna(value)
                else f"{100.0 * float(value):+.3f}%"
            )
    h1_threshold_display = pd.DataFrame(h1_threshold_summaries)
    for column in (
        "share of still-open H1",
        "P(win)",
        "P(loss)",
        "held E[net]",
        "close H1 mean",
    ):
        h1_threshold_display[column] = h1_threshold_display[column].map(
            lambda value: "n/a"
            if pd.isna(value)
            else f"{100.0 * float(value):+.3f}%"
        )
    loss_h1_vs_h3_display = pd.DataFrame(loss_h1_vs_h3_summaries)
    for column in (
        "share of trades",
        "close H1 gross mean",
        "close H3 gross mean",
        "H1 improvement mean",
        "H1 improvement median",
        "P(H1 better than H3)",
        "H1 E[net]",
        "H3 E[net]",
        "H1 large-loss rate",
        "H3 large-loss rate",
    ):
        loss_h1_vs_h3_display[column] = loss_h1_vs_h3_display[column].map(
            lambda value: "n/a"
            if pd.isna(value)
            else f"{100.0 * float(value):+.3f}%"
        )
    h2_vs_h3_display = pd.DataFrame(h2_vs_h3_summaries)
    for column in (
        "share of trades",
        "H3 TP rate",
        "close H2 gross mean",
        "hold H3 gross mean",
        "H2 minus H3 mean",
        "P(H2 better than H3)",
        "close H2 E[net]",
        "hold H3 E[net]",
        "close H2 win rate",
        "hold H3 win rate",
        "whole-strategy E[net] delta",
    ):
        h2_vs_h3_display[column] = h2_vs_h3_display[column].map(
            lambda value: "n/a"
            if pd.isna(value)
            else f"{100.0 * float(value):+.3f}%"
        )
    high_h1_display = pd.DataFrame(high_h1_severity_summaries)
    for column in (
        "share of still-open H1",
        "high H1 mean",
        "high H1 Q25",
        "high H1 median",
        "high H1 Q75",
        "high H1 Q90",
        "TP progress median",
        "P(H1 < 25% TP)",
        "P(H1 < 50% TP)",
    ):
        high_h1_display[column] = high_h1_display[column].map(
            lambda value: "n/a"
            if pd.isna(value)
            else f"{100.0 * float(value):+.3f}%"
        )
    tp_level_display = pd.DataFrame(tp_level_summaries)
    for column in (
        "share",
        "pred TP mean",
        "TP hit rate",
        "loss rate",
        "large-loss rate",
        "E[net]",
    ):
        tp_level_display[column] = tp_level_display[column].map(
            lambda value: "n/a"
            if pd.isna(value)
            else f"{100.0 * float(value):+.3f}%"
        )

    fig = plt.figure(figsize=(19, 26), constrained_layout=True)
    grid = fig.add_gridspec(
        11,
        1,
        height_ratios=[0.8, 0.8, 0.7, 1.3, 0.9, 0.8, 0.8, 1.1, 1.2, 1.5, 1.5],
    )
    ax_table = fig.add_subplot(grid[0])
    ax_clustering = fig.add_subplot(grid[1])
    ax_loss_runs = fig.add_subplot(grid[2])
    ax_h1_close = fig.add_subplot(grid[3])
    ax_h1_threshold = fig.add_subplot(grid[4])
    ax_loss_h1_vs_h3 = fig.add_subplot(grid[5])
    ax_h2_vs_h3 = fig.add_subplot(grid[6])
    ax_high_h1 = fig.add_subplot(grid[7])
    ax_tp_level = fig.add_subplot(grid[8])
    ax_curve = fig.add_subplot(grid[9])
    ax_scatter = fig.add_subplot(grid[10])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.45)
    ax_table.set_title(title, fontsize=12, pad=12)

    ax_clustering.axis("off")
    clustering_table = ax_clustering.table(
        cellText=clustering_display.values,
        colLabels=clustering_display.columns,
        loc="center",
        cellLoc="center",
    )
    clustering_table.auto_set_font_size(False)
    clustering_table.set_fontsize(7.5)
    clustering_table.scale(1.0, 1.45)
    ax_clustering.set_title(
        "Loss clustering after realized exit | next 5 complete bars",
        fontsize=11,
        pad=10,
    )

    ax_loss_runs.axis("off")
    loss_run_table = ax_loss_runs.table(
        cellText=loss_run_display.values,
        colLabels=loss_run_display.columns,
        loc="center",
        cellLoc="center",
    )
    loss_run_table.auto_set_font_size(False)
    loss_run_table.set_fontsize(7.2)
    loss_run_table.scale(1.0, 1.45)
    ax_loss_runs.set_title(
        "Consecutive net-loss runs | each cell: run count | share of all losing trades",
        fontsize=11,
        pad=10,
    )

    ax_h1_close.axis("off")
    h1_close_table = ax_h1_close.table(
        cellText=h1_close_display.values,
        colLabels=h1_close_display.columns,
        loc="center",
        cellLoc="center",
    )
    h1_close_table.auto_set_font_size(False)
    h1_close_table.set_fontsize(8)
    h1_close_table.scale(1.0, 1.35)
    ax_h1_close.set_title(
        "Close H1 return: all trades and positions still open after H1",
        fontsize=11,
        pad=10,
    )

    ax_h1_threshold.axis("off")
    h1_threshold_table = ax_h1_threshold.table(
        cellText=h1_threshold_display.values,
        colLabels=h1_threshold_display.columns,
        loc="center",
        cellLoc="center",
    )
    h1_threshold_table.auto_set_font_size(False)
    h1_threshold_table.set_fontsize(8)
    h1_threshold_table.scale(1.0, 1.35)
    ax_h1_threshold.set_title(
        "Final outcome conditional on close H1 threshold | positions still open after H1",
        fontsize=11,
        pad=10,
    )

    ax_loss_h1_vs_h3.axis("off")
    loss_h1_vs_h3_table = ax_loss_h1_vs_h3.table(
        cellText=loss_h1_vs_h3_display.values,
        colLabels=loss_h1_vs_h3_display.columns,
        loc="center",
        cellLoc="center",
    )
    loss_h1_vs_h3_table.auto_set_font_size(False)
    loss_h1_vs_h3_table.set_fontsize(7.4)
    loss_h1_vs_h3_table.scale(1.0, 1.35)
    ax_loss_h1_vs_h3.set_title(
        "No-SL losing close exits: hypothetical close H1 versus actual close H3",
        fontsize=11,
        pad=10,
    )

    ax_h2_vs_h3.axis("off")
    h2_vs_h3_table = ax_h2_vs_h3.table(
        cellText=h2_vs_h3_display.values,
        colLabels=h2_vs_h3_display.columns,
        loc="center",
        cellLoc="center",
    )
    h2_vs_h3_table.auto_set_font_size(False)
    h2_vs_h3_table.set_fontsize(7.0)
    h2_vs_h3_table.scale(1.0, 1.35)
    ax_h2_vs_h3.set_title(
        "Trades unresolved after H2: exit close H2 versus hold for TP/close H3",
        fontsize=11,
        pad=10,
    )

    ax_high_h1.axis("off")
    high_h1_table = ax_high_h1.table(
        cellText=high_h1_display.values,
        colLabels=high_h1_display.columns,
        loc="center",
        cellLoc="center",
    )
    high_h1_table.auto_set_font_size(False)
    high_h1_table.set_fontsize(7.5)
    high_h1_table.scale(1.0, 1.35)
    ax_high_h1.set_title(
        "High H1 versus later outcome severity | positions still open after H1",
        fontsize=11,
        pad=10,
    )

    ax_tp_level.axis("off")
    tp_level_table = ax_tp_level.table(
        cellText=tp_level_display.values,
        colLabels=tp_level_display.columns,
        loc="center",
        cellLoc="center",
    )
    tp_level_table.auto_set_font_size(False)
    tp_level_table.set_fontsize(8)
    tp_level_table.scale(1.0, 1.25)
    ax_tp_level.set_title(
        "Outcome by predicted dynamic-TP level | quintile thresholds fitted on Val",
        fontsize=11,
        pad=10,
    )

    colors = {"val": "#2563eb", "test": "#dc2626"}
    for result in results:
        if result.trades.empty:
            continue
        color = colors.get(result.split, None)
        ax_curve.plot(
            result.trades.index,
            100.0 * result.trades["cum_net"],
            label=result.split,
            color=color,
            linewidth=1.1,
        )
        sample = result.trades
        if len(sample) > 20000:
            sample = sample.iloc[:: int(np.ceil(len(sample) / 20000))]
        ax_scatter.scatter(
            100.0 * sample["prediction"],
            100.0 * sample["actual_mfe"],
            s=4,
            alpha=0.18,
            label=result.split,
            color=color,
        )
    ax_curve.axhline(0.0, color="black", linewidth=0.8)
    ax_curve.set_ylabel("Cumulative net return (%)")
    ax_curve.set_title("Cumulative return in chronological trade order")
    ax_curve.grid(alpha=0.2)
    ax_curve.legend()
    ax_scatter.set_xlabel("Predicted MFE Q (%)")
    ax_scatter.set_ylabel("Actual MFE (%)")
    ax_scatter.set_title("Predicted dynamic TP versus actual MFE")
    ax_scatter.grid(alpha=0.2)
    ax_scatter.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _render_random_large_loss_windows(
    raw: pd.DataFrame,
    results: list[SplitResult],
    *,
    bar_delta: pd.Timedelta,
    output: Path,
    bars_per_panel: int,
    panel_count: int,
    random_seed: int,
    chart_style: str = "line",
    bear_selected_index: pd.Index | None = None,
    bear_top_fraction: float = 0.40,
    bear_precision: float = np.nan,
    bear_recall: float = np.nan,
    slope_selected_index: pd.Index | None = None,
    slope_top_fraction: float = 0.10,
    loss_threshold: float = -0.001,
) -> None:
    """Plot random price windows containing gross losses beyond a threshold."""
    marked_parts: list[pd.DataFrame] = []
    selected_entry_parts: list[pd.DatetimeIndex] = []
    outcome_parts: list[pd.DataFrame] = []
    for result in results:
        result_entry_times = pd.DatetimeIndex(result.trades.index) + bar_delta
        selected_entry_parts.append(result_entry_times)
        outcomes = pd.DataFrame(
            {
                "entry_time": result_entry_times,
                "net_return": result.trades["net_return"].to_numpy(float),
                "gross_return": result.trades["gross_return"].to_numpy(float),
            }
        )
        outcome_parts.append(outcomes)
        marked = result.trades.loc[
            result.trades["gross_return"].lt(float(loss_threshold)),
            ["entry_open", "gross_return"],
        ].copy()
        marked["split"] = result.split
        marked["entry_time"] = pd.DatetimeIndex(marked.index) + bar_delta
        marked_parts.append(marked)
    marked_trades = pd.concat(marked_parts, axis=0, ignore_index=True)
    selected_entry_index = pd.DatetimeIndex([])
    for part in selected_entry_parts:
        selected_entry_index = selected_entry_index.union(part)
    outcome_frame = pd.concat(outcome_parts, ignore_index=True)
    winning_entry_times = set(
        pd.DatetimeIndex(outcome_frame.loc[outcome_frame["net_return"].gt(0.0), "entry_time"])
    )
    large_loss_entry_times = set(
        pd.DatetimeIndex(
            outcome_frame.loc[
                outcome_frame["gross_return"].lt(float(loss_threshold)),
                "entry_time",
            ]
        )
    )
    marked_trades = marked_trades.dropna(subset=["entry_time", "entry_open"])
    if marked_trades.empty:
        logger.warning("No gross-loss trades available for random-window plot.")
        return

    price = pd.to_numeric(raw["close"], errors="coerce")
    price_index = pd.DatetimeIndex(raw.index)
    index_positions = pd.Series(np.arange(len(price_index)), index=price_index)
    marked_trades["entry_pos"] = marked_trades["entry_time"].map(index_positions)
    marked_trades = marked_trades.dropna(subset=["entry_pos"])
    if marked_trades.empty:
        logger.warning("Gross-loss entry timestamps do not align with price data.")
        return

    bars_per_panel = min(int(bars_per_panel), len(raw))
    rng = np.random.default_rng(int(random_seed))
    anchor_order = rng.permutation(len(marked_trades))
    starts: list[int] = []
    for anchor_idx in anchor_order:
        anchor_pos = int(marked_trades.iloc[int(anchor_idx)]["entry_pos"])
        offset = int(rng.integers(0, bars_per_panel))
        start = int(np.clip(anchor_pos - offset, 0, len(raw) - bars_per_panel))
        if all(abs(start - previous) >= bars_per_panel // 2 for previous in starts):
            starts.append(start)
        if len(starts) >= int(panel_count):
            break
    if len(starts) < int(panel_count):
        for anchor_idx in anchor_order:
            anchor_pos = int(marked_trades.iloc[int(anchor_idx)]["entry_pos"])
            start = int(
                np.clip(anchor_pos - bars_per_panel // 2, 0, len(raw) - bars_per_panel)
            )
            if start not in starts:
                starts.append(start)
            if len(starts) >= int(panel_count):
                break

    fig, axes = plt.subplots(
        len(starts),
        1,
        figsize=(20, 4.2 * len(starts)),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    for panel_number, (ax, start) in enumerate(zip(axes_array, starts), start=1):
        stop = start + bars_per_panel
        segment = price.iloc[start:stop]
        candle_segment = raw.iloc[start:stop]
        start_time = segment.index[0]
        end_time = segment.index[-1]
        entries = marked_trades[
            marked_trades["entry_time"].between(start_time, end_time)
        ]
        selected_mask = pd.Series(
            segment.index.isin(selected_entry_index),
            index=segment.index,
            dtype=bool,
        )
        selected_run_id = selected_mask.ne(
            selected_mask.shift(fill_value=False)
        ).cumsum()
        first_selected_region = True
        for _, region in selected_mask[selected_mask].groupby(
            selected_run_id[selected_mask]
        ):
            ax.axvspan(
                region.index[0] - bar_delta / 2,
                region.index[-1] + bar_delta / 2,
                color="#22c55e",
                alpha=0.13,
                linewidth=0.0,
                label="strategy-selected entry region" if first_selected_region else None,
                zorder=0,
            )
            first_selected_region = False
        if bear_selected_index is not None:
            bear_mask = pd.Series(
                segment.index.isin(bear_selected_index),
                index=segment.index,
                dtype=bool,
            )
            run_id = bear_mask.ne(bear_mask.shift(fill_value=False)).cumsum()
            first_bear_region = True
            for _, region in bear_mask[bear_mask].groupby(run_id[bear_mask]):
                region_start = region.index[0] - bar_delta / 2
                region_end = region.index[-1] + bar_delta / 2
                ax.axvspan(
                    region_start,
                    region_end,
                    color="#ef4444",
                    alpha=0.12,
                    linewidth=0.0,
                    label=(
                        f"Bear rank 1 top {100.0 * bear_top_fraction:.0f}%"
                        if first_bear_region
                        else None
                    ),
                    zorder=0,
                )
                first_bear_region = False
        if slope_selected_index is not None:
            slope_mask = pd.Series(
                segment.index.isin(slope_selected_index),
                index=segment.index,
                dtype=bool,
            )
            slope_run_id = slope_mask.ne(
                slope_mask.shift(fill_value=False)
            ).cumsum()
            first_slope_region = True
            for _, region in slope_mask[slope_mask].groupby(
                slope_run_id[slope_mask]
            ):
                ax.axvspan(
                    region.index[0] - bar_delta / 2,
                    region.index[-1] + bar_delta / 2,
                    color="#facc15",
                    alpha=0.28,
                    linewidth=0.0,
                    label=(
                        f"MA3 reversal top {100.0 * slope_top_fraction:.0f}% + MA3 slope > 0"
                        if first_slope_region
                        else None
                    ),
                    zorder=1,
                )
                first_slope_region = False
        if chart_style == "candles":
            x_values = mdates.date2num(pd.DatetimeIndex(candle_segment.index).to_pydatetime())
            width = max(bar_delta.total_seconds() / 86400.0 * 0.72, 1e-9)
            open_values = pd.to_numeric(candle_segment["open"], errors="coerce").to_numpy(float)
            high_values = pd.to_numeric(candle_segment["high"], errors="coerce").to_numpy(float)
            low_values = pd.to_numeric(candle_segment["low"], errors="coerce").to_numpy(float)
            close_values = pd.to_numeric(candle_segment["close"], errors="coerce").to_numpy(float)
            finite_prices = np.concatenate(
                [open_values, high_values, low_values, close_values]
            )
            finite_prices = finite_prices[np.isfinite(finite_prices)]
            min_body_height = (
                max(float(np.ptp(finite_prices)) * 0.0006, 1e-8)
                if len(finite_prices)
                else 1e-8
            )
            wick_segments = []
            wick_colors = []
            body_patches = []
            body_faces = []
            body_edges = []
            for timestamp, x, open_price, high_price, low_price, close_price in zip(
                candle_segment.index,
                x_values,
                open_values,
                high_values,
                low_values,
                close_values,
            ):
                if not np.all(
                    np.isfinite([open_price, high_price, low_price, close_price])
                ):
                    continue
                candle_time = pd.Timestamp(timestamp)
                is_win_entry = candle_time in winning_entry_times
                is_large_loss_entry = candle_time in large_loss_entry_times
                if is_win_entry:
                    edge, face = "#15803d", "#22c55e"
                elif is_large_loss_entry:
                    edge, face = "#b91c1c", "#ef4444"
                else:
                    edge, face = "#475569", "#ffffff"
                wick_segments.append([(x, low_price), (x, high_price)])
                wick_colors.append(edge)
                body_low = min(open_price, close_price)
                body_height = max(abs(close_price - open_price), min_body_height)
                body_patches.append(
                    Rectangle((x - width / 2.0, body_low), width, body_height)
                )
                body_faces.append(face)
                body_edges.append(edge)
            ax.add_collection(
                LineCollection(
                    wick_segments,
                    colors=wick_colors,
                    linewidths=0.65,
                    zorder=2,
                )
            )
            ax.add_collection(
                PatchCollection(
                    body_patches,
                    facecolors=body_faces,
                    edgecolors=body_edges,
                    linewidths=0.65,
                    zorder=3,
                    match_original=False,
                )
            )
            ax.xaxis_date()
            ax.autoscale_view()
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            legend_handles.extend(
                [
                    Patch(facecolor="#ffffff", edgecolor="#475569", label="regular candle"),
                    Patch(
                        facecolor="#22c55e",
                        edgecolor="#15803d",
                        label="winning trade entry",
                    ),
                    Patch(
                        facecolor="#ef4444",
                        edgecolor="#b91c1c",
                        label=f"gross loss > {abs(100.0 * loss_threshold):.2f}%",
                    ),
                ]
            )
            legend_labels.extend(
                ["regular candle", "winning trade entry", "large gross-loss entry"]
            )
            ax.legend(legend_handles, legend_labels, loc="upper left", fontsize=8)
        else:
            ax.plot(
                segment.index,
                segment.to_numpy(float),
                color="#334155",
                linewidth=0.9,
            )
            ax.scatter(
                entries["entry_time"],
                entries["entry_open"],
                marker="v",
                s=38,
                color="#dc2626",
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
                label=f"gross loss > {abs(100.0 * loss_threshold):.2f}%",
            )
            ax.legend(loc="upper left", fontsize=8)
        ax.set_title(
            f"Panel {panel_number} | {start_time} to {end_time} | "
            f"{len(segment):,} bars | marked entries={len(entries):,}",
            fontsize=10,
        )
        ax.set_ylabel("BTC price")
        ax.grid(alpha=0.2)
    axes_array[-1].set_xlabel("Time")
    overlay_metrics = ""
    if bear_selected_index is not None:
        overlay_metrics = (
            f" | Bear precision={bear_precision:.2%}, recall={bear_recall:.2%}"
        )
    if slope_selected_index is not None:
        overlay_metrics += (
            f" | yellow=MA3 reversal top {100.0 * slope_top_fraction:.0f}%"
            " and MA3 slope > 0"
        )
    fig.suptitle(
        "Random 5m price windows containing trades with gross loss > 0.10%\n"
        "Green regions show strategy-selected entries; "
        + (
            "green candles show winning entries; red candles only show gross losses "
            f"> {abs(100.0 * loss_threshold):.2f}%; "
            if chart_style == "candles"
            else "red markers show gross-loss entries; "
        )
        + f"trade cost {'included in candle outcome' if chart_style == 'candles' else 'excluded'}"
        + overlay_metrics,
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _ma3_positive_slope(close: pd.Series) -> pd.Series:
    """Observable eligibility gate used by the MA3 reversal Long label."""
    numeric = pd.to_numeric(close, errors="coerce")
    ma3 = numeric.rolling(int(config.MA_SLOPE_FAST_WINDOW)).mean()
    slope = ma3 - ma3.shift(int(config.MA_SLOPE_FAST_SHIFT))
    return slope.gt(0.0).where(slope.notna())


def _ma_slope_trade_outcome_summary(
    results: list[SplitResult],
    raw: pd.DataFrame,
    *,
    large_loss_threshold: float = -0.001,
) -> list[dict[str, float | int | str]]:
    """Compare observable MA3/MA10 slopes for wins and large gross losses."""
    close = pd.to_numeric(raw["close"], errors="coerce")
    definitions = (("MA3", 3, 2), ("MA10", 10, 3))
    slope_pct: dict[str, pd.Series] = {}
    for name, window, shift in definitions:
        moving_average = close.rolling(window).mean()
        previous = moving_average.shift(shift)
        slope_pct[name] = moving_average.sub(previous).div(previous)

    split_trades = [(result.split, result.trades) for result in results]
    combined = pd.concat(
        [result.trades for result in results],
        axis=0,
    ).sort_index()
    split_trades.append(("val+test", combined))

    rows: list[dict[str, float | int | str]] = []
    for split_name, trades in split_trades:
        groups = (
            ("win (net > 0)", trades["net_return"].gt(0.0)),
            (
                "gross loss > 0.10%",
                trades["gross_return"].lt(float(large_loss_threshold)),
            ),
        )
        for group_name, group_mask in groups:
            selected_index = trades.index[group_mask]
            for ma_name, _, _ in definitions:
                values = slope_pct[ma_name].reindex(selected_index).dropna()
                quantiles = values.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
                rows.append(
                    {
                        "split": split_name,
                        "group": group_name,
                        "slope": f"{ma_name} pct",
                        "n": len(values),
                        "mean": float(values.mean()),
                        "Q10": float(quantiles.loc[0.10]),
                        "Q25": float(quantiles.loc[0.25]),
                        "median": float(quantiles.loc[0.50]),
                        "Q75": float(quantiles.loc[0.75]),
                        "Q90": float(quantiles.loc[0.90]),
                        "P(slope > 0)": float(values.gt(0.0).mean()),
                    }
                )
    return rows


def _slope_reversal_band_recall(
    bundle: Any,
    raw: pd.DataFrame,
    *,
    band_step: float = 0.05,
) -> list[dict[str, float | int | str]]:
    """Recall contribution by score band inside the observable Long gate.

    Cumulative top-fraction cutoffs are fitted on eligible Val rows. Identical
    score cutoffs are then applied to Test. Recall always divides by all label
    positives among rows where the current MA3 slope is positive.
    """
    step = float(band_step)
    if not 0.0 < step <= 1.0:
        raise ValueError("slope recall band step must be in (0, 1].")
    fractions = np.arange(step, 1.0 + step / 2.0, step)
    slope_positive = _ma3_positive_slope(raw["close"])

    split_data: dict[str, pd.DataFrame] = {}
    for split_name, signal in (("val", bundle.val), ("test", bundle.test)):
        data = signal.data.copy()
        eligible = slope_positive.reindex(data.index).fillna(False).astype(bool)
        split_data[split_name] = data.loc[eligible].sort_values(
            "pred", ascending=False
        )
    val_data = split_data["val"]
    if val_data.empty:
        return []

    cutoffs: list[float] = []
    for fraction in fractions:
        count = min(
            len(val_data),
            max(1, int(np.ceil(len(val_data) * float(fraction)))),
        )
        cutoffs.append(float(val_data.iloc[count - 1]["pred"]))

    rows: list[dict[str, float | int | str]] = []
    for split_name, data in split_data.items():
        total_positive = int(data["label"].eq(1).sum())
        previous_index = pd.Index([])
        for band_number, (fraction, cutoff) in enumerate(
            zip(fractions, cutoffs, strict=True)
        ):
            cumulative = pd.Index(data.index[data["pred"].ge(cutoff)])
            band_index = cumulative.difference(previous_index, sort=False)
            band = data.reindex(band_index).dropna(subset=["label", "pred"])
            cumulative_data = data.reindex(cumulative).dropna(
                subset=["label", "pred"]
            )
            band_true = int(band["label"].eq(1).sum())
            cumulative_true = int(cumulative_data["label"].eq(1).sum())
            band_start = float(fraction - step)
            rows.append(
                {
                    "split": split_name,
                    "score band": (
                        f"top {100.0 * band_start:.0f}-{100.0 * fraction:.0f}%"
                    ),
                    "eligible MA3 slope > 0": len(data),
                    "eligible positives": total_positive,
                    "band n": len(band),
                    "band true": band_true,
                    "band precision": (
                        band_true / len(band) if len(band) else np.nan
                    ),
                    "band recall contribution": (
                        band_true / total_positive if total_positive else np.nan
                    ),
                    "cumulative n": len(cumulative_data),
                    "cumulative recall": (
                        cumulative_true / total_positive
                        if total_positive
                        else np.nan
                    ),
                    "Val cutoff": cutoff,
                }
            )
            previous_index = cumulative
    return rows


def _render_slope_reversal_recall_table(
    rows: list[dict[str, float | int | str]],
    output: Path,
) -> None:
    """Save the MA3-positive score-band recall table as a compact image."""
    if not rows:
        return
    frame = pd.DataFrame(rows).copy()
    display = frame[
        [
            "split",
            "score band",
            "eligible MA3 slope > 0",
            "eligible positives",
            "band n",
            "band true",
            "band precision",
            "band recall contribution",
            "cumulative n",
            "cumulative recall",
        ]
    ].copy()
    for column in (
        "band precision",
        "band recall contribution",
        "cumulative recall",
    ):
        display[column] = display[column].map(
            lambda value: "-" if pd.isna(value) else f"{float(value):.2%}"
        )
    for column in (
        "eligible MA3 slope > 0",
        "eligible positives",
        "band n",
        "band true",
        "cumulative n",
    ):
        display[column] = display[column].map(lambda value: f"{int(value):,}")

    figure_height = max(5.0, 0.34 * len(display) + 1.8)
    fig, ax = plt.subplots(figsize=(18, figure_height))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#fff7ed")
    ax.set_title(
        "MA3 slope-reversal Rank 1 recall by score band\n"
        "Cutoffs fitted on MA3-slope-positive Val rows and applied unchanged "
        "to Test",
        fontsize=13,
        pad=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _bear_loss_overlap_summary(
    results: list[SplitResult],
    bear_selected_by_split: dict[str, pd.Index],
    *,
    loss_threshold: float = -0.001,
) -> list[dict[str, float | int | str]]:
    """Measure Bear-selection precision/recall for large MFE trade losses."""
    rows: list[dict[str, float | int | str]] = []
    combined_red = pd.Index([])
    combined_bear = pd.Index([])
    combined_trades = pd.Index([])
    for result in results:
        trades_index = pd.Index(result.trades.index)
        red_index = pd.Index(
            result.trades.index[
                result.trades["gross_return"].lt(float(loss_threshold))
            ]
        )
        bear_index = pd.Index(bear_selected_by_split.get(result.split, pd.Index([])))
        overlap = red_index.intersection(bear_index)
        bear_on_trades = bear_index.intersection(trades_index)
        rows.append(
            {
                "split": result.split,
                "red losses": len(red_index),
                "Bear selected": len(bear_index),
                "overlap": len(overlap),
                "recall": len(overlap) / len(red_index) if len(red_index) else np.nan,
                "precision": (
                    len(overlap) / len(bear_index) if len(bear_index) else np.nan
                ),
                "Bear selected among MFE trades": len(bear_on_trades),
                "precision | MFE trades": (
                    len(overlap) / len(bear_on_trades)
                    if len(bear_on_trades)
                    else np.nan
                ),
            }
        )
        combined_red = combined_red.union(red_index)
        combined_bear = combined_bear.union(bear_index)
        combined_trades = combined_trades.union(trades_index)
    combined_overlap = combined_red.intersection(combined_bear)
    combined_bear_on_trades = combined_bear.intersection(combined_trades)
    rows.append(
        {
            "split": "all",
            "red losses": len(combined_red),
            "Bear selected": len(combined_bear),
            "overlap": len(combined_overlap),
            "recall": (
                len(combined_overlap) / len(combined_red)
                if len(combined_red)
                else np.nan
            ),
            "precision": (
                len(combined_overlap) / len(combined_bear)
                if len(combined_bear)
                else np.nan
            ),
            "Bear selected among MFE trades": len(combined_bear_on_trades),
            "precision | MFE trades": (
                len(combined_overlap) / len(combined_bear_on_trades)
                if len(combined_bear_on_trades)
                else np.nan
            ),
        }
    )
    return rows


def run(args: argparse.Namespace) -> Path:
    archive_path = Path(args.archive)
    data_path = Path(args.data)
    metadata, entry = _load_archive(archive_path, args.rank)
    if config.canonical_label_mode(metadata.get("label_mode")) != "quantile_trade":
        raise ValueError("Archive label_mode must be quantile_trade.")
    if config.canonical_quantile_target(metadata.get("quantile_target")) != "mfe":
        raise ValueError("This backtest requires quantile_target=mfe.")
    horizons = [int(value) for value in metadata.get("horizons", [])]
    if len(horizons) != 1 or horizons[0] < 1:
        raise ValueError("This backtest requires exactly one positive archive horizon.")
    horizon = horizons[0]
    quantile = config.validate_quantile_alpha(metadata.get("quantile_alpha"))
    policy = _split_policy(metadata)
    purge = config.purge_bars_for_horizons(horizons)

    raw = load_ohlcv(data_path)
    bear_entry: dict[str, Any] | None = None
    bear_bundle = None
    bear_horizons: list[int] = []
    bear_policy: dict[str, Any] = {}
    bear_spec: ModelSpec | None = None
    filter_entry: dict[str, Any] | None = None
    filter_bundle = None
    filter_horizons: list[int] = []
    filter_policy: dict[str, Any] = {}
    filter_spec: ModelSpec | None = None
    slope_entry: dict[str, Any] | None = None
    slope_bundle = None
    slope_horizons: list[int] = []
    slope_policy: dict[str, Any] = {}
    slope_spec: ModelSpec | None = None
    if not args.no_bear_overlay:
        bear_path = Path(args.bear_archive)
        bear_metadata, bear_entry = _load_archive(bear_path, args.bear_rank)
        if config.canonical_label_mode(bear_metadata.get("label_mode")) != "bear":
            raise ValueError("--bear-archive must have label_mode=bear.")
        bear_horizons = _archive_horizons(
            bear_path,
            fallback=[1],
            label="Bear archive",
        )
        bear_policy = _split_policy(bear_metadata)
        bear_spec = ModelSpec(
            archive_path=bear_path,
            rank=int(args.bear_rank),
            label_mode="bear",
            label_threshold=float(bear_metadata.get("label_threshold", 0.0)),
            top_fraction=float(args.bear_top_fraction),
            label_direction=config.canonical_label_direction(
                str(bear_metadata.get("label_direction", "long"))
            ),
        )
    if args.and_filter_archive is not None:
        filter_path = Path(args.and_filter_archive)
        filter_metadata, filter_entry = _load_archive(
            filter_path,
            args.and_filter_rank,
        )
        filter_mode = config.canonical_label_mode(filter_metadata.get("label_mode"))
        if filter_mode != "adverse_floor":
            raise ValueError("--and-filter-archive must use label_mode=adverse_floor.")
        filter_horizons = _archive_horizons(
            filter_path,
            fallback=[1],
            label="AND filter archive",
        )
        filter_policy = _split_policy(filter_metadata)
        filter_spec = ModelSpec(
            archive_path=filter_path,
            rank=int(args.and_filter_rank),
            label_mode=filter_mode,
            label_threshold=float(filter_metadata.get("label_threshold", 0.0)),
            top_fraction=float(args.and_filter_top_fraction),
            label_direction=config.canonical_label_direction(
                str(filter_metadata.get("label_direction", "long"))
            ),
        )
    if not args.no_slope_reversal_overlay:
        slope_path = Path(args.slope_reversal_archive)
        slope_metadata, slope_entry = _load_archive(
            slope_path,
            args.slope_reversal_rank,
        )
        slope_mode = config.canonical_label_mode(
            slope_metadata.get("label_mode")
        )
        if slope_mode != "ma_slope_reversal":
            raise ValueError(
                "--slope-reversal-archive must use label_mode=ma_slope_reversal."
            )
        slope_horizons = _archive_horizons(
            slope_path,
            fallback=[1],
            label="MA slope-reversal archive",
        )
        slope_policy = _split_policy(slope_metadata)
        slope_spec = ModelSpec(
            archive_path=slope_path,
            rank=int(args.slope_reversal_rank),
            label_mode=slope_mode,
            label_threshold=0.0,
            top_fraction=float(args.slope_reversal_top_fraction),
            label_direction="long",
        )
    labeled = add_binary_labels(
        raw,
        horizons=horizons,
        label_mode="quantile_trade",
    )
    train, val, test = split_labeled_by_dates(
        labeled,
        val_start=policy["val_start"],
        test_start=policy["test_start"],
        test_end=policy["test_end"],
        purge_bars=purge,
    )
    wf_source = labeled[labeled.index < pd.Timestamp(policy["wf_end"])]
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

    feature_entries = [entry]
    if bear_entry is not None:
        feature_entries.append(bear_entry)
    if filter_entry is not None:
        feature_entries.append(filter_entry)
    if slope_entry is not None:
        feature_entries.append(slope_entry)
    windows = _required_windows_for_entries(feature_entries)
    logger.info("Building required feature windows: %s", windows)
    feature_frame = build_feature_frame(
        raw,
        windows=windows,
        quality_index=folds[0].train_df.index,
    )
    feature_space = CryptoFeatureSpace(
        feature_frame,
        selectable_features(feature_frame),
    )
    if bear_spec is not None and bear_entry is not None:
        bear_bundle = _train_spec_bundle(
            spec=bear_spec,
            entry=bear_entry,
            raw_df=raw,
            feature_space=feature_space,
            horizons=bear_horizons,
            val_start=str(bear_policy["val_start"]),
            test_start=str(bear_policy["test_start"]),
            test_end=bear_policy["test_end"],
            purge_bars=config.purge_bars_for_horizons(bear_horizons),
        )
    if filter_spec is not None and filter_entry is not None:
        filter_bundle = _train_spec_bundle(
            spec=filter_spec,
            entry=filter_entry,
            raw_df=raw,
            feature_space=feature_space,
            horizons=filter_horizons,
            val_start=str(filter_policy["val_start"]),
            test_start=str(filter_policy["test_start"]),
            test_end=filter_policy["test_end"],
            purge_bars=config.purge_bars_for_horizons(filter_horizons),
        )
    if slope_spec is not None and slope_entry is not None:
        slope_bundle = _train_spec_bundle(
            spec=slope_spec,
            entry=slope_entry,
            raw_df=raw,
            feature_space=feature_space,
            horizons=slope_horizons,
            val_start=str(slope_policy["val_start"]),
            test_start=str(slope_policy["test_start"]),
            test_end=slope_policy["test_end"],
            purge_bars=config.purge_bars_for_horizons(slope_horizons),
        )
    individual = CryptoIndividual(
        features=[str(value) for value in entry["features"]],
        generation=int(entry.get("generation", 0) or 0),
        score=float(entry.get("score", np.nan)),
    )
    evaluator = QuantileFitnessEvaluator(
        horizons=horizons,
        target="mfe",
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
    assert test_prediction is not None
    path = _price_path(raw, horizon)
    baseline_results = [
        _make_result(
            "val",
            valid_val,
            val_prediction,
            path,
            args.min_prediction,
            args.stop_loss,
            args.same_candle_policy,
            args.trade_cost,
        ),
        _make_result(
            "test",
            valid_test,
            test_prediction,
            path,
            args.min_prediction,
            args.stop_loss,
            args.same_candle_policy,
            args.trade_cost,
        ),
    ]
    results = baseline_results
    and_filter_impact_summaries: list[dict[str, float | int | str]] = []
    if filter_bundle is not None:
        allowed_by_split = {
            "val": pd.Index(filter_bundle.val.selected_index),
            "test": pd.Index(filter_bundle.test.selected_index),
        }
        results = [
            _make_result(
                "val",
                valid_val,
                val_prediction,
                path,
                args.min_prediction,
                args.stop_loss,
                args.same_candle_policy,
                args.trade_cost,
                allowed_index=allowed_by_split["val"],
            ),
            _make_result(
                "test",
                valid_test,
                test_prediction,
                path,
                args.min_prediction,
                args.stop_loss,
                args.same_candle_policy,
                args.trade_cost,
                allowed_index=allowed_by_split["test"],
            ),
        ]
        and_filter_impact_summaries = _and_filter_impact_summary(
            baseline_results,
            results,
        )
    raw_deltas = raw.index.to_series().diff().dropna()
    bar_delta = raw_deltas.median()
    if not isinstance(bar_delta, pd.Timedelta) or bar_delta <= pd.Timedelta(0):
        raise ValueError("Unable to infer a positive candle interval from input data.")
    slope_selected_entry_index: pd.Index | None = None
    slope_recall_rows: list[dict[str, float | int | str]] = []
    ma3_cooldown_impact_summaries: list[dict[str, float | int | str]] = []
    if slope_bundle is not None:
        slope_selected_signal_index = pd.DatetimeIndex(
            pd.Index(slope_bundle.val.selected_index).union(
                pd.Index(slope_bundle.test.selected_index)
            )
        )
        slope_positive = _ma3_positive_slope(raw["close"])
        slope_positive_selected = slope_positive.reindex(
            slope_selected_signal_index
        ).fillna(False).astype(bool)
        slope_selected_signal_index = slope_selected_signal_index[
            slope_positive_selected.to_numpy()
        ]
        slope_selected_entry_index = slope_selected_signal_index + bar_delta
        logger.info(
            "MA3 reversal overlay filter | selected=%d | MA3 slope > 0 kept=%d",
            len(slope_bundle.val.selected_index) + len(slope_bundle.test.selected_index),
            len(slope_selected_signal_index),
        )
        slope_recall_rows = _slope_reversal_band_recall(
            slope_bundle,
            raw,
            band_step=args.slope_reversal_band_step,
        )
        if args.ma3_cooldown_filter:
            triggers_by_split = {
                "val": pd.DatetimeIndex(slope_bundle.val.selected_index),
                "test": pd.DatetimeIndex(slope_bundle.test.selected_index),
            }
            candidates_by_split = {
                "val": valid_val.index,
                "test": valid_test.index,
            }
            allowed_by_split: dict[str, pd.DatetimeIndex] = {}
            for split_name in ("val", "test"):
                triggers = triggers_by_split[split_name]
                valid_trigger = slope_positive.reindex(triggers).fillna(False)
                triggers = triggers[valid_trigger.to_numpy(dtype=bool)]
                allowed_by_split[split_name] = _ma3_cooldown_allowed_index(
                    candidates_by_split[split_name],
                    triggers,
                    slope_positive,
                    cooldown_bars=args.ma3_cooldown_bars,
                )
            before_cooldown = results
            results = [
                SplitResult(
                    split=result.split,
                    trades=result.trades.loc[
                        result.trades.index.isin(allowed_by_split[result.split])
                    ].copy(),
                    candidate_rows=result.candidate_rows,
                )
                for result in results
            ]
            ma3_cooldown_impact_summaries = _ma3_cooldown_impact_summary(
                before_cooldown,
                results,
            )

    summaries = [_summary(result) for result in results]
    two_sided_excursion_summaries = _two_sided_excursion_summary(
        results,
        args.two_sided_threshold,
    )
    bear_overlap_summaries: list[dict[str, float | int | str]] = []
    bear_selected_entry_index: pd.Index | None = None
    if bear_bundle is not None:
        bear_selected_by_split = {
            "val": pd.Index(bear_bundle.val.selected_index),
            "test": pd.Index(bear_bundle.test.selected_index),
        }
        bear_overlap_summaries = _bear_loss_overlap_summary(
            results,
            bear_selected_by_split,
        )
        bear_selected_entry_index = pd.DatetimeIndex(
            bear_selected_by_split["val"].union(bear_selected_by_split["test"])
        ) + bar_delta
    clustering_summaries = [
        _loss_clustering_summary(
            result,
            bar_delta=bar_delta,
            cooldown_bars=5,
        )
        for result in results
    ]
    loss_run_summaries = [_loss_run_distribution(result) for result in results]
    h1_close_summaries = [
        row
        for result in results
        for row in _h1_close_outcome_summary(result, args.h1_low_threshold)
    ]
    losing_low_h1_summaries = _losing_trade_low_h1_summary(
        results,
        args.h1_low_threshold,
    )
    close_threshold_outcome_summaries = _close_threshold_by_outcome_summary(
        results,
        args.h1_low_threshold,
    )
    h1_threshold_summaries = [
        row
        for result in results
        for row in _h1_close_threshold_summary(
            result,
            args.h1_close_threshold,
        )
    ]
    loss_h1_vs_h3_summaries = [
        _loss_close_h1_vs_h3_summary(
            result,
            trade_cost=args.trade_cost,
            large_loss_threshold=args.large_loss_threshold,
        )
        for result in results
    ]
    h2_vs_h3_summaries = [
        _close_h2_vs_hold_h3_summary(
            result,
            trade_cost=args.trade_cost,
        )
        for result in results
    ]
    first_1m_low_summaries: list[dict[str, float | int | str]] = []
    if args.data_1m is not None:
        first_1m_low_summaries = _first_1m_low_summary(
            results,
            data_1m_path=args.data_1m,
            entry_delay=bar_delta,
            threshold=args.first_1m_low_threshold,
            simulated_stop_loss=args.first_1m_stop_loss,
            trade_cost=args.trade_cost,
            large_loss_threshold=args.first_1m_loss_threshold,
        )
    high_h1_severity_summaries = [
        row
        for result in results
        for row in _high_h1_loss_severity_summary(
            result,
            args.large_loss_threshold,
        )
    ]
    tp_level_summaries = _tp_level_summary(
        results,
        args.large_loss_threshold,
    )
    ma_slope_outcome_summaries = _ma_slope_trade_outcome_summary(
        results,
        raw,
        large_loss_threshold=-0.001,
    )
    summary_frame = pd.DataFrame(summaries)
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print("\n=== Quantile MFE dynamic TP backtest ===")
        print(summary_frame.to_string(index=False))
        print("\n=== Selected trades crossing both MFE and MAE thresholds ===")
        print(pd.DataFrame(two_sided_excursion_summaries).to_string(index=False))
        if and_filter_impact_summaries:
            print("\n=== MFE AND adverse-floor filter impact ===")
            print(pd.DataFrame(and_filter_impact_summaries).to_string(index=False))
        if ma3_cooldown_impact_summaries:
            print(
                "\n=== Stateful MA3 reversal cooldown filter impact | "
                f"wait={args.ma3_cooldown_bars} future bars ==="
            )
            print(
                pd.DataFrame(ma3_cooldown_impact_summaries).to_string(index=False)
            )
        print("\n=== Loss clustering after realized exit: next 5 bars ===")
        print(pd.DataFrame(clustering_summaries).to_string(index=False))
        print("\n=== Consecutive net-loss runs: count | share of losing trades ===")
        print(pd.DataFrame(loss_run_summaries).to_string(index=False))
        print("\n=== Close H1 return: strategy wins versus losses ===")
        print(pd.DataFrame(h1_close_summaries).to_string(index=False))
        print("\n=== Final losing trades split by low H1 ===")
        print(pd.DataFrame(losing_low_h1_summaries).to_string(index=False))
        print("\n=== Close H1/H3 threshold by final outcome ===")
        print(
            pd.DataFrame(close_threshold_outcome_summaries).to_string(index=False)
        )
        print("\n=== Final outcome conditional on close H1 threshold ===")
        print(pd.DataFrame(h1_threshold_summaries).to_string(index=False))
        print("\n=== No-SL losing trades: close H1 versus close H3 ===")
        print(pd.DataFrame(loss_h1_vs_h3_summaries).to_string(index=False))
        print("\n=== Unresolved after H2: close H2 versus hold H3 ===")
        print(pd.DataFrame(h2_vs_h3_summaries).to_string(index=False))
        if first_1m_low_summaries:
            print("\n=== First 1m candle after signal: low versus open H1 ===")
            print(pd.DataFrame(first_1m_low_summaries).to_string(index=False))
        print("\n=== High H1 versus later outcome severity ===")
        print(pd.DataFrame(high_h1_severity_summaries).to_string(index=False))
        print("\n=== Outcome by Val-derived predicted TP quintile ===")
        print(pd.DataFrame(tp_level_summaries).to_string(index=False))
        print(
            "\n=== Observable MA slope at signal t | wins vs gross loss > 0.10% ==="
        )
        print(
            pd.DataFrame(ma_slope_outcome_summaries).to_string(index=False)
        )
        if bear_overlap_summaries:
            print("\n=== Bear top-fraction coverage of gross losses > 0.10% ===")
            print(pd.DataFrame(bear_overlap_summaries).to_string(index=False))
        if slope_recall_rows:
            print(
                "\n=== MA3 reversal recall by score band | denominator: "
                "all positives with current MA3 slope > 0 ==="
            )
            print(pd.DataFrame(slope_recall_rows).to_string(index=False))

    output = Path(args.out_dir) / "quantile_mfe_backtest.png"
    title = (
        f"MFE Q{100.0 * quantile:.0f} dynamic TP | rank {args.rank} | H{horizon} | "
        f"MFE > {100.0 * args.min_prediction:.3f}% | "
        f"SL={100.0 * args.stop_loss:.3f}% | {args.same_candle_policy} | "
        f"cost={100.0 * args.trade_cost:.3f}%"
    )
    if args.ma3_cooldown_filter:
        title += f" | MA3 cooldown={args.ma3_cooldown_bars} bars"
    _render(
        summaries,
        clustering_summaries,
        loss_run_summaries,
        h1_close_summaries,
        h1_threshold_summaries,
        loss_h1_vs_h3_summaries,
        h2_vs_h3_summaries,
        high_h1_severity_summaries,
        tp_level_summaries,
        results,
        output,
        title,
    )
    slope_recall_output: Path | None = None
    if slope_recall_rows:
        slope_recall_output = output.with_name("ma3_reversal_recall_bands.png")
        _render_slope_reversal_recall_table(
            slope_recall_rows,
            slope_recall_output,
        )
    examples_outputs: list[Path] = []
    for image_number in range(int(args.example_images)):
        image_seed = int(args.example_seed) + image_number * int(args.example_seed_step)
        suffix = (
            f"_seed{image_seed}"
            if int(args.example_images) > 1
            else ""
        )
        examples_output = output.with_name(f"ma3_reversal_overlay{suffix}.png")
        _render_random_large_loss_windows(
            raw,
            results,
            bar_delta=bar_delta,
            output=examples_output,
            bars_per_panel=args.example_window_bars,
            panel_count=args.example_panels,
            random_seed=image_seed,
            chart_style=args.example_chart_style,
            bear_selected_index=bear_selected_entry_index,
            bear_top_fraction=args.bear_top_fraction,
            bear_precision=(
                float(bear_overlap_summaries[-1]["precision"])
                if bear_overlap_summaries
                else np.nan
            ),
            bear_recall=(
                float(bear_overlap_summaries[-1]["recall"])
                if bear_overlap_summaries
                else np.nan
            ),
            slope_selected_index=slope_selected_entry_index,
            slope_top_fraction=args.slope_reversal_top_fraction,
        )
        examples_outputs.append(examples_output)
    logger.info("Saved report: %s", output)
    if slope_recall_output is not None:
        logger.info("Saved MA3 reversal recall bands: %s", slope_recall_output)
    for examples_output in examples_outputs:
        logger.info("Saved gross-loss examples: %s", examples_output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--bear-archive", type=Path, default=DEFAULT_BEAR_ARCHIVE)
    parser.add_argument("--bear-rank", type=int, default=1)
    parser.add_argument("--bear-top-fraction", type=float, default=0.40)
    parser.add_argument("--no-bear-overlay", action="store_true")
    parser.add_argument(
        "--slope-reversal-archive",
        type=Path,
        default=DEFAULT_SLOPE_REVERSAL_ARCHIVE,
    )
    parser.add_argument("--slope-reversal-rank", type=int, default=1)
    parser.add_argument(
        "--slope-reversal-top-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--slope-reversal-band-step",
        type=float,
        default=0.05,
    )
    parser.add_argument("--no-slope-reversal-overlay", action="store_true")
    parser.add_argument(
        "--ma3-cooldown-filter",
        action="store_true",
        help=(
            "Block MFE trades on MA3 reversal signals, wait for the configured "
            "future bars, then require observable MA3 slope > 0 before resuming."
        ),
    )
    parser.add_argument("--ma3-cooldown-bars", type=int, default=2)
    parser.add_argument("--and-filter-archive", type=Path, default=None)
    parser.add_argument("--and-filter-rank", type=int, default=1)
    parser.add_argument("--and-filter-top-fraction", type=float, default=0.80)
    parser.add_argument("--min-prediction", type=float, default=0.0002)
    parser.add_argument("--two-sided-threshold", type=float, default=0.0002)
    parser.add_argument("--stop-loss", type=float, default=0.001)
    parser.add_argument("--h1-close-threshold", type=float, default=-0.0005)
    parser.add_argument("--h1-low-threshold", type=float, default=-0.00015)
    parser.add_argument("--large-loss-threshold", type=float, default=-0.002)
    parser.add_argument(
        "--same-candle-policy",
        choices=("stop_first", "tp_first"),
        default="stop_first",
    )
    parser.add_argument("--trade-cost", type=float, default=config.TRADE_COST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--data-1m", type=Path, default=None)
    parser.add_argument("--first-1m-low-threshold", type=float, default=-0.001)
    parser.add_argument("--first-1m-stop-loss", type=float, default=0.0005)
    parser.add_argument("--first-1m-loss-threshold", type=float, default=-0.001)
    parser.add_argument("--example-window-bars", type=int, default=500)
    parser.add_argument("--example-panels", type=int, default=4)
    parser.add_argument("--example-seed", type=int, default=1)
    parser.add_argument("--example-images", type=int, default=1)
    parser.add_argument("--example-seed-step", type=int, default=18)
    parser.add_argument(
        "--example-chart-style",
        choices=("line", "candles"),
        default="line",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if not np.isfinite(args.min_prediction) or args.min_prediction < 0.0:
        parser.error("--min-prediction must be finite and non-negative.")
    if not np.isfinite(args.two_sided_threshold) or args.two_sided_threshold < 0.0:
        parser.error("--two-sided-threshold must be finite and non-negative.")
    if not np.isfinite(args.trade_cost) or args.trade_cost < 0.0:
        parser.error("--trade-cost must be finite and non-negative.")
    if not np.isfinite(args.stop_loss) or args.stop_loss <= 0.0:
        parser.error("--stop-loss must be finite and positive.")
    if not np.isfinite(args.h1_close_threshold):
        parser.error("--h1-close-threshold must be finite.")
    if not np.isfinite(args.h1_low_threshold):
        parser.error("--h1-low-threshold must be finite.")
    if not np.isfinite(args.large_loss_threshold) or args.large_loss_threshold >= 0.0:
        parser.error("--large-loss-threshold must be finite and negative.")
    if not np.isfinite(args.first_1m_low_threshold):
        parser.error("--first-1m-low-threshold must be finite.")
    if not np.isfinite(args.first_1m_stop_loss) or args.first_1m_stop_loss <= 0.0:
        parser.error("--first-1m-stop-loss must be finite and positive.")
    if (
        not np.isfinite(args.first_1m_loss_threshold)
        or args.first_1m_loss_threshold >= 0.0
    ):
        parser.error("--first-1m-loss-threshold must be finite and negative.")
    if args.rank < 1:
        parser.error("--rank must be positive.")
    if args.bear_rank < 1:
        parser.error("--bear-rank must be positive.")
    if args.slope_reversal_rank < 1:
        parser.error("--slope-reversal-rank must be positive.")
    if args.ma3_cooldown_bars < 1:
        parser.error("--ma3-cooldown-bars must be positive.")
    if args.ma3_cooldown_filter and args.no_slope_reversal_overlay:
        parser.error(
            "--ma3-cooldown-filter requires the MA3 slope-reversal model."
        )
    if args.and_filter_rank < 1:
        parser.error("--and-filter-rank must be positive.")
    if not 0.0 < args.bear_top_fraction <= 1.0:
        parser.error("--bear-top-fraction must be in (0, 1].")
    if not 0.0 < args.slope_reversal_top_fraction <= 1.0:
        parser.error("--slope-reversal-top-fraction must be in (0, 1].")
    if not 0.0 < args.slope_reversal_band_step <= 1.0:
        parser.error("--slope-reversal-band-step must be in (0, 1].")
    if not 0.0 < args.and_filter_top_fraction <= 1.0:
        parser.error("--and-filter-top-fraction must be in (0, 1].")
    if args.example_window_bars < 10:
        parser.error("--example-window-bars must be at least 10.")
    if args.example_panels < 1:
        parser.error("--example-panels must be positive.")
    if args.example_images < 1:
        parser.error("--example-images must be positive.")
    if args.example_seed_step < 1:
        parser.error("--example-seed-step must be positive.")
    return args


if __name__ == "__main__":
    run(parse_args())
