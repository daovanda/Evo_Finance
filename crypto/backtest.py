"""Backtest the live two-stage crypto strategy.

This module evaluates the practical flow:

1. One or more base archive individuals create the original trade signal at t.
2. If the trade has not hit take profit before the exit model is due, the exit
   model is evaluated at the correct delayed bar for its label mode.
3. Report how often base signals survive to that exit point and how often the
   exit model selects those cases.

Examples:
    python -m crypto.backtest ^
      --base crypto/results/crypto_btc_mfe_h5_seed1_12h.json#1#mfe#0.003 ^
      --base crypto/results/crypto_btc_close_exit_h5_seed1_12h.json#1#close_exit#0.001 ^
      --base-ensemble and ^
      --exit crypto/results/crypto_btc_exit_after_h1_h5_noh1_tp04_seed1_12h.json#1#exit_after_h1#0.004 ^
      --tp-threshold 0.004
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
from crypto.analyze import (
    _entry_to_individual,
    _load_rank_entry,
    _required_windows_for_entries,
    _return_context_by_horizon,
    _threshold_filename_token,
    _train_final_booster,
    _valid_frame,
)
from crypto.data import add_binary_labels, load_ohlcv, split_labeled_by_dates
from crypto.evolution import CryptoIndividual
from crypto.expression import CryptoFeatureSpace
from crypto.features import build_feature_frame, selectable_features


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crypto.backtest")


DEFAULT_OUT_DIR = config.RESULTS_DIR / "backtest"
TP_SWEEP_START: float = -0.002
TP_SWEEP_END: float = 0.005
TP_SWEEP_STEP: float = 0.0005
EXIT_TOP_FRACTIONS: list[float] = [0.10 ,0.20, 0.30, 0.40, 0.50, 0.6, 0.7]

_FEATURE_SPACE_CACHE: dict[tuple[Any, ...], CryptoFeatureSpace] = {}


@dataclass(frozen=True)
class ModelSpec:
    archive_path: Path
    rank: int
    label_mode: str
    label_threshold: float
    top_fraction: float


@dataclass(frozen=True)
class SplitSignals:
    split: str
    data: pd.DataFrame
    selected_index: pd.Index
    pred_threshold: float
    top_fraction: float


@dataclass(frozen=True)
class BundleSignals:
    label: str
    val: SplitSignals
    test: SplitSignals


@dataclass(frozen=True)
class BacktestResult:
    summary: pd.DataFrame
    tp_sweep: pd.DataFrame
    csv_path: Path
    tp_sweep_csv_path: Path
    chart_path: Path


def run_backtest(
    base_specs: list[ModelSpec],
    exit_spec: ModelSpec,
    data_path: str | Path = config.DATA_PATH,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    base_ensemble: str = "and",
    tp_threshold: float | None = None,
    val_start: str = config.VAL_START,
    test_start: str = config.TEST_START,
    test_end: str | None = config.TEST_END,
) -> BacktestResult:
    if not base_specs:
        raise ValueError("At least one --base spec is required.")
    base_ensemble = str(base_ensemble).strip().lower()
    if base_ensemble not in {"and", "or"}:
        raise ValueError("--base-ensemble must be 'and' or 'or'.")

    tp = float(tp_threshold if tp_threshold is not None else exit_spec.label_threshold)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Backtest setup: base_specs=%d | base_ensemble=%s | exit=%s rank %d",
        len(base_specs),
        base_ensemble.upper(),
        exit_spec.archive_path,
        exit_spec.rank,
    )
    logger.info("Loading crypto data from %s", data_path)
    raw_df = load_ohlcv(data_path)
    horizon = int(max(config.HOLDING_HORIZONS))
    purge_bars = config.purge_bars_for_horizons(config.HOLDING_HORIZONS)

    entries = [_load_rank_entry(spec.archive_path, spec.rank) for spec in [*base_specs, exit_spec]]
    quality_train = _quality_train_index(
        raw_df=raw_df,
        spec=base_specs[0],
        val_start=val_start,
        test_start=test_start,
        test_end=test_end,
        purge_bars=purge_bars,
    )
    required_windows = _required_windows_for_entries(entries)
    feature_space = _cached_feature_space(
        raw_df=raw_df,
        data_path=data_path,
        required_windows=required_windows,
        quality_index=quality_train,
    )

    base_bundles: list[BundleSignals] = []
    for spec, entry in zip(base_specs, entries[: len(base_specs)], strict=True):
        bundle = _train_spec_bundle(
            spec=spec,
            entry=entry,
            raw_df=raw_df,
            feature_space=feature_space,
            val_start=val_start,
            test_start=test_start,
            test_end=test_end,
            purge_bars=purge_bars,
        )
        base_bundles.append(bundle)

    exit_bundle = _train_spec_bundle(
        spec=exit_spec,
        entry=entries[-1],
        raw_df=raw_df,
        feature_space=feature_space,
        val_start=val_start,
        test_start=test_start,
        test_end=test_end,
        purge_bars=purge_bars,
    )
    base_bundle = _combine_base_bundles(base_bundles, selection=base_ensemble)
    exit_delay_bars = _exit_delay_bars(exit_spec.label_mode)

    _, path_by_horizon, _ = _return_context_by_horizon(
        raw_df,
        [horizon],
        label_mode="mfe",
    )
    path_returns = path_by_horizon[horizon]

    summary_rows: list[dict[str, Any]] = []
    tp_sweep_rows: list[dict[str, Any]] = []
    tp_sweep_thresholds = _tp_sweep_thresholds()
    for split_name, base_split, exit_split in [
        ("val", base_bundle.val, exit_bundle.val),
        ("test", base_bundle.test, exit_bundle.test),
    ]:
        summary_row, split_tp_rows = _summarize_split(
            split=split_name,
            base_split=base_split,
            exit_split=exit_split,
            path_returns=path_returns,
            raw_index=pd.DatetimeIndex(raw_df.index),
            tp_threshold=tp,
            tp_sweep_thresholds=tp_sweep_thresholds,
            exit_delay_bars=exit_delay_bars,
        )
        summary_rows.append(summary_row)
        tp_sweep_rows.extend(split_tp_rows)

    summary = pd.DataFrame(summary_rows)
    tp_sweep = pd.DataFrame(tp_sweep_rows)
    run_name = _backtest_name(base_specs, exit_spec, base_ensemble, tp)
    csv_path = out_path / f"{run_name}.csv"
    tp_sweep_csv_path = out_path / f"{run_name}_tp_sweep.csv"
    chart_path = out_path / f"{run_name}.png"
    summary.to_csv(csv_path, index=False)
    tp_sweep.to_csv(tp_sweep_csv_path, index=False)
    _plot_summary(
        summary=summary,
        tp_sweep=tp_sweep,
        chart_path=chart_path,
        base_label=base_bundle.label,
        exit_label=exit_bundle.label,
        base_ensemble=base_ensemble,
        tp_threshold=tp,
        exit_delay_bars=exit_delay_bars,
    )
    logger.info("Saved summary: %s", csv_path)
    logger.info("Saved TP sweep: %s", tp_sweep_csv_path)
    logger.info("Saved chart: %s", chart_path)
    return BacktestResult(
        summary=summary,
        tp_sweep=tp_sweep,
        csv_path=csv_path,
        tp_sweep_csv_path=tp_sweep_csv_path,
        chart_path=chart_path,
    )


def _quality_train_index(
    raw_df: pd.DataFrame,
    spec: ModelSpec,
    val_start: str,
    test_start: str,
    test_end: str | None,
    purge_bars: int,
) -> pd.Index:
    labeled = add_binary_labels(
        raw_df,
        horizons=config.HOLDING_HORIZONS,
        threshold=spec.label_threshold,
        return_fn=config.get_label_return_fn(spec.label_mode),
        label_mode=spec.label_mode,
    )
    train_df, _, _ = split_labeled_by_dates(
        labeled,
        val_start=val_start,
        test_start=test_start,
        test_end=test_end,
        purge_bars=purge_bars,
    )
    return train_df.index


def _cached_feature_space(
    raw_df: pd.DataFrame,
    data_path: str | Path,
    required_windows: list[int],
    quality_index: pd.Index,
) -> CryptoFeatureSpace:
    data_index = pd.DatetimeIndex(raw_df.index)
    quality_dt = pd.DatetimeIndex(quality_index)
    cache_key = (
        str(Path(data_path)),
        tuple(int(window) for window in required_windows),
        len(raw_df),
        str(data_index.min()) if len(data_index) else "",
        str(data_index.max()) if len(data_index) else "",
        len(quality_dt),
        str(quality_dt.min()) if len(quality_dt) else "",
        str(quality_dt.max()) if len(quality_dt) else "",
    )
    cached = _FEATURE_SPACE_CACHE.get(cache_key)
    if cached is not None:
        logger.info("Using cached feature matrix | windows=%s", required_windows)
        return cached

    logger.info("Building feature matrix | windows=%s", required_windows)
    feature_df = build_feature_frame(
        raw_df,
        windows=required_windows,
        quality_index=quality_index,
    )
    feature_pool = selectable_features(feature_df)
    feature_space = CryptoFeatureSpace(feature_df, feature_pool)
    _FEATURE_SPACE_CACHE[cache_key] = feature_space
    return feature_space


def _train_spec_bundle(
    spec: ModelSpec,
    entry: dict[str, Any],
    raw_df: pd.DataFrame,
    feature_space: CryptoFeatureSpace,
    val_start: str,
    test_start: str,
    test_end: str | None,
    purge_bars: int,
) -> BundleSignals:
    logger.info(
        "Training %s rank %d | mode=%s threshold=%.6f top=%.2f%%",
        spec.archive_path,
        spec.rank,
        spec.label_mode,
        spec.label_threshold,
        spec.top_fraction * 100.0,
    )
    individual = _entry_to_individual(entry)
    labeled = add_binary_labels(
        raw_df,
        horizons=config.HOLDING_HORIZONS,
        threshold=spec.label_threshold,
        return_fn=config.get_label_return_fn(spec.label_mode),
        label_mode=spec.label_mode,
    )
    train_df, val_df, test_df = split_labeled_by_dates(
        labeled,
        val_start=val_start,
        test_start=test_start,
        test_end=test_end,
        purge_bars=purge_bars,
    )

    horizon_results: list[tuple[int, SplitSignals, SplitSignals]] = []
    for horizon in config.HOLDING_HORIZONS:
        result = _train_one_horizon_signal(
            individual=individual,
            horizon=int(horizon),
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_space=feature_space,
            top_fraction=spec.top_fraction,
        )
        if result is not None:
            horizon_results.append(result)

    if not horizon_results:
        raise ValueError(f"No valid horizon model for {spec.archive_path} rank {spec.rank}.")

    label = (
        f"{spec.archive_path.stem} r{spec.rank:02d} "
        f"{spec.label_mode} thr={spec.label_threshold:.4g} top={spec.top_fraction:.0%}"
    )
    if len(horizon_results) == 1:
        _, val, test = horizon_results[0]
        return BundleSignals(label=label, val=val, test=test)

    val = _combine_horizons(
        split="val",
        split_results=[item[1] for item in horizon_results],
        top_fraction=spec.top_fraction,
    )
    test = _combine_horizons(
        split="test",
        split_results=[item[2] for item in horizon_results],
        top_fraction=spec.top_fraction,
    )
    return BundleSignals(label=f"{label} h-ensemble", val=val, test=test)


def _train_one_horizon_signal(
    individual: CryptoIndividual,
    horizon: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_space: CryptoFeatureSpace,
    top_fraction: float,
) -> tuple[int, SplitSignals, SplitSignals] | None:
    label_col = f"label_h{horizon}"
    ret_col = f"future_return_h{horizon}"
    train = _valid_frame(train_df, label_col, ret_col)
    val = _valid_frame(val_df, label_col, ret_col)
    test = _valid_frame(test_df, label_col, ret_col)
    if train.empty or val.empty or test.empty:
        logger.warning("h%d skipped: empty train/val/test after label filtering.", horizon)
        return None

    X_train = feature_space.matrix(individual.features, train.index)
    X_val = feature_space.matrix(individual.features, val.index)
    X_test = feature_space.matrix(individual.features, test.index)
    y_train = train[label_col].astype(int)
    y_val = val[label_col].astype(int)
    y_test = test[label_col].astype(int)
    if y_train.nunique() < 2:
        logger.warning("h%d skipped: train label is constant.", horizon)
        return None

    booster = _train_final_booster(X_train, y_train, X_val, y_val)
    val_pred = pd.Series(booster.predict(X_val), index=val.index, name="pred")
    test_pred = pd.Series(booster.predict(X_test), index=test.index, name="pred")
    val_signal = _split_signals(
        split="val",
        label=y_val,
        pred=val_pred,
        top_fraction=top_fraction,
    )
    test_signal = _split_signals(
        split="test",
        label=y_test,
        pred=test_pred,
        top_fraction=top_fraction,
        pred_threshold=val_signal.pred_threshold,
    )
    return horizon, val_signal, test_signal


def _split_signals(
    split: str,
    label: pd.Series,
    pred: pd.Series,
    top_fraction: float,
    pred_threshold: float | None = None,
) -> SplitSignals:
    data = (
        pd.DataFrame({"label": label, "pred": pred})
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["label", "pred"])
    )
    if data.empty:
        return SplitSignals(
            split=split,
            data=data,
            selected_index=pd.Index([]),
            pred_threshold=float("nan"),
            top_fraction=float(top_fraction),
        )

    if pred_threshold is None:
        n_select = min(
            len(data),
            max(
                int(config.MIN_TRADES_PER_SPLIT),
                int(np.ceil(len(data) * float(top_fraction))),
            ),
        )
        selected = data.nlargest(n_select, "pred")
        threshold = float(selected["pred"].min()) if len(selected) else float("inf")
    else:
        threshold = float(pred_threshold)
        selected = data[data["pred"] >= threshold]
    return SplitSignals(
        split=split,
        data=data,
        selected_index=pd.Index(selected.index),
        pred_threshold=threshold,
        top_fraction=float(top_fraction),
    )


def _combine_horizons(
    split: str,
    split_results: list[SplitSignals],
    top_fraction: float,
) -> SplitSignals:
    if not split_results:
        return SplitSignals(
            split=split,
            data=pd.DataFrame(),
            selected_index=pd.Index([]),
            pred_threshold=float("nan"),
            top_fraction=float(top_fraction),
        )
    selected_index = _combine_indices(
        [item.selected_index for item in split_results],
        selection="and",
    )
    common_index = split_results[0].data.index
    for item in split_results[1:]:
        common_index = common_index.intersection(item.data.index)
    pred_frame = pd.concat(
        [item.data["pred"].reindex(common_index) for item in split_results],
        axis=1,
    )
    label = split_results[-1].data["label"].reindex(common_index)
    data = pd.DataFrame({"label": label, "pred": pred_frame.mean(axis=1)}).dropna()
    selected = pd.Index(data.index.intersection(selected_index))
    return SplitSignals(
        split=split,
        data=data,
        selected_index=selected,
        pred_threshold=_selected_pred_threshold(data, selected),
        top_fraction=float(top_fraction),
    )


def _combine_base_bundles(
    bundles: list[BundleSignals],
    selection: str,
) -> BundleSignals:
    if len(bundles) == 1:
        return bundles[0]
    label = f"base {str(selection).upper()} (" + " + ".join(item.label for item in bundles) + ")"
    return BundleSignals(
        label=label,
        val=_combine_bundle_splits("val", [item.val for item in bundles], selection),
        test=_combine_bundle_splits("test", [item.test for item in bundles], selection),
    )


def _combine_bundle_splits(
    split: str,
    splits: list[SplitSignals],
    selection: str,
) -> SplitSignals:
    selected_index = _combine_indices([item.selected_index for item in splits], selection)
    common_index = splits[0].data.index
    for item in splits[1:]:
        common_index = common_index.union(item.data.index)
    pred_frame = pd.concat([item.data["pred"].reindex(common_index) for item in splits], axis=1)
    label_frame = pd.concat(
        [item.data["label"].reindex(common_index) for item in splits],
        axis=1,
    )
    data = pd.DataFrame(
        {
            "label": label_frame.mean(axis=1),
            "pred": pred_frame.mean(axis=1),
        }
    ).dropna(subset=["pred"])
    selected = pd.Index(data.index.intersection(selected_index))
    return SplitSignals(
        split=split,
        data=data,
        selected_index=selected,
        pred_threshold=_selected_pred_threshold(data, selected),
        top_fraction=float(config.TRADE_TOP_FRACTION),
    )


def _selected_pred_threshold(data: pd.DataFrame, selected_index: pd.Index) -> float:
    selected = data.reindex(data.index.intersection(selected_index)).dropna(subset=["pred"])
    if selected.empty:
        return float("nan")
    return float(pd.to_numeric(selected["pred"], errors="coerce").min())


def _combine_indices(indices: list[pd.Index], selection: str) -> pd.Index:
    selected: pd.Index | None = None
    for index in indices:
        current = pd.Index(index)
        if selected is None:
            selected = current
        elif str(selection).lower() == "or":
            selected = selected.union(current)
        else:
            selected = selected.intersection(current)
    return selected if selected is not None else pd.Index([])


def _summarize_split(
    split: str,
    base_split: SplitSignals,
    exit_split: SplitSignals,
    path_returns: pd.DataFrame,
    raw_index: pd.DatetimeIndex,
    tp_threshold: float,
    tp_sweep_thresholds: list[float],
    exit_delay_bars: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_selected = pd.Index(base_split.selected_index)
    base_path = path_returns.reindex(base_selected).dropna(subset=["high_h1"])
    base_signals = int(len(base_path))
    if base_signals == 0:
        summary = {
            "split": split,
            "base_signals": 0,
            "base_no_h1": 0,
            "base_no_h1_rate": 0.0,
            "exit_selected_after_no_h1": 0,
            "exit_selected_after_no_h1_rate": 0.0,
            "exit_delay_bars": int(exit_delay_bars),
            "base_no_pre_exit_hit": 0,
            "base_no_pre_exit_hit_rate": 0.0,
            "exit_selected_after_pre_exit": 0,
            "exit_selected_after_pre_exit_rate": 0.0,
            "base_threshold_source": "val-top",
            "exit_top_fraction": float(exit_split.top_fraction),
            "tp_threshold": float(tp_threshold),
        }
        return summary, (
            _empty_tp_sweep_rows(split, tp_sweep_thresholds, group="all_signal")
            + _empty_tp_sweep_rows(split, tp_sweep_thresholds, group="base_no_h1")
            + _empty_tp_sweep_rows(split, tp_sweep_thresholds, group="base_no_pre_exit_hit")
            + _empty_tp_sweep_rows(split, tp_sweep_thresholds, group="exit_selected")
            + _empty_tp_sweep_rows(split, tp_sweep_thresholds, group="no_exit_selected")
        )

    no_h1_index = pd.Index(base_path.index[base_path["high_h1"] <= float(tp_threshold)])
    no_h1_count = int(len(no_h1_index))
    no_pre_exit_index = _no_pre_exit_hit_index(
        base_path,
        exit_delay_bars=int(exit_delay_bars),
        tp_threshold=float(tp_threshold),
    )
    no_pre_exit_count = int(len(no_pre_exit_index))
    exit_mapping = _future_bar_mapping(no_pre_exit_index, raw_index, offset=int(exit_delay_bars))
    exit_selected = pd.Index(exit_split.selected_index)
    exit_selected_mapping = exit_mapping[exit_mapping["exit_index"].isin(exit_selected)]
    no_exit_selected_mapping = exit_mapping[~exit_mapping["exit_index"].isin(exit_selected)]
    exit_after_pre_exit = int(len(exit_selected_mapping))
    exit_selected_base_index = pd.Index(exit_selected_mapping["base_index"])
    no_exit_selected_base_index = pd.Index(no_exit_selected_mapping["base_index"])
    summary = {
        "split": split,
        "base_signals": base_signals,
        "base_no_h1": no_h1_count,
        "base_no_h1_rate": no_h1_count / base_signals if base_signals else 0.0,
        "exit_selected_after_no_h1": exit_after_pre_exit,
        "exit_selected_after_no_h1_rate": (
            exit_after_pre_exit / no_h1_count if no_h1_count else 0.0
        ),
        "exit_delay_bars": int(exit_delay_bars),
        "base_no_pre_exit_hit": no_pre_exit_count,
        "base_no_pre_exit_hit_rate": (
            no_pre_exit_count / base_signals if base_signals else 0.0
        ),
        "exit_selected_after_pre_exit": exit_after_pre_exit,
        "exit_selected_after_pre_exit_rate": (
            exit_after_pre_exit / no_pre_exit_count if no_pre_exit_count else 0.0
        ),
        "base_pred_threshold": _json_safe_float(base_split.pred_threshold),
        "exit_pred_threshold": _json_safe_float(exit_split.pred_threshold),
        "exit_top_fraction": float(exit_split.top_fraction),
        "tp_threshold": float(tp_threshold),
    }
    all_signal_tp_rows = _tp_sweep_rows(
        split=split,
        group="all_signal",
        selected_base_index=pd.Index(base_path.index),
        path_returns=path_returns,
        thresholds=tp_sweep_thresholds,
        min_h=1,
    )
    base_no_h1_tp_rows = _tp_sweep_rows(
        split=split,
        group="base_no_h1",
        selected_base_index=no_h1_index,
        path_returns=path_returns,
        thresholds=tp_sweep_thresholds,
        min_h=2,
    )
    base_no_pre_exit_tp_rows = _tp_sweep_rows(
        split=split,
        group="base_no_pre_exit_hit",
        selected_base_index=no_pre_exit_index,
        path_returns=path_returns,
        thresholds=tp_sweep_thresholds,
        min_h=int(exit_delay_bars) + 1,
    )
    exit_selected_tp_rows = _tp_sweep_rows(
        split=split,
        group="exit_selected",
        selected_base_index=exit_selected_base_index,
        path_returns=path_returns,
        thresholds=tp_sweep_thresholds,
        min_h=int(exit_delay_bars) + 1,
    )
    no_exit_selected_tp_rows = _tp_sweep_rows(
        split=split,
        group="no_exit_selected",
        selected_base_index=no_exit_selected_base_index,
        path_returns=path_returns,
        thresholds=tp_sweep_thresholds,
        min_h=int(exit_delay_bars) + 1,
    )
    return (
        summary,
        all_signal_tp_rows
        + base_no_h1_tp_rows
        + base_no_pre_exit_tp_rows
        + exit_selected_tp_rows
        + no_exit_selected_tp_rows,
    )


def _future_bar_mapping(
    index: pd.Index,
    raw_index: pd.DatetimeIndex,
    offset: int,
) -> pd.DataFrame:
    if len(index) == 0:
        return pd.DataFrame(columns=["base_index", "exit_index"])
    positions = raw_index.get_indexer(pd.DatetimeIndex(index))
    valid = positions >= 0
    base_values = np.asarray(index, dtype=object)[valid]
    positions = positions[valid] + max(int(offset), 0)
    in_range = positions < len(raw_index)
    return pd.DataFrame(
        {
            "base_index": pd.DatetimeIndex(base_values[in_range]),
            "exit_index": raw_index.take(positions[in_range]),
        }
    )


def _no_pre_exit_hit_index(
    base_path: pd.DataFrame,
    exit_delay_bars: int,
    tp_threshold: float,
) -> pd.Index:
    hit_columns = [
        f"high_h{step}"
        for step in range(1, max(int(exit_delay_bars), 1) + 1)
        if f"high_h{step}" in base_path.columns
    ]
    if not hit_columns:
        return pd.Index([])
    high_values = base_path[hit_columns].apply(pd.to_numeric, errors="coerce")
    no_hit = ~(high_values > float(tp_threshold)).any(axis=1)
    return pd.Index(base_path.index[no_hit.fillna(False)])


def _exit_delay_bars(label_mode: str) -> int:
    mode = config.canonical_label_mode(label_mode)
    if mode == "exit_after_h2":
        return 2
    return 1


def _pre_exit_label(exit_delay_bars: int) -> str:
    delay = max(int(exit_delay_bars), 1)
    if delay == 1:
        return "no H1 hit"
    return f"no H1-H{delay} hit"


def _tp_sweep_thresholds() -> list[float]:
    count = int(np.floor((TP_SWEEP_END - TP_SWEEP_START) / TP_SWEEP_STEP + 1e-12)) + 1
    return [float(TP_SWEEP_START + idx * TP_SWEEP_STEP) for idx in range(max(count, 0))]


def _empty_tp_sweep_rows(
    split: str,
    thresholds: list[float],
    group: str,
) -> list[dict[str, Any]]:
    return [
        {
            "split": split,
            "group": group,
            "tp_threshold": float(threshold),
            "sample_count": 0,
            "hit_count": 0,
            "hit_rate": 0.0,
        }
        for threshold in thresholds
    ]


def _tp_sweep_rows(
    split: str,
    group: str,
    selected_base_index: pd.Index,
    path_returns: pd.DataFrame,
    thresholds: list[float],
    min_h: int = 2,
) -> list[dict[str, Any]]:
    selected_path = path_returns.reindex(selected_base_index)
    high_cols = _future_high_columns(selected_path, min_h=min_h)
    selected_path = selected_path.dropna(subset=high_cols) if high_cols else selected_path.iloc[0:0]
    total = int(len(selected_path))
    if total == 0 or not high_cols:
        return _empty_tp_sweep_rows(split, thresholds, group=group)

    high_values = selected_path[high_cols].apply(pd.to_numeric, errors="coerce")
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        hit = (high_values > float(threshold)).any(axis=1)
        hit_count = int(hit.sum())
        rows.append(
            {
                "split": split,
                "group": group,
                "tp_threshold": float(threshold),
                "sample_count": total,
                "hit_count": hit_count,
                "hit_rate": hit_count / total if total else 0.0,
            }
        )
    return rows


def _future_high_columns(frame: pd.DataFrame, min_h: int) -> list[str]:
    columns: list[tuple[int, str]] = []
    min_h = max(int(min_h), 1)
    for column in frame.columns:
        text = str(column)
        if not text.startswith("high_h"):
            continue
        suffix = text[len("high_h") :]
        if not suffix.isdigit():
            continue
        step = int(suffix)
        if step >= min_h:
            columns.append((step, text))
    return [name for _, name in sorted(columns)]


def _post_h1_high_columns(frame: pd.DataFrame) -> list[str]:
    return _future_high_columns(frame, min_h=2)


def _plot_summary(
    summary: pd.DataFrame,
    tp_sweep: pd.DataFrame,
    chart_path: Path,
    base_label: str,
    exit_label: str,
    base_ensemble: str,
    tp_threshold: float,
    exit_delay_bars: int,
) -> None:
    pre_exit_label = _pre_exit_label(exit_delay_bars)
    post_exit_start_h = max(int(exit_delay_bars), 1) + 1
    fig, (
        ax_bar,
        ax_sweep,
        ax_all_table,
        ax_base_table,
        ax_pre_exit_table,
        ax_exit_table,
        ax_no_exit_table,
        ax_table,
    ) = plt.subplots(
        8,
        1,
        figsize=(15.5, 18.0),
        gridspec_kw={
            "height_ratios": [2.0, 2.0, 1.10, 1.10, 1.10, 1.10, 1.10, 1.55]
        },
        constrained_layout=True,
    )
    x = np.arange(len(summary))
    width = 0.34
    ax_bar.bar(
        x - width / 2,
        summary["base_no_pre_exit_hit_rate"].astype(float) * 100.0,
        width,
        label=f"base signal {pre_exit_label} / base signal",
        color="#4c78a8",
    )
    ax_bar.bar(
        x + width / 2,
        summary["exit_selected_after_pre_exit_rate"].astype(float) * 100.0,
        width,
        label=f"exit selected / base {pre_exit_label}",
        color="#f58518",
    )
    ax_bar.set_xticks(x, summary["split"].astype(str).tolist())
    ax_bar.set_ylim(0.0, 100.0)
    ax_bar.set_ylabel("Rate %")
    ax_bar.grid(True, axis="y", alpha=0.25)
    ax_bar.legend(loc="upper left")
    ax_bar.set_title(
        f"Two-stage backtest | base={base_ensemble.upper()} | TP={tp_threshold:.4g}\n"
        f"{base_label}\nexit: {exit_label}",
        fontsize=10,
    )

    if not tp_sweep.empty:
        styles = {
            ("all_signal", "val"): ("#9c755f", "-"),
            ("all_signal", "test"): ("#9c755f", "--"),
            ("base_no_h1", "val"): ("#4c78a8", "-"),
            ("base_no_h1", "test"): ("#4c78a8", "--"),
            ("base_no_pre_exit_hit", "val"): ("#b279a2", "-"),
            ("base_no_pre_exit_hit", "test"): ("#b279a2", "--"),
            ("exit_selected", "val"): ("#f58518", "-"),
            ("exit_selected", "test"): ("#f58518", "--"),
            ("no_exit_selected", "val"): ("#54a24b", "-"),
            ("no_exit_selected", "test"): ("#54a24b", "--"),
        }
        for (group_name, split), group in tp_sweep.groupby(["group", "split"], sort=False):
            color, linestyle = styles.get((str(group_name), str(split)), ("#777777", "-"))
            ax_sweep.plot(
                group["tp_threshold"].astype(float) * 100.0,
                group["hit_rate"].astype(float) * 100.0,
                marker="o",
                linestyle=linestyle,
                linewidth=1.4,
                label=f"{group_name} {split}",
                color=color,
            )
        ax_sweep.set_ylabel("Hit rate %")
        ax_sweep.set_xlabel("TP threshold %")
        ax_sweep.set_ylim(0.0, 100.0)
        ax_sweep.grid(True, alpha=0.25)
        ax_sweep.legend(loc="upper right")
        ax_sweep.set_title("TP hit rate by remaining horizon window")
    else:
        ax_sweep.axis("off")

    _draw_table(
        ax_all_table,
        _sweep_table(tp_sweep, group_name="all_signal"),
        title="all_signal: TP hit rate in H1+ / total base signals",
        font_size=6.3,
    )
    _draw_table(
        ax_base_table,
        _sweep_table(tp_sweep, group_name="base_no_h1"),
        title="base_no_h1: TP hit rate in H2-H5 / total base_no_h1",
        font_size=6.3,
    )
    _draw_table(
        ax_pre_exit_table,
        _sweep_table(tp_sweep, group_name="base_no_pre_exit_hit"),
        title=(
            f"base_no_pre_exit_hit: TP hit rate in H{post_exit_start_h}+ / "
            f"total base {pre_exit_label}"
        ),
        font_size=6.3,
    )
    _draw_table(
        ax_exit_table,
        _sweep_table(tp_sweep, group_name="exit_selected"),
        title=f"exit_selected: TP hit rate in H{post_exit_start_h}+ / total exit_selected",
        font_size=6.3,
    )
    _draw_table(
        ax_no_exit_table,
        _sweep_table(tp_sweep, group_name="no_exit_selected"),
        title=(
            f"no_exit_selected: TP hit rate in H{post_exit_start_h}+ / "
            "total no_exit_selected"
        ),
        font_size=6.3,
    )

    table_df = summary.copy()
    for column in [
        "base_no_h1_rate",
        "exit_selected_after_no_h1_rate",
        "base_no_pre_exit_hit_rate",
        "exit_selected_after_pre_exit_rate",
        "tp_threshold",
        "exit_top_fraction",
    ]:
        if column in table_df.columns:
            table_df[column] = table_df[column].astype(float).map(lambda value: f"{value:.2%}")
    for column in ["base_pred_threshold", "exit_pred_threshold"]:
        if column in table_df.columns:
            table_df[column] = table_df[column].map(
                lambda value: "n/a" if pd.isna(value) else f"{float(value):.6f}"
            )
    _draw_table(
        ax_table,
        table_df,
        title="summary",
        font_size=7.0,
    )
    fig.savefig(chart_path, dpi=170)
    plt.close(fig)


def _sweep_table(tp_sweep: pd.DataFrame, group_name: str) -> pd.DataFrame:
    columns = ["split"] + [_format_threshold_pct(threshold) for threshold in _tp_sweep_thresholds()]
    if tp_sweep.empty or "group" not in tp_sweep.columns:
        return pd.DataFrame(columns=columns)
    subset = tp_sweep[tp_sweep["group"].astype(str) == str(group_name)]
    if subset.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, str]] = []
    for split, group in subset.groupby("split", sort=False):
        sorted_group = group.sort_values("tp_threshold")
        n_value = pd.to_numeric(sorted_group["sample_count"], errors="coerce").max()
        n = int(n_value) if pd.notna(n_value) else 0
        row: dict[str, str] = {"split": f"{split} n={n}"}
        for _, item in sorted_group.iterrows():
            row[_format_threshold_pct(float(item["tp_threshold"]))] = _format_pct(
                float(item["hit_rate"])
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns).fillna("")


def _draw_table(
    ax: plt.Axes,
    table_df: pd.DataFrame,
    title: str,
    font_size: float,
) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=9, loc="left", pad=3)
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        bbox=[0.0, 0.0, 1.0, 0.88],
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.10)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.30)
        if row == 0:
            cell.set_facecolor("#222831")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f7fbff" if row % 2 else "#fff7e8")


def _format_threshold_pct(value: float) -> str:
    return f"{float(value) * 100.0:.2f}%"


def _format_pct(value: float) -> str:
    return f"{float(value):.2%}"


def _backtest_name(
    base_specs: list[ModelSpec],
    exit_spec: ModelSpec,
    base_ensemble: str,
    tp_threshold: float,
) -> str:
    base_name = "_".join(
        f"{_safe_name(spec.archive_path.stem)}_r{spec.rank:02d}"
        for spec in base_specs
    )
    exit_name = f"{_safe_name(exit_spec.archive_path.stem)}_r{exit_spec.rank:02d}"
    tp_name = _threshold_filename_token(float(tp_threshold))
    exit_top_name = _threshold_filename_token(float(exit_spec.top_fraction))
    return (
        f"strategy_exit_top_{exit_top_name}_{base_ensemble}_{base_name}_"
        f"exit_{exit_name}_tp_{tp_name}"
    )


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("._") or "run"


def _parse_spec(
    value: str,
    default_top_fraction: float,
    require_top_fraction: bool = False,
) -> ModelSpec:
    parts = [part.strip() for part in str(value).split("#")]
    if len(parts) not in {4, 5}:
        raise ValueError(
            "Spec must be ARCHIVE#RANK#MODE#THRESHOLD[#TOP_FRACTION], "
            f"got: {value!r}"
        )
    archive_text, rank_text, mode_text, threshold_text = parts[:4]
    mode_text = config.canonical_label_mode(mode_text)
    if require_top_fraction and len(parts) < 5:
        raise ValueError("--exit spec must include TOP_FRACTION or use --exit-top-fraction.")
    top_fraction = float(parts[4]) if len(parts) == 5 and parts[4] else float(default_top_fraction)
    if not 0 < top_fraction <= 1:
        raise ValueError(f"top fraction must be in (0, 1], got {top_fraction}.")
    return ModelSpec(
        archive_path=Path(archive_text),
        rank=int(rank_text),
        label_mode=mode_text,
        label_threshold=float(threshold_text),
        top_fraction=top_fraction,
    )


def _json_safe_float(value: float) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _validate_top_fractions(values: list[float] | tuple[float, ...]) -> list[float]:
    unique_values: list[float] = []
    for value in values:
        number = float(value)
        if not 0 < number <= 1:
            raise ValueError(f"exit top fraction must be in (0, 1], got {number}.")
        if number not in unique_values:
            unique_values.append(number)
    if not unique_values:
        raise ValueError("At least one exit top fraction is required.")
    return unique_values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        action="append",
        nargs="+",
        required=True,
        help=(
            "Base signal spec(s): ARCHIVE#RANK#MODE#THRESHOLD. "
            "Base top fraction is config.TRADE_TOP_FRACTION unless a fifth field is supplied."
        ),
    )
    parser.add_argument(
        "--base-ensemble",
        choices=["and", "or"],
        default="and",
        help="How to combine multiple base specs. Default: and.",
    )
    parser.add_argument(
        "--exit",
        required=True,
        help=(
            "Exit model spec: ARCHIVE#RANK#MODE#THRESHOLD[#TOP_FRACTION]. "
            "If TOP_FRACTION and --exit-top-fraction are omitted, "
            "EXIT_TOP_FRACTIONS from this file is used."
        ),
    )
    parser.add_argument(
        "--exit-top-fraction",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Override exit model top fraction(s). Example: "
            "--exit-top-fraction 0.10 0.20 0.30."
        ),
    )
    parser.add_argument(
        "--tp-threshold",
        type=float,
        default=None,
        help=(
            "TP threshold used to decide whether H1 was hit. "
            "Default: exit spec label threshold."
        ),
    )
    parser.add_argument("--data", default=str(config.DATA_PATH), help="Crypto OHLCV CSV path.")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Output directory. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument("--val-start", default=config.VAL_START)
    parser.add_argument("--test-start", default=config.TEST_START)
    parser.add_argument("--test-end", default=config.TEST_END)
    args = parser.parse_args()

    raw_base_specs = [raw for group in args.base for raw in group]
    base_specs = [
        _parse_spec(
            raw,
            default_top_fraction=float(config.TRADE_TOP_FRACTION),
            require_top_fraction=False,
        )
        for raw in raw_base_specs
    ]
    exit_spec_has_top_fraction = len(str(args.exit).split("#")) >= 5
    exit_default_top = float(EXIT_TOP_FRACTIONS[0])
    exit_spec = _parse_spec(
        args.exit,
        default_top_fraction=exit_default_top,
        require_top_fraction=False,
    )
    if args.exit_top_fraction is not None:
        exit_top_fractions = _validate_top_fractions(args.exit_top_fraction)
    elif exit_spec_has_top_fraction:
        exit_top_fractions = _validate_top_fractions([exit_spec.top_fraction])
    else:
        exit_top_fractions = _validate_top_fractions(EXIT_TOP_FRACTIONS)

    results: list[BacktestResult] = []
    for exit_top_fraction in exit_top_fractions:
        active_exit_spec = ModelSpec(
            archive_path=exit_spec.archive_path,
            rank=exit_spec.rank,
            label_mode=exit_spec.label_mode,
            label_threshold=exit_spec.label_threshold,
            top_fraction=float(exit_top_fraction),
        )
        results.append(
            run_backtest(
                base_specs=base_specs,
                exit_spec=active_exit_spec,
                data_path=args.data,
                out_dir=args.out_dir,
                base_ensemble=args.base_ensemble,
                tp_threshold=args.tp_threshold,
                val_start=args.val_start,
                test_start=args.test_start,
                test_end=args.test_end,
            )
        )
    print(
        json.dumps(
            [
                {
                    "csv": str(result.csv_path),
                    "tp_sweep_csv": str(result.tp_sweep_csv_path),
                    "chart": str(result.chart_path),
                }
                for result in results
            ],
            indent=2,
        )
    )
    for result in results:
        print(result.summary.to_string(index=False))
        if not result.tp_sweep.empty:
            print(result.tp_sweep.to_string(index=False))


if __name__ == "__main__":
    main()
