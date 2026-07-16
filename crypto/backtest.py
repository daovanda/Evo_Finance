"""Backtest the live staged crypto strategy.

This module evaluates the practical flow:

1. One or more base archive individuals create the original trade signal at t.
2. If the trade has not hit take profit during H1, exit_after_h1 is evaluated
   at t+1.
3. If exit_after_h1 does not select the trade and H2 also does not hit take
   profit, exit_after_h2 is evaluated at t+2.
4. Report each stage with table-only charts.

Examples:
    python -m crypto.backtest ^
      --base crypto/results/crypto_btc_mfe_h5_seed1_12h.json#1#mfe#0.003 ^
      --base crypto/results/crypto_btc_close_exit_h5_seed1_12h.json#1#close_exit#0.001 ^
      --base-ensemble and ^
      --exit1 crypto/results/crypto_btc_exit_after_h1_h5_noh1_tp04_seed1_12h.json#1#exit_after_h1#0.004 ^
      --exit2 crypto/results/crypto_btc_exit_after_h2_h5_tp04_seed1_12h.json#1#exit_after_h2#0.004 ^
      --tp-threshold 0.004
"""

from __future__ import annotations

import argparse
import hashlib
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
    exit1_spec: ModelSpec,
    exit2_spec: ModelSpec,
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

    tp = float(tp_threshold if tp_threshold is not None else exit1_spec.label_threshold)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Backtest setup: base_specs=%d | base_ensemble=%s | exit1=%s rank %d | exit2=%s rank %d",
        len(base_specs),
        base_ensemble.upper(),
        exit1_spec.archive_path,
        exit1_spec.rank,
        exit2_spec.archive_path,
        exit2_spec.rank,
    )
    logger.info("Loading crypto data from %s", data_path)
    raw_df = load_ohlcv(data_path)
    base_horizons = _normalize_horizons(config.HOLDING_HORIZONS, "config.HOLDING_HORIZONS")
    exit1_horizons = _archive_horizons(
        exit1_spec.archive_path,
        fallback=base_horizons,
        label="exit1",
    )
    exit2_horizons = _archive_horizons(
        exit2_spec.archive_path,
        fallback=base_horizons,
        label="exit2",
    )
    all_horizons = sorted(set(base_horizons + exit1_horizons + exit2_horizons))
    horizon = int(max(all_horizons))
    purge_bars = config.purge_bars_for_horizons(all_horizons)
    logger.info(
        "Backtest horizons: base=%s | exit1=%s | exit2=%s | path_horizon=h%d",
        base_horizons,
        exit1_horizons,
        exit2_horizons,
        horizon,
    )

    entries = [
        _load_rank_entry(spec.archive_path, spec.rank)
        for spec in [*base_specs, exit1_spec, exit2_spec]
    ]
    quality_train = _quality_train_index(
        raw_df=raw_df,
        spec=base_specs[0],
        horizons=all_horizons,
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
            horizons=base_horizons,
            val_start=val_start,
            test_start=test_start,
            test_end=test_end,
            purge_bars=purge_bars,
        )
        base_bundles.append(bundle)

    exit1_bundle = _train_spec_bundle(
        spec=exit1_spec,
        entry=entries[-2],
        raw_df=raw_df,
        feature_space=feature_space,
        horizons=exit1_horizons,
        val_start=val_start,
        test_start=test_start,
        test_end=test_end,
        purge_bars=purge_bars,
    )
    exit2_bundle = _train_spec_bundle(
        spec=exit2_spec,
        entry=entries[-1],
        raw_df=raw_df,
        feature_space=feature_space,
        horizons=exit2_horizons,
        val_start=val_start,
        test_start=test_start,
        test_end=test_end,
        purge_bars=purge_bars,
    )
    base_bundle = _combine_base_bundles(base_bundles, selection=base_ensemble)

    _, path_by_horizon, _ = _return_context_by_horizon(
        raw_df,
        [horizon],
        label_mode="mfe",
    )
    path_returns = path_by_horizon[horizon]

    summary_rows: list[dict[str, Any]] = []
    tp_sweep_rows: list[dict[str, Any]] = []
    tp_sweep_thresholds = _tp_sweep_thresholds()
    for split_name, base_split, exit1_split, exit2_split in [
        ("val", base_bundle.val, exit1_bundle.val, exit2_bundle.val),
        ("test", base_bundle.test, exit1_bundle.test, exit2_bundle.test),
    ]:
        summary_row, split_tp_rows = _summarize_split(
            split=split_name,
            base_split=base_split,
            exit1_split=exit1_split,
            exit2_split=exit2_split,
            path_returns=path_returns,
            raw_index=pd.DatetimeIndex(raw_df.index),
            tp_threshold=tp,
            tp_sweep_thresholds=tp_sweep_thresholds,
        )
        summary_rows.append(summary_row)
        tp_sweep_rows.extend(split_tp_rows)

    summary = pd.DataFrame(summary_rows)
    tp_sweep = pd.DataFrame(tp_sweep_rows)
    run_name = _backtest_name(base_specs, exit1_spec, exit2_spec, base_ensemble, tp)
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
        exit1_label=exit1_bundle.label,
        exit2_label=exit2_bundle.label,
        base_ensemble=base_ensemble,
        tp_threshold=tp,
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
    horizons: list[int],
    val_start: str,
    test_start: str,
    test_end: str | None,
    purge_bars: int,
) -> pd.Index:
    labeled = add_binary_labels(
        raw_df,
        horizons=horizons,
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


def _archive_horizons(path: Path, fallback: list[int], label: str) -> list[int]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:
        logger.warning(
            "Could not read metadata.horizons for %s archive %s; using fallback %s. Error: %s",
            label,
            path,
            fallback,
            exc,
        )
        return list(fallback)

    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    raw_horizons = metadata.get("horizons") if isinstance(metadata, dict) else None
    if raw_horizons is None:
        logger.warning(
            "%s archive %s has no metadata.horizons; using fallback %s.",
            label,
            path,
            fallback,
        )
        return list(fallback)
    horizons = _normalize_horizons(raw_horizons, f"{label} metadata.horizons")
    logger.info("%s archive horizons from metadata: %s", label, horizons)
    return horizons


def _normalize_horizons(values: Any, label: str) -> list[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple, set)):
        raise ValueError(f"{label} must be a list of positive integers, got: {values!r}")
    horizons = sorted({int(value) for value in values})
    if not horizons:
        raise ValueError(f"{label} must not be empty.")
    if any(value < 1 for value in horizons):
        raise ValueError(f"{label} must contain positive integers, got: {values!r}")
    return horizons


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
    horizons: list[int],
    val_start: str,
    test_start: str,
    test_end: str | None,
    purge_bars: int,
) -> BundleSignals:
    logger.info(
        "Training %s rank %d | mode=%s threshold=%.6f top=%.2f%% | horizons=%s",
        spec.archive_path,
        spec.rank,
        spec.label_mode,
        spec.label_threshold,
        spec.top_fraction * 100.0,
        horizons,
    )
    individual = _entry_to_individual(entry)
    labeled = add_binary_labels(
        raw_df,
        horizons=horizons,
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
    for horizon in horizons:
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
    exit1_split: SplitSignals,
    exit2_split: SplitSignals,
    path_returns: pd.DataFrame,
    raw_index: pd.DatetimeIndex,
    tp_threshold: float,
    tp_sweep_thresholds: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_selected = pd.Index(base_split.selected_index)
    base_path = path_returns.reindex(base_selected).dropna(subset=["high_h1", "high_h2"])
    base_signals = int(len(base_path))
    if base_signals == 0:
        summary = {
            "split": split,
            "base_signals": 0,
            "base_no_h1": 0,
            "base_no_h1_rate": 0.0,
            "exit1_selected": 0,
            "exit1_selected_rate": 0.0,
            "exit1_no_selected": 0,
            "exit1_no_selected_rate": 0.0,
            "exit1_top_fraction": float(exit1_split.top_fraction),
            "exit1_pred_threshold": _json_safe_float(exit1_split.pred_threshold),
            "exit1_no_selected_no_h2": 0,
            "exit1_no_selected_no_h2_rate": 0.0,
            "exit2_selected": 0,
            "exit2_selected_rate": 0.0,
            "exit2_no_selected": 0,
            "exit2_no_selected_rate": 0.0,
            "exit2_top_fraction": float(exit2_split.top_fraction),
            "exit2_pred_threshold": _json_safe_float(exit2_split.pred_threshold),
            "base_pred_threshold": _json_safe_float(base_split.pred_threshold),
            "tp_threshold": float(tp_threshold),
        }
        return summary, (
            _empty_stage_tp_rows(split, tp_sweep_thresholds)
        )

    no_h1_index = pd.Index(base_path.index[base_path["high_h1"] <= float(tp_threshold)])
    no_h1_count = int(len(no_h1_index))

    exit1_mapping = _future_bar_mapping(no_h1_index, raw_index, offset=1)
    exit1_selected_index = pd.Index(exit1_split.selected_index)
    exit1_selected_mapping = exit1_mapping[
        exit1_mapping["exit_index"].isin(exit1_selected_index)
    ]
    exit1_no_selected_mapping = exit1_mapping[
        ~exit1_mapping["exit_index"].isin(exit1_selected_index)
    ]
    exit1_selected_base_index = pd.Index(exit1_selected_mapping["base_index"])
    exit1_no_selected_base_index = pd.Index(exit1_no_selected_mapping["base_index"])
    exit1_selected_count = int(len(exit1_selected_base_index))
    exit1_no_selected_count = int(len(exit1_no_selected_base_index))

    exit1_no_selected_path = path_returns.reindex(exit1_no_selected_base_index)
    exit1_no_selected_no_h2_index = pd.Index(
        exit1_no_selected_path.index[
            pd.to_numeric(exit1_no_selected_path["high_h2"], errors="coerce")
            <= float(tp_threshold)
        ]
    )
    exit1_no_selected_no_h2_count = int(len(exit1_no_selected_no_h2_index))

    exit2_mapping = _future_bar_mapping(exit1_no_selected_no_h2_index, raw_index, offset=2)
    exit2_selected_index = pd.Index(exit2_split.selected_index)
    exit2_selected_mapping = exit2_mapping[
        exit2_mapping["exit_index"].isin(exit2_selected_index)
    ]
    exit2_no_selected_mapping = exit2_mapping[
        ~exit2_mapping["exit_index"].isin(exit2_selected_index)
    ]
    exit2_selected_base_index = pd.Index(exit2_selected_mapping["base_index"])
    exit2_no_selected_base_index = pd.Index(exit2_no_selected_mapping["base_index"])
    exit2_selected_count = int(len(exit2_selected_base_index))
    exit2_no_selected_count = int(len(exit2_no_selected_base_index))

    summary = {
        "split": split,
        "base_signals": base_signals,
        "base_no_h1": no_h1_count,
        "base_no_h1_rate": no_h1_count / base_signals if base_signals else 0.0,
        "exit1_selected": exit1_selected_count,
        "exit1_selected_rate": exit1_selected_count / no_h1_count if no_h1_count else 0.0,
        "exit1_no_selected": exit1_no_selected_count,
        "exit1_no_selected_rate": (
            exit1_no_selected_count / no_h1_count if no_h1_count else 0.0
        ),
        "exit1_top_fraction": float(exit1_split.top_fraction),
        "exit1_no_selected_no_h2": exit1_no_selected_no_h2_count,
        "exit1_no_selected_no_h2_rate": (
            exit1_no_selected_no_h2_count / exit1_no_selected_count
            if exit1_no_selected_count
            else 0.0
        ),
        "exit2_selected": exit2_selected_count,
        "exit2_selected_rate": (
            exit2_selected_count / exit1_no_selected_no_h2_count
            if exit1_no_selected_no_h2_count
            else 0.0
        ),
        "exit2_no_selected": exit2_no_selected_count,
        "exit2_no_selected_rate": (
            exit2_no_selected_count / exit1_no_selected_no_h2_count
            if exit1_no_selected_no_h2_count
            else 0.0
        ),
        "exit2_top_fraction": float(exit2_split.top_fraction),
        "base_pred_threshold": _json_safe_float(base_split.pred_threshold),
        "exit1_pred_threshold": _json_safe_float(exit1_split.pred_threshold),
        "exit2_pred_threshold": _json_safe_float(exit2_split.pred_threshold),
        "tp_threshold": float(tp_threshold),
    }
    all_signal_tp_rows = _tp_sweep_rows(
        split=split,
        group="p1_base_signal",
        selected_base_index=pd.Index(base_path.index),
        path_returns=path_returns,
        thresholds=tp_sweep_thresholds,
        min_h=1,
    )
    base_no_h1_tp_rows = _tp_sweep_rows(
        split=split,
        group="p1_base_no_h1",
        selected_base_index=no_h1_index,
        path_returns=path_returns,
        thresholds=tp_sweep_thresholds,
        min_h=2,
    )
    exit1_selected_tp_rows = _tp_sweep_rows(
        split=split,
        group="p1_exit_after_h1_selected",
        selected_base_index=exit1_selected_base_index,
        path_returns=path_returns,
        thresholds=tp_sweep_thresholds,
        min_h=2,
    )
    exit1_no_selected_tp_rows = _tp_sweep_rows(
        split=split,
        group="p1_exit_after_h1_no_selected",
        selected_base_index=exit1_no_selected_base_index,
        path_returns=path_returns,
        thresholds=tp_sweep_thresholds,
        min_h=2,
    )
    exit1_no_selected_h2_tp_rows = _tp_sweep_rows(
        split=split,
        group="p1_exit_after_h1_no_selected_h2",
        selected_base_index=exit1_no_selected_base_index,
        path_returns=path_returns,
        thresholds=tp_sweep_thresholds,
        min_h=2,
        max_h=2,
    )

    exit1_no_selected_no_h2_tp_rows = _tp_sweep_rows(
        split=split,
        group="p2_exit_after_h1_no_selected_no_h2",
        selected_base_index=exit1_no_selected_no_h2_index,
        path_returns=path_returns,
        thresholds=tp_sweep_thresholds,
        min_h=3,
    )
    exit2_selected_tp_rows = _tp_sweep_rows(
        split=split,
        group="p2_exit_after_h2_selected",
        selected_base_index=exit2_selected_base_index,
        path_returns=path_returns,
        thresholds=tp_sweep_thresholds,
        min_h=3,
    )
    exit2_no_selected_tp_rows = _tp_sweep_rows(
        split=split,
        group="p2_exit_after_h2_no_selected",
        selected_base_index=exit2_no_selected_base_index,
        path_returns=path_returns,
        thresholds=tp_sweep_thresholds,
        min_h=3,
    )
    return (
        summary,
        all_signal_tp_rows
        + base_no_h1_tp_rows
        + exit1_selected_tp_rows
        + exit1_no_selected_tp_rows
        + exit1_no_selected_h2_tp_rows
        + exit1_no_selected_no_h2_tp_rows
        + exit2_selected_tp_rows
        + exit2_no_selected_tp_rows,
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
            "miss_count": 0,
            "miss_return_mean": float("nan"),
        }
        for threshold in thresholds
    ]


def _empty_stage_tp_rows(split: str, thresholds: list[float]) -> list[dict[str, Any]]:
    groups = [
        "p1_base_signal",
        "p1_base_no_h1",
        "p1_exit_after_h1_selected",
        "p1_exit_after_h1_no_selected",
        "p1_exit_after_h1_no_selected_h2",
        "p2_exit_after_h1_no_selected_no_h2",
        "p2_exit_after_h2_selected",
        "p2_exit_after_h2_no_selected",
    ]
    rows: list[dict[str, Any]] = []
    for group in groups:
        rows.extend(_empty_tp_sweep_rows(split, thresholds, group=group))
    return rows


def _tp_sweep_rows(
    split: str,
    group: str,
    selected_base_index: pd.Index,
    path_returns: pd.DataFrame,
    thresholds: list[float],
    min_h: int = 2,
    max_h: int | None = None,
) -> list[dict[str, Any]]:
    selected_path = path_returns.reindex(selected_base_index)
    high_cols = _future_high_columns(selected_path, min_h=min_h, max_h=max_h)
    selected_path = selected_path.dropna(subset=high_cols) if high_cols else selected_path.iloc[0:0]
    total = int(len(selected_path))
    if total == 0 or not high_cols:
        return _empty_tp_sweep_rows(split, thresholds, group=group)

    high_values = selected_path[high_cols].apply(pd.to_numeric, errors="coerce")
    close_col = f"close_h{_max_h_from_high_columns(high_cols)}"
    close_values = (
        pd.to_numeric(selected_path[close_col], errors="coerce")
        if close_col in selected_path.columns
        else pd.Series(np.nan, index=selected_path.index, dtype=float)
    )
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        hit = (high_values > float(threshold)).any(axis=1)
        hit_count = int(hit.sum())
        miss = ~hit
        miss_count = int(miss.sum())
        miss_close = close_values[miss & close_values.notna()]
        miss_return_mean = float(miss_close.mean()) if not miss_close.empty else float("nan")
        rows.append(
            {
                "split": split,
                "group": group,
                "tp_threshold": float(threshold),
                "sample_count": total,
                "hit_count": hit_count,
                "hit_rate": hit_count / total if total else 0.0,
                "miss_count": miss_count,
                "miss_return_mean": miss_return_mean,
            }
        )
    return rows


def _future_high_columns(
    frame: pd.DataFrame,
    min_h: int,
    max_h: int | None = None,
) -> list[str]:
    columns: list[tuple[int, str]] = []
    min_h = max(int(min_h), 1)
    max_h_value = int(max_h) if max_h is not None else None
    for column in frame.columns:
        text = str(column)
        if not text.startswith("high_h"):
            continue
        suffix = text[len("high_h") :]
        if not suffix.isdigit():
            continue
        step = int(suffix)
        if max_h_value is not None and step > max_h_value:
            continue
        if step >= min_h:
            columns.append((step, text))
    return [name for _, name in sorted(columns)]


def _max_h_from_high_columns(columns: list[str]) -> int:
    steps: list[int] = []
    for column in columns:
        text = str(column)
        if not text.startswith("high_h"):
            continue
        suffix = text[len("high_h") :]
        if suffix.isdigit():
            steps.append(int(suffix))
    return max(steps) if steps else 1


def _plot_summary(
    summary: pd.DataFrame,
    tp_sweep: pd.DataFrame,
    chart_path: Path,
    base_label: str,
    exit1_label: str,
    exit2_label: str,
    base_ensemble: str,
    tp_threshold: float,
) -> None:
    fig, (
        ax_p1_summary,
        ax_p1_base,
        ax_p1_no_h1,
        ax_p1_selected,
        ax_p1_no_selected,
        ax_p1_no_selected_h2,
        ax_p2_summary,
        ax_p2_base,
        ax_p2_selected,
        ax_p2_no_selected,
    ) = plt.subplots(
        10,
        1,
        figsize=(17.5, 25.0),
        gridspec_kw={
            "height_ratios": [
                1.25,
                1.35,
                1.35,
                1.35,
                1.35,
                1.35,
                1.25,
                1.35,
                1.35,
                1.35,
            ]
        },
        constrained_layout=True,
    )

    _draw_table(
        ax_p1_summary,
        _part1_summary_table(summary),
        title=(
            f"Part 1: exit_after_h1 | base={base_ensemble.upper()} | TP={tp_threshold:.4g}\n"
            f"base: {base_label}\nexit_after_h1: {exit1_label}"
        ),
        font_size=7.0,
    )
    _draw_table(
        ax_p1_base,
        _sweep_table(tp_sweep, group_name="p1_base_signal"),
        title="base signal: TP hitrate in H1-H5 / total base signal",
        font_size=6.3,
    )
    _draw_table(
        ax_p1_no_h1,
        _sweep_table(tp_sweep, group_name="p1_base_no_h1"),
        title="base no H1: TP hitrate in H2-H5 / total base no H1",
        font_size=6.3,
    )
    _draw_table(
        ax_p1_selected,
        _sweep_table(tp_sweep, group_name="p1_exit_after_h1_selected"),
        title="exit_after_h1 selected: TP hitrate in H2-H5 / total exit_after_h1 selected",
        font_size=6.3,
    )
    _draw_table(
        ax_p1_no_selected,
        _sweep_table(tp_sweep, group_name="p1_exit_after_h1_no_selected"),
        title=(
            "exit_after_h1 no selected: "
            "TP hitrate in H2-H5 / total exit_after_h1 no selected"
        ),
        font_size=6.3,
    )
    _draw_table(
        ax_p1_no_selected_h2,
        _sweep_table(tp_sweep, group_name="p1_exit_after_h1_no_selected_h2"),
        title="exit_after_h1 no selected: TP hitrate in H2 / total exit_after_h1 no selected",
        font_size=6.3,
    )

    _draw_table(
        ax_p2_summary,
        _part2_summary_table(summary),
        title=f"Part 2: exit_after_h2\nexit_after_h2: {exit2_label}",
        font_size=7.0,
    )
    _draw_table(
        ax_p2_base,
        _sweep_table(tp_sweep, group_name="p2_exit_after_h1_no_selected_no_h2"),
        title=(
            "base_e_after_h1_no_selected_no_H2: "
            "TP hitrate in H3-H5 / total base_e_after_h1_no_selected_no_H2"
        ),
        font_size=6.3,
    )
    _draw_table(
        ax_p2_selected,
        _sweep_table(tp_sweep, group_name="p2_exit_after_h2_selected"),
        title="exit_after_h2 selected: TP hitrate in H3-H5 / total exit_after_h2 selected",
        font_size=6.3,
    )
    _draw_table(
        ax_p2_no_selected,
        _sweep_table(tp_sweep, group_name="p2_exit_after_h2_no_selected"),
        title="exit_after_h2 no selected: TP hitrate in H3-H5 / total exit_after_h2 no selected",
        font_size=6.3,
    )
    fig.savefig(chart_path, dpi=170)
    plt.close(fig)


def _part1_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in summary.iterrows():
        base_no_h1 = _count_rate_cell(row, "base_no_h1", "base_no_h1_rate")
        exit1_selected = _count_rate_cell(row, "exit1_selected", "exit1_selected_rate")
        exit1_no_selected = _count_rate_cell(
            row,
            "exit1_no_selected",
            "exit1_no_selected_rate",
        )
        rows.append(
            {
                "split": str(row.get("split", "")),
                "base signal": _count_cell(row.get("base_signals")),
                "base no H1": base_no_h1,
                "e_after_h1 selected": exit1_selected,
                "e_after_h1 no selected": exit1_no_selected,
                "exit_h1_top_fraction": _format_pct(float(row.get("exit1_top_fraction", 0.0))),
                "TP_threshold": _format_pct(float(row.get("tp_threshold", 0.0))),
            }
        )
    return pd.DataFrame(rows)


def _part2_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in summary.iterrows():
        h1_no_selected = _count_cell(row.get("exit1_no_selected"))
        no_h2 = _count_rate_cell(
            row,
            "exit1_no_selected_no_h2",
            "exit1_no_selected_no_h2_rate",
        )
        exit2_selected = _count_rate_cell(row, "exit2_selected", "exit2_selected_rate")
        exit2_no_selected = _count_rate_cell(
            row,
            "exit2_no_selected",
            "exit2_no_selected_rate",
        )
        rows.append(
            {
                "split": str(row.get("split", "")),
                "e_after_h1 no selected": h1_no_selected,
                "base_e_after_h1_no_selected_no_H2": no_h2,
                "e_after_h2 selected": exit2_selected,
                "e_after_h2 no selected": exit2_no_selected,
                "exit_h2_top_fraction": _format_pct(float(row.get("exit2_top_fraction", 0.0))),
                "TP_threshold": _format_pct(float(row.get("tp_threshold", 0.0))),
            }
        )
    return pd.DataFrame(rows)


def _count_rate_cell(row: pd.Series, count_col: str, rate_col: str) -> str:
    return f"{_count_cell(row.get(count_col))} ({_format_pct(float(row.get(rate_col, 0.0)))})"


def _count_cell(value: Any) -> str:
    try:
        if pd.isna(value):
            return "0"
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


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
        hit_row: dict[str, str] = {"split": f"{split} hit n={n}"}
        miss_return_row: dict[str, str] = {"split": f"{split} miss ret"}
        for _, item in sorted_group.iterrows():
            threshold_label = _format_threshold_pct(float(item["tp_threshold"]))
            hit_row[threshold_label] = _format_pct(
                float(item["hit_rate"])
            )
            miss_return = item.get("miss_return_mean", float("nan"))
            miss_return_row[threshold_label] = _format_signed_pct(miss_return)
        rows.append(hit_row)
        rows.append(miss_return_row)
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


def _format_signed_pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:+.2%}"


def _backtest_name(
    base_specs: list[ModelSpec],
    exit1_spec: ModelSpec,
    exit2_spec: ModelSpec,
    base_ensemble: str,
    tp_threshold: float,
) -> str:
    base_name = "_".join(
        f"b{i}_{_short_safe_name(spec.archive_path.stem)}_r{spec.rank:02d}"
        for i, spec in enumerate(base_specs, start=1)
    )
    exit1_name = (
        f"x1_{_short_safe_name(exit1_spec.archive_path.stem)}_r{exit1_spec.rank:02d}"
    )
    exit2_name = (
        f"x2_{_short_safe_name(exit2_spec.archive_path.stem)}_r{exit2_spec.rank:02d}"
    )
    tp_name = _threshold_filename_token(float(tp_threshold))
    exit1_top_name = _threshold_filename_token(float(exit1_spec.top_fraction))
    exit2_top_name = _threshold_filename_token(float(exit2_spec.top_fraction))
    return _safe_name(
        "_".join(
            [
                f"strategy_e1top_{exit1_top_name}",
                f"e2top_{exit2_top_name}",
                base_ensemble,
                base_name,
                exit1_name,
                exit2_name,
                f"tp_{tp_name}",
            ]
        )
    )


def _short_safe_name(value: str, max_len: int = 22) -> str:
    text = _safe_name(value)
    if len(text) <= max_len:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]
    head_len = max(8, max_len - len(digest) - 1)
    return f"{text[:head_len]}_{digest}"


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
        raise ValueError("Exit spec must include TOP_FRACTION or use a top-fraction CLI option.")
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


def _resolve_top_fraction_values(
    override: list[float] | None,
    spec_has_top_fraction: bool,
    spec: ModelSpec,
) -> list[float]:
    if override is not None:
        return _validate_top_fractions(override)
    if spec_has_top_fraction:
        return _validate_top_fractions([spec.top_fraction])
    return _validate_top_fractions(EXIT_TOP_FRACTIONS)


def _pair_top_fractions(
    exit1_values: list[float],
    exit2_values: list[float],
) -> list[tuple[float, float]]:
    if len(exit1_values) == len(exit2_values):
        return list(zip(exit1_values, exit2_values, strict=True))
    if len(exit1_values) == 1:
        return [(exit1_values[0], value) for value in exit2_values]
    if len(exit2_values) == 1:
        return [(value, exit2_values[0]) for value in exit1_values]
    return [(value1, value2) for value1 in exit1_values for value2 in exit2_values]


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
        "--exit1",
        required=True,
        help=(
            "Exit-after-H1 model spec: ARCHIVE#RANK#MODE#THRESHOLD[#TOP_FRACTION]. "
            "Use mode exit_after_h1."
        ),
    )
    parser.add_argument(
        "--exit2",
        required=True,
        help=(
            "Exit-after-H2 model spec: ARCHIVE#RANK#MODE#THRESHOLD[#TOP_FRACTION]. "
            "Use mode exit_after_h2."
        ),
    )
    parser.add_argument(
        "--exit1-top-fraction",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Override exit_after_h1 top fraction(s). Example: "
            "--exit1-top-fraction 0.10 0.20 0.30."
        ),
    )
    parser.add_argument(
        "--exit2-top-fraction",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Override exit_after_h2 top fraction(s). Example: "
            "--exit2-top-fraction 0.10 0.20 0.30."
        ),
    )
    parser.add_argument(
        "--tp-threshold",
        type=float,
        default=None,
        help=(
            "TP threshold used to decide whether H1/H2 was hit. "
            "Default: exit1 spec label threshold."
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
    exit1_spec_has_top_fraction = len(str(args.exit1).split("#")) >= 5
    exit2_spec_has_top_fraction = len(str(args.exit2).split("#")) >= 5
    exit1_spec = _parse_spec(
        args.exit1,
        default_top_fraction=float(EXIT_TOP_FRACTIONS[0]),
        require_top_fraction=False,
    )
    exit2_spec = _parse_spec(
        args.exit2,
        default_top_fraction=float(EXIT_TOP_FRACTIONS[0]),
        require_top_fraction=False,
    )
    exit1_top_fractions = _resolve_top_fraction_values(
        override=args.exit1_top_fraction,
        spec_has_top_fraction=exit1_spec_has_top_fraction,
        spec=exit1_spec,
    )
    exit2_top_fractions = _resolve_top_fraction_values(
        override=args.exit2_top_fraction,
        spec_has_top_fraction=exit2_spec_has_top_fraction,
        spec=exit2_spec,
    )

    results: list[BacktestResult] = []
    for exit1_top_fraction, exit2_top_fraction in _pair_top_fractions(
        exit1_top_fractions,
        exit2_top_fractions,
    ):
        active_exit1_spec = ModelSpec(
            archive_path=exit1_spec.archive_path,
            rank=exit1_spec.rank,
            label_mode=exit1_spec.label_mode,
            label_threshold=exit1_spec.label_threshold,
            top_fraction=float(exit1_top_fraction),
        )
        active_exit2_spec = ModelSpec(
            archive_path=exit2_spec.archive_path,
            rank=exit2_spec.rank,
            label_mode=exit2_spec.label_mode,
            label_threshold=exit2_spec.label_threshold,
            top_fraction=float(exit2_top_fraction),
        )
        results.append(
            run_backtest(
                base_specs=base_specs,
                exit1_spec=active_exit1_spec,
                exit2_spec=active_exit2_spec,
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
