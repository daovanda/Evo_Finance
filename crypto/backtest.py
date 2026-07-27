"""Analyze base-model signals by disjoint prediction-score bands.

The report contains only:

1. A compact base-signal overview.
2. High TP hit rates through H24 for 5%-wide score bands up to top 70%.
3. Low TP hit rates through H24 for the same bands.
4. Two-sided excursion rates through H24.
5. A simultaneous Long/Short TP strategy through close H24.

Model spec format:
    ARCHIVE#RANK#MODE#THRESHOLD[#TOP_FRACTION[#DIRECTION]]

Always quote model specs in PowerShell because an unquoted ``#`` starts a
comment.

PowerShell, one payoff base:
    python -m crypto.backtest `
      --base "crypto/results/crypto_btc_payoff_h5_seed1_resume_seed2_48h.json#1#payoff#0.002#0.05#Long" `
      --base-ensemble and `
      --tp-threshold 0.004 `
      --label-direction Long `
      --data data/crypto/BTCUSDT_15m.csv `
      --out-dir crypto/results/backtest

PowerShell, require agreement between two base individuals:
    python -m crypto.backtest `
      --base "crypto/results/crypto_btc_mfe_seed1_12h.json#1#mfe#0.003#0.10#Long" `
      --base "crypto/results/crypto_btc_close_exit_seed1_12h.json#1#close_exit#0.001#0.10#Long" `
      --base-ensemble and `
      --tp-threshold 0.004 `
      --label-direction Long `
      --data data/crypto/BTCUSDT_15m.csv `
      --out-dir crypto/results/backtest

Bash/VM, one payoff base:
    python -m crypto.backtest \
      --base "crypto/results/crypto_btc_payoff_h5_seed1_resume_seed2_36h.checkpoint.json#1#payoff#0.002#0.10#Long" \
      --base-ensemble and \
      --tp-threshold 0.004 \
      --label-direction Long \
      --data data/crypto/BTCUSDT_15m.csv \
      --out-dir crypto/results/backtest

The report image is written to crypto/results/backtest unless --out-dir is
provided. No CSV files are created.
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
TP_SWEEP_END: float = 0.007
TP_SWEEP_STEP: float = 0.0005
BASE_FRACTION_BAND_STEP: float = 0.05
BASE_FRACTION_BAND_MAX: float = 0.70
BASE_SIGNAL_ANALYSIS_HORIZON: int = 24
BASE_SIGNAL_HIGH_GROUP: str = (
    f"base_fraction_band_high_h1_h{BASE_SIGNAL_ANALYSIS_HORIZON}"
)
BASE_SIGNAL_LOW_GROUP: str = (
    f"base_fraction_band_low_h1_h{BASE_SIGNAL_ANALYSIS_HORIZON}"
)
BASE_SIGNAL_TWO_SIDED_GROUP: str = (
    f"base_fraction_band_two_sided_h1_h{BASE_SIGNAL_ANALYSIS_HORIZON}"
)
SCORE_BAND_SWEEP_START: float = 0.0
SCORE_BAND_SWEEP_END: float = 0.0100
SCORE_BAND_SWEEP_STEP: float = 0.0005
SCORE_BAND_RANGES: list[tuple[float, float]] = [
    (
        step_index * BASE_FRACTION_BAND_STEP,
        (step_index + 1) * BASE_FRACTION_BAND_STEP,
    )
    for step_index in range(
        int(round(BASE_FRACTION_BAND_MAX / BASE_FRACTION_BAND_STEP))
    )
]
SCORE_BAND_TP_H1_H4: float = 0.007
SCORE_BAND_TP_H5: float = -0.002
SCORE_BAND_CUTLOSS: float = -0.5
SCORE_BAND_ENTRY_MIN_LOW_THRESHOLD: float = -0.5
SCORE_BAND_MAX_HIGH_BELOW_THRESHOLD: float = -0.5
SCORE_BAND_WEAK_H1_TP_H2_H4: float = -0.003
SCORE_BAND_TWO_SIDED_OPTIMIZE_TP: bool = True
SCORE_BAND_TWO_SIDED_TP_LONG: float = 0.01
SCORE_BAND_TWO_SIDED_TP_SHORT: float = 0.01
TWO_SIDED_SWEEP_START: float = 0.0005
TWO_SIDED_SWEEP_END: float = 0.0100
TWO_SIDED_SWEEP_STEP: float = 0.0005
EXIT_TOP_FRACTIONS: list[float] = [0.10, 0.20, 0.30, 0.40, 0.50, 0.6, 0.7]
TP_OPT_START: float = 0.0020
TP_OPT_END: float = 0.0070
TP_OPT_STEP: float = 0.0005
TP_OPT_TOP_K: int = 5

_FEATURE_SPACE_CACHE: dict[tuple[Any, ...], CryptoFeatureSpace] = {}


@dataclass(frozen=True)
class ModelSpec:
    archive_path: Path
    rank: int
    label_mode: str
    label_threshold: float
    top_fraction: float
    label_direction: str = "long"
    exit_after_k: int | None = None


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
    val_horizons: tuple[SplitSignals, ...] = ()
    test_horizons: tuple[SplitSignals, ...] = ()


@dataclass(frozen=True)
class BacktestResult:
    summary: pd.DataFrame
    score_band_rates: pd.DataFrame
    two_sided_score_band: pd.DataFrame
    two_sided_trade_path: pd.DataFrame
    chart_path: Path


def run_backtest(
    base_specs: list[ModelSpec],
    exit1_spec: ModelSpec | None = None,
    exit2_spec: ModelSpec | None = None,
    data_path: str | Path = config.DATA_PATH,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    base_ensemble: str = "and",
    tp_threshold: float | None = None,
    label_direction: str | None = None,
    val_start: str = config.VAL_START,
    test_start: str = config.TEST_START,
    test_end: str | None = config.TEST_END,
) -> BacktestResult:
    if not base_specs:
        raise ValueError("At least one --base spec is required.")
    base_ensemble = str(base_ensemble).strip().lower()
    if base_ensemble not in {"and", "or"}:
        raise ValueError("--base-ensemble must be 'and' or 'or'.")
    if label_direction not in (None, ""):
        label_direction = config.canonical_label_direction(label_direction)
    else:
        directions = {spec.label_direction for spec in base_specs}
        if len(directions) != 1:
            raise ValueError(
                "Mixed Long and Short base specs require --label-direction "
                "to select the common strategy evaluation direction."
            )
        label_direction = directions.pop()

    tp = float(
        tp_threshold
        if tp_threshold is not None
        else base_specs[0].label_threshold
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Base score-band setup: evaluation_direction=%s | "
        "train_directions=%s | base_specs=%d | base_ensemble=%s",
        label_direction,
        [spec.label_direction for spec in base_specs],
        len(base_specs),
        base_ensemble.upper(),
    )
    if exit1_spec is not None or exit2_spec is not None:
        logger.info(
            "Legacy exit specs were supplied but are no longer trained or "
            "included in the base score-band report."
        )
    logger.info("Loading crypto data from %s", data_path)
    raw_df = load_ohlcv(data_path)
    default_base_horizons = _normalize_horizons(
        config.HOLDING_HORIZONS, "config.HOLDING_HORIZONS"
    )
    base_horizons_by_spec = [
        _archive_horizons(
            spec.archive_path,
            fallback=default_base_horizons,
            label=f"base[{index}]",
        )
        for index, spec in enumerate(base_specs, start=1)
    ]
    all_horizons = sorted(
        {h for horizons in base_horizons_by_spec for h in horizons}
    )
    horizon = max(BASE_SIGNAL_ANALYSIS_HORIZON, int(max(all_horizons)))
    purge_bars = config.purge_bars_for_horizons([*all_horizons, horizon])
    logger.info(
        "Backtest horizons: base=%s | score-band path=h%d",
        base_horizons_by_spec,
        horizon,
    )

    entries = [
        _load_rank_entry(spec.archive_path, spec.rank)
        for spec in base_specs
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
    for spec, entry, base_horizons in zip(
        base_specs,
        entries[: len(base_specs)],
        base_horizons_by_spec,
        strict=True,
    ):
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

    base_bundle = _combine_base_bundles(base_bundles, selection=base_ensemble)

    _, raw_path_by_horizon, _ = _return_context_by_horizon(
        raw_df,
        [horizon],
        label_mode="mfe",
        label_direction="long",
    )
    raw_price_path_returns = raw_path_by_horizon[horizon]

    summary = _base_overview(
        base_bundle=base_bundle,
        raw_path_returns=raw_price_path_returns,
        tp_threshold=tp,
    )
    score_band_rates = pd.DataFrame(
        _base_fraction_band_tp_rows(
            base_bundles=base_bundles,
            selection=base_ensemble,
            raw_path_returns=raw_price_path_returns,
            thresholds=_score_band_sweep_thresholds(),
        )
    )
    two_sided_score_band = _score_band_two_sided_tp_strategy(
        base_bundles=base_bundles,
        selection=base_ensemble,
        raw_path_returns=raw_price_path_returns,
        tp_long=SCORE_BAND_TWO_SIDED_TP_LONG,
        tp_short=SCORE_BAND_TWO_SIDED_TP_SHORT,
        optimize_tp=SCORE_BAND_TWO_SIDED_OPTIMIZE_TP,
    )
    two_sided_trade_path = _two_sided_score_band_trade_path(
        base_bundle=base_bundle,
        raw_path_returns=raw_price_path_returns,
        tp_long=SCORE_BAND_TWO_SIDED_TP_LONG,
        tp_short=SCORE_BAND_TWO_SIDED_TP_SHORT,
        optimize_tp=SCORE_BAND_TWO_SIDED_OPTIMIZE_TP,
        base_bundles=base_bundles,
        selection=base_ensemble,
        score_band_strategy=two_sided_score_band,
    )
    run_name = _base_score_band_name(
        base_specs,
        base_ensemble,
        tp,
        label_direction=label_direction,
    )
    chart_path = out_path / f"{run_name}.png"
    _plot_summary(
        summary=summary,
        score_band_rates=score_band_rates,
        two_sided_score_band=two_sided_score_band,
        two_sided_trade_path=two_sided_trade_path,
        crypto_close=raw_df["close"],
        chart_path=chart_path,
        base_label=base_bundle.label,
        base_ensemble=base_ensemble,
        tp_threshold=tp,
        label_direction=label_direction,
    )
    logger.info("Saved chart: %s", chart_path)
    return BacktestResult(
        summary=summary,
        score_band_rates=score_band_rates,
        two_sided_score_band=two_sided_score_band,
        two_sided_trade_path=two_sided_trade_path,
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
        label_direction=spec.label_direction,
        exit_after_k=spec.exit_after_k,
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


def _archive_label_direction(
    path: Path,
    explicit_direction: str | None = None,
) -> str:
    if explicit_direction not in (None, ""):
        return config.canonical_label_direction(explicit_direction)
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return config.canonical_label_direction("long")
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    metadata_direction = (
        metadata.get("label_direction") if isinstance(metadata, dict) else None
    )
    return config.canonical_label_direction(
        metadata_direction if metadata_direction not in (None, "") else "long"
    )


def _archive_exit_after_k(path: Path, label_mode: str) -> int | None:
    default_k = config.resolve_exit_after_k(label_mode)
    if default_k is None:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default_k
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    archived_k = metadata.get("exit_after_k") if isinstance(metadata, dict) else None
    return config.resolve_exit_after_k(label_mode, archived_k)


def _normalize_horizons(values: Any, label: str) -> list[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple, set)):
        raise ValueError(
            f"{label} must be a list of positive integers, got: {values!r}"
        )
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
    label_direction = spec.label_direction
    logger.info(
        "Training %s rank %d | mode=%s direction=%s threshold=%.6f "
        "exit_after_k=%s top=%.2f%% | horizons=%s",
        spec.archive_path,
        spec.rank,
        spec.label_mode,
        label_direction,
        spec.label_threshold,
        spec.exit_after_k,
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
        label_direction=label_direction,
        exit_after_k=spec.exit_after_k,
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
        raise ValueError(
            f"No valid horizon model for {spec.archive_path} rank {spec.rank}."
        )

    label = (
        f"{spec.archive_path.stem} r{spec.rank:02d} "
        f"{spec.label_mode} {label_direction} "
        f"thr={spec.label_threshold:.4g} top={spec.top_fraction:.0%}"
    )
    if len(horizon_results) == 1:
        _, val, test = horizon_results[0]
        return BundleSignals(
            label=label,
            val=val,
            test=test,
            val_horizons=(val,),
            test_horizons=(test,),
        )

    val_horizons = tuple(item[1] for item in horizon_results)
    test_horizons = tuple(item[2] for item in horizon_results)
    val = _combine_horizons(
        split="val",
        split_results=list(val_horizons),
        top_fraction=spec.top_fraction,
    )
    test = _combine_horizons(
        split="test",
        split_results=list(test_horizons),
        top_fraction=spec.top_fraction,
    )
    return BundleSignals(
        label=f"{label} h-ensemble",
        val=val,
        test=test,
        val_horizons=val_horizons,
        test_horizons=test_horizons,
    )


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
        logger.warning(
            "h%d skipped: empty train/val/test after label filtering.", horizon
        )
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
    label = (
        f"base {str(selection).upper()} ("
        + " + ".join(item.label for item in bundles)
        + ")"
    )
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
    selected_index = _combine_indices(
        [item.selected_index for item in splits], selection
    )
    common_index = splits[0].data.index
    for item in splits[1:]:
        common_index = common_index.union(item.data.index)
    pred_frame = pd.concat(
        [item.data["pred"].reindex(common_index) for item in splits], axis=1
    )
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
    selected = data.reindex(data.index.intersection(selected_index)).dropna(
        subset=["pred"]
    )
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


def _base_fraction_band_tp_rows(
    base_bundles: list[BundleSignals],
    selection: str,
    raw_path_returns: pd.DataFrame,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    """Measure raw-price excursions in disjoint 5%-wide score bands.

    Each Horizon learns its cumulative top-fraction cutoff from validation.
    The same Horizon cutoff is applied to test, Horizon selections are ANDed
    inside each individual, and individual selections are then combined with
    the configured AND/OR rule.
    """
    if not base_bundles:
        return []
    rows: list[dict[str, Any]] = []
    bands_by_split = _base_fraction_band_indices(
        base_bundles,
        selection=selection,
        max_fraction=BASE_FRACTION_BAND_MAX,
    )
    for split_name, split_bands in bands_by_split.items():
        for band_start, band_end, band_selected in split_bands:
            high_rows = _raw_extreme_sweep_rows(
                split=split_name,
                group=BASE_SIGNAL_HIGH_GROUP,
                selected_base_index=band_selected,
                raw_path_returns=raw_path_returns,
                levels=thresholds,
                extreme="high",
                min_h=1,
                max_h=BASE_SIGNAL_ANALYSIS_HORIZON,
            )
            low_rows = _raw_extreme_sweep_rows(
                split=split_name,
                group=BASE_SIGNAL_LOW_GROUP,
                selected_base_index=band_selected,
                raw_path_returns=raw_path_returns,
                levels=thresholds,
                extreme="low",
                min_h=1,
                max_h=BASE_SIGNAL_ANALYSIS_HORIZON,
            )
            two_sided_rows = _tp_sweep_rows(
                split=split_name,
                group=BASE_SIGNAL_TWO_SIDED_GROUP,
                selected_base_index=band_selected,
                path_returns=raw_path_returns,
                thresholds=_two_sided_sweep_thresholds(),
                min_h=1,
                max_h=BASE_SIGNAL_ANALYSIS_HORIZON,
                label_direction="long",
                include_two_sided_move=True,
            )
            for row in [*high_rows, *low_rows, *two_sided_rows]:
                row["band_start"] = band_start
                row["band_end"] = float(band_end)
            rows.extend(high_rows)
            rows.extend(low_rows)
            rows.extend(two_sided_rows)
    return rows


def _base_overview(
    base_bundle: BundleSignals,
    raw_path_returns: pd.DataFrame,
    tp_threshold: float,
) -> pd.DataFrame:
    """Return the compact first table retained in the score-band report."""
    required = [
        f"high_h{BASE_SIGNAL_ANALYSIS_HORIZON}",
        f"low_h{BASE_SIGNAL_ANALYSIS_HORIZON}",
        f"close_h{BASE_SIGNAL_ANALYSIS_HORIZON}",
    ]
    rows: list[dict[str, Any]] = []
    for split_name, split_signals in (
        ("val", base_bundle.val),
        ("test", base_bundle.test),
    ):
        selected_path = raw_path_returns.reindex(split_signals.selected_index)
        if set(required).issubset(selected_path.columns):
            selected_path = selected_path.dropna(subset=required)
        else:
            selected_path = selected_path.iloc[0:0]
        rows.append(
            {
                "split": split_name,
                "base_signals": int(len(selected_path)),
                "trades_per_day": _average_trades_per_day(
                    split_signals.data.index,
                    int(len(selected_path)),
                ),
                "tp_threshold": float(tp_threshold),
            }
        )
    return pd.DataFrame(rows)


def _average_trades_per_day(index: pd.Index, trade_count: int) -> float:
    """Average selected signals over calendar days represented by a split."""
    timestamps = pd.to_datetime(pd.Index(index), errors="coerce")
    valid = timestamps[~pd.isna(timestamps)]
    if len(valid) == 0:
        return float("nan")
    day_count = int(pd.Index(valid.normalize()).nunique())
    return float(trade_count) / day_count if day_count else float("nan")


def _raw_extreme_sweep_rows(
    split: str,
    group: str,
    selected_base_index: pd.Index,
    raw_path_returns: pd.DataFrame,
    levels: list[float],
    extreme: str,
    min_h: int,
    max_h: int,
) -> list[dict[str, Any]]:
    """Measure raw high-up or low-down hit rates over an inclusive H range."""
    if extreme not in {"high", "low"}:
        raise ValueError("extreme must be 'high' or 'low'.")
    columns = _future_price_columns(
        raw_path_returns,
        extreme,
        min_h=min_h,
        max_h=max_h,
    )
    selected = raw_path_returns.reindex(selected_base_index)
    selected = selected.dropna(subset=columns) if columns else selected.iloc[0:0]
    values = selected[columns].apply(pd.to_numeric, errors="coerce")
    total = int(len(values))
    extreme_value = (
        values.max(axis=1, skipna=False)
        if extreme == "high"
        else values.min(axis=1, skipna=False)
    )

    rows: list[dict[str, Any]] = []
    for level in levels:
        level = float(level)
        threshold = level if extreme == "high" else -level
        hit = (
            extreme_value.gt(level)
            if extreme == "high"
            else extreme_value.lt(-level)
        )
        hit_count = int(hit.sum())
        rows.append(
            {
                "split": split,
                "group": group,
                "tp_threshold": threshold,
                "sample_count": total,
                "hit_count": hit_count,
                "hit_rate": hit_count / total if total else 0.0,
                "miss_count": total - hit_count,
                "miss_return_mean": float("nan"),
                "close_h2_return_mean": float("nan"),
                "two_sided_count": float("nan"),
                "two_sided_rate": float("nan"),
            }
        )
    return rows


def _base_fraction_band_indices(
    base_bundles: list[BundleSignals],
    selection: str,
    max_fraction: float,
) -> dict[str, list[tuple[float, float, pd.Index]]]:
    band_count = int(round(float(max_fraction) / BASE_FRACTION_BAND_STEP))
    band_ends = [
        float(step_index * BASE_FRACTION_BAND_STEP)
        for step_index in range(1, band_count + 1)
    ]
    horizon_cutoffs = [
        [
            [
                _top_fraction_cutoff(horizon_split.data, fraction)
                for fraction in band_ends
            ]
            for horizon_split in _bundle_horizon_splits(bundle, split="val")
        ]
        for bundle in base_bundles
    ]
    result: dict[str, list[tuple[float, float, pd.Index]]] = {}
    for split_name in ("val", "test"):
        previous_cumulative = pd.Index([])
        split_bands: list[tuple[float, float, pd.Index]] = []
        for band_index, band_end in enumerate(band_ends):
            member_indices: list[pd.Index] = []
            for member_index, bundle in enumerate(base_bundles):
                horizon_indices: list[pd.Index] = []
                horizon_splits = _bundle_horizon_splits(
                    bundle,
                    split=split_name,
                )
                for horizon_index, horizon_split in enumerate(horizon_splits):
                    if split_name == "val":
                        selected = _top_fraction_indices(
                            horizon_split.data,
                            band_end,
                        )
                    else:
                        selected = _indices_at_or_above_cutoff(
                            horizon_split.data,
                            horizon_cutoffs[member_index][horizon_index][
                                band_index
                            ],
                        )
                    horizon_indices.append(selected)
                member_indices.append(
                    _combine_indices(horizon_indices, selection="and")
                )
            cumulative = _combine_indices(member_indices, selection=selection)
            band_selected = cumulative.difference(previous_cumulative, sort=False)
            band_start = float(band_end - BASE_FRACTION_BAND_STEP)
            split_bands.append((band_start, band_end, band_selected))
            previous_cumulative = cumulative
        result[split_name] = split_bands
    return result


def _bundle_horizon_splits(
    bundle: BundleSignals,
    split: str,
) -> tuple[SplitSignals, ...]:
    """Return per-H splits, falling back for legacy/synthetic bundles."""
    if split == "val":
        return bundle.val_horizons or (bundle.val,)
    if split == "test":
        return bundle.test_horizons or (bundle.test,)
    raise ValueError("split must be 'val' or 'test'.")


def _top_fraction_cutoff(data: pd.DataFrame, fraction: float) -> float:
    selected = _top_fraction_indices(data, fraction)
    if len(selected) == 0:
        return float("nan")
    pred = pd.to_numeric(data.loc[selected, "pred"], errors="coerce").dropna()
    return float(pred.min()) if not pred.empty else float("nan")


def _top_fraction_indices(data: pd.DataFrame, fraction: float) -> pd.Index:
    if data.empty or "pred" not in data.columns:
        return pd.Index([])
    pred = pd.to_numeric(data["pred"], errors="coerce").dropna()
    if pred.empty:
        return pd.Index([])
    n_select = min(
        len(pred),
        max(1, int(np.ceil(len(pred) * float(fraction) - 1e-12))),
    )
    return pd.Index(pred.nlargest(n_select).index)


def _indices_at_or_above_cutoff(data: pd.DataFrame, cutoff: float) -> pd.Index:
    if data.empty or "pred" not in data.columns or not np.isfinite(cutoff):
        return pd.Index([])
    pred = pd.to_numeric(data["pred"], errors="coerce")
    return pd.Index(data.index[pred >= float(cutoff)])


def _low_h1_sweep_rows(
    split: str,
    selected_base_index: pd.Index,
    path_returns: pd.DataFrame,
    thresholds: list[float],
    label_direction: str,
) -> list[dict[str, Any]]:
    selected_path = path_returns.reindex(selected_base_index)
    if "low_h1" not in selected_path.columns:
        selected_path = selected_path.iloc[0:0]
    else:
        selected_path = selected_path.dropna(subset=["low_h1"])
    total = int(len(selected_path))
    directional_low = (
        pd.to_numeric(selected_path["low_h1"], errors="coerce")
        if "low_h1" in selected_path.columns
        else pd.Series(dtype=float)
    )
    raw_low = (
        -directional_low
        if config.canonical_label_direction(label_direction) == "short"
        else directional_low
    )

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        hit_count: int | float = float("nan")
        hit_rate = float("nan")
        miss_count: int | float = float("nan")
        if float(threshold) >= 0.0:
            hit = raw_low < -float(threshold)
            hit_count = int(hit.sum())
            hit_rate = hit_count / total if total else 0.0
            miss_count = total - hit_count
        rows.append(
            {
                "split": split,
                "group": "base_fraction_band_low_h1",
                "tp_threshold": float(threshold),
                "sample_count": total,
                "hit_count": hit_count,
                "hit_rate": hit_rate,
                "miss_count": miss_count,
                "miss_return_mean": float("nan"),
                "close_h2_return_mean": float("nan"),
                "two_sided_count": float("nan"),
                "two_sided_rate": float("nan"),
            }
        )
    return rows


def _summarize_split(
    split: str,
    base_split: SplitSignals,
    exit1_split: SplitSignals,
    exit2_split: SplitSignals,
    path_returns: pd.DataFrame,
    raw_index: pd.DatetimeIndex,
    tp_threshold: float,
    tp_sweep_thresholds: list[float],
    label_direction: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hit_prefix = _hit_price_prefix(label_direction)
    adverse_prefix = _adverse_price_prefix(label_direction)
    hit_h1_col = f"{hit_prefix}_h1"
    hit_h2_col = f"{hit_prefix}_h2"
    adverse_h1_col = f"{adverse_prefix}_h1"
    base_selected = pd.Index(base_split.selected_index)
    base_path = path_returns.reindex(base_selected).dropna(
        subset=[hit_h1_col, hit_h2_col, adverse_h1_col]
    )
    base_signals = int(len(base_path))
    if base_signals == 0:
        summary = {
            "split": split,
            "base_signals": 0,
            "base_no_h1": 0,
            "base_no_h1_rate": 0.0,
            "base_low_h1_le_neg01": 0,
            "base_low_h1_le_neg01_rate": 0.0,
            "base_low_h1_le_neg005": 0,
            "base_low_h1_le_neg005_rate": 0.0,
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
        return summary, (_empty_stage_tp_rows(split, tp_sweep_thresholds))

    no_h1_index = pd.Index(
        base_path.index[base_path[hit_h1_col] <= float(tp_threshold)]
    )
    no_h1_count = int(len(no_h1_index))
    adverse_h1 = pd.to_numeric(base_path[adverse_h1_col], errors="coerce")
    low_h1_le_neg01_count = int((adverse_h1 <= -0.001).sum())
    low_h1_le_neg005_count = int((adverse_h1 <= -0.0005).sum())

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
            pd.to_numeric(exit1_no_selected_path[hit_h2_col], errors="coerce")
            <= float(tp_threshold)
        ]
    )
    exit1_no_selected_no_h2_count = int(len(exit1_no_selected_no_h2_index))

    exit2_mapping = _future_bar_mapping(
        exit1_no_selected_no_h2_index, raw_index, offset=2
    )
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
        "base_low_h1_le_neg01": low_h1_le_neg01_count,
        "base_low_h1_le_neg01_rate": (
            low_h1_le_neg01_count / base_signals if base_signals else 0.0
        ),
        "base_low_h1_le_neg005": low_h1_le_neg005_count,
        "base_low_h1_le_neg005_rate": (
            low_h1_le_neg005_count / base_signals if base_signals else 0.0
        ),
        "exit1_selected": exit1_selected_count,
        "exit1_selected_rate": exit1_selected_count / no_h1_count
        if no_h1_count
        else 0.0,
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
        max_h=BASE_SIGNAL_ANALYSIS_HORIZON,
        label_direction=label_direction,
    )
    base_no_h1_tp_rows = _tp_sweep_rows(
        split=split,
        group="p1_base_no_h1",
        selected_base_index=no_h1_index,
        path_returns=path_returns,
        thresholds=tp_sweep_thresholds,
        min_h=2,
        label_direction=label_direction,
    )
    exit1_selected_tp_rows = _tp_sweep_rows(
        split=split,
        group="p1_exit_k1_selected",
        selected_base_index=exit1_selected_base_index,
        path_returns=path_returns,
        thresholds=tp_sweep_thresholds,
        min_h=2,
        label_direction=label_direction,
    )
    exit1_no_selected_tp_rows = _tp_sweep_rows(
        split=split,
        group="p1_exit_k1_no_selected",
        selected_base_index=exit1_no_selected_base_index,
        path_returns=path_returns,
        thresholds=tp_sweep_thresholds,
        min_h=2,
        label_direction=label_direction,
    )
    exit1_no_selected_h2_tp_rows = _tp_sweep_rows(
        split=split,
        group="p1_exit_k1_no_selected_h2",
        selected_base_index=exit1_no_selected_base_index,
        path_returns=path_returns,
        thresholds=tp_sweep_thresholds,
        min_h=2,
        max_h=2,
        label_direction=label_direction,
    )

    exit1_no_selected_no_h2_tp_rows = _tp_sweep_rows(
        split=split,
        group="p2_exit_k1_no_selected_no_h2",
        selected_base_index=exit1_no_selected_no_h2_index,
        path_returns=path_returns,
        thresholds=tp_sweep_thresholds,
        min_h=3,
        label_direction=label_direction,
    )
    exit2_selected_tp_rows = _tp_sweep_rows(
        split=split,
        group="p2_exit_k2_selected",
        selected_base_index=exit2_selected_base_index,
        path_returns=path_returns,
        thresholds=tp_sweep_thresholds,
        min_h=3,
        label_direction=label_direction,
    )
    exit2_no_selected_tp_rows = _tp_sweep_rows(
        split=split,
        group="p2_exit_k2_no_selected",
        selected_base_index=exit2_no_selected_base_index,
        path_returns=path_returns,
        thresholds=tp_sweep_thresholds,
        min_h=3,
        label_direction=label_direction,
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


def _tp_optimization_levels() -> list[float]:
    count = int(np.floor((TP_OPT_END - TP_OPT_START) / TP_OPT_STEP + 1e-12)) + 1
    levels = [float(TP_OPT_START + idx * TP_OPT_STEP) for idx in range(max(count, 0))]
    if not levels:
        raise ValueError("Dynamic TP optimization requires at least one TP level.")
    if any(level <= float(config.TRADE_COST) for level in levels):
        logger.warning(
            "Dynamic TP grid contains levels at/below TRADE_COST=%.4g.",
            config.TRADE_COST,
        )
    return levels


def _dynamic_tp_strategy_frame(
    base_split: SplitSignals,
    exit1_split: SplitSignals,
    exit2_split: SplitSignals,
    path_returns: pd.DataFrame,
    raw_index: pd.DatetimeIndex,
    label_direction: str,
) -> pd.DataFrame:
    hit_cols = _future_hit_columns(
        path_returns,
        min_h=1,
        label_direction=label_direction,
    )
    adverse_cols = _future_price_columns(
        path_returns,
        _adverse_price_prefix(label_direction),
        min_h=1,
    )
    if len(hit_cols) < 3:
        return pd.DataFrame()
    final_h = _max_h_from_hit_columns(hit_cols, label_direction=label_direction)
    close_col = f"close_h{final_h}"
    required = [
        *hit_cols,
        *adverse_cols,
        "open_h5",
        "close_h2",
        "close_h5",
        close_col,
    ]
    base_path = path_returns.reindex(pd.Index(base_split.selected_index)).dropna(
        subset=required
    )
    if base_path.empty:
        return pd.DataFrame()

    frame = pd.DataFrame(index=base_path.index)
    hit_prefix = _hit_price_prefix(label_direction)
    frame["high_h1"] = pd.to_numeric(base_path[f"{hit_prefix}_h1"], errors="coerce")
    frame["high_h2"] = pd.to_numeric(base_path[f"{hit_prefix}_h2"], errors="coerce")
    frame["high_h3"] = pd.to_numeric(base_path[f"{hit_prefix}_h3"], errors="coerce")
    frame["high_h4"] = pd.to_numeric(base_path[f"{hit_prefix}_h4"], errors="coerce")
    frame["high_h5"] = pd.to_numeric(base_path[f"{hit_prefix}_h5"], errors="coerce")
    if "entry_filter_low_h0" in base_path.columns:
        frame["entry_filter_low_h0"] = pd.to_numeric(
            base_path["entry_filter_low_h0"], errors="coerce"
        )
    for horizon in range(1, 6):
        raw_low = pd.to_numeric(base_path[f"low_h{horizon}"], errors="coerce")
        if config.canonical_label_direction(label_direction) == "short":
            raw_low = -raw_low
        frame[f"raw_low_h{horizon}"] = raw_low
    frame["open_h5"] = pd.to_numeric(base_path["open_h5"], errors="coerce")
    adverse_prefix = _adverse_price_prefix(label_direction)
    frame["adverse_h1"] = pd.to_numeric(
        base_path[f"{adverse_prefix}_h1"], errors="coerce"
    )
    frame["adverse_h2"] = pd.to_numeric(
        base_path[f"{adverse_prefix}_h2"], errors="coerce"
    )
    frame["adverse_h3"] = pd.to_numeric(
        base_path[f"{adverse_prefix}_h3"], errors="coerce"
    )
    frame["adverse_h4"] = pd.to_numeric(
        base_path[f"{adverse_prefix}_h4"], errors="coerce"
    )
    frame["adverse_h5"] = pd.to_numeric(
        base_path[f"{adverse_prefix}_h5"], errors="coerce"
    )
    h2_plus = _future_hit_columns(base_path, min_h=2, label_direction=label_direction)
    h3_plus = _future_hit_columns(base_path, min_h=3, label_direction=label_direction)
    frame["max_high_h2_plus"] = base_path[h2_plus].max(axis=1, skipna=False)
    frame["max_high_h3_plus"] = base_path[h3_plus].max(axis=1, skipna=False)
    adverse_h3_plus = _future_price_columns(
        base_path,
        adverse_prefix,
        min_h=3,
    )
    frame["min_adverse_h3_plus"] = base_path[adverse_h3_plus].min(
        axis=1,
        skipna=False,
    )
    frame["close_h2"] = pd.to_numeric(base_path["close_h2"], errors="coerce")
    frame["close_h3"] = pd.to_numeric(base_path["close_h3"], errors="coerce")
    frame["close_h4"] = pd.to_numeric(base_path["close_h4"], errors="coerce")
    frame["close_h5"] = pd.to_numeric(base_path["close_h5"], errors="coerce")
    frame["close_final"] = pd.to_numeric(base_path[close_col], errors="coerce")

    exit1_mapping = _future_bar_mapping(frame.index, raw_index, offset=1)
    exit2_mapping = _future_bar_mapping(frame.index, raw_index, offset=2)
    exit1_selected = pd.Series(
        exit1_mapping["exit_index"]
        .isin(pd.Index(exit1_split.selected_index))
        .to_numpy(),
        index=pd.Index(exit1_mapping["base_index"]),
        dtype=bool,
    )
    exit2_selected = pd.Series(
        exit2_mapping["exit_index"]
        .isin(pd.Index(exit2_split.selected_index))
        .to_numpy(),
        index=pd.Index(exit2_mapping["base_index"]),
        dtype=bool,
    )
    frame["exit1_selected"] = (
        exit1_selected.reindex(frame.index).fillna(False).astype(bool)
    )
    frame["exit2_selected"] = (
        exit2_selected.reindex(frame.index).fillna(False).astype(bool)
    )
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def _score_band_fixed_h5_strategy(
    base_bundles: list[BundleSignals],
    selection: str,
    exit1_bundle: BundleSignals,
    exit2_bundle: BundleSignals,
    path_returns: pd.DataFrame,
    raw_index: pd.DatetimeIndex,
    label_direction: str,
) -> pd.DataFrame:
    columns = [
        "split",
        "score_band",
        "band_start",
        "band_end",
        "signals_before_filter",
        "trades",
        "hit_h1",
        "hit_h2",
        "hit_h3",
        "hit_h4",
        "weak_h1_tp_h2_h4",
        "hit_open_h5",
        "hit_h5",
        "cutloss",
        "close_h5",
        "hit_h1_return_mean",
        "hit_h2_return_mean",
        "hit_h3_return_mean",
        "hit_h4_return_mean",
        "weak_h1_tp_h2_h4_return_mean",
        "hit_open_h5_return_mean",
        "hit_h5_return_mean",
        "cutloss_return_mean",
        "close_h5_return_mean",
        "high_h1_below_threshold",
        "high_h1_below_threshold_rate",
        "high_h1_below_threshold_high_h2_mean",
        "high_h1_below_threshold_close_h2_mean",
        "high_h1_below_threshold_high_h3_mean",
        "high_h1_below_threshold_close_h3_mean",
        "high_h1_below_threshold_high_h4_mean",
        "high_h1_below_threshold_close_h4_mean",
        "gross_mean",
        "e_net",
    ]
    max_fraction = max(end for _, end in SCORE_BAND_RANGES)
    bands_by_split = _base_fraction_band_indices(
        base_bundles,
        selection=selection,
        max_fraction=max_fraction,
    )
    rows: list[dict[str, Any]] = []
    for split_name, split_bands in bands_by_split.items():
        exit1_split = exit1_bundle.val if split_name == "val" else exit1_bundle.test
        exit2_split = exit2_bundle.val if split_name == "val" else exit2_bundle.test
        detailed_frames: list[pd.DataFrame] = []
        signals_before_filter_total = 0
        for band_start, band_end, selected_index in split_bands:
            if not any(
                np.isclose(band_start, configured_start)
                and np.isclose(band_end, configured_end)
                for configured_start, configured_end in SCORE_BAND_RANGES
            ):
                continue
            base_split = SplitSignals(
                split=split_name,
                data=pd.DataFrame(index=selected_index),
                selected_index=selected_index,
                pred_threshold=float("nan"),
                top_fraction=float(band_end),
            )
            frame = _dynamic_tp_strategy_frame(
                base_split=base_split,
                exit1_split=exit1_split,
                exit2_split=exit2_split,
                path_returns=path_returns,
                raw_index=raw_index,
                label_direction=label_direction,
            )
            signals_before_filter = int(len(frame))
            signals_before_filter_total += signals_before_filter
            frame = _apply_score_band_entry_filter(frame)
            detailed = _simulate_score_band_fixed_h5_frame(frame)
            detailed_frames.append(detailed)
            rows.append(
                _score_band_fixed_h5_metrics(
                    detailed,
                    split=split_name,
                    band_start=band_start,
                    band_end=band_end,
                    signals_before_filter=signals_before_filter,
                )
            )
        if detailed_frames:
            all_detailed = pd.concat(detailed_frames, axis=0)
            rows.append(
                _score_band_fixed_h5_metrics(
                    all_detailed,
                    split=split_name,
                    band_start=float("nan"),
                    band_end=float("nan"),
                    signals_before_filter=signals_before_filter_total,
                )
            )
    return pd.DataFrame(rows, columns=columns)


def _apply_score_band_entry_filter(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep signals whose H0 low stays above the H0-open floor."""
    column = "entry_filter_low_h0"
    if frame.empty or column not in frame.columns:
        return frame.copy()
    low_h0_return = pd.to_numeric(frame[column], errors="coerce")
    keep = low_h0_return.gt(float(SCORE_BAND_ENTRY_MIN_LOW_THRESHOLD))
    return frame.loc[keep].copy()


def _score_band_two_sided_tp_strategy(
    base_bundles: list[BundleSignals],
    selection: str,
    raw_path_returns: pd.DataFrame,
    tp_long: float = SCORE_BAND_TWO_SIDED_TP_LONG,
    tp_short: float = SCORE_BAND_TWO_SIDED_TP_SHORT,
    optimize_tp: bool = SCORE_BAND_TWO_SIDED_OPTIMIZE_TP,
) -> pd.DataFrame:
    """Evaluate simultaneous Long/Short TP orders in disjoint score bands."""
    columns = [
        "split",
        "score_band",
        "band_start",
        "band_end",
        "trades",
        "long_hit",
        "short_hit",
        "both_hit",
        "long_only",
        "short_only",
        "neither_hit",
        "long_return_mean",
        "short_return_mean",
        "gross_mean",
        "e_net",
        "tp_long",
        "tp_short",
    ]
    bands_by_split = _base_fraction_band_indices(
        base_bundles,
        selection=selection,
        max_fraction=BASE_FRACTION_BAND_MAX,
    )
    rows: list[dict[str, Any]] = []
    available_horizons = [
        horizon
        for horizon in range(1, BASE_SIGNAL_ANALYSIS_HORIZON + 1)
        if {
            f"high_h{horizon}",
            f"low_h{horizon}",
            f"close_h{horizon}",
        }.issubset(raw_path_returns.columns)
    ]
    exit_horizon = (
        max(available_horizons)
        if available_horizons
        else BASE_SIGNAL_ANALYSIS_HORIZON
    )
    detailed_by_split: dict[str, list[pd.DataFrame]] = {"val": [], "test": []}
    rows_by_split: dict[str, list[dict[str, Any]]] = {"val": [], "test": []}
    val_bands = bands_by_split.get("val", [])
    test_bands = bands_by_split.get("test", [])
    for val_band, test_band in zip(val_bands, test_bands, strict=True):
        band_start, band_end, val_index = val_band
        _, _, test_index = test_band
        if optimize_tp:
            selected_tp_long, selected_tp_short = _optimize_two_sided_tp_pair(
                raw_path_returns.reindex(val_index),
                max_tp_long=tp_long,
                max_tp_short=tp_short,
                exit_horizon=exit_horizon,
            )
        else:
            selected_tp_long = float(tp_long)
            selected_tp_short = float(tp_short)
        for split_name, selected_index in (
            ("val", val_index),
            ("test", test_index),
        ):
            detailed = _simulate_two_sided_tp_frame(
                raw_path_returns.reindex(selected_index),
                tp_long=selected_tp_long,
                tp_short=selected_tp_short,
                exit_horizon=exit_horizon,
            )
            detailed_by_split[split_name].append(detailed)
            rows_by_split[split_name].append(
                _two_sided_tp_metrics(
                    detailed,
                    split=split_name,
                    band_start=band_start,
                    band_end=band_end,
                    tp_long=selected_tp_long,
                    tp_short=selected_tp_short,
                )
            )
    for split_name in ("val", "test"):
        rows.extend(rows_by_split[split_name])
        detailed_frames = detailed_by_split[split_name]
        if detailed_frames:
            rows.append(
                _two_sided_tp_metrics(
                    pd.concat(detailed_frames, axis=0),
                    split=split_name,
                    band_start=float("nan"),
                    band_end=float("nan"),
                )
            )
    return pd.DataFrame(rows, columns=columns)


def _two_sided_score_band_trade_path(
    base_bundle: BundleSignals,
    raw_path_returns: pd.DataFrame,
    tp_long: float = SCORE_BAND_TWO_SIDED_TP_LONG,
    tp_short: float = SCORE_BAND_TWO_SIDED_TP_SHORT,
    optimize_tp: bool = SCORE_BAND_TWO_SIDED_OPTIMIZE_TP,
    base_bundles: list[BundleSignals] | None = None,
    selection: str = "and",
    score_band_strategy: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return configured base trades using the TP pair assigned to each band."""
    columns = [
        "split",
        "score_band",
        "gross_return",
        "net_return",
        "cumulative_net_return",
        "tp_long",
        "tp_short",
    ]
    available_horizons = [
        horizon
        for horizon in range(1, BASE_SIGNAL_ANALYSIS_HORIZON + 1)
        if {
            f"high_h{horizon}",
            f"low_h{horizon}",
            f"close_h{horizon}",
        }.issubset(raw_path_returns.columns)
    ]
    exit_horizon = (
        max(available_horizons)
        if available_horizons
        else BASE_SIGNAL_ANALYSIS_HORIZON
    )
    if base_bundles:
        bands_by_split = _base_fraction_band_indices(
            base_bundles,
            selection=selection,
            max_fraction=BASE_FRACTION_BAND_MAX,
        )
        val_pair_by_band = _two_sided_tp_pairs_by_band(score_band_strategy)
        configured_indices = {
            "val": pd.Index(base_bundle.val.selected_index),
            "test": pd.Index(base_bundle.test.selected_index),
        }
        split_paths: list[pd.DataFrame] = []
        for split_name in ("val", "test"):
            split_frames: list[pd.DataFrame] = []
            for band_start, band_end, band_index in bands_by_split.get(
                split_name, []
            ):
                selected_index = pd.Index(band_index).intersection(
                    configured_indices[split_name],
                    sort=False,
                )
                if selected_index.empty:
                    continue
                if optimize_tp:
                    pair = val_pair_by_band.get(
                        _score_band_key(band_start, band_end)
                    )
                    if pair is None:
                        continue
                    selected_tp_long, selected_tp_short = pair
                else:
                    selected_tp_long = float(tp_long)
                    selected_tp_short = float(tp_short)
                detailed = _simulate_two_sided_tp_frame(
                    raw_path_returns.reindex(selected_index),
                    tp_long=selected_tp_long,
                    tp_short=selected_tp_short,
                    exit_horizon=exit_horizon,
                )
                if detailed.empty:
                    continue
                split_frame = pd.DataFrame(index=detailed.index)
                split_frame["split"] = split_name
                split_frame["score_band"] = (
                    f"top {band_start:.0%}-{band_end:.0%}"
                )
                split_frame["gross_return"] = pd.to_numeric(
                    detailed["gross_return"], errors="coerce"
                )
                split_frame["net_return"] = (
                    split_frame["gross_return"] - float(config.TRADE_COST)
                )
                split_frame["tp_long"] = selected_tp_long
                split_frame["tp_short"] = selected_tp_short
                split_frames.append(split_frame)
            if not split_frames:
                continue
            split_path = pd.concat(split_frames, axis=0).sort_index()
            split_path = split_path.dropna(subset=["net_return"])
            split_path["cumulative_net_return"] = (
                split_path["net_return"].cumsum()
            )
            split_paths.append(split_path)
        if not split_paths:
            return pd.DataFrame(columns=columns)
        return pd.concat(split_paths, axis=0)[columns]

    if optimize_tp:
        selected_tp_long, selected_tp_short = _optimize_two_sided_tp_pair(
            raw_path_returns.reindex(base_bundle.val.selected_index),
            max_tp_long=tp_long,
            max_tp_short=tp_short,
            exit_horizon=exit_horizon,
        )
    else:
        selected_tp_long = float(tp_long)
        selected_tp_short = float(tp_short)
    split_paths: list[pd.DataFrame] = []
    for split_name, split_signals in (
        ("val", base_bundle.val),
        ("test", base_bundle.test),
    ):
        detailed = _simulate_two_sided_tp_frame(
            raw_path_returns.reindex(split_signals.selected_index),
            tp_long=selected_tp_long,
            tp_short=selected_tp_short,
            exit_horizon=exit_horizon,
        )
        if detailed.empty:
            continue
        split_path = pd.DataFrame(index=detailed.index)
        split_path["split"] = split_name
        split_path["score_band"] = "configured base selection"
        split_path["gross_return"] = pd.to_numeric(
            detailed["gross_return"], errors="coerce"
        )
        split_path["net_return"] = (
            split_path["gross_return"] - float(config.TRADE_COST)
        )
        split_path["tp_long"] = selected_tp_long
        split_path["tp_short"] = selected_tp_short
        split_path = split_path.sort_index()
        split_path = split_path.dropna(subset=["net_return"])
        split_path["cumulative_net_return"] = split_path["net_return"].cumsum()
        split_paths.append(split_path)
    if not split_paths:
        return pd.DataFrame(columns=columns)
    return pd.concat(split_paths, axis=0)[columns]


def _two_sided_tp_pairs_by_band(
    strategy: pd.DataFrame | None,
) -> dict[tuple[float, float], tuple[float, float]]:
    """Read each validation band's optimized pair from the strategy table."""
    if strategy is None or strategy.empty:
        return {}
    required = {"split", "band_start", "band_end", "tp_long", "tp_short"}
    if not required.issubset(strategy.columns):
        return {}
    val_rows = strategy[strategy["split"].astype(str).str.lower() == "val"]
    pairs: dict[tuple[float, float], tuple[float, float]] = {}
    for _, row in val_rows.iterrows():
        values = pd.to_numeric(
            row[["band_start", "band_end", "tp_long", "tp_short"]],
            errors="coerce",
        )
        if values.isna().any():
            continue
        pairs[_score_band_key(values["band_start"], values["band_end"])] = (
            float(values["tp_long"]),
            float(values["tp_short"]),
        )
    return pairs


def _score_band_key(start: float, end: float) -> tuple[float, float]:
    return round(float(start), 10), round(float(end), 10)


def _optimize_two_sided_tp_pair(
    frame: pd.DataFrame,
    max_tp_long: float,
    max_tp_short: float,
    exit_horizon: int,
) -> tuple[float, float]:
    """Choose the joint Long/Short TP pair with highest validation E[net]."""
    long_levels = _two_sided_tp_grid(max_tp_long)
    short_levels = _two_sided_tp_grid(max_tp_short)
    h = int(exit_horizon)
    required = [
        *[f"high_h{horizon}" for horizon in range(1, h + 1)],
        *[f"low_h{horizon}" for horizon in range(1, h + 1)],
        f"close_h{h}",
    ]
    if frame.empty or h < 1 or not set(required).issubset(frame.columns):
        return 0.0, 0.0
    valid = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    if valid.empty:
        return 0.0, 0.0

    max_high = valid[
        [f"high_h{horizon}" for horizon in range(1, h + 1)]
    ].max(axis=1).to_numpy(dtype=float)
    min_low = valid[
        [f"low_h{horizon}" for horizon in range(1, h + 1)]
    ].min(axis=1).to_numpy(dtype=float)
    close_final = pd.to_numeric(
        valid[f"close_h{h}"], errors="coerce"
    ).to_numpy(dtype=float)

    best_mean = -float("inf")
    best_long = 0.0
    best_short = 0.0
    tolerance = 1e-15
    for candidate_long in long_levels:
        long_return = np.where(
            max_high >= float(candidate_long),
            float(candidate_long),
            close_final,
        )
        for candidate_short in short_levels:
            short_return = np.where(
                min_low <= -float(candidate_short),
                float(candidate_short),
                -close_final,
            )
            gross_mean = float(np.mean(long_return + short_return))
            if gross_mean > best_mean + tolerance:
                best_mean = gross_mean
                best_long = float(candidate_long)
                best_short = float(candidate_short)
                continue
            if abs(gross_mean - best_mean) <= tolerance:
                candidate_total = float(candidate_long + candidate_short)
                best_total = best_long + best_short
                if candidate_total < best_total - tolerance:
                    best_long = float(candidate_long)
                    best_short = float(candidate_short)
    return best_long, best_short


def _two_sided_tp_grid(maximum: float) -> np.ndarray:
    """Return an inclusive 0..maximum TP grid using the configured step."""
    upper = float(maximum)
    step = float(TWO_SIDED_SWEEP_STEP)
    if not np.isfinite(upper) or upper < 0.0:
        raise ValueError("two-sided TP maximum must be finite and non-negative.")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("TWO_SIDED_SWEEP_STEP must be finite and positive.")
    levels = np.arange(0.0, upper + step * 0.5, step, dtype=float)
    levels = levels[levels <= upper + 1e-12]
    if len(levels) == 0 or not np.isclose(levels[-1], upper):
        levels = np.append(levels, upper)
    return np.unique(np.round(levels, 12))


def _simulate_two_sided_tp_frame(
    frame: pd.DataFrame,
    tp_long: float = SCORE_BAND_TWO_SIDED_TP_LONG,
    tp_short: float = SCORE_BAND_TWO_SIDED_TP_SHORT,
    exit_horizon: int | None = None,
) -> pd.DataFrame:
    """Settle simultaneous Long and Short positions through exit_horizon.

    A hit leg realizes its own TP. Unfilled legs exit at the final close.
    """
    long_threshold = float(tp_long)
    short_threshold = float(tp_short)
    if not np.isfinite(long_threshold) or long_threshold < 0.0:
        raise ValueError("two-sided Long TP must be finite and non-negative.")
    if not np.isfinite(short_threshold) or short_threshold < 0.0:
        raise ValueError("two-sided Short TP must be finite and non-negative.")
    if exit_horizon is None:
        candidates = [
            horizon
            for horizon in range(1, BASE_SIGNAL_ANALYSIS_HORIZON + 1)
            if {
                f"high_h{horizon}",
                f"low_h{horizon}",
                f"close_h{horizon}",
            }.issubset(frame.columns)
        ]
        exit_horizon = max(candidates) if candidates else 0
    exit_horizon = int(exit_horizon)
    if exit_horizon < 1:
        return pd.DataFrame(
            columns=[
                *frame.columns,
                "long_hit",
                "short_hit",
                "long_return",
                "short_return",
                "gross_return",
            ]
        )
    required = [
        *[f"high_h{horizon}" for horizon in range(1, exit_horizon + 1)],
        *[f"low_h{horizon}" for horizon in range(1, exit_horizon + 1)],
        f"close_h{exit_horizon}",
    ]
    if frame.empty or not set(required).issubset(frame.columns):
        return pd.DataFrame(
            columns=[*frame.columns, "long_hit", "short_hit", "long_return", "short_return", "gross_return"]
        )
    result = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    if result.empty:
        result["long_hit"] = pd.Series(dtype=bool)
        result["short_hit"] = pd.Series(dtype=bool)
        result["long_return"] = pd.Series(dtype=float)
        result["short_return"] = pd.Series(dtype=float)
        result["gross_return"] = pd.Series(dtype=float)
        return result

    max_high = result[
        [f"high_h{horizon}" for horizon in range(1, exit_horizon + 1)]
    ].max(axis=1)
    min_low = result[
        [f"low_h{horizon}" for horizon in range(1, exit_horizon + 1)]
    ].min(axis=1)
    close_final = pd.to_numeric(
        result[f"close_h{exit_horizon}"], errors="coerce"
    )
    long_hit = max_high >= long_threshold
    short_hit = min_low <= -short_threshold
    result["long_hit"] = long_hit
    result["short_hit"] = short_hit
    result["long_return"] = close_final.where(~long_hit, long_threshold)
    result["short_return"] = (-close_final).where(~short_hit, short_threshold)
    result["gross_return"] = result["long_return"] + result["short_return"]
    return result


def _two_sided_tp_metrics(
    detailed: pd.DataFrame,
    split: str,
    band_start: float,
    band_end: float,
    tp_long: float = float("nan"),
    tp_short: float = float("nan"),
) -> dict[str, Any]:
    n = int(len(detailed))
    long_hit = detailed.get("long_hit", pd.Series(False, index=detailed.index)).astype(bool)
    short_hit = detailed.get("short_hit", pd.Series(False, index=detailed.index)).astype(bool)
    gross = pd.to_numeric(detailed.get("gross_return"), errors="coerce")
    long_return = pd.to_numeric(detailed.get("long_return"), errors="coerce")
    short_return = pd.to_numeric(detailed.get("short_return"), errors="coerce")
    gross_mean = float(gross.mean()) if n and not gross.empty else float("nan")
    is_all = not np.isfinite(band_start)
    return {
        "split": split,
        "score_band": (
            f"ALL top 0-{BASE_FRACTION_BAND_MAX:.0%}"
            if is_all
            else f"top {float(band_start):.0%}-{float(band_end):.0%}"
        ),
        "band_start": band_start,
        "band_end": band_end,
        "trades": n,
        "long_hit": int(long_hit.sum()),
        "short_hit": int(short_hit.sum()),
        "both_hit": int((long_hit & short_hit).sum()),
        "long_only": int((long_hit & ~short_hit).sum()),
        "short_only": int((~long_hit & short_hit).sum()),
        "neither_hit": int((~long_hit & ~short_hit).sum()),
        "long_return_mean": (
            float(long_return.mean()) if n and not long_return.empty else float("nan")
        ),
        "short_return_mean": (
            float(short_return.mean()) if n and not short_return.empty else float("nan")
        ),
        "gross_mean": gross_mean,
        "e_net": (
            gross_mean - float(config.TRADE_COST)
            if np.isfinite(gross_mean)
            else float("nan")
        ),
        "tp_long": float(tp_long),
        "tp_short": float(tp_short),
    }


def _simulate_score_band_fixed_h5_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply TP and cut-loss in candle order, with stop-first OHLC ties."""
    if frame.empty:
        return pd.DataFrame(columns=[*frame.columns, "realized", "outcome"])
    result = frame.copy()
    outcome = pd.Series("close_h5", index=result.index, dtype=object)
    realized = pd.to_numeric(result["close_h5"], errors="coerce").copy()

    active = pd.Series(True, index=result.index, dtype=bool)

    cutloss_h1 = active & (result["adverse_h1"] <= float(SCORE_BAND_CUTLOSS))
    realized.loc[cutloss_h1] = float(SCORE_BAND_CUTLOSS)
    outcome.loc[cutloss_h1] = "cutloss_h1"
    hit_tp_h1 = (
        active
        & ~cutloss_h1
        & (result["high_h1"] >= float(SCORE_BAND_TP_H1_H4))
    )
    realized.loc[hit_tp_h1] = float(SCORE_BAND_TP_H1_H4)
    outcome.loc[hit_tp_h1] = "tp_h1"
    active &= ~(cutloss_h1 | hit_tp_h1)

    weak_h1 = active & (
        result["high_h1"] < float(SCORE_BAND_MAX_HIGH_BELOW_THRESHOLD)
    )
    result["weak_h1_branch"] = weak_h1

    for horizon in range(2, 5):
        cutloss = active & (
            result[f"adverse_h{horizon}"] <= float(SCORE_BAND_CUTLOSS)
        )
        realized.loc[cutloss] = float(SCORE_BAND_CUTLOSS)
        outcome.loc[cutloss] = f"cutloss_h{horizon}"

        weak_hit_tp = (
            active
            & weak_h1
            & ~cutloss
            & (result[f"high_h{horizon}"] >= float(SCORE_BAND_WEAK_H1_TP_H2_H4))
        )
        realized.loc[weak_hit_tp] = float(SCORE_BAND_WEAK_H1_TP_H2_H4)
        outcome.loc[weak_hit_tp] = f"weak_tp_h{horizon}"

        regular_hit_tp = (
            active
            & ~weak_h1
            & ~cutloss
            & (result[f"high_h{horizon}"] >= float(SCORE_BAND_TP_H1_H4))
        )
        realized.loc[regular_hit_tp] = float(SCORE_BAND_TP_H1_H4)
        outcome.loc[regular_hit_tp] = f"tp_h{horizon}"
        active &= ~(cutloss | weak_hit_tp | regular_hit_tp)

    gap_cutloss_h5 = active & (result["open_h5"] <= float(SCORE_BAND_CUTLOSS))
    realized.loc[gap_cutloss_h5] = float(SCORE_BAND_CUTLOSS)
    outcome.loc[gap_cutloss_h5] = "cutloss_h5"
    active &= ~gap_cutloss_h5

    hit_open_h5 = active & (result["open_h5"] >= float(SCORE_BAND_TP_H5))
    realized.loc[hit_open_h5] = result.loc[hit_open_h5, "open_h5"]
    outcome.loc[hit_open_h5] = "open_h5"
    active &= ~hit_open_h5

    cutloss_h5 = active & (result["adverse_h5"] <= float(SCORE_BAND_CUTLOSS))
    realized.loc[cutloss_h5] = float(SCORE_BAND_CUTLOSS)
    outcome.loc[cutloss_h5] = "cutloss_h5"
    active &= ~cutloss_h5

    hit_h5 = active & (result["high_h5"] >= float(SCORE_BAND_TP_H5))
    realized.loc[hit_h5] = float(SCORE_BAND_TP_H5)
    outcome.loc[hit_h5] = "tp_h5"

    result["realized"] = realized
    result["outcome"] = outcome
    return result


def _score_band_fixed_h5_metrics(
    detailed: pd.DataFrame,
    split: str,
    band_start: float,
    band_end: float,
    signals_before_filter: int | None = None,
) -> dict[str, Any]:
    n = int(len(detailed))
    n_before_filter = (
        n if signals_before_filter is None else int(signals_before_filter)
    )
    outcome = detailed.get("outcome", pd.Series(dtype=object))
    realized = pd.to_numeric(detailed.get("realized"), errors="coerce")

    def branch_return_mean(branch: str) -> float:
        values = realized[outcome == branch].dropna()
        return float(values.mean()) if not values.empty else float("nan")

    def outcome_return_mean(mask: pd.Series) -> float:
        values = realized[mask].dropna()
        return float(values.mean()) if not values.empty else float("nan")

    tp_masks = {
        horizon: outcome.isin([f"tp_h{horizon}", f"weak_tp_h{horizon}"])
        for horizon in range(1, 5)
    }
    weak_tp_mask = outcome.astype(str).str.startswith("weak_tp_h")

    gross_mean = (
        float(realized.mean())
        if n
        else float("nan")
    )
    followup_means: dict[str, float] = {}
    if "high_h1" in detailed.columns:
        high_h1 = pd.to_numeric(detailed["high_h1"], errors="coerce")
        high_h1_below = high_h1.lt(float(SCORE_BAND_MAX_HIGH_BELOW_THRESHOLD))
        high_h1_below_count = int(high_h1_below.sum())
        for horizon in (2, 3, 4):
            for price_name in ("high", "close"):
                column = f"{price_name}_h{horizon}"
                source = detailed.get(
                    column,
                    pd.Series(np.nan, index=detailed.index, dtype=float),
                )
                values = pd.to_numeric(
                    source.loc[high_h1_below], errors="coerce"
                ).dropna()
                followup_means[column] = (
                    float(values.mean()) if not values.empty else float("nan")
                )
    else:
        high_h1_below_count = 0
        for horizon in (2, 3, 4):
            followup_means[f"high_h{horizon}"] = float("nan")
            followup_means[f"close_h{horizon}"] = float("nan")
    is_all = not np.isfinite(band_start)
    return {
        "split": split,
        "score_band": (
            "ALL top 0-30%"
            if is_all
            else f"top {float(band_start):.0%}-{float(band_end):.0%}"
        ),
        "band_start": band_start,
        "band_end": band_end,
        "signals_before_filter": n_before_filter,
        "trades": n,
        "hit_h1": int(tp_masks[1].sum()),
        "hit_h2": int(tp_masks[2].sum()),
        "hit_h3": int(tp_masks[3].sum()),
        "hit_h4": int(tp_masks[4].sum()),
        "weak_h1_tp_h2_h4": int(weak_tp_mask.sum()),
        "hit_open_h5": int((outcome == "open_h5").sum()),
        "hit_h5": int((outcome == "tp_h5").sum()),
        "cutloss": int(outcome.astype(str).str.startswith("cutloss_").sum()),
        "close_h5": int((outcome == "close_h5").sum()),
        "hit_h1_return_mean": outcome_return_mean(tp_masks[1]),
        "hit_h2_return_mean": outcome_return_mean(tp_masks[2]),
        "hit_h3_return_mean": outcome_return_mean(tp_masks[3]),
        "hit_h4_return_mean": outcome_return_mean(tp_masks[4]),
        "weak_h1_tp_h2_h4_return_mean": outcome_return_mean(weak_tp_mask),
        "hit_open_h5_return_mean": branch_return_mean("open_h5"),
        "hit_h5_return_mean": branch_return_mean("tp_h5"),
        "cutloss_return_mean": (
            float(realized[outcome.astype(str).str.startswith("cutloss_")].mean())
            if outcome.astype(str).str.startswith("cutloss_").any()
            else float("nan")
        ),
        "close_h5_return_mean": branch_return_mean("close_h5"),
        "high_h1_below_threshold": high_h1_below_count,
        "high_h1_below_threshold_rate": (
            high_h1_below_count / n if n else 0.0
        ),
        "high_h1_below_threshold_high_h2_mean": followup_means["high_h2"],
        "high_h1_below_threshold_close_h2_mean": followup_means["close_h2"],
        "high_h1_below_threshold_high_h3_mean": followup_means["high_h3"],
        "high_h1_below_threshold_close_h3_mean": followup_means["close_h3"],
        "high_h1_below_threshold_high_h4_mean": followup_means["high_h4"],
        "high_h1_below_threshold_close_h4_mean": followup_means["close_h4"],
        "gross_mean": gross_mean,
        "e_net": gross_mean - float(config.TRADE_COST) if n else float("nan"),
    }


def _optimize_dynamic_tp(
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    levels: list[float],
    top_k: int = 5,
) -> pd.DataFrame:
    columns = [
        "rank",
        "tp_h1",
        "tp_exit1_selected",
        "tp_h2",
        "tp_exit2_selected",
        "tp_exit2_no_selected",
        "val_e_net",
        "test_e_net",
        "val_hit_rate",
        "test_hit_rate",
        "val_trades",
        "test_trades",
    ]
    if val_frame.empty or test_frame.empty:
        return pd.DataFrame(columns=columns)

    val_arrays = _dynamic_tp_arrays(val_frame)
    test_arrays = _dynamic_tp_arrays(test_frame)
    candidates = _top_dynamic_tp_candidates(val_arrays, levels, top_k)

    rows: list[dict[str, Any]] = []
    for rank, (total_return, hit_count, combo) in enumerate(candidates, start=1):
        val_metrics = {
            "e_net": total_return / len(val_arrays["close_final"])
            - float(config.TRADE_COST),
            "hit_rate": hit_count / len(val_arrays["close_final"]),
            "n_trades": float(len(val_arrays["close_final"])),
        }
        test_metrics = _simulate_dynamic_tp_arrays(test_arrays, combo)
        rows.append(
            {
                "rank": rank,
                "tp_h1": combo[0],
                "tp_exit1_selected": combo[1],
                "tp_h2": combo[2],
                "tp_exit2_selected": combo[3],
                "tp_exit2_no_selected": combo[4],
                "val_e_net": val_metrics["e_net"],
                "test_e_net": test_metrics["e_net"],
                "val_hit_rate": val_metrics["hit_rate"],
                "test_hit_rate": test_metrics["hit_rate"],
                "val_trades": int(val_metrics["n_trades"]),
                "test_trades": int(test_metrics["n_trades"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _top_dynamic_tp_candidates(
    arrays: dict[str, np.ndarray],
    levels: list[float],
    top_k: int,
) -> list[tuple[float, int, tuple[float, ...]]]:
    """Return the exact top dynamic-TP grid combinations without a 5-D product."""
    keep = max(int(top_k), 0)
    n = len(arrays["close_final"])
    if keep == 0 or n == 0:
        return []

    close_final = arrays["close_final"]
    all_rows = np.ones(n, dtype=bool)
    top_candidates: list[tuple[float, int, tuple[float, ...]]] = []

    for tp_h1 in levels:
        hit_h1 = all_rows & (arrays["high_h1"] > tp_h1)
        after_h1 = ~hit_h1
        h1_total = float(hit_h1.sum()) * float(tp_h1)
        h1_hits = int(hit_h1.sum())

        exit1_selected = after_h1 & arrays["exit1_selected"]
        exit1_options = _tp_branch_options(
            close_final,
            arrays["max_high_h2_plus"],
            exit1_selected,
            levels,
        )

        exit1_no_selected = after_h1 & ~arrays["exit1_selected"]
        downstream_options: list[tuple[float, int, tuple[float, ...]]] = []
        for tp_h2 in levels:
            hit_h2 = exit1_no_selected & (arrays["high_h2"] > tp_h2)
            after_h2 = exit1_no_selected & ~hit_h2
            h2_total = float(hit_h2.sum()) * float(tp_h2)
            h2_hits = int(hit_h2.sum())

            exit2_selected = after_h2 & arrays["exit2_selected"]
            exit2_selected_options = _tp_branch_options(
                close_final,
                arrays["max_high_h3_plus"],
                exit2_selected,
                levels,
            )
            exit2_no_selected = after_h2 & ~arrays["exit2_selected"]
            exit2_no_options = _tp_branch_options(
                close_final,
                arrays["max_high_h3_plus"],
                exit2_no_selected,
                levels,
            )

            for selected_total, selected_hits, tp_exit2 in exit2_selected_options:
                for no_total, no_hits, tp_exit2_no in exit2_no_options:
                    downstream_options.append(
                        (
                            h2_total + selected_total + no_total,
                            h2_hits + selected_hits + no_hits,
                            (float(tp_h2), float(tp_exit2), float(tp_exit2_no)),
                        )
                    )
        downstream_options = _take_top_candidates(downstream_options, keep)

        combined: list[tuple[float, int, tuple[float, ...]]] = []
        for exit1_total, exit1_hits, tp_exit1 in exit1_options:
            for (
                downstream_total,
                downstream_hits,
                downstream_combo,
            ) in downstream_options:
                combined.append(
                    (
                        h1_total + exit1_total + downstream_total,
                        h1_hits + exit1_hits + downstream_hits,
                        (float(tp_h1), float(tp_exit1), *downstream_combo),
                    )
                )
        top_candidates.extend(_take_top_candidates(combined, keep))

    return _take_top_candidates(top_candidates, keep)


def _tp_branch_options(
    close_final: np.ndarray,
    favorable_move: np.ndarray,
    mask: np.ndarray,
    levels: list[float],
) -> list[tuple[float, int, float]]:
    branch_close = close_final[mask]
    branch_move = favorable_move[mask]
    options: list[tuple[float, int, float]] = []
    for threshold in levels:
        hit = branch_move > float(threshold)
        total = float(np.where(hit, float(threshold), branch_close).sum())
        options.append((total, int(hit.sum()), float(threshold)))
    return options


def _take_top_candidates(
    candidates: list[tuple[float, int, tuple[float, ...]]],
    top_k: int,
) -> list[tuple[float, int, tuple[float, ...]]]:
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[: max(int(top_k), 0)]


def _dynamic_tp_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "high_h1": frame["high_h1"].to_numpy(dtype=float),
        "high_h2": frame["high_h2"].to_numpy(dtype=float),
        "max_high_h2_plus": frame["max_high_h2_plus"].to_numpy(dtype=float),
        "max_high_h3_plus": frame["max_high_h3_plus"].to_numpy(dtype=float),
        "close_final": frame["close_final"].to_numpy(dtype=float),
        "exit1_selected": frame["exit1_selected"].to_numpy(dtype=bool),
        "exit2_selected": frame["exit2_selected"].to_numpy(dtype=bool),
    }


def _simulate_dynamic_tp_arrays(
    arrays: dict[str, np.ndarray],
    thresholds: tuple[float, ...],
) -> dict[str, float]:
    if len(thresholds) != 5:
        raise ValueError("Dynamic TP simulation requires exactly five thresholds.")
    n = len(arrays["close_final"])
    if n == 0:
        return {"e_net": float("nan"), "hit_rate": 0.0, "n_trades": 0.0}

    tp_h1, tp_exit1, tp_h2, tp_exit2, tp_exit2_no = map(float, thresholds)
    realized = arrays["close_final"].copy()
    hit = np.zeros(n, dtype=bool)

    hit_h1 = arrays["high_h1"] > tp_h1
    realized[hit_h1] = tp_h1
    hit |= hit_h1

    after_h1 = ~hit_h1
    exit1_selected = after_h1 & arrays["exit1_selected"]
    hit_exit1 = exit1_selected & (arrays["max_high_h2_plus"] > tp_exit1)
    realized[hit_exit1] = tp_exit1
    hit |= hit_exit1

    exit1_no_selected = after_h1 & ~arrays["exit1_selected"]
    hit_h2 = exit1_no_selected & (arrays["high_h2"] > tp_h2)
    realized[hit_h2] = tp_h2
    hit |= hit_h2

    after_h2 = exit1_no_selected & ~hit_h2
    exit2_selected = after_h2 & arrays["exit2_selected"]
    hit_exit2 = exit2_selected & (arrays["max_high_h3_plus"] > tp_exit2)
    realized[hit_exit2] = tp_exit2
    hit |= hit_exit2

    exit2_no_selected = after_h2 & ~arrays["exit2_selected"]
    hit_exit2_no = exit2_no_selected & (arrays["max_high_h3_plus"] > tp_exit2_no)
    realized[hit_exit2_no] = tp_exit2_no
    hit |= hit_exit2_no

    return {
        "e_net": float(realized.mean() - float(config.TRADE_COST)),
        "hit_rate": float(hit.mean()),
        "n_trades": float(n),
    }


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


def _two_sided_sweep_thresholds() -> list[float]:
    count = (
        int(
            np.floor(
                (TWO_SIDED_SWEEP_END - TWO_SIDED_SWEEP_START)
                / TWO_SIDED_SWEEP_STEP
                + 1e-12
            )
        )
        + 1
    )
    return [
        float(TWO_SIDED_SWEEP_START + idx * TWO_SIDED_SWEEP_STEP)
        for idx in range(max(count, 0))
    ]


def _score_band_sweep_thresholds() -> list[float]:
    count = (
        int(
            np.floor(
                (SCORE_BAND_SWEEP_END - SCORE_BAND_SWEEP_START)
                / SCORE_BAND_SWEEP_STEP
                + 1e-12
            )
        )
        + 1
    )
    return [
        float(SCORE_BAND_SWEEP_START + idx * SCORE_BAND_SWEEP_STEP)
        for idx in range(max(count, 0))
    ]


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
            "close_h2_return_mean": float("nan"),
            "two_sided_count": float("nan"),
            "two_sided_rate": float("nan"),
        }
        for threshold in thresholds
    ]


def _empty_stage_tp_rows(split: str, thresholds: list[float]) -> list[dict[str, Any]]:
    groups = [
        "p1_base_signal",
        "p1_base_no_h1",
        "p1_exit_k1_selected",
        "p1_exit_k1_no_selected",
        "p1_exit_k1_no_selected_h2",
        "p2_exit_k1_no_selected_no_h2",
        "p2_exit_k2_selected",
        "p2_exit_k2_no_selected",
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
    label_direction: str = config.LABEL_DIRECTION,
    include_two_sided_move: bool = False,
) -> list[dict[str, Any]]:
    selected_path = path_returns.reindex(selected_base_index)
    hit_cols = _future_hit_columns(
        selected_path,
        min_h=min_h,
        max_h=max_h,
        label_direction=label_direction,
    )
    selected_path = (
        selected_path.dropna(subset=hit_cols) if hit_cols else selected_path.iloc[0:0]
    )
    total = int(len(selected_path))
    if total == 0 or not hit_cols:
        return _empty_tp_sweep_rows(split, thresholds, group=group)

    hit_values = selected_path[hit_cols].apply(pd.to_numeric, errors="coerce")
    high_cols = _future_price_columns(selected_path, "high", min_h=min_h, max_h=max_h)
    low_cols = _future_price_columns(selected_path, "low", min_h=min_h, max_h=max_h)
    high_values = selected_path[high_cols].apply(pd.to_numeric, errors="coerce")
    low_values = selected_path[low_cols].apply(pd.to_numeric, errors="coerce")
    if config.canonical_label_direction(label_direction) == "short":
        # Path returns are direction-normalized; restore raw price returns.
        high_values = -high_values
        low_values = -low_values
    close_col = (
        f"close_h{_max_h_from_hit_columns(hit_cols, label_direction=label_direction)}"
    )
    close_values = (
        pd.to_numeric(selected_path[close_col], errors="coerce")
        if close_col in selected_path.columns
        else pd.Series(np.nan, index=selected_path.index, dtype=float)
    )
    close_h2_values = (
        pd.to_numeric(selected_path["close_h2"], errors="coerce").dropna()
        if "close_h2" in selected_path.columns
        else pd.Series(dtype=float)
    )
    close_h2_return_mean = (
        float(close_h2_values.mean()) if not close_h2_values.empty else float("nan")
    )
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        hit = (hit_values > float(threshold)).any(axis=1)
        hit_count = int(hit.sum())
        miss = ~hit
        miss_count = int(miss.sum())
        miss_close = close_values[miss & close_values.notna()]
        miss_return_mean = (
            float(miss_close.mean()) if not miss_close.empty else float("nan")
        )
        two_sided_count: int | float = float("nan")
        two_sided_rate = float("nan")
        if (
            include_two_sided_move
            and float(threshold) >= 0.0
            and high_cols
            and low_cols
        ):
            two_sided = (high_values > float(threshold)).any(axis=1) & (
                low_values < -float(threshold)
            ).any(axis=1)
            two_sided_count = int(two_sided.sum())
            two_sided_rate = two_sided_count / total if total else 0.0
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
                "close_h2_return_mean": close_h2_return_mean,
                "two_sided_count": two_sided_count,
                "two_sided_rate": two_sided_rate,
            }
        )
    return rows


def _hit_price_prefix(label_direction: str = config.LABEL_DIRECTION) -> str:
    return (
        "low"
        if config.canonical_label_direction(label_direction) == "short"
        else "high"
    )


def _adverse_price_prefix(label_direction: str = config.LABEL_DIRECTION) -> str:
    return (
        "high"
        if config.canonical_label_direction(label_direction) == "short"
        else "low"
    )


def _future_hit_columns(
    frame: pd.DataFrame,
    min_h: int,
    max_h: int | None = None,
    label_direction: str = config.LABEL_DIRECTION,
) -> list[str]:
    return _future_price_columns(
        frame,
        _hit_price_prefix(label_direction),
        min_h=min_h,
        max_h=max_h,
    )


def _future_price_columns(
    frame: pd.DataFrame,
    price_prefix: str,
    min_h: int,
    max_h: int | None = None,
) -> list[str]:
    columns: list[tuple[int, str]] = []
    prefix = f"{price_prefix}_h"
    min_h = max(int(min_h), 1)
    max_h_value = int(max_h) if max_h is not None else None
    for column in frame.columns:
        text = str(column)
        if not text.startswith(prefix):
            continue
        suffix = text[len(prefix) :]
        if not suffix.isdigit():
            continue
        step = int(suffix)
        if max_h_value is not None and step > max_h_value:
            continue
        if step >= min_h:
            columns.append((step, text))
    return [name for _, name in sorted(columns)]


def _max_h_from_hit_columns(
    columns: list[str],
    label_direction: str = config.LABEL_DIRECTION,
) -> int:
    steps: list[int] = []
    prefix = f"{_hit_price_prefix(label_direction)}_h"
    for column in columns:
        text = str(column)
        if not text.startswith(prefix):
            continue
        suffix = text[len(prefix) :]
        if suffix.isdigit():
            steps.append(int(suffix))
    return max(steps) if steps else 1


def _plot_summary(
    summary: pd.DataFrame,
    score_band_rates: pd.DataFrame,
    two_sided_score_band: pd.DataFrame,
    two_sided_trade_path: pd.DataFrame,
    crypto_close: pd.Series,
    chart_path: Path,
    base_label: str,
    base_ensemble: str,
    tp_threshold: float,
    label_direction: str,
) -> None:
    fig, axes = plt.subplots(
        7,
        1,
        figsize=(24.0, 43.0),
        gridspec_kw={
            "height_ratios": [0.9, 6.2, 6.2, 6.2, 5.5, 2.6, 3.2]
        },
        constrained_layout=True,
    )
    (
        ax_overview,
        ax_high,
        ax_low,
        ax_two_sided_rate,
        ax_two_sided_strategy,
        ax_two_sided_trade_path,
        ax_two_sided_time_path,
    ) = axes

    _draw_table(
        ax_overview,
        _overview_table(summary),
        title=(
            f"Base signal overview | evaluation direction={label_direction} | "
            f"ensemble={base_ensemble.upper()} | TP reference={tp_threshold:.2%}\n"
            f"{base_label}"
        ),
        font_size=8.0,
    )
    _draw_table(
        ax_high,
        _fraction_band_sweep_table(
            score_band_rates,
            group_name=BASE_SIGNAL_HIGH_GROUP,
            thresholds=_score_band_sweep_thresholds(),
        ),
        title=(
            "Base signal score bands: TP high hit rate in "
            f"H1-H{BASE_SIGNAL_ANALYSIS_HORIZON} | high > TP | "
            "0.00%..1.00% step=0.05% | disjoint 5% bands to "
            f"top {BASE_FRACTION_BAND_MAX:.0%}"
        ),
        font_size=5.4,
    )
    _draw_table(
        ax_low,
        _fraction_band_sweep_table(
            score_band_rates,
            group_name=BASE_SIGNAL_LOW_GROUP,
            thresholds=[-value for value in _score_band_sweep_thresholds()],
        ),
        title=(
            "Base signal score bands: TP low hit rate in "
            f"H1-H{BASE_SIGNAL_ANALYSIS_HORIZON} | low < TP | "
            "-0.00%..-1.00% step=0.05% | disjoint 5% bands to "
            f"top {BASE_FRACTION_BAND_MAX:.0%}"
        ),
        font_size=5.4,
    )
    _draw_table(
        ax_two_sided_rate,
        _fraction_band_two_sided_table(score_band_rates),
        title=(
            "Base signal score bands: P(max high "
            f"H1-H{BASE_SIGNAL_ANALYSIS_HORIZON} > +x AND min low "
            f"H1-H{BASE_SIGNAL_ANALYSIS_HORIZON} < -x) | "
            "x=0.05%..1.00% step=0.05% | disjoint 5% bands to "
            f"top {BASE_FRACTION_BAND_MAX:.0%}"
        ),
        font_size=5.4,
    )
    _draw_table(
        ax_two_sided_strategy,
        _two_sided_score_band_table(two_sided_score_band),
        title=(
            _two_sided_strategy_title()
            + " | "
            "unfilled legs exit at close "
            f"H{BASE_SIGNAL_ANALYSIS_HORIZON} | "
            "E[net] subtracts TRADE_COST once"
        ),
        font_size=6.0,
    )
    _draw_two_sided_trade_path(ax_two_sided_trade_path, two_sided_trade_path)
    _draw_two_sided_time_path(
        ax_two_sided_time_path,
        two_sided_trade_path,
        crypto_close=crypto_close,
    )
    fig.savefig(chart_path, dpi=170)
    plt.close(fig)


def _two_sided_strategy_title() -> str:
    if SCORE_BAND_TWO_SIDED_OPTIMIZE_TP:
        return (
            "Val-optimized two-sided score-band strategy | open Long + Short | "
            "joint TP grid 0.00%.."
            f"{SCORE_BAND_TWO_SIDED_TP_LONG:.2%} Long and 0.00%.."
            f"{SCORE_BAND_TWO_SIDED_TP_SHORT:.2%} Short | "
            "Val pair applied unchanged to Test"
        )
    return (
        "Fixed-TP two-sided score-band strategy | open Long + Short | "
        f"TP Long={SCORE_BAND_TWO_SIDED_TP_LONG:.2%} | "
        f"TP Short={SCORE_BAND_TWO_SIDED_TP_SHORT:.2%}"
    )


def _overview_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = ["split", "base signal", "avg trades/day", "TP_threshold"]
    rows = [
        {
            "split": str(row.get("split", "")),
            "base signal": _count_cell(row.get("base_signals")),
            "avg trades/day": _format_decimal(row.get("trades_per_day"), digits=2),
            "TP_threshold": _format_pct(float(row.get("tp_threshold", 0.0))),
        }
        for _, row in summary.iterrows()
    ]
    return pd.DataFrame(rows, columns=columns)


def _draw_two_sided_trade_path(ax: Any, trade_path: pd.DataFrame) -> None:
    """Draw cumulative net return in chronological trade order."""
    tp_suffix = ""
    if not trade_path.empty:
        tp_long = pd.to_numeric(trade_path.get("tp_long"), errors="coerce").dropna()
        tp_short = pd.to_numeric(trade_path.get("tp_short"), errors="coerce").dropna()
        if not tp_long.empty and not tp_short.empty:
            pair_count = len(
                pd.DataFrame(
                    {"tp_long": tp_long, "tp_short": tp_short}
                ).drop_duplicates()
            )
            if pair_count == 1:
                tp_suffix = (
                    f" | TP Long=+{float(tp_long.iloc[0]):.2%}, "
                    f"Short=-{float(tp_short.iloc[0]):.2%}"
                )
            else:
                tp_suffix = f" | {pair_count} band-specific Val-opt TP pairs"
    ax.set_title(
        "Two-sided strategy trade path | cumulative sum of per-trade net return "
        "| configured --base top fraction(s) | TRADE_COST once per signal"
        f"{tp_suffix}",
        fontsize=10,
        pad=8,
    )
    ax.axhline(0.0, color="#4b5563", linewidth=0.9, linestyle="--", alpha=0.8)
    if trade_path.empty:
        ax.text(
            0.5,
            0.5,
            "No valid trades",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_axis_off()
        return

    colors = {"val": "#2563eb", "test": "#dc2626"}
    plotted = False
    for split_name in ("val", "test"):
        split_path = trade_path[
            trade_path["split"].astype(str).str.lower() == split_name
        ].copy()
        if split_path.empty:
            continue
        split_path = split_path.sort_index()
        cumulative = (
            pd.to_numeric(
                split_path["cumulative_net_return"], errors="coerce"
            )
            * 100.0
        )
        cumulative = cumulative.dropna()
        if cumulative.empty:
            continue
        trade_number = np.arange(1, len(cumulative) + 1)
        ending = float(cumulative.iloc[-1])
        mean_net = float(
            pd.to_numeric(split_path["net_return"], errors="coerce").mean()
            * 100.0
        )
        ax.plot(
            trade_number,
            cumulative.to_numpy(),
            linewidth=1.4,
            color=colors[split_name],
            label=(
                f"{split_name.upper()} n={len(cumulative):,} | "
                f"end={ending:+.2f}% | mean/trade={mean_net:+.3f}%"
            ),
        )
        plotted = True

    ax.set_xlabel("Chronological trade number")
    ax.set_ylabel("Cumulative net return (percentage points)")
    ax.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.65)
    if plotted:
        ax.legend(loc="best", frameon=False, fontsize=8)


def _draw_two_sided_time_path(
    ax: Any,
    trade_path: pd.DataFrame,
    crypto_close: pd.Series | None = None,
) -> None:
    """Draw the Val-to-Test return path, price shape, and daily trade counts."""
    ax.set_title(
        "Two-sided strategy timeline | continuous Val + Test cumulative net "
        "return | normalized crypto price shape | daily trade count",
        fontsize=10,
        pad=8,
    )
    ax.axhline(0.0, color="#4b5563", linewidth=0.9, linestyle="--", alpha=0.8)
    if trade_path.empty:
        ax.text(
            0.5,
            0.5,
            "No valid trades",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_axis_off()
        return

    timeline = trade_path.copy()
    timeline.index = pd.to_datetime(timeline.index, errors="coerce")
    timeline["net_return"] = pd.to_numeric(
        timeline.get("net_return"), errors="coerce"
    )
    timeline = timeline[
        timeline.index.notna() & timeline["net_return"].notna()
    ].sort_index()
    if timeline.empty:
        ax.text(
            0.5,
            0.5,
            "No timestamped trades",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_axis_off()
        return

    cumulative = timeline["net_return"].cumsum() * 100.0
    ending = float(cumulative.iloc[-1])
    mean_net = float(timeline["net_return"].mean() * 100.0)
    return_line = ax.plot(
        timeline.index,
        cumulative,
        color="#0f766e",
        linewidth=1.5,
        label=(
            f"Cumulative net | n={len(timeline):,} | end={ending:+.2f}% "
            f"| mean/trade={mean_net:+.3f}%"
        ),
    )
    ax.set_xlabel("Signal time")
    ax.set_ylabel("Cumulative net return (percentage points)", color="#0f766e")
    ax.tick_params(axis="y", labelcolor="#0f766e")
    ax.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.65)

    price_line: list[Any] = []
    if crypto_close is not None:
        close = pd.to_numeric(crypto_close, errors="coerce").copy()
        close.index = pd.to_datetime(close.index, errors="coerce")
        close = close[
            close.index.notna()
            & close.notna()
            & (close.index >= timeline.index.min())
            & (close.index <= timeline.index.max())
        ].sort_index()
        if not close.empty and float(close.max()) > float(close.min()):
            y_min = min(0.0, float(cumulative.min()))
            y_max = max(0.0, float(cumulative.max()))
            if np.isclose(y_min, y_max):
                y_min, y_max = -1.0, 1.0
            normalized_close = (
                (close - float(close.min()))
                / (float(close.max()) - float(close.min()))
                * (y_max - y_min)
                + y_min
            )
            price_line = ax.plot(
                close.index,
                normalized_close,
                color="#6b7280",
                linewidth=1.0,
                alpha=0.72,
                label="Crypto close price shape (normalized; no price scale)",
            )
            ax.set_ylim(y_min, y_max)

    trade_days = pd.Series(1, index=timeline.index, dtype=int).resample("D").sum()
    full_days = pd.date_range(
        trade_days.index.min(),
        trade_days.index.max(),
        freq="D",
    )
    trade_days = trade_days.reindex(full_days, fill_value=0)
    count_ax = ax.twinx()
    count_bars = count_ax.bar(
        trade_days.index,
        trade_days.to_numpy(),
        width=0.85,
        color="#f59e0b",
        alpha=0.24,
        label="Trades/day",
    )
    count_ax.set_ylabel("Trades per day", color="#b45309")
    count_ax.tick_params(axis="y", labelcolor="#b45309")
    count_ax.set_ylim(bottom=0.0)

    handles = [*return_line, *price_line, count_bars]
    labels = [handle.get_label() for handle in handles]
    ax.legend(handles, labels, loc="best", frameon=False, fontsize=8)


def _count_cell(value: Any) -> str:
    try:
        if pd.isna(value):
            return "0"
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _sweep_table(
    tp_sweep: pd.DataFrame,
    group_name: str,
    include_close_h2: bool = False,
) -> pd.DataFrame:
    columns = ["split"] + [
        _format_threshold_pct(threshold) for threshold in _tp_sweep_thresholds()
    ]
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
        two_sided_row: dict[str, str] = {"split": f"{split} high>thr & low<-thr"}
        miss_return_row: dict[str, str] = {"split": f"{split} miss ret"}
        has_two_sided = (
            pd.to_numeric(
                sorted_group.get("two_sided_rate"),
                errors="coerce",
            )
            .notna()
            .any()
        )
        for _, item in sorted_group.iterrows():
            threshold_label = _format_threshold_pct(float(item["tp_threshold"]))
            hit_row[threshold_label] = _format_pct(float(item["hit_rate"]))
            miss_return = item.get("miss_return_mean", float("nan"))
            miss_return_row[threshold_label] = _format_signed_pct(miss_return)
            two_sided_rate = item.get("two_sided_rate", float("nan"))
            two_sided_row[threshold_label] = (
                _format_pct(float(two_sided_rate)) if pd.notna(two_sided_rate) else ""
            )
        rows.append(hit_row)
        if has_two_sided:
            rows.append(two_sided_row)
        rows.append(miss_return_row)
        if include_close_h2:
            close_h2 = pd.to_numeric(
                sorted_group.get("close_h2_return_mean"),
                errors="coerce",
            ).dropna()
            close_h2_row: dict[str, str] = {"split": f"{split} mean close H2"}
            if len(columns) > 1 and not close_h2.empty:
                close_h2_row[columns[1]] = _format_signed_pct(float(close_h2.iloc[0]))
            rows.append(close_h2_row)
    return pd.DataFrame(rows, columns=columns).fillna("")


def _fraction_band_sweep_table(
    tp_sweep: pd.DataFrame,
    group_name: str,
    thresholds: list[float],
) -> pd.DataFrame:
    columns = ["split / score band"] + [
        _format_threshold_pct(threshold) for threshold in thresholds
    ]
    required = {"group", "band_start", "band_end", "tp_threshold", "hit_rate"}
    if tp_sweep.empty or not required.issubset(tp_sweep.columns):
        return pd.DataFrame(columns=columns)
    subset = tp_sweep[tp_sweep["group"].astype(str) == str(group_name)]
    if subset.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, str]] = []
    for (split, band_start, band_end), group in subset.groupby(
        ["split", "band_start", "band_end"],
        sort=False,
        dropna=False,
    ):
        sorted_group = group.sort_values("tp_threshold")
        n_value = pd.to_numeric(sorted_group["sample_count"], errors="coerce").max()
        n = int(n_value) if pd.notna(n_value) else 0
        row: dict[str, str] = {
            "split / score band": (
                f"{split} top {float(band_start):.0%}-{float(band_end):.0%} n={n}"
            )
        }
        for _, item in sorted_group.iterrows():
            hit_rate = item.get("hit_rate", float("nan"))
            row[_format_threshold_pct(float(item["tp_threshold"]))] = (
                _format_pct(float(hit_rate)) if pd.notna(hit_rate) else ""
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns).fillna("")


def _fraction_band_two_sided_table(tp_sweep: pd.DataFrame) -> pd.DataFrame:
    columns = ["split / score band"] + [
        _format_threshold_pct(threshold)
        for threshold in _two_sided_sweep_thresholds()
    ]
    required = {
        "group",
        "band_start",
        "band_end",
        "tp_threshold",
        "two_sided_rate",
    }
    if tp_sweep.empty or not required.issubset(tp_sweep.columns):
        return pd.DataFrame(columns=columns)
    subset = tp_sweep[
        tp_sweep["group"].astype(str) == BASE_SIGNAL_TWO_SIDED_GROUP
    ]
    if subset.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, str]] = []
    for (split, band_start, band_end), group in subset.groupby(
        ["split", "band_start", "band_end"],
        sort=False,
        dropna=False,
    ):
        sorted_group = group.sort_values("tp_threshold")
        n_value = pd.to_numeric(sorted_group["sample_count"], errors="coerce").max()
        n = int(n_value) if pd.notna(n_value) else 0
        row: dict[str, str] = {
            "split / score band": (
                f"{split} top {float(band_start):.0%}-{float(band_end):.0%} n={n}"
            )
        }
        for _, item in sorted_group.iterrows():
            threshold_label = _format_threshold_pct(float(item["tp_threshold"]))
            rate = item.get("two_sided_rate", float("nan"))
            row[threshold_label] = (
                _format_pct(float(rate)) if pd.notna(rate) else ""
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns).fillna("")


def _score_band_strategy_table(results: pd.DataFrame) -> pd.DataFrame:
    threshold_text = _format_threshold_pct(SCORE_BAND_MAX_HIGH_BELOW_THRESHOLD)
    high_h1_below_label = (
        f"high H1<{threshold_text} / all"
    )
    followup_labels = {
        horizon: (
            f"high H1<{threshold_text} "
            f"mean high H{horizon} | mean close H{horizon}"
        )
        for horizon in (2, 3, 4)
    }
    columns = [
        "split / score band",
        "n",
        "n_after_filter",
        "hit H1",
        "hit H2",
        "hit H3",
        "hit H4",
        "weak H1 TP H2-H4",
        "hit open H5",
        "hit H5",
        "cutloss",
        "close H5",
        high_h1_below_label,
        followup_labels[2],
        followup_labels[3],
        followup_labels[4],
        "gross mean",
        "E[net]",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, str]] = []
    for _, row in results.iterrows():
        n_after_filter = int(row.get("trades", 0) or 0)
        n = int(row.get("signals_before_filter", n_after_filter) or 0)
        rows.append(
            {
                "split / score band": f"{row.get('split', '')} {row.get('score_band', '')}",
                "n": str(n),
                "n_after_filter": str(n_after_filter),
                "hit H1": _count_fraction_return_cell(
                    row.get("hit_h1"),
                    n_after_filter,
                    row.get("hit_h1_return_mean"),
                ),
                "hit H2": _count_fraction_return_cell(
                    row.get("hit_h2"),
                    n_after_filter,
                    row.get("hit_h2_return_mean"),
                ),
                "hit H3": _count_fraction_return_cell(
                    row.get("hit_h3"),
                    n_after_filter,
                    row.get("hit_h3_return_mean"),
                ),
                "hit H4": _count_fraction_return_cell(
                    row.get("hit_h4"),
                    n_after_filter,
                    row.get("hit_h4_return_mean"),
                ),
                "weak H1 TP H2-H4": _count_fraction_return_cell(
                    row.get("weak_h1_tp_h2_h4"),
                    n_after_filter,
                    row.get("weak_h1_tp_h2_h4_return_mean"),
                ),
                "hit open H5": _count_fraction_return_cell(
                    row.get("hit_open_h5"),
                    n_after_filter,
                    row.get("hit_open_h5_return_mean"),
                ),
                "hit H5": _count_fraction_return_cell(
                    row.get("hit_h5"),
                    n_after_filter,
                    row.get("hit_h5_return_mean"),
                ),
                "cutloss": _count_fraction_return_cell(
                    row.get("cutloss"),
                    n_after_filter,
                    row.get("cutloss_return_mean"),
                ),
                "close H5": _count_fraction_return_cell(
                    row.get("close_h5"),
                    n_after_filter,
                    row.get("close_h5_return_mean"),
                ),
                high_h1_below_label: _count_fraction_cell(
                    row.get("high_h1_below_threshold"),
                    n_after_filter,
                ),
                **{
                    followup_labels[horizon]: (
                        f"{_format_signed_pct(row.get(f'high_h1_below_threshold_high_h{horizon}_mean')) or 'n/a'}"
                        " | "
                        f"{_format_signed_pct(row.get(f'high_h1_below_threshold_close_h{horizon}_mean')) or 'n/a'}"
                    )
                    for horizon in (2, 3, 4)
                },
                "gross mean": _format_signed_pct(row.get("gross_mean")),
                "E[net]": _format_signed_pct(row.get("e_net")),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _two_sided_score_band_table(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "split / score band",
        "n",
        "Long hit",
        "Short hit",
        "both TP",
        "Long only",
        "Short only",
        "neither",
        "Long mean",
        "Short mean",
        "gross mean",
        "E[net]",
        "TP Long opt",
        "TP Short opt",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, str]] = []
    for _, row in results.iterrows():
        n = int(row.get("trades", 0) or 0)
        rows.append(
            {
                "split / score band": f"{row.get('split', '')} {row.get('score_band', '')}",
                "n": str(n),
                "Long hit": _count_fraction_cell(row.get("long_hit"), n),
                "Short hit": _count_fraction_cell(row.get("short_hit"), n),
                "both TP": _count_fraction_cell(row.get("both_hit"), n),
                "Long only": _count_fraction_cell(row.get("long_only"), n),
                "Short only": _count_fraction_cell(row.get("short_only"), n),
                "neither": _count_fraction_cell(row.get("neither_hit"), n),
                "Long mean": _format_signed_pct(row.get("long_return_mean")),
                "Short mean": _format_signed_pct(row.get("short_return_mean")),
                "gross mean": _format_signed_pct(row.get("gross_mean")),
                "E[net]": _format_signed_pct(row.get("e_net")),
                "TP Long opt": _format_unsigned_pct(row.get("tp_long")),
                "TP Short opt": _format_unsigned_pct(row.get("tp_short")),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _count_fraction_cell(value: Any, denominator: int) -> str:
    count = int(value or 0)
    fraction = count / denominator if denominator else 0.0
    return f"{count} ({fraction:.1%})"


def _count_fraction_return_cell(
    value: Any,
    denominator: int,
    return_mean: Any,
) -> str:
    base = _count_fraction_cell(value, denominator)
    formatted_return = _format_signed_pct(return_mean)
    return f"{base} | {formatted_return or 'n/a'}"


def _dynamic_tp_table(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rank",
        "TP H1",
        "TP E1 selected",
        "TP H2",
        "TP E2 selected",
        "TP E2 no-selected",
        "val E[net]",
        "test E[net]",
        "val hit",
        "test hit",
        "val n",
        "test n",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, str]] = []
    for _, row in results.iterrows():
        rows.append(
            {
                "rank": _count_cell(row.get("rank")),
                "TP H1": _format_pct(float(row.get("tp_h1", 0.0))),
                "TP E1 selected": _format_pct(float(row.get("tp_exit1_selected", 0.0))),
                "TP H2": _format_pct(float(row.get("tp_h2", 0.0))),
                "TP E2 selected": _format_pct(float(row.get("tp_exit2_selected", 0.0))),
                "TP E2 no-selected": _format_pct(
                    float(row.get("tp_exit2_no_selected", 0.0))
                ),
                "val E[net]": _format_signed_pct(row.get("val_e_net")),
                "test E[net]": _format_signed_pct(row.get("test_e_net")),
                "val hit": _format_pct(float(row.get("val_hit_rate", 0.0))),
                "test hit": _format_pct(float(row.get("test_hit_rate", 0.0))),
                "val n": _count_cell(row.get("val_trades")),
                "test n": _count_cell(row.get("test_trades")),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _draw_table(
    ax: plt.Axes,
    table_df: pd.DataFrame,
    title: str,
    font_size: float,
) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=9, loc="left", pad=3)
    if table_df.empty:
        ax.text(
            0.5,
            0.44,
            "No samples for this strategy/filter configuration.",
            ha="center",
            va="center",
            fontsize=max(8.0, float(font_size) + 1.0),
            color="#6b7280",
            transform=ax.transAxes,
        )
        return
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


def _format_unsigned_pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.2%}"


def _format_decimal(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.{int(digits)}f}"


def _format_signed_pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:+.2%}"


def _base_score_band_name(
    base_specs: list[ModelSpec],
    base_ensemble: str,
    tp_threshold: float,
    label_direction: str = config.LABEL_DIRECTION,
) -> str:
    base_name = "_".join(
        (
            f"b{i}_{_short_safe_name(spec.archive_path.stem)}_r{spec.rank:02d}"
            f"_top{_threshold_filename_token(float(spec.top_fraction))}"
        )
        for i, spec in enumerate(base_specs, start=1)
    )
    tp_name = _threshold_filename_token(float(tp_threshold))
    direction_name = config.canonical_label_direction(label_direction)
    full_name = _safe_name(
        "_".join(
            [
                "base_score_bands",
                direction_name,
                base_ensemble,
                base_name,
                f"tp_{tp_name}",
                f"h{BASE_SIGNAL_ANALYSIS_HORIZON}",
            ]
        )
    )
    if len(full_name) <= 150:
        return full_name
    digest = hashlib.sha1(full_name.encode("utf-8")).hexdigest()[:12]
    return _safe_name(
        "_".join(
            [
                "base_score_bands",
                direction_name,
                base_ensemble,
                f"bases{len(base_specs)}",
                f"tp_{tp_name}",
                f"h{BASE_SIGNAL_ANALYSIS_HORIZON}",
                digest,
            ]
        )
    )


def _backtest_name(
    base_specs: list[ModelSpec],
    exit1_spec: ModelSpec,
    exit2_spec: ModelSpec,
    base_ensemble: str,
    tp_threshold: float,
    label_direction: str = config.LABEL_DIRECTION,
) -> str:
    base_name = "_".join(
        (
            f"b{i}_{_short_safe_name(spec.archive_path.stem)}_r{spec.rank:02d}"
            f"_top{_threshold_filename_token(float(spec.top_fraction))}"
        )
        for i, spec in enumerate(base_specs, start=1)
    )
    exit1_name = (
        f"x1_{_short_safe_name(exit1_spec.archive_path.stem)}_r{exit1_spec.rank:02d}"
    )
    exit2_name = (
        f"x2_{_short_safe_name(exit2_spec.archive_path.stem)}_r{exit2_spec.rank:02d}"
    )
    tp_name = _threshold_filename_token(float(tp_threshold))
    direction_name = config.canonical_label_direction(label_direction)
    exit1_top_name = _threshold_filename_token(float(exit1_spec.top_fraction))
    exit2_top_name = _threshold_filename_token(float(exit2_spec.top_fraction))
    full_name = _safe_name(
        "_".join(
            [
                f"strategy_e1top_{exit1_top_name}",
                f"e2top_{exit2_top_name}",
                direction_name,
                base_ensemble,
                base_name,
                exit1_name,
                exit2_name,
                f"tp_{tp_name}",
            ]
        )
    )
    if len(full_name) <= 150:
        return full_name

    # Keep Windows output paths comfortably below MAX_PATH while retaining the
    # parameters most useful when scanning a directory. The digest covers every
    # archive, rank, threshold and top fraction contained in the full name.
    digest = hashlib.sha1(full_name.encode("utf-8")).hexdigest()[:12]
    return _safe_name(
        "_".join(
            [
                f"strategy_e1top_{exit1_top_name}",
                f"e2top_{exit2_top_name}",
                direction_name,
                base_ensemble,
                f"bases{len(base_specs)}",
                f"tp_{tp_name}",
                digest,
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
    if len(parts) not in {4, 5, 6, 7}:
        raise ValueError(
            "Spec must be ARCHIVE#RANK#MODE#THRESHOLD"
            "[#TOP_FRACTION[#DIRECTION[#EXIT_AFTER_K]]], "
            f"got: {value!r}"
        )
    archive_text, rank_text, mode_text, threshold_text = parts[:4]
    mode_text = config.canonical_label_mode(mode_text)
    if require_top_fraction and len(parts) < 5:
        raise ValueError(
            "Exit spec must include TOP_FRACTION or use a top-fraction CLI option."
        )
    top_fraction = (
        float(parts[4]) if len(parts) == 5 and parts[4] else float(default_top_fraction)
    )
    if len(parts) >= 6:
        top_fraction = float(parts[4]) if parts[4] else float(default_top_fraction)
    direction = _archive_label_direction(
        Path(archive_text),
        explicit_direction=parts[5] if len(parts) >= 6 and parts[5] else None,
    )
    explicit_k = int(parts[6]) if len(parts) == 7 and parts[6] else None
    if not 0 < top_fraction <= 1:
        raise ValueError(f"top fraction must be in (0, 1], got {top_fraction}.")
    return ModelSpec(
        archive_path=Path(archive_text),
        rank=int(rank_text),
        label_mode=mode_text,
        label_direction=direction,
        label_threshold=float(threshold_text),
        top_fraction=top_fraction,
        exit_after_k=(
            config.resolve_exit_after_k(mode_text, explicit_k)
            if explicit_k is not None
            else _archive_exit_after_k(Path(archive_text), mode_text)
        ),
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
            "Base signal spec(s): ARCHIVE#RANK#MODE#THRESHOLD"
            "[#TOP_FRACTION[#DIRECTION]]. "
            "Each base may use its own fifth-field top fraction; otherwise that base "
            "falls back to config.TRADE_TOP_FRACTION. Quote each spec in the shell."
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
        default=None,
        help=(
            "Deprecated compatibility option. Exit models are no longer "
            "trained or included in this base score-band report."
        ),
    )
    parser.add_argument(
        "--exit2",
        default=None,
        help=(
            "Deprecated compatibility option. Exit models are no longer "
            "trained or included in this base score-band report."
        ),
    )
    parser.add_argument(
        "--exit1-top-fraction",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Deprecated compatibility option; ignored."
        ),
    )
    parser.add_argument(
        "--exit2-top-fraction",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Deprecated compatibility option; ignored."
        ),
    )
    parser.add_argument(
        "--tp-threshold",
        type=float,
        default=None,
        help=(
            "Reference TP shown in the overview. Default: first base spec "
            "label threshold."
        ),
    )
    parser.add_argument(
        "--label-direction",
        default=None,
        help=(
            "Common strategy evaluation direction. Long uses high as TP; Short "
            "uses low as TP. This does not override model training directions: "
            "each base uses its own spec/archive direction. Required when base "
            "specs mix Long and Short."
        ),
    )
    parser.add_argument(
        "--data", default=str(config.DATA_PATH), help="Crypto OHLCV CSV path."
    )
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
    exit1_spec = (
        _parse_spec(
            args.exit1,
            default_top_fraction=float(config.TRADE_TOP_FRACTION),
            require_top_fraction=False,
        )
        if args.exit1
        else None
    )
    exit2_spec = (
        _parse_spec(
            args.exit2,
            default_top_fraction=float(config.TRADE_TOP_FRACTION),
            require_top_fraction=False,
        )
        if args.exit2
        else None
    )
    if args.exit1_top_fraction is not None or args.exit2_top_fraction is not None:
        logger.warning(
            "--exit1-top-fraction/--exit2-top-fraction are deprecated and ignored."
        )

    results = [
        run_backtest(
            base_specs=base_specs,
            exit1_spec=exit1_spec,
            exit2_spec=exit2_spec,
            data_path=args.data,
            out_dir=args.out_dir,
            base_ensemble=args.base_ensemble,
            tp_threshold=args.tp_threshold,
            label_direction=args.label_direction,
            val_start=args.val_start,
            test_start=args.test_start,
            test_end=args.test_end,
        )
    ]
    print(
        json.dumps(
            [
                {
                    "chart": str(result.chart_path),
                }
                for result in results
            ],
            indent=2,
        )
    )
    for result in results:
        print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
