"""Train a simple crypto meta learner from walk-forward OOF predictions.

The meta learner is intentionally narrow: it only learns on rows where the
selected archive individual would have produced a trade signal. This makes it a
signal filter, not a replacement for the evolutionary model.

Example:
    python -m crypto.meta_learner.train \
      --archive crypto/results/crypto_btc_mfe_seed1_12h.json \
      --rank 1 \
      --base-label-mode mfe \
      --meta-label-mode payoff
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto import config
from crypto.data import (
    add_binary_labels,
    load_ohlcv,
    make_walk_forward_folds,
    split_labeled_by_dates,
)
from crypto.expression import CryptoFeatureSpace
from crypto.features import build_feature_frame, selectable_features
from crypto.fitness import _internal_early_stop_split


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crypto.meta_learner.train")


DEFAULT_OUTPUT_DIR = Path("crypto/meta_learner/output")
DEFAULT_META_MODEL_NAME = "meta_model.txt"
DEFAULT_META_DATASET_NAME = "meta_dataset.csv"
DEFAULT_HOLDOUT_PRED_NAME = "meta_holdout_predictions.csv"
DEFAULT_FINAL_VAL_PRED_NAME = "final_val_predictions.csv"
DEFAULT_FINAL_TEST_PRED_NAME = "final_test_predictions.csv"
DEFAULT_CHART_NAME = "meta_summary.png"

MINIMAL_CONTEXT_FEATURES = [
    "vol_regime_z_50",
    "range_pct_mean_14",
    "realized_vol_z_120",
    "buy_pressure_mean_10",
    "signed_volume_sum_ratio_50",
    "taker_buy_base_volume_ratio_14",
    "quote_volume_proxy_log_delta_40",
    "ret_close_20",
]

META_LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 7,
    "max_depth": 3,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "min_data_in_leaf": 80,
    "lambda_l1": 1.0,
    "lambda_l2": 10.0,
    "force_col_wise": True,
    "verbose": -1,
    "seed": 42,
    "feature_fraction_seed": 42,
    "bagging_seed": 42,
    "data_random_seed": 42,
}
META_NUM_BOOST_ROUND = 200
META_EARLY_STOPPING = 20

_WINDOW_ARG_RE = re.compile(r",\s*(\d+)(?=\))")
_WINDOW_SUFFIX_RE = re.compile(r"_(\d+)\b")


@dataclass(frozen=True)
class ArchiveSelection:
    archive_path: Path
    rank: int
    score: float | None
    generation: int
    features: list[str]


@dataclass(frozen=True)
class MetaArtifacts:
    output_dir: Path
    manifest_path: Path
    model_path: Path
    dataset_path: Path
    holdout_predictions_path: Path | None
    final_val_predictions_path: Path | None
    final_test_predictions_path: Path | None
    chart_path: Path | None


def train_meta_learner(
    archive_path: str | Path,
    rank: int,
    data_path: str | Path = config.DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_name: str | None = None,
    horizons: list[int] | tuple[int, ...] = tuple(config.HOLDING_HORIZONS),
    base_label_mode: str = config.LABEL_MODE,
    base_label_threshold: float | None = None,
    meta_label_mode: str = "payoff",
    meta_label_threshold: float | None = None,
    meta_exit_horizon: int | None = None,
    signals_only: bool = True,
    meta_valid_fraction: float = 0.2,
    val_start: str = config.VAL_START,
    test_start: str = config.TEST_START,
    test_end: str | None = config.TEST_END,
    wf_end: str = config.WF_END,
    wf_min_train_months: int = config.WF_MIN_TRAIN_MONTHS,
    wf_val_months: int = config.WF_VAL_MONTHS,
    wf_step_months: int = config.WF_STEP_MONTHS,
) -> MetaArtifacts:
    config.validate_config()
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    horizons = [int(h) for h in horizons]
    if not horizons:
        raise ValueError("At least one horizon is required.")
    meta_exit_horizon = int(meta_exit_horizon or max(horizons))

    base_label_mode = str(base_label_mode).strip().lower()
    meta_label_mode = str(meta_label_mode).strip().lower()
    base_label_threshold = config.default_label_threshold(
        base_label_mode,
        base_label_threshold,
    )
    meta_label_threshold = config.default_label_threshold(
        meta_label_mode,
        meta_label_threshold,
    )

    selection = load_archive_selection(archive_path, rank)
    run_name = run_name or (
        f"{archive_path.stem}_r{int(rank):02d}_meta_"
        f"{meta_label_mode}_h{meta_exit_horizon}"
    )
    run_dir = output_dir / _safe_name(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading crypto data from %s", data_path)
    raw_df = load_ohlcv(data_path)
    labeled_df = add_binary_labels(
        raw_df,
        horizons=horizons,
        threshold=base_label_threshold,
        label_mode=base_label_mode,
    )
    meta_return = config.get_label_return_fn(meta_label_mode)(raw_df, meta_exit_horizon)
    labeled_df[f"meta_future_return_h{meta_exit_horizon}"] = meta_return
    labeled_df[f"meta_label_h{meta_exit_horizon}"] = (
        meta_return > float(meta_label_threshold)
    ).astype(float)
    labeled_df.loc[
        meta_return.isna(),
        f"meta_label_h{meta_exit_horizon}",
    ] = np.nan

    purge_bars = config.purge_bars_for_horizons(
        list(dict.fromkeys([*horizons, meta_exit_horizon]))
    )
    folds = make_walk_forward_folds(
        labeled_df,
        wf_end=wf_end,
        min_train_months=wf_min_train_months,
        val_months=wf_val_months,
        step_months=wf_step_months,
        purge_bars=purge_bars,
    )
    for fold in folds:
        logger.info(
            "WF %s: train=%d val=%d [%s -> %s)",
            fold.name,
            len(fold.train_df),
            len(fold.val_df),
            fold.val_start,
            fold.val_end,
        )

    windows = _required_windows(selection.features + MINIMAL_CONTEXT_FEATURES)
    logger.info("Building feature matrix with required windows=%s", windows)
    feature_df = build_feature_frame(
        raw_df,
        windows=windows,
        quality_filter=False,
    )
    feature_pool = selectable_features(feature_df)
    feature_space = CryptoFeatureSpace(feature_df, feature_pool)

    meta_dataset = build_meta_dataset(
        selection=selection,
        folds=folds,
        feature_space=feature_space,
        feature_df=feature_df,
        horizons=horizons,
        meta_exit_horizon=meta_exit_horizon,
        signals_only=signals_only,
    )
    if meta_dataset.empty:
        raise ValueError("Meta dataset is empty. Try --all-predictions or a larger top fraction.")

    dataset_path = run_dir / DEFAULT_META_DATASET_NAME
    meta_dataset.to_csv(dataset_path, index=True)
    logger.info("Saved meta dataset: %s | rows=%d", dataset_path, len(meta_dataset))

    meta_feature_cols = _meta_feature_columns(horizons, meta_dataset)
    meta_result = train_meta_model(
        meta_dataset=meta_dataset,
        feature_cols=meta_feature_cols,
        valid_fraction=meta_valid_fraction,
    )
    model_path = run_dir / DEFAULT_META_MODEL_NAME
    meta_result["booster"].save_model(str(model_path))

    holdout_path: Path | None = None
    if meta_result["holdout_predictions"] is not None:
        holdout_path = run_dir / DEFAULT_HOLDOUT_PRED_NAME
        meta_result["holdout_predictions"].to_csv(holdout_path, index=True)

    train_df, val_df, test_df = split_labeled_by_dates(
        labeled_df,
        val_start=val_start,
        test_start=test_start,
        test_end=test_end,
        purge_bars=purge_bars,
    )
    final_eval = evaluate_final_meta(
        selection=selection,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_space=feature_space,
        feature_df=feature_df,
        horizons=horizons,
        meta_exit_horizon=meta_exit_horizon,
        meta_booster=meta_result["booster"],
        meta_feature_cols=meta_feature_cols,
        signals_only=signals_only,
    )
    final_val_path: Path | None = None
    final_test_path: Path | None = None
    if final_eval["val_predictions"] is not None:
        final_val_path = run_dir / DEFAULT_FINAL_VAL_PRED_NAME
        final_eval["val_predictions"].to_csv(final_val_path, index=True)
    if final_eval["test_predictions"] is not None:
        final_test_path = run_dir / DEFAULT_FINAL_TEST_PRED_NAME
        final_eval["test_predictions"].to_csv(final_test_path, index=True)
    chart_path = run_dir / DEFAULT_CHART_NAME
    plot_meta_summary(
        output_path=chart_path,
        oof_dataset=meta_dataset,
        holdout_predictions=meta_result["holdout_predictions"],
        final_val_predictions=final_eval["val_predictions"],
        final_test_predictions=final_eval["test_predictions"],
        metrics={
            "oof": meta_result["metrics"],
            "final": final_eval["metrics"],
        },
    )

    manifest = {
        "pipeline": "crypto_meta_learner",
        "archive": str(archive_path),
        "rank": int(rank),
        "score": selection.score,
        "generation": selection.generation,
        "data": str(data_path),
        "run_name": run_name,
        "output_dir": str(run_dir),
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "holdout_predictions_path": str(holdout_path) if holdout_path else None,
        "final_val_predictions_path": str(final_val_path) if final_val_path else None,
        "final_test_predictions_path": str(final_test_path) if final_test_path else None,
        "chart_path": str(chart_path),
        "base": {
            "label_mode": base_label_mode,
            "label_threshold": float(base_label_threshold),
            "horizons": horizons,
            "trade_top_fraction": float(config.TRADE_TOP_FRACTION),
            "min_trades_per_split": int(config.MIN_TRADES_PER_SPLIT),
        },
        "meta": {
            "label_mode": meta_label_mode,
            "label_threshold": float(meta_label_threshold),
            "exit_horizon": int(meta_exit_horizon),
            "signals_only": bool(signals_only),
            "feature_cols": meta_feature_cols,
            "context_features": [
                name for name in MINIMAL_CONTEXT_FEATURES if name in meta_dataset.columns
            ],
            "missing_context_features": [
                name for name in MINIMAL_CONTEXT_FEATURES if name not in meta_dataset.columns
            ],
            "valid_fraction": float(meta_valid_fraction),
            "final_val_start": val_start,
            "final_test_start": test_start,
            "final_test_end": test_end,
        },
        "folds": [
            {
                "name": fold.name,
                "train_rows": int(len(fold.train_df)),
                "val_rows": int(len(fold.val_df)),
                "val_start": _ts_str(fold.val_start),
                "val_end": _ts_str(fold.val_end),
            }
            for fold in folds
        ],
        "features": selection.features,
        "metrics": {
            "oof": meta_result["metrics"],
            "final": final_eval["metrics"],
        },
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved meta model: %s", model_path)
    logger.info("Saved manifest: %s", manifest_path)
    logger.info("Meta metrics: %s", manifest["metrics"])
    return MetaArtifacts(
        output_dir=run_dir,
        manifest_path=manifest_path,
        model_path=model_path,
        dataset_path=dataset_path,
        holdout_predictions_path=holdout_path,
        final_val_predictions_path=final_val_path,
        final_test_predictions_path=final_test_path,
        chart_path=chart_path,
    )


def build_meta_dataset(
    selection: ArchiveSelection,
    folds: list[Any],
    feature_space: CryptoFeatureSpace,
    feature_df: pd.DataFrame,
    horizons: list[int],
    meta_exit_horizon: int,
    signals_only: bool = True,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold in folds:
        horizon_frames: list[pd.DataFrame] = []
        for horizon in horizons:
            horizon_frame = _oof_predictions_for_horizon(
                selection=selection,
                fold=fold,
                feature_space=feature_space,
                horizon=int(horizon),
            )
            if horizon_frame is not None and not horizon_frame.empty:
                horizon_frames.append(horizon_frame)

        if len(horizon_frames) != len(horizons):
            logger.warning(
                "Skip fold %s because not all horizons produced predictions.",
                fold.name,
            )
            continue

        frame = _assemble_prediction_frame(
            frames=horizon_frames,
            label_source_df=fold.val_df,
            feature_df=feature_df,
            horizons=horizons,
            meta_exit_horizon=meta_exit_horizon,
            fold_name=fold.name,
            selection=selection,
            signals_only=signals_only,
        )
        if not frame.empty:
            rows.append(frame)

    if not rows:
        return pd.DataFrame()
    dataset = pd.concat(rows, axis=0).sort_index()
    dataset.index.name = "date"
    return dataset


def _assemble_prediction_frame(
    frames: list[pd.DataFrame],
    label_source_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    horizons: list[int],
    meta_exit_horizon: int,
    fold_name: str,
    selection: ArchiveSelection,
    signals_only: bool,
) -> pd.DataFrame:
    frame = pd.concat(frames, axis=1, join="inner").sort_index()
    if frame.empty:
        return frame

    pred_cols = [f"pred_h{int(h)}" for h in horizons]
    margin_cols = [f"margin_h{int(h)}" for h in horizons]
    signal_cols = [f"signal_h{int(h)}" for h in horizons]
    frame["pred_mean"] = frame[pred_cols].mean(axis=1)
    frame["pred_std"] = frame[pred_cols].std(axis=1, ddof=0).fillna(0.0)
    frame["margin_min"] = frame[margin_cols].min(axis=1)
    frame["margin_mean"] = frame[margin_cols].mean(axis=1)
    frame["signal_count"] = frame[signal_cols].sum(axis=1)
    frame["all_horizon_signal"] = frame["signal_count"] == len(horizons)

    context = feature_df.reindex(frame.index)
    available_context = [name for name in MINIMAL_CONTEXT_FEATURES if name in context.columns]
    frame = frame.join(context[available_context])

    ret_col = f"meta_future_return_h{meta_exit_horizon}"
    label_col = f"meta_label_h{meta_exit_horizon}"
    label_data = label_source_df.reindex(frame.index)[[ret_col, label_col]]
    frame = frame.join(label_data)
    frame = frame.rename(
        columns={
            ret_col: "meta_future_return",
            label_col: "meta_label",
        }
    )
    frame["fold"] = fold_name
    frame["archive"] = str(selection.archive_path)
    frame["rank"] = int(selection.rank)

    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["meta_label", "meta_future_return"])
    if signals_only:
        frame = frame[frame["all_horizon_signal"]].copy()
    return frame


def evaluate_final_meta(
    selection: ArchiveSelection,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_space: CryptoFeatureSpace,
    feature_df: pd.DataFrame,
    horizons: list[int],
    meta_exit_horizon: int,
    meta_booster: lgb.Booster,
    meta_feature_cols: list[str],
    signals_only: bool = True,
) -> dict[str, Any]:
    val_predictions = _final_meta_predictions_for_split(
        selection=selection,
        train_df=train_df,
        eval_df=val_df,
        eval_name="val",
        feature_space=feature_space,
        feature_df=feature_df,
        horizons=horizons,
        meta_exit_horizon=meta_exit_horizon,
        meta_booster=meta_booster,
        meta_feature_cols=meta_feature_cols,
        signals_only=signals_only,
        meta_threshold=None,
        use_own_threshold=True,
    )
    val_threshold = _frame_meta_threshold(val_predictions)
    test_predictions = _final_meta_predictions_for_split(
        selection=selection,
        train_df=train_df,
        eval_df=test_df,
        eval_name="test",
        feature_space=feature_space,
        feature_df=feature_df,
        horizons=horizons,
        meta_exit_horizon=meta_exit_horizon,
        meta_booster=meta_booster,
        meta_feature_cols=meta_feature_cols,
        signals_only=signals_only,
        meta_threshold=val_threshold,
        use_own_threshold=False,
    )
    return {
        "val_predictions": val_predictions,
        "test_predictions": test_predictions,
        "metrics": {
            "val": _final_prediction_metrics(val_predictions),
            "test": _final_prediction_metrics(test_predictions),
        },
    }


def _final_meta_predictions_for_split(
    selection: ArchiveSelection,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    eval_name: str,
    feature_space: CryptoFeatureSpace,
    feature_df: pd.DataFrame,
    horizons: list[int],
    meta_exit_horizon: int,
    meta_booster: lgb.Booster,
    meta_feature_cols: list[str],
    signals_only: bool = True,
    meta_threshold: float | None = None,
    use_own_threshold: bool = True,
) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    for horizon in horizons:
        frame = _final_predictions_for_horizon(
            selection=selection,
            train_df=train_df,
            eval_df=eval_df,
            feature_space=feature_space,
            horizon=int(horizon),
            eval_name=eval_name,
        )
        if frame is not None and not frame.empty:
            frames.append(frame)

    if len(frames) != len(horizons):
        logger.warning(
            "Final %s skipped because not all horizons produced predictions.",
            eval_name,
        )
        return None

    data = _assemble_prediction_frame(
        frames=frames,
        label_source_df=eval_df,
        feature_df=feature_df,
        horizons=horizons,
        meta_exit_horizon=meta_exit_horizon,
        fold_name=f"final_{eval_name}",
        selection=selection,
        signals_only=signals_only,
    )
    if data.empty:
        return data
    valid = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=meta_feature_cols + ["meta_label", "meta_future_return"]
    )
    if valid.empty:
        return valid
    valid["meta_pred"] = pd.Series(
        meta_booster.predict(valid[meta_feature_cols]),
        index=valid.index,
    )
    threshold = meta_threshold
    if threshold is None and use_own_threshold:
        threshold = _top_prediction_threshold(valid["meta_pred"])
    valid["meta_threshold"] = float(threshold) if threshold is not None else np.nan
    valid["meta_signal"] = (
        valid["meta_pred"] >= float(threshold)
        if threshold is not None
        else False
    )
    valid["split"] = eval_name
    return valid


def _final_predictions_for_horizon(
    selection: ArchiveSelection,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_space: CryptoFeatureSpace,
    horizon: int,
    eval_name: str,
) -> pd.DataFrame | None:
    label_col = f"label_h{horizon}"
    ret_col = f"future_return_h{horizon}"
    train = _valid_frame(train_df, label_col, ret_col)
    eval_data = _valid_frame(eval_df, label_col, ret_col)
    if train.empty or eval_data.empty:
        return None

    X_train = feature_space.matrix(selection.features, train.index)
    y_train = train[label_col].astype(int)
    if y_train.nunique() < 2:
        logger.warning("Final %s h%d skipped: train label is constant.", eval_name, horizon)
        return None

    X_eval = feature_space.matrix(selection.features, eval_data.index)
    booster = _train_base_booster(X_train, y_train)
    pred = pd.Series(booster.predict(X_eval), index=eval_data.index)
    threshold = _top_prediction_threshold(pred)
    signal = pred >= threshold if threshold is not None else pd.Series(False, index=pred.index)
    return pd.DataFrame(
        {
            f"pred_h{horizon}": pred,
            f"threshold_h{horizon}": float(threshold) if threshold is not None else np.nan,
            f"margin_h{horizon}": pred - float(threshold) if threshold is not None else np.nan,
            f"signal_h{horizon}": signal.astype(float),
            f"base_label_h{horizon}": eval_data[label_col].astype(float),
            f"base_future_return_h{horizon}": eval_data[ret_col].astype(float),
        },
        index=eval_data.index,
    )


def _oof_predictions_for_horizon(
    selection: ArchiveSelection,
    fold: Any,
    feature_space: CryptoFeatureSpace,
    horizon: int,
) -> pd.DataFrame | None:
    label_col = f"label_h{horizon}"
    ret_col = f"future_return_h{horizon}"
    train = _valid_frame(fold.train_df, label_col, ret_col)
    val = _valid_frame(fold.val_df, label_col, ret_col)
    if train.empty or val.empty:
        return None

    X_train = feature_space.matrix(selection.features, train.index)
    X_val = feature_space.matrix(selection.features, val.index)
    y_train = train[label_col].astype(int)
    y_val = val[label_col].astype(int)
    if y_train.nunique() < 2:
        logger.warning("%s h%d skipped: train label is constant.", fold.name, horizon)
        return None

    booster = _train_base_booster(X_train, y_train)
    pred = pd.Series(booster.predict(X_val), index=val.index)
    threshold = _top_prediction_threshold(pred)
    signal = pred >= threshold if threshold is not None else pd.Series(False, index=pred.index)
    frame = pd.DataFrame(
        {
            f"pred_h{horizon}": pred,
            f"threshold_h{horizon}": float(threshold) if threshold is not None else np.nan,
            f"margin_h{horizon}": pred - float(threshold) if threshold is not None else np.nan,
            f"signal_h{horizon}": signal.astype(float),
            f"base_label_h{horizon}": y_val.astype(float),
            f"base_future_return_h{horizon}": val[ret_col].astype(float),
        },
        index=val.index,
    )
    return frame


def train_meta_model(
    meta_dataset: pd.DataFrame,
    feature_cols: list[str],
    valid_fraction: float = 0.2,
) -> dict[str, Any]:
    data = meta_dataset.replace([np.inf, -np.inf], np.nan).dropna(
        subset=feature_cols + ["meta_label", "meta_future_return"]
    )
    if data.empty:
        raise ValueError("No valid rows remain for meta model training.")
    data = data.sort_index()

    n_valid = int(np.ceil(len(data) * float(valid_fraction)))
    n_valid = max(n_valid, 0)
    n_valid = min(n_valid, len(data) // 2)
    if n_valid > 0:
        train = data.iloc[:-n_valid]
        valid = data.iloc[-n_valid:]
    else:
        train = data
        valid = pd.DataFrame(columns=data.columns)

    if train["meta_label"].nunique() < 2:
        raise ValueError("Meta train label is constant; cannot train binary model.")

    train_set = lgb.Dataset(
        train[feature_cols],
        label=train["meta_label"].astype(int),
        free_raw_data=False,
    )
    callbacks = [lgb.log_evaluation(period=-1)]
    valid_sets = None
    if len(valid) and valid["meta_label"].nunique() >= 2:
        valid_set = lgb.Dataset(
            valid[feature_cols],
            label=valid["meta_label"].astype(int),
            reference=train_set,
            free_raw_data=False,
        )
        valid_sets = [valid_set]
        callbacks.insert(0, lgb.early_stopping(META_EARLY_STOPPING, verbose=False))

    booster = lgb.train(
        params=dict(META_LGBM_PARAMS),
        train_set=train_set,
        num_boost_round=META_NUM_BOOST_ROUND,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )

    train_pred = pd.Series(booster.predict(train[feature_cols]), index=train.index)
    train_metrics = _classification_trade_metrics(
        y_true=train["meta_label"],
        pred=train_pred,
        future_return=train["meta_future_return"],
    )

    holdout_predictions = None
    valid_metrics = None
    if len(valid):
        valid_pred = pd.Series(booster.predict(valid[feature_cols]), index=valid.index)
        valid_metrics = _classification_trade_metrics(
            y_true=valid["meta_label"],
            pred=valid_pred,
            future_return=valid["meta_future_return"],
        )
        holdout_predictions = valid[
            ["meta_label", "meta_future_return", "fold", "all_horizon_signal"]
        ].copy()
        holdout_predictions["meta_pred"] = valid_pred
        threshold = valid_metrics["trade_threshold"]
        holdout_predictions["meta_signal"] = (
            holdout_predictions["meta_pred"] >= threshold
            if threshold is not None
            else False
        )

    before_meta = {
        "rows": int(len(data)),
        "base_rate": float(data["meta_label"].mean()),
        "net_return_mean": float(
            (data["meta_future_return"].astype(float) - float(config.TRADE_COST)).mean()
        ),
    }
    metrics = {
        "before_meta": before_meta,
        "train": train_metrics,
        "holdout": valid_metrics,
        "rows_total": int(len(data)),
        "rows_train": int(len(train)),
        "rows_holdout": int(len(valid)),
        "best_iteration": int(booster.best_iteration or META_NUM_BOOST_ROUND),
    }
    return {
        "booster": booster,
        "metrics": _json_safe(metrics),
        "holdout_predictions": holdout_predictions,
    }


def _final_prediction_metrics(frame: pd.DataFrame | None) -> dict[str, Any] | None:
    if frame is None or frame.empty:
        return None
    data = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["meta_label", "meta_future_return", "meta_pred"]
    )
    if data.empty:
        return None
    before = {
        "rows": int(len(data)),
        "base_rate": float(data["meta_label"].mean()),
        "net_return_mean": float(
            (data["meta_future_return"].astype(float) - float(config.TRADE_COST)).mean()
        ),
    }
    y = data["meta_label"].astype(int)
    auc = _binary_auc(y, data["meta_pred"])
    threshold = _frame_meta_threshold(data)
    meta_signal = (
        data["meta_signal"].astype(bool)
        if "meta_signal" in data.columns
        else pd.Series(False, index=data.index)
    )
    selected = data[meta_signal]
    selected_precision = float(selected["meta_label"].mean()) if len(selected) else 0.0
    selected_net_return = (
        float((selected["meta_future_return"].astype(float) - float(config.TRADE_COST)).mean())
        if len(selected)
        else 0.0
    )
    after = {
        "auc": float(auc),
        "base_rate": float(y.mean()),
        "trade_threshold": float(threshold) if threshold is not None else None,
        "n_samples": int(len(data)),
        "n_trades": int(len(selected)),
        "precision_at_trade": selected_precision,
        "precision_excess": selected_precision - float(y.mean()),
        "net_return_mean": selected_net_return,
        "selected_rows": int(len(selected)),
        "selected_rate": float(len(selected) / len(data)) if len(data) else 0.0,
        "selected_net_return_mean": selected_net_return,
        "selected_precision": selected_precision,
    }
    return {
        "before_meta": before,
        "after_meta": after,
    }


def _frame_meta_threshold(frame: pd.DataFrame | None) -> float | None:
    if frame is None or frame.empty or "meta_threshold" not in frame.columns:
        return None
    values = (
        pd.to_numeric(frame["meta_threshold"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if values.empty:
        return None
    return float(values.iloc[0])


def plot_meta_summary(
    output_path: str | Path,
    oof_dataset: pd.DataFrame,
    holdout_predictions: pd.DataFrame | None,
    final_val_predictions: pd.DataFrame | None,
    final_test_predictions: pd.DataFrame | None,
    metrics: dict[str, Any],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections = [
        ("OOF train/all", oof_dataset, None),
        ("OOF holdout", holdout_predictions, "meta_pred"),
        ("Final val", final_val_predictions, "meta_pred"),
        ("Final test", final_test_predictions, "meta_pred"),
    ]
    rows = []
    for name, frame, pred_col in sections:
        rows.append(_summary_row(name, frame, pred_col))
    table = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Crypto Meta Learner Summary", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.axis("off")
    display_cols = [
        "section",
        "rows",
        "label_rate",
        "net_before",
        "selected_rows",
        "selected_label_rate",
        "net_after",
    ]
    cell_text = []
    for _, row in table.iterrows():
        cell_text.append(
            [
                row["section"],
                str(int(row["rows"])),
                _fmt_pct(row["label_rate"]),
                _fmt_pct(row["net_before"]),
                "" if pd.isna(row["selected_rows"]) else str(int(row["selected_rows"])),
                "" if pd.isna(row["selected_label_rate"]) else _fmt_pct(row["selected_label_rate"]),
                "" if pd.isna(row["net_after"]) else _fmt_pct(row["net_after"]),
            ]
        )
    tbl = ax.table(
        cellText=cell_text,
        colLabels=display_cols,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.05, 1.35)

    ax = axes[0, 1]
    plot_data = table.dropna(subset=["net_before"]).copy()
    x = np.arange(len(plot_data))
    ax.bar(x - 0.18, plot_data["net_before"] * 100.0, width=0.36, label="before meta")
    after_vals = plot_data["net_after"].fillna(0.0) * 100.0
    ax.bar(x + 0.18, after_vals, width=0.36, label="after meta")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_data["section"], rotation=20, ha="right")
    ax.set_ylabel("Mean net return (%)")
    ax.set_title("Net Return Before/After Meta")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(table["section"], table["label_rate"] * 100.0, marker="o", label="before/base")
    if table["selected_label_rate"].notna().any():
        ax.plot(
            table["section"],
            table["selected_label_rate"] * 100.0,
            marker="o",
            label="after meta",
        )
    ax.set_ylabel("Positive label rate (%)")
    ax.set_title("Meta Label Rate")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()

    ax = axes[1, 1]
    ax.bar(table["section"], table["rows"], label="base signal rows")
    if table["selected_rows"].notna().any():
        ax.bar(table["section"], table["selected_rows"].fillna(0), label="meta selected")
    ax.set_ylabel("Rows")
    ax.set_title("Trade Count")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _summary_row(
    name: str,
    frame: pd.DataFrame | None,
    pred_col: str | None,
) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {
            "section": name,
            "rows": 0,
            "label_rate": np.nan,
            "net_before": np.nan,
            "selected_rows": np.nan,
            "selected_label_rate": np.nan,
            "net_after": np.nan,
        }
    data = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["meta_label", "meta_future_return"]
    )
    if data.empty:
        return {
            "section": name,
            "rows": 0,
            "label_rate": np.nan,
            "net_before": np.nan,
            "selected_rows": np.nan,
            "selected_label_rate": np.nan,
            "net_after": np.nan,
        }
    net_before = data["meta_future_return"].astype(float) - float(config.TRADE_COST)
    selected = pd.DataFrame()
    if pred_col and pred_col in data.columns:
        if "meta_signal" in data.columns:
            selected = data[data["meta_signal"].astype(bool)]
        else:
            threshold = _top_prediction_threshold(data[pred_col])
            if threshold is not None:
                selected = data[data[pred_col] >= float(threshold)]
    net_after = (
        selected["meta_future_return"].astype(float) - float(config.TRADE_COST)
        if len(selected)
        else pd.Series(dtype=float)
    )
    return {
        "section": name,
        "rows": int(len(data)),
        "label_rate": float(data["meta_label"].mean()),
        "net_before": float(net_before.mean()),
        "selected_rows": int(len(selected)) if pred_col else np.nan,
        "selected_label_rate": float(selected["meta_label"].mean()) if len(selected) else np.nan,
        "net_after": float(net_after.mean()) if len(net_after) else np.nan,
    }


def _fmt_pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number * 100.0:.2f}%"


def _train_base_booster(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> lgb.Booster:
    split = _internal_early_stop_split(X_train, y_train)
    callbacks = [lgb.log_evaluation(period=-1)]
    valid_sets = None
    if split is None or config.LGBM_EARLY_STOPPING <= 0:
        train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    else:
        X_fit, y_fit, X_stop, y_stop = split
        train_set = lgb.Dataset(X_fit, label=y_fit, free_raw_data=False)
        stop_set = lgb.Dataset(
            X_stop,
            label=y_stop,
            reference=train_set,
            free_raw_data=False,
        )
        valid_sets = [stop_set]
        callbacks.insert(0, lgb.early_stopping(config.LGBM_EARLY_STOPPING, verbose=False))

    return lgb.train(
        params=dict(config.LGBM_PARAMS),
        train_set=train_set,
        num_boost_round=int(config.LGBM_NUM_BOOST_ROUND),
        valid_sets=valid_sets,
        callbacks=callbacks,
    )


def _classification_trade_metrics(
    y_true: pd.Series,
    pred: pd.Series,
    future_return: pd.Series,
) -> dict[str, Any]:
    data = (
        pd.DataFrame({"y": y_true, "pred": pred, "ret": future_return})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if data.empty:
        return {
            "auc": 0.5,
            "base_rate": 0.0,
            "precision_at_trade": 0.0,
            "precision_excess": 0.0,
            "trade_threshold": None,
            "n_samples": 0,
            "n_trades": 0,
            "net_return_mean": 0.0,
        }
    y = data["y"].astype(int)
    auc = _binary_auc(y, data["pred"])
    threshold = _top_prediction_threshold(data["pred"])
    traded = data[data["pred"] >= float(threshold)] if threshold is not None else data.iloc[0:0]
    precision = float(traded["y"].mean()) if len(traded) else 0.0
    net_return = traded["ret"].astype(float) - float(config.TRADE_COST)
    return {
        "auc": float(auc),
        "base_rate": float(y.mean()),
        "precision_at_trade": precision,
        "precision_excess": precision - float(y.mean()),
        "trade_threshold": float(threshold) if threshold is not None else None,
        "n_samples": int(len(data)),
        "n_trades": int(len(traded)),
        "net_return_mean": float(net_return.mean()) if len(net_return) else 0.0,
    }


def _top_prediction_threshold(pred: pd.Series) -> float | None:
    pred = pd.to_numeric(pred, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if pred.empty:
        return None
    n_select = min(
        len(pred),
        max(
            int(config.MIN_TRADES_PER_SPLIT),
            int(np.ceil(len(pred) * float(config.TRADE_TOP_FRACTION))),
        ),
    )
    return float(pred.nlargest(n_select).min())


def _binary_auc(y_true: pd.Series, pred: pd.Series) -> float:
    y = pd.Series(y_true).astype(int)
    scores = pd.Series(pred, index=y.index).astype(float)
    data = pd.DataFrame({"y": y, "score": scores}).dropna()
    n_pos = int((data["y"] == 1).sum())
    n_neg = int((data["y"] == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = data["score"].rank(method="average")
    pos_rank_sum = float(ranks[data["y"] == 1].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def load_archive_selection(path: str | Path, rank: int) -> ArchiveSelection:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", payload if isinstance(payload, list) else [])
    if not isinstance(entries, list):
        raise ValueError(f"Archive has no entries list: {path}")
    wanted = int(rank)
    matches = [entry for entry in entries if int(entry.get("rank", -1)) == wanted]
    if not matches:
        raise ValueError(f"Archive {path} does not contain rank {wanted}.")
    entry = dict(matches[0])
    features = _clean_features(entry)
    score = entry.get("score")
    return ArchiveSelection(
        archive_path=path,
        rank=wanted,
        score=float(score) if score is not None else None,
        generation=int(entry.get("generation", 0) or 0),
        features=features,
    )


def _clean_features(entry: dict[str, Any]) -> list[str]:
    raw_features = entry.get("features")
    if not isinstance(raw_features, list) or not raw_features:
        raise ValueError(f"Archive rank {entry.get('rank')} has no features list.")
    features: list[str] = []
    for feature in raw_features:
        feature_text = str(feature).strip()
        if feature_text and feature_text not in features:
            features.append(feature_text)
    if not features:
        raise ValueError(f"Archive rank {entry.get('rank')} has no usable features.")
    return features


def _valid_frame(df: pd.DataFrame, label_col: str, ret_col: str) -> pd.DataFrame:
    if label_col not in df.columns or ret_col not in df.columns:
        raise ValueError(f"Missing label/return columns: {label_col}, {ret_col}")
    return df.dropna(subset=[label_col, ret_col]).copy()


def _meta_feature_columns(horizons: list[int], dataset: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for horizon in horizons:
        h = int(horizon)
        columns.extend(
            [
                f"pred_h{h}",
                f"threshold_h{h}",
                f"margin_h{h}",
            ]
        )
    columns.extend(
        [
            "pred_mean",
            "pred_std",
            "margin_min",
            "margin_mean",
            "signal_count",
        ]
    )
    columns.extend([name for name in MINIMAL_CONTEXT_FEATURES if name in dataset.columns])
    return [name for name in columns if name in dataset.columns]


def _required_windows(features: list[str]) -> list[int]:
    windows: set[int] = set()
    for feature in features:
        for match in _WINDOW_SUFFIX_RE.finditer(str(feature)):
            windows.add(int(match.group(1)))
        for match in _WINDOW_ARG_RE.finditer(str(feature)):
            windows.add(int(match.group(1)))
    return sorted(window for window in windows if window > 1)


def _parse_horizons(text: str) -> list[int]:
    horizons = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not horizons:
        raise argparse.ArgumentTypeError("horizons must not be empty.")
    if any(h < 1 for h in horizons):
        raise argparse.ArgumentTypeError("all horizons must be positive.")
    return horizons


def _safe_name(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return safe.strip("._") or "meta_learner"


def _ts_str(value: Any) -> str:
    if value is None:
        return ""
    return str(pd.Timestamp(value))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Crypto archive JSON path.")
    parser.add_argument("--rank", type=int, required=True, help="Archive rank to use.")
    parser.add_argument("--data", default=str(config.DATA_PATH), help="Crypto OHLCV CSV path.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--horizons",
        type=_parse_horizons,
        default=list(config.HOLDING_HORIZONS),
        help="Comma-separated base model horizons. Default comes from config.",
    )
    parser.add_argument(
        "--base-label-mode",
        choices=sorted(config.LABEL_RETURN_FNS),
        default=config.LABEL_MODE,
        help="Label mode used to retrain the archive individual per fold.",
    )
    parser.add_argument(
        "--base-label-threshold",
        type=float,
        default=None,
        help="Base label threshold. Default follows config.default_label_threshold.",
    )
    parser.add_argument(
        "--meta-label-mode",
        choices=sorted(config.LABEL_RETURN_FNS),
        default="payoff",
        help="Label mode used for the meta learner target. Default: payoff.",
    )
    parser.add_argument(
        "--meta-label-threshold",
        type=float,
        default=None,
        help="Meta label threshold. Default follows config.default_label_threshold.",
    )
    parser.add_argument(
        "--meta-exit-horizon",
        type=int,
        default=None,
        help="Horizon used for meta target. Default: max(--horizons).",
    )
    parser.add_argument("--wf-end", default=config.WF_END)
    parser.add_argument("--wf-min-train-months", type=int, default=config.WF_MIN_TRAIN_MONTHS)
    parser.add_argument("--wf-val-months", type=int, default=config.WF_VAL_MONTHS)
    parser.add_argument("--wf-step-months", type=int, default=config.WF_STEP_MONTHS)
    parser.add_argument(
        "--val-start",
        default=config.VAL_START,
        help="Final validation start date. Default comes from crypto.config.",
    )
    parser.add_argument(
        "--test-start",
        default=config.TEST_START,
        help="Final test start date. Default comes from crypto.config.",
    )
    parser.add_argument(
        "--test-end",
        default=config.TEST_END,
        help="Final test end date. Default comes from crypto.config.",
    )
    parser.add_argument("--meta-valid-fraction", type=float, default=0.2)
    parser.add_argument(
        "--all-predictions",
        action="store_true",
        help="Train meta learner on all OOF predictions instead of signal rows only.",
    )
    args = parser.parse_args()

    artifacts = train_meta_learner(
        archive_path=args.archive,
        rank=args.rank,
        data_path=args.data,
        output_dir=args.out_dir,
        run_name=args.run_name,
        horizons=args.horizons,
        base_label_mode=args.base_label_mode,
        base_label_threshold=args.base_label_threshold,
        meta_label_mode=args.meta_label_mode,
        meta_label_threshold=args.meta_label_threshold,
        meta_exit_horizon=args.meta_exit_horizon,
        signals_only=not args.all_predictions,
        meta_valid_fraction=args.meta_valid_fraction,
        val_start=args.val_start,
        test_start=args.test_start,
        test_end=args.test_end,
        wf_end=args.wf_end,
        wf_min_train_months=args.wf_min_train_months,
        wf_val_months=args.wf_val_months,
        wf_step_months=args.wf_step_months,
    )
    logger.info("Done. Manifest: %s", artifacts.manifest_path)


if __name__ == "__main__":
    main()
