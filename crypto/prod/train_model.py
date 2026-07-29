"""Train production crypto models from selected archive individuals.

Example:
    python -m crypto.prod.train_model --archive crypto/results/crypto_btc_seed1_12h.json --rank 1
    python -m crypto.prod.train_model --archive crypto/results/crypto_btc_seed1_12h.json --top 3
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from crypto import config
from crypto.analyze import _required_windows_for_entries
from crypto.data import add_binary_labels, load_ohlcv, split_labeled_by_dates
from crypto.expression import CryptoFeatureSpace
from crypto.features import build_feature_frame, selectable_features


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crypto.prod.train_model")


DEFAULT_MODEL_DIR = Path("crypto/prod/model")
SCORE_BAND_FRACTIONS: tuple[float, ...] = tuple(
    step / 100.0 for step in range(5, 101, 5)
)


@dataclass(frozen=True)
class EnsembleIndividualSpec:
    archive_path: Path
    rank: int
    label_mode: str | None = None
    label_threshold: float | None = None
    label_direction: str | None = None
    exit_after_k: int | None = None


def train_from_archive(
    archive_path: str | Path,
    data_path: str | Path = config.DATA_PATH,
    output_dir: str | Path = DEFAULT_MODEL_DIR,
    top: int | None = None,
    ranks: list[int] | None = None,
    run_name: str | None = None,
    val_start: str = config.VAL_START,
    test_start: str = config.TEST_START,
    test_end: str | None = config.TEST_END,
    label_mode: str = config.LABEL_MODE,
    label_direction: str | None = None,
    label_threshold: float | None = None,
    exit_after_k: int | None = None,
    trade_top_fraction: float = config.TRADE_TOP_FRACTION,
) -> Path:
    """Train one LightGBM model per selected individual and horizon."""
    config.validate_config()
    trade_top_fraction = _validate_trade_top_fraction(trade_top_fraction)
    archive_path = Path(archive_path)
    horizons = _archive_horizons(archive_path)
    selected_entries = _filter_entries(
        _load_archive_entries(archive_path), top=top, ranks=ranks
    )

    run_name = run_name or archive_path.stem
    model_dir = Path(output_dir) / _safe_name(run_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading crypto data from %s", data_path)
    raw_df = load_ohlcv(data_path)
    label_mode = config.canonical_label_mode(label_mode)
    exit_after_k = _resolve_archive_exit_after_k(
        archive_path,
        label_mode,
        explicit_k=exit_after_k,
    )
    label_direction = _resolve_archive_label_direction(
        archive_path,
        explicit_direction=label_direction,
    )
    label_threshold = config.default_label_threshold(label_mode, label_threshold)
    labeled_df = add_binary_labels(
        raw_df,
        horizons=horizons,
        threshold=label_threshold,
        return_fn=config.get_label_return_fn(label_mode),
        label_mode=label_mode,
        label_direction=label_direction,
        exit_after_k=exit_after_k,
    )
    purge_bars = config.purge_bars_for_horizons(horizons)
    train_df, val_df, test_df = split_labeled_by_dates(
        labeled_df,
        val_start=val_start,
        test_start=test_start,
        test_end=test_end,
        purge_bars=purge_bars,
    )
    logger.info(
        "Final split: train=%d | val=%d | test=%d | purge=%d bars",
        len(train_df),
        len(val_df),
        len(test_df),
        purge_bars,
    )

    required_windows = _required_windows_for_entries(selected_entries)
    logger.info(
        "Building crypto feature matrix; quality filter uses final train rows "
        "| windows=%s",
        required_windows,
    )
    feature_df = build_feature_frame(
        raw_df,
        windows=required_windows,
        quality_index=train_df.index,
    )
    feature_pool = selectable_features(feature_df)
    feature_space = CryptoFeatureSpace(feature_df, feature_pool)

    manifest: dict[str, Any] = {
        "pipeline": "crypto_prod_train",
        "archive": str(archive_path),
        "data": str(data_path),
        "run_name": run_name,
        "model_dir": str(model_dir),
        "config": _config_snapshot(
            val_start=val_start,
            test_start=test_start,
            test_end=test_end,
            purge_bars=purge_bars,
            label_mode=label_mode,
            label_direction=label_direction,
            label_threshold=float(label_threshold),
            exit_after_k=exit_after_k,
            horizons=horizons,
            trade_top_fraction=trade_top_fraction,
            feature_windows=required_windows,
        ),
        "entries": [],
    }

    for entry in selected_entries:
        rank = int(entry.get("rank", 0) or 0)
        features = _clean_features(entry)
        logger.info(
            "Training rank %02d | score=%s | features=%d",
            rank,
            entry.get("score"),
            len(features),
        )

        entry_record: dict[str, Any] = {
            "rank": rank,
            "entry_id": _entry_id(
                archive_path=archive_path,
                rank=rank,
                label_mode=label_mode,
                label_direction=label_direction,
                label_threshold=float(label_threshold),
            ),
            "archive": str(archive_path),
            "label_mode": label_mode,
            "label_direction": label_direction,
            "label_threshold": float(label_threshold),
            "exit_after_k": exit_after_k,
            "score": _json_safe(entry.get("score")),
            "generation": int(entry.get("generation", 0) or 0),
            "features": features,
            "models": [],
        }
        for horizon in horizons:
            horizon = int(horizon)
            model_record = _train_one_horizon(
                rank=rank,
                horizon=horizon,
                features=features,
                train_df=train_df,
                val_df=val_df,
                feature_space=feature_space,
                model_dir=model_dir,
                trade_top_fraction=trade_top_fraction,
            )
            entry_record["models"].append(model_record)
        manifest["entries"].append(entry_record)

    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved manifest: %s", manifest_path)
    return manifest_path


def train_ensemble_from_specs(
    specs: list[EnsembleIndividualSpec],
    data_path: str | Path = config.DATA_PATH,
    output_dir: str | Path = DEFAULT_MODEL_DIR,
    run_name: str | None = None,
    val_start: str = config.VAL_START,
    test_start: str = config.TEST_START,
    test_end: str | None = config.TEST_END,
    default_label_mode: str = config.LABEL_MODE,
    default_label_direction: str | None = None,
    default_label_threshold: float | None = None,
    default_exit_after_k: int | None = None,
    trade_top_fraction: float = config.TRADE_TOP_FRACTION,
) -> Path:
    """Train one production model bundle from archive/rank specs."""
    if len(specs) < 1:
        raise ValueError("Need at least one ensemble individual spec.")
    config.validate_config()
    trade_top_fraction = _validate_trade_top_fraction(trade_top_fraction)
    run_name = run_name or "crypto_ensemble"
    default_label_mode = config.canonical_label_mode(default_label_mode)
    default_exit_after_k = config.resolve_exit_after_k(
        default_label_mode,
        default_exit_after_k,
    )
    member_directions = [
        _resolve_archive_label_direction(
            spec.archive_path,
            explicit_direction=spec.label_direction or default_label_direction,
        )
        for spec in specs
    ]
    if len(set(member_directions)) != 1:
        raise ValueError(
            "Production all-members ensemble requires one direction, got "
            f"{member_directions}. Train Long and Short bundles separately."
        )
    default_label_direction = member_directions[0]
    default_label_threshold = config.default_label_threshold(
        default_label_mode,
        default_label_threshold,
    )
    model_dir = Path(output_dir) / _safe_name(run_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading crypto data from %s", data_path)
    raw_df = load_ohlcv(data_path)
    purge_bars = config.purge_bars_for_horizons(config.HOLDING_HORIZONS)

    logger.info(
        "Building crypto feature matrix; quality filter uses default final train rows."
    )
    default_labeled_df = add_binary_labels(
        raw_df,
        horizons=config.HOLDING_HORIZONS,
        threshold=default_label_threshold,
        return_fn=config.get_label_return_fn(default_label_mode),
        label_mode=default_label_mode,
        label_direction=default_label_direction,
        exit_after_k=default_exit_after_k,
    )
    default_train_df, _, _ = split_labeled_by_dates(
        default_labeled_df,
        val_start=val_start,
        test_start=test_start,
        test_end=test_end,
        purge_bars=purge_bars,
    )
    feature_df = build_feature_frame(raw_df, quality_index=default_train_df.index)
    feature_pool = selectable_features(feature_df)
    feature_space = CryptoFeatureSpace(feature_df, feature_pool)

    manifest: dict[str, Any] = {
        "pipeline": "crypto_prod_train",
        "manifest_version": 2,
        "bundle_type": "individual_ensemble",
        "data": str(data_path),
        "run_name": run_name,
        "model_dir": str(model_dir),
        "config": _config_snapshot(
            val_start=val_start,
            test_start=test_start,
            test_end=test_end,
            purge_bars=purge_bars,
            label_mode=default_label_mode,
            label_direction=default_label_direction,
            label_threshold=default_label_threshold,
            exit_after_k=default_exit_after_k,
            trade_top_fraction=trade_top_fraction,
        ),
        "entries": [],
        "ensemble": {
            "type": "all_entries",
            "members": [],
            "exit_horizon": int(max(config.HOLDING_HORIZONS)),
        },
    }

    label_cache: dict[
        tuple[str, str, float, int | None],
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ] = {}
    for spec, label_direction in zip(specs, member_directions, strict=True):
        label_mode = config.canonical_label_mode(spec.label_mode or default_label_mode)
        exit_after_k = _resolve_archive_exit_after_k(
            spec.archive_path,
            label_mode,
            explicit_k=spec.exit_after_k,
        )
        label_threshold = _resolve_spec_label_threshold(spec, default_label_threshold)
        cache_key = (label_mode, label_direction, label_threshold, exit_after_k)
        if cache_key not in label_cache:
            labeled_df = add_binary_labels(
                raw_df,
                horizons=config.HOLDING_HORIZONS,
                threshold=label_threshold,
                return_fn=config.get_label_return_fn(label_mode),
                label_mode=label_mode,
                label_direction=label_direction,
                exit_after_k=exit_after_k,
            )
            label_cache[cache_key] = split_labeled_by_dates(
                labeled_df,
                val_start=val_start,
                test_start=test_start,
                test_end=test_end,
                purge_bars=purge_bars,
            )
        train_df, val_df, _test_df = label_cache[cache_key]

        entry = _load_rank_entry(spec.archive_path, spec.rank)
        features = _clean_features(entry)
        rank = int(entry.get("rank", spec.rank) or spec.rank)
        entry_id = _entry_id(
            archive_path=spec.archive_path,
            rank=rank,
            label_mode=label_mode,
            label_direction=label_direction,
            label_threshold=label_threshold,
        )
        logger.info(
            "Training ensemble member %s | label=%s threshold=%.6f | features=%d",
            entry_id,
            label_mode,
            label_threshold,
            len(features),
        )
        entry_record: dict[str, Any] = {
            "rank": rank,
            "entry_id": entry_id,
            "archive": str(spec.archive_path),
            "label_mode": label_mode,
            "label_direction": label_direction,
            "label_threshold": label_threshold,
            "exit_after_k": exit_after_k,
            "score": _json_safe(entry.get("score")),
            "generation": int(entry.get("generation", 0) or 0),
            "features": features,
            "models": [],
        }
        for horizon in config.HOLDING_HORIZONS:
            model_record = _train_one_horizon(
                rank=rank,
                horizon=int(horizon),
                features=features,
                train_df=train_df,
                val_df=val_df,
                feature_space=feature_space,
                model_dir=model_dir,
                entry_id=entry_id,
                trade_top_fraction=trade_top_fraction,
            )
            model_record["label_mode"] = label_mode
            model_record["label_direction"] = label_direction
            model_record["label_threshold"] = label_threshold
            entry_record["models"].append(model_record)
        manifest["entries"].append(entry_record)
        manifest["ensemble"]["members"].append(entry_id)

    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved ensemble manifest: %s", manifest_path)
    return manifest_path


def _train_one_horizon(
    rank: int,
    horizon: int,
    features: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_space: CryptoFeatureSpace,
    model_dir: Path,
    entry_id: str | None = None,
    trade_top_fraction: float = config.TRADE_TOP_FRACTION,
) -> dict[str, Any]:
    label_col = f"label_h{horizon}"
    ret_col = f"future_return_h{horizon}"
    train = _valid_frame(train_df, label_col, ret_col)
    val = _valid_frame(val_df, label_col, ret_col)
    if train.empty:
        raise ValueError(f"h{horizon}: empty train data after label filtering.")

    X_train = feature_space.matrix(features, train.index)
    y_train = train[label_col].astype(int)
    if y_train.nunique() < 2:
        raise ValueError(f"h{horizon}: train label is constant.")

    X_val = (
        feature_space.matrix(features, val.index) if not val.empty else pd.DataFrame()
    )
    y_val = val[label_col].astype(int) if not val.empty else pd.Series(dtype=int)

    booster = _train_booster(X_train, y_train, X_val, y_val)
    val_pred = (
        pd.Series(booster.predict(X_val), index=val.index, name="pred")
        if len(X_val)
        else pd.Series(dtype=float)
    )
    trade_top_fraction = _validate_trade_top_fraction(trade_top_fraction)
    val_trade_threshold = _top_prediction_threshold(
        val_pred,
        trade_top_fraction=trade_top_fraction,
    )
    val_score_band_cutoffs = _score_band_cutoffs(val_pred)
    model_name = (
        f"{_safe_name(entry_id)}_h{horizon}.txt"
        if entry_id
        else f"rank_{rank:02d}_h{horizon}.txt"
    )
    model_path = model_dir / model_name
    booster.save_model(str(model_path))
    logger.info(
        "Saved model rank %02d h%d: %s | best_iteration=%s",
        rank,
        horizon,
        model_path,
        booster.best_iteration,
    )
    return {
        "horizon": horizon,
        "model_path": model_name,
        "label_col": label_col,
        "return_col": ret_col,
        "n_features": len(features),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "train_base_rate": float(y_train.mean()),
        "val_base_rate": float(y_val.mean()) if len(y_val) else None,
        "val_trade_threshold": val_trade_threshold,
        "val_score_band_cutoffs": val_score_band_cutoffs,
        "score_band_fractions": list(SCORE_BAND_FRACTIONS),
        "trade_top_fraction": float(trade_top_fraction),
        "min_trades_per_split": int(config.MIN_TRADES_PER_SPLIT),
        "best_iteration": int(booster.best_iteration or config.LGBM_NUM_BOOST_ROUND),
    }


def _top_prediction_threshold(
    pred: pd.Series,
    trade_top_fraction: float = config.TRADE_TOP_FRACTION,
) -> float | None:
    pred = (
        pd.to_numeric(pred, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    )
    if pred.empty:
        return None
    trade_top_fraction = _validate_trade_top_fraction(trade_top_fraction)
    n_select = min(
        len(pred),
        max(
            int(config.MIN_TRADES_PER_SPLIT),
            int(np.ceil(len(pred) * float(trade_top_fraction))),
        ),
    )
    return float(pred.nlargest(n_select).min())


def _validate_trade_top_fraction(value: float) -> float:
    fraction = float(value)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("trade_top_fraction must be in (0, 1].")
    return fraction


def _score_band_cutoffs(pred: pd.Series) -> dict[str, float | None]:
    clean = (
        pd.to_numeric(pred, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    cutoffs: dict[str, float | None] = {}
    for band_index, fraction in enumerate(SCORE_BAND_FRACTIONS, start=1):
        if clean.empty:
            cutoffs[f"q{band_index}"] = None
            continue
        n_select = min(
            len(clean),
            max(1, int(np.ceil(len(clean) * float(fraction) - 1e-12))),
        )
        cutoffs[f"q{band_index}"] = float(clean.nlargest(n_select).min())
    return cutoffs


def _train_booster(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> lgb.Booster:
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    callbacks = [lgb.log_evaluation(period=-1)]
    valid_sets = None
    if config.LGBM_EARLY_STOPPING > 0 and len(X_val) > 0 and y_val.nunique() >= 2:
        valid_sets = [
            lgb.Dataset(
                X_val,
                label=y_val,
                reference=train_set,
                free_raw_data=False,
            )
        ]
        callbacks.insert(
            0, lgb.early_stopping(config.LGBM_EARLY_STOPPING, verbose=False)
        )

    return lgb.train(
        params=dict(config.LGBM_PARAMS),
        train_set=train_set,
        num_boost_round=int(config.LGBM_NUM_BOOST_ROUND),
        valid_sets=valid_sets,
        callbacks=callbacks,
    )


def _load_archive_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", payload if isinstance(payload, list) else [])
    if not isinstance(entries, list):
        raise ValueError(f"Archive has no entries list: {path}")
    logger.info("Loaded %d archive entries from %s", len(entries), path)
    return [dict(entry) for entry in entries]


def _load_archive_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    return dict(metadata) if isinstance(metadata, dict) else {}


def _archive_horizons(path: Path) -> list[int]:
    raw_horizons = _load_archive_metadata(path).get("horizons")
    if not isinstance(raw_horizons, list) or not raw_horizons:
        horizons = [int(value) for value in config.HOLDING_HORIZONS]
        logger.warning(
            "Archive %s has no metadata horizons; falling back to config: %s",
            path,
            horizons,
        )
        return horizons
    horizons = sorted({int(value) for value in raw_horizons})
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError(f"Archive has invalid metadata horizons: {raw_horizons!r}")
    logger.info("Using archive metadata horizons: %s", horizons)
    return horizons


def _resolve_archive_label_direction(
    path: Path,
    explicit_direction: str | None = None,
) -> str:
    if explicit_direction not in (None, ""):
        return config.canonical_label_direction(explicit_direction)
    metadata_direction = _load_archive_metadata(path).get("label_direction")
    return config.canonical_label_direction(
        metadata_direction if metadata_direction not in (None, "") else "long"
    )


def _resolve_archive_exit_after_k(
    path: Path,
    label_mode: str,
    explicit_k: int | None = None,
) -> int | None:
    requested_k = config.resolve_exit_after_k(label_mode, explicit_k)
    if (
        requested_k is None
        or explicit_k is not None
        or config.canonical_label_mode(label_mode) != "exit_after_k"
    ):
        return requested_k
    metadata = _load_archive_metadata(path)
    archived_mode = metadata.get("label_mode")
    archived_k = config.resolve_exit_after_k(
        archived_mode,
        metadata.get("exit_after_k"),
    )
    return archived_k if archived_k is not None else requested_k


def _load_rank_entry(path: Path, rank: int) -> dict[str, Any]:
    entries = _filter_entries(_load_archive_entries(path), top=None, ranks=[int(rank)])
    entry = dict(entries[0])
    entry["_archive_path"] = str(path)
    return entry


def _filter_entries(
    entries: list[dict[str, Any]],
    top: int | None,
    ranks: list[int] | None,
) -> list[dict[str, Any]]:
    if ranks:
        wanted = {int(rank) for rank in ranks}
        selected = [entry for entry in entries if int(entry.get("rank", -1)) in wanted]
        found = {int(entry.get("rank", -1)) for entry in selected}
        missing = sorted(wanted - found)
        if missing:
            logger.warning("Archive does not contain requested rank(s): %s", missing)
        if not selected:
            raise ValueError(f"No archive entries matched rank(s): {sorted(wanted)}")
        return selected
    if top is None:
        top = 1
    return entries[: int(top)]


def _clean_features(entry: dict[str, Any]) -> list[str]:
    raw_features = entry.get("features")
    if not isinstance(raw_features, list) or not raw_features:
        raise ValueError(f"Archive rank {entry.get('rank')} has no features list.")
    features: list[str] = []
    for feature in raw_features:
        feature = str(feature).strip()
        if feature and feature not in features:
            features.append(feature)
    if not features:
        raise ValueError(f"Archive rank {entry.get('rank')} has no valid features.")
    return features


def _valid_frame(df: pd.DataFrame, label_col: str, ret_col: str) -> pd.DataFrame:
    if label_col not in df.columns or ret_col not in df.columns:
        raise ValueError(f"Missing required columns: {label_col}, {ret_col}")
    return df.dropna(subset=[label_col, ret_col]).copy()


def _config_snapshot(
    val_start: str,
    test_start: str,
    test_end: str | None,
    purge_bars: int,
    label_mode: str,
    label_direction: str,
    label_threshold: float,
    exit_after_k: int | None = None,
    horizons: list[int] | None = None,
    trade_top_fraction: float = config.TRADE_TOP_FRACTION,
    feature_windows: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "horizons": list(horizons or config.HOLDING_HORIZONS),
        "label_mode": label_mode,
        "label_direction": label_direction,
        "direction_neutral": config.is_direction_neutral_label_mode(label_mode),
        "label_threshold": float(label_threshold),
        "exit_after_k": exit_after_k,
        "payoff_tp": float(config.PAYOFF_TP),
        "payoff_adverse_floor": float(config.PAYOFF_ADVERSE_FLOOR),
        "tp_safe_path": float(config.TP_SAFE_PATH),
        "safe_adverse_floor": (
            float(label_threshold)
            if label_mode == "safe_path_mfe"
            else float(config.SAFE_ADVERSE_FLOOR)
        ),
        "safe_path_rule": config.SAFE_PATH_RULE,
        "precision_only": config.is_precision_only_label_mode(label_mode),
        "trade_top_fraction": float(
            _validate_trade_top_fraction(trade_top_fraction)
        ),
        "trade_cost": float(config.TRADE_COST),
        "val_start": val_start,
        "test_start": test_start,
        "test_end": test_end,
        "purge_bars": int(purge_bars),
        "feature_windows": list(
            config.WINDOWS if feature_windows is None else feature_windows
        ),
        "feature_corr_threshold": float(config.FEATURE_CORR_THRESHOLD),
        "lgbm_params": dict(config.LGBM_PARAMS),
        "lgbm_num_boost_round": int(config.LGBM_NUM_BOOST_ROUND),
        "lgbm_early_stopping": int(config.LGBM_EARLY_STOPPING),
    }


def _safe_name(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return clean.strip("._") or "crypto_model"


def _entry_id(
    archive_path: Path,
    rank: int,
    label_mode: str,
    label_direction: str,
    label_threshold: float,
) -> str:
    suffix = ""
    if str(label_mode).strip().lower() == "payoff":
        suffix = (
            f"_tp_{_threshold_token(float(config.PAYOFF_TP))}"
            f"_floor_{_threshold_token(float(config.PAYOFF_ADVERSE_FLOOR))}"
        )
    elif str(label_mode).strip().lower() == "safe_path_mfe":
        suffix = f"_tp_{_threshold_token(float(config.TP_SAFE_PATH))}"
    return _safe_name(
        f"{Path(archive_path).stem}_r{int(rank):02d}_{label_mode}_{label_direction}_thr_"
        f"{_threshold_token(float(label_threshold))}{suffix}"
    )


def _threshold_token(value: float) -> str:
    return f"{value * 100.0:.3f}pct".replace(".", "p").replace("-", "m")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    return value


def _parse_ranks(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    return [int(value) for value in values]


def _resolve_spec_label_threshold(
    spec: EnsembleIndividualSpec,
    default_label_threshold: float,
) -> float:
    if spec.label_threshold is not None:
        return float(spec.label_threshold)
    if spec.label_mode is not None:
        return config.default_label_threshold(spec.label_mode)
    return float(default_label_threshold)


def _parse_ensemble_specs(values: list[str] | None) -> list[EnsembleIndividualSpec]:
    if not values:
        return []
    specs: list[EnsembleIndividualSpec] = []
    for raw_value in values:
        value = str(raw_value).strip()
        if not value:
            continue
        mode_text: str | None = None
        threshold_text: str | None = None
        direction_text: str | None = None
        exit_after_k_text: str | None = None
        if "#" in value:
            parts = [part.strip() for part in value.split("#")]
            if len(parts) not in {2, 3, 4, 5, 6}:
                raise ValueError(
                    "Invalid --ensemble-individual spec. Use "
                    "ARCHIVE#RANK[#MODE[#THRESHOLD[#DIRECTION[#EXIT_AFTER_K]]]], "
                    f"got: {raw_value!r}"
                )
            path_text, rank_text = parts[0], parts[1]
            if len(parts) >= 3 and parts[2]:
                mode_text = parts[2]
            if len(parts) >= 4 and parts[3]:
                threshold_text = parts[3]
            if len(parts) >= 5 and parts[4]:
                direction_text = parts[4]
            if len(parts) >= 6 and parts[5]:
                exit_after_k_text = parts[5]
        else:
            raise ValueError(
                "Invalid --ensemble-individual spec. Use ARCHIVE#RANK[#MODE[#THRESHOLD]], "
                f"got: {raw_value!r}"
            )
        if mode_text is not None:
            mode_text = config.canonical_label_mode(mode_text)
        if direction_text is not None:
            direction_text = config.canonical_label_direction(direction_text)
        specs.append(
            EnsembleIndividualSpec(
                archive_path=Path(path_text),
                rank=int(rank_text),
                label_mode=mode_text,
                label_threshold=float(threshold_text)
                if threshold_text is not None
                else None,
                label_direction=direction_text,
                exit_after_k=(
                    int(exit_after_k_text)
                    if exit_after_k_text is not None
                    else None
                ),
            )
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=None, help="Crypto archive JSON path.")
    parser.add_argument(
        "--data", default=str(config.DATA_PATH), help="Crypto OHLCV CSV path."
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_MODEL_DIR),
        help=f"Model output root directory. Default: {DEFAULT_MODEL_DIR}",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output subdirectory name under --out-dir. Default: archive filename stem.",
    )
    parser.add_argument(
        "--top", type=int, default=None, help="Train only top N entries."
    )
    parser.add_argument(
        "--rank",
        nargs="+",
        default=None,
        help="Train specific archive rank(s), for example --rank 1 3 10.",
    )
    parser.add_argument(
        "--ensemble-individual",
        nargs="+",
        default=None,
        help=(
            "Train a production ensemble bundle from specs "
            "ARCHIVE#RANK[#MODE[#THRESHOLD[#DIRECTION[#EXIT_AFTER_K]]]]. Example: "
            "crypto/results/a.json#1#mfe#0.003#long"
        ),
    )
    parser.add_argument("--val-start", default=config.VAL_START)
    parser.add_argument("--test-start", default=config.TEST_START)
    parser.add_argument("--test-end", default=config.TEST_END)
    parser.add_argument(
        "--label-mode",
        default=config.LABEL_MODE,
        help=(
            "Label mode used when training production models. "
            f"Allowed: {', '.join(sorted(config.LABEL_RETURN_FNS))}. "
            "first_hit_safe_close/safe_close -> safe_path_mfe. "
            f"Default: {config.LABEL_MODE}."
        ),
    )
    parser.add_argument(
        "--label-direction",
        default=None,
        help=(
            "Long or Short. An explicit value overrides archive metadata. "
            "Archives created before direction metadata are treated as Long."
        ),
    )
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=None,
        help=(
            "Label threshold used when training production models. Default is "
            "LABEL_THRESHOLD for ordinary modes, TRADE_COST for payoff, and "
            "SAFE_ADVERSE_FLOOR for safe_path_mfe. For safe_path_mfe this is "
            "the stop-first adverse low/high floor; TP is config.TP_SAFE_PATH. "
            "For adverse_floor use a positive distance such as 0.003. For "
            "high_exit this is the directional threshold of the exact H candle. "
            "For two_sided_tp it is the positive absolute TP on both sides."
        ),
    )
    parser.add_argument(
        "--trade-top-fraction",
        type=float,
        default=config.TRADE_TOP_FRACTION,
        help=(
            "Top fraction of Val predictions used to create the production "
            "signal threshold. Stored in manifest.json."
        ),
    )
    parser.add_argument(
        "--exit-after-k",
        type=int,
        default=None,
        help=(
            "Decision candle k for exit_after_k. Defaults to archive metadata, "
            f"then config.EXIT_AFTER_K={config.EXIT_AFTER_K}."
        ),
    )
    args = parser.parse_args()

    specs = _parse_ensemble_specs(args.ensemble_individual)
    if specs:
        manifest_path = train_ensemble_from_specs(
            specs=specs,
            data_path=args.data,
            output_dir=args.out_dir,
            run_name=args.run_name,
            val_start=args.val_start,
            test_start=args.test_start,
            test_end=args.test_end,
            default_label_mode=args.label_mode,
            default_label_direction=args.label_direction,
            default_label_threshold=args.label_threshold,
            default_exit_after_k=args.exit_after_k,
            trade_top_fraction=args.trade_top_fraction,
        )
    else:
        if not args.archive:
            parser.error(
                "--archive is required unless --ensemble-individual is provided."
            )
        manifest_path = train_from_archive(
            archive_path=args.archive,
            data_path=args.data,
            output_dir=args.out_dir,
            top=args.top,
            ranks=_parse_ranks(args.rank),
            run_name=args.run_name,
            val_start=args.val_start,
            test_start=args.test_start,
            test_end=args.test_end,
            label_mode=args.label_mode,
            label_direction=args.label_direction,
            label_threshold=args.label_threshold,
            exit_after_k=args.exit_after_k,
            trade_top_fraction=args.trade_top_fraction,
        )
    logger.info("Done. Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
