"""Train production crypto models from selected archive individuals.

Example:
    python -m crypto.prod.train_model --archive crypto/results/crypto_btc_seed1_12h.json --rank 1
    python -m crypto.prod.train_model --archive crypto/results/crypto_btc_seed1_12h.json --top 3
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from crypto import config
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
) -> Path:
    """Train one LightGBM model per selected individual and horizon."""
    config.validate_config()
    archive_path = Path(archive_path)
    selected_entries = _filter_entries(_load_archive_entries(archive_path), top=top, ranks=ranks)

    run_name = run_name or archive_path.stem
    model_dir = Path(output_dir) / _safe_name(run_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading crypto data from %s", data_path)
    raw_df = load_ohlcv(data_path)
    labeled_df = add_binary_labels(
        raw_df,
        horizons=config.HOLDING_HORIZONS,
        threshold=config.LABEL_THRESHOLD,
    )
    purge_bars = config.purge_bars_for_horizons(config.HOLDING_HORIZONS)
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

    logger.info("Building crypto feature matrix; quality filter uses final train rows.")
    feature_df = build_feature_frame(raw_df, quality_index=train_df.index)
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
            "score": _json_safe(entry.get("score")),
            "generation": int(entry.get("generation", 0) or 0),
            "features": features,
            "models": [],
        }
        for horizon in config.HOLDING_HORIZONS:
            horizon = int(horizon)
            model_record = _train_one_horizon(
                rank=rank,
                horizon=horizon,
                features=features,
                train_df=train_df,
                val_df=val_df,
                feature_space=feature_space,
                model_dir=model_dir,
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


def _train_one_horizon(
    rank: int,
    horizon: int,
    features: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_space: CryptoFeatureSpace,
    model_dir: Path,
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

    X_val = feature_space.matrix(features, val.index) if not val.empty else pd.DataFrame()
    y_val = val[label_col].astype(int) if not val.empty else pd.Series(dtype=int)

    booster = _train_booster(X_train, y_train, X_val, y_val)
    val_pred = (
        pd.Series(booster.predict(X_val), index=val.index, name="pred")
        if len(X_val)
        else pd.Series(dtype=float)
    )
    val_trade_threshold = _top_prediction_threshold(val_pred)
    model_name = f"rank_{rank:02d}_h{horizon}.txt"
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
        "trade_top_fraction": float(config.TRADE_TOP_FRACTION),
        "min_trades_per_split": int(config.MIN_TRADES_PER_SPLIT),
        "best_iteration": int(booster.best_iteration or config.LGBM_NUM_BOOST_ROUND),
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
        callbacks.insert(0, lgb.early_stopping(config.LGBM_EARLY_STOPPING, verbose=False))

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
) -> dict[str, Any]:
    return {
        "horizons": list(config.HOLDING_HORIZONS),
        "label_threshold": float(config.LABEL_THRESHOLD),
        "val_start": val_start,
        "test_start": test_start,
        "test_end": test_end,
        "purge_bars": int(purge_bars),
        "feature_windows": list(config.WINDOWS),
        "feature_corr_threshold": float(config.FEATURE_CORR_THRESHOLD),
        "lgbm_params": dict(config.LGBM_PARAMS),
        "lgbm_num_boost_round": int(config.LGBM_NUM_BOOST_ROUND),
        "lgbm_early_stopping": int(config.LGBM_EARLY_STOPPING),
    }


def _safe_name(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return clean.strip("._") or "crypto_model"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Crypto archive JSON path.")
    parser.add_argument("--data", default=str(config.DATA_PATH), help="Crypto OHLCV CSV path.")
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
    parser.add_argument("--top", type=int, default=None, help="Train only top N entries.")
    parser.add_argument(
        "--rank",
        nargs="+",
        default=None,
        help="Train specific archive rank(s), for example --rank 1 3 10.",
    )
    parser.add_argument("--val-start", default=config.VAL_START)
    parser.add_argument("--test-start", default=config.TEST_START)
    parser.add_argument("--test-end", default=config.TEST_END)
    args = parser.parse_args()

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
    )
    logger.info("Done. Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
