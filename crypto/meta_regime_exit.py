"""Leakage-safe Bull/Bear OOF targets for intrabar regime exits."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crypto import config
from crypto.data import CryptoFold
from crypto.evolution import CryptoIndividual
from crypto.expression import CryptoFeatureSpace
from crypto.fitness import CryptoFitnessEvaluator
from crypto.meta_targets import MetaFeatureAlignment, make_meta_fold


logger = logging.getLogger(__name__)

REGIME_SIDE_COLUMN = "meta_regime_side"
REGIME_BULL_COLUMN = "meta_regime_bull_signal"
REGIME_BEAR_COLUMN = "meta_regime_bear_signal"
REGIME_EXIT_TIME_COLUMN = "meta_regime_exit_time"
REGIME_EXIT_PRICE_COLUMN = "meta_regime_exit_price"
REGIME_ACTION_TIME_COLUMN = "meta_regime_action_time"
REGIME_ACTION_PRICE_COLUMN = "meta_regime_action_price"
REGIME_CLOSE_TIME_COLUMN = "meta_regime_close_time"
REGIME_CLOSE_PRICE_COLUMN = "meta_regime_close_price"


@dataclass(frozen=True)
class RegimeArchive:
    archive_path: Path
    archive_sha256: str
    rank: int
    mode: str
    horizon: int
    top_fraction: float
    individual: CryptoIndividual
    metadata: dict[str, Any]
    tolerance: float
    min_move: float
    min_bars: int


@dataclass(frozen=True)
class MetaRegimeExitBase:
    bull: RegimeArchive
    bear: RegimeArchive
    top_fraction: float


@dataclass(frozen=True)
class MetaRegimeExitData:
    folds: list[CryptoFold]
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    base: MetaRegimeExitBase


@dataclass(frozen=True)
class RegimeSimulation:
    trades: pd.DataFrame
    early_exits: int
    locked_rows: int


def load_meta_regime_base(
    bull_archive: str | Path,
    bear_archive: str | Path,
    *,
    bull_rank: int = 1,
    bear_rank: int = 1,
    top_fraction: float | None = None,
) -> MetaRegimeExitBase:
    """Load and validate the two direction archives used as OOF bases."""
    bull = _load_direction_archive(bull_archive, bull_rank, "bull")
    bear = _load_direction_archive(bear_archive, bear_rank, "bear")
    if bull.horizon != bear.horizon:
        raise ValueError(
            "meta_regime_exit Bull/Bear archives must use the same horizon: "
            f"bull=H{bull.horizon}, bear=H{bear.horizon}."
        )
    if bull.horizon != 1:
        raise ValueError(
            "meta_regime_exit currently requires Bull/Bear archives with H1."
        )
    if top_fraction is None:
        if not np.isclose(bull.top_fraction, bear.top_fraction, atol=1e-12, rtol=0.0):
            raise ValueError(
                "Bull/Bear archives use different trade_top_fraction values; "
                "pass --meta-regime-base-top-fraction explicitly."
            )
        fraction = bull.top_fraction
    else:
        fraction = float(top_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("meta regime base top fraction must be in (0, 1].")
    return MetaRegimeExitBase(bull=bull, bear=bear, top_fraction=fraction)


def required_regime_windows(base: MetaRegimeExitBase) -> list[int]:
    """Return rolling windows referenced by both archived individuals."""
    from crypto.meta_targets import required_feature_windows

    return sorted(
        set(required_feature_windows(base.bull.individual))
        | set(required_feature_windows(base.bear.individual))
    )


def build_meta_regime_exit_data(
    *,
    raw_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    alignment: MetaFeatureAlignment,
    original_folds: list[CryptoFold],
    final_train_df: pd.DataFrame,
    final_val_df: pd.DataFrame,
    final_test_df: pd.DataFrame,
    base_feature_space: CryptoFeatureSpace,
    base: MetaRegimeExitBase,
    exit_threshold: float,
    meta_val_fraction: float,
    purge_bars: int,
    test_start: str | pd.Timestamp,
) -> MetaRegimeExitData:
    """Create OOF regime signals and causal intrabar exit labels."""
    threshold = float(exit_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("meta regime exit threshold must be finite and positive.")
    if int(alignment.lookahead_bars) < 1:
        raise ValueError("meta_regime_exit requires at least one observed feature bar.")
    bars_per_target = int(
        alignment.target_interval.value // alignment.feature_interval.value
    )
    if int(alignment.lookahead_bars) >= bars_per_target:
        raise ValueError(
            "meta_regime_exit must leave an executable intrabar open after the "
            f"observation: lookahead={alignment.lookahead_bars}, "
            f"bars_per_target={bars_per_target}."
        )

    bull_labels = _archive_label_frame(raw_df, base.bull)
    bear_labels = _archive_label_frame(raw_df, base.bear)
    meta_folds: list[CryptoFold] = []
    oof_frames: list[pd.DataFrame] = []
    for fold in original_folds:
        bull_prediction = _base_prediction(
            base.bull,
            base_feature_space,
            bull_labels.reindex(fold.train_df.index),
            fold.val_df,
        )
        bear_prediction = _base_prediction(
            base.bear,
            base_feature_space,
            bear_labels.reindex(fold.train_df.index),
            fold.val_df,
        )
        targeted = attach_meta_regime_exit_targets(
            fold.val_df,
            raw_df=raw_df,
            minute_df=minute_df,
            alignment=alignment,
            bull_prediction=bull_prediction,
            bear_prediction=bear_prediction,
            top_fraction=base.top_fraction,
            exit_threshold=threshold,
        )
        meta_fold = make_meta_fold(
            fold.name,
            targeted,
            val_fraction=meta_val_fraction,
            purge_bars=purge_bars,
        )
        meta_folds.append(meta_fold)
        oof_frames.append(targeted)
        _log_target_counts(meta_fold.name, targeted)

    if not oof_frames:
        raise ValueError("No original walk-forward folds available for meta_regime_exit.")
    meta_train = (
        pd.concat(oof_frames)
        .sort_index()
        .loc[lambda frame: ~frame.index.duplicated(keep="last")]
    )

    final_bull_val = _base_prediction(
        base.bull,
        base_feature_space,
        bull_labels.reindex(final_train_df.index),
        final_val_df,
    )
    final_bear_val = _base_prediction(
        base.bear,
        base_feature_space,
        bear_labels.reindex(final_train_df.index),
        final_val_df,
    )
    meta_val = attach_meta_regime_exit_targets(
        final_val_df,
        raw_df=raw_df,
        minute_df=minute_df,
        alignment=alignment,
        bull_prediction=final_bull_val,
        bear_prediction=final_bear_val,
        top_fraction=base.top_fraction,
        exit_threshold=threshold,
    )

    del test_start
    final_bull_test = _base_prediction(
        base.bull,
        base_feature_space,
        bull_labels.reindex(final_train_df.index),
        final_test_df,
    )
    final_bear_test = _base_prediction(
        base.bear,
        base_feature_space,
        bear_labels.reindex(final_train_df.index),
        final_test_df,
    )
    meta_test = attach_meta_regime_exit_targets(
        final_test_df,
        raw_df=raw_df,
        minute_df=minute_df,
        alignment=alignment,
        bull_prediction=final_bull_test,
        bear_prediction=final_bear_test,
        top_fraction=base.top_fraction,
        exit_threshold=threshold,
        bull_cutoff=float(meta_val.attrs["bull_cutoff"]),
        bear_cutoff=float(meta_val.attrs["bear_cutoff"]),
    )
    return MetaRegimeExitData(
        folds=meta_folds,
        train_df=meta_train,
        val_df=meta_val,
        test_df=meta_test,
        base=base,
    )


def attach_meta_regime_exit_targets(
    frame: pd.DataFrame,
    *,
    raw_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    alignment: MetaFeatureAlignment,
    bull_prediction: pd.Series,
    bear_prediction: pd.Series,
    top_fraction: float,
    exit_threshold: float,
    bull_cutoff: float | None = None,
    bear_cutoff: float | None = None,
) -> pd.DataFrame:
    """Attach exclusive base signals, exit labels, and executable timestamps."""
    result = frame.sort_index().copy()
    bull_pred = pd.to_numeric(bull_prediction.reindex(result.index), errors="coerce")
    bear_pred = pd.to_numeric(bear_prediction.reindex(result.index), errors="coerce")
    selected_bull_cutoff = (
        top_fraction_cutoff(bull_pred, top_fraction)
        if bull_cutoff is None
        else float(bull_cutoff)
    )
    selected_bear_cutoff = (
        top_fraction_cutoff(bear_pred, top_fraction)
        if bear_cutoff is None
        else float(bear_cutoff)
    )
    bull = (
        bull_pred.ge(selected_bull_cutoff)
        if np.isfinite(selected_bull_cutoff)
        else bull_pred.notna() & False
    )
    bear = (
        bear_pred.ge(selected_bear_cutoff)
        if np.isfinite(selected_bear_cutoff)
        else bear_pred.notna() & False
    )
    side = pd.Series(
        np.where(bull & ~bear, 1, np.where(bear & ~bull, -1, 0)),
        index=result.index,
        dtype="int8",
    )

    action_time = raw_df.index.to_series().shift(-1).reindex(result.index)
    action_price = pd.to_numeric(raw_df["open"].shift(-1), errors="coerce").reindex(
        result.index
    )
    close_time = action_time + alignment.target_interval
    close_price = pd.to_numeric(raw_df["close"].shift(-1), errors="coerce").reindex(
        result.index
    )
    mapped = alignment.target_to_feature.reindex(result.index)
    exit_time = mapped + alignment.feature_interval
    exit_price = pd.Series(
        minute_df["open"].reindex(pd.DatetimeIndex(exit_time.dropna())).to_numpy(),
        index=exit_time.dropna().index,
        dtype="float64",
    ).reindex(result.index)

    directional_close = side.astype(float) * (close_price.div(action_price) - 1.0)
    eligible = (
        side.ne(0)
        & action_time.notna()
        & action_price.notna()
        & action_price.gt(0.0)
        & close_price.notna()
        & exit_time.notna()
        & exit_price.notna()
    )
    label = directional_close.lt(-float(exit_threshold)).astype(float).where(eligible)

    result["meta_regime_bull_prediction"] = bull_pred
    result["meta_regime_bear_prediction"] = bear_pred
    result[REGIME_BULL_COLUMN] = bull.astype(bool)
    result[REGIME_BEAR_COLUMN] = bear.astype(bool)
    result[REGIME_SIDE_COLUMN] = side
    result[REGIME_ACTION_TIME_COLUMN] = pd.to_datetime(action_time)
    result[REGIME_ACTION_PRICE_COLUMN] = action_price
    result[REGIME_EXIT_TIME_COLUMN] = pd.to_datetime(exit_time)
    result[REGIME_EXIT_PRICE_COLUMN] = exit_price
    result[REGIME_CLOSE_TIME_COLUMN] = pd.to_datetime(close_time)
    result[REGIME_CLOSE_PRICE_COLUMN] = close_price
    result["meta_regime_directional_close"] = directional_close.where(eligible)
    result["label_h1"] = label
    # This diagnostic return is not used directly by regime strategy fitness.
    result["future_return_h1"] = directional_close.where(eligible)
    result.attrs["bull_cutoff"] = selected_bull_cutoff
    result.attrs["bear_cutoff"] = selected_bear_cutoff
    return result


def top_fraction_cutoff(prediction: pd.Series, fraction: float) -> float:
    clean = pd.to_numeric(prediction, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if clean.empty:
        return float("nan")
    n = min(len(clean), max(1, int(np.ceil(len(clean) * float(fraction)))))
    return float(clean.nlargest(n).min())


def simulate_regime_exit(
    frame: pd.DataFrame,
    prediction: pd.Series | None,
    *,
    prediction_cutoff: float,
    trade_cost: float,
) -> RegimeSimulation:
    """Simulate one entry per base-signal episode with causal intrabar exits."""
    data = frame.sort_index()
    pred = (
        pd.to_numeric(prediction.reindex(data.index), errors="coerce")
        if prediction is not None
        else pd.Series(np.nan, index=data.index, dtype="float64")
    )
    rows: list[dict[str, Any]] = []
    current_side = 0
    entry_time = pd.NaT
    entry_price = float("nan")
    episode_side = 0
    episode_consumed = False
    early_exits = 0
    locked_rows = 0

    def close_trade(exit_time: Any, exit_price: float, reason: str) -> None:
        nonlocal current_side, entry_time, entry_price
        gross = current_side * (float(exit_price) / float(entry_price) - 1.0)
        rows.append(
            {
                "side": "long" if current_side > 0 else "short",
                "entry_time": entry_time,
                "exit_time": pd.Timestamp(exit_time),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "gross_return": float(gross),
                "net_return": float(gross - float(trade_cost)),
                "exit_reason": reason,
            }
        )
        current_side = 0
        entry_time = pd.NaT
        entry_price = float("nan")

    for timestamp, row in data.iterrows():
        wanted = int(row.get(REGIME_SIDE_COLUMN, 0) or 0)
        if wanted != episode_side:
            episode_side = wanted
            episode_consumed = False

        action_time = row.get(REGIME_ACTION_TIME_COLUMN)
        action_price = float(row.get(REGIME_ACTION_PRICE_COLUMN, np.nan))
        executable_action = pd.notna(action_time) and np.isfinite(action_price)
        closed_this_open = False
        if current_side != 0 and wanted != current_side and executable_action:
            close_trade(action_time, action_price, "base_signal_end")
            closed_this_open = True

        if (
            current_side == 0
            and wanted != 0
            and not episode_consumed
            and not closed_this_open
            and executable_action
        ):
            current_side = wanted
            entry_time = pd.Timestamp(action_time)
            entry_price = action_price
            episode_consumed = True

        if current_side == 0:
            if wanted != 0 and episode_consumed:
                locked_rows += 1
            continue
        if wanted != current_side:
            continue

        should_exit = np.isfinite(pred.get(timestamp, np.nan)) and float(
            pred.get(timestamp)
        ) >= float(prediction_cutoff)
        meta_time = row.get(REGIME_EXIT_TIME_COLUMN)
        meta_price = float(row.get(REGIME_EXIT_PRICE_COLUMN, np.nan))
        if should_exit and pd.notna(meta_time) and np.isfinite(meta_price):
            close_trade(meta_time, meta_price, "meta_early_exit")
            early_exits += 1

    if current_side != 0:
        valid = data.dropna(subset=[REGIME_CLOSE_TIME_COLUMN, REGIME_CLOSE_PRICE_COLUMN])
        if not valid.empty:
            final = valid.iloc[-1]
            close_trade(
                final[REGIME_CLOSE_TIME_COLUMN],
                float(final[REGIME_CLOSE_PRICE_COLUMN]),
                "split_end",
            )
    trades = pd.DataFrame(rows)
    return RegimeSimulation(
        trades=trades,
        early_exits=int(early_exits),
        locked_rows=int(locked_rows),
    )


def _load_direction_archive(
    archive_path: str | Path,
    rank: int,
    expected_mode: str,
) -> RegimeArchive:
    path = Path(archive_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    entries = payload.get("entries", [])
    mode = config.canonical_label_mode(metadata.get("label_mode"))
    if mode != expected_mode:
        raise ValueError(f"Expected {expected_mode} archive, got {mode}: {path}")
    horizons = [int(value) for value in metadata.get("horizons", [])]
    if len(horizons) != 1:
        raise ValueError(f"{expected_mode} archive must contain exactly one horizon.")
    row = next((item for item in entries if int(item.get("rank", 0)) == int(rank)), None)
    if row is None and 1 <= int(rank) <= len(entries):
        row = entries[int(rank) - 1]
    if not isinstance(row, dict) or not row.get("features"):
        raise ValueError(f"Rank {rank} not found or has no features in {path}.")
    if expected_mode == "bull":
        tolerance = float(metadata.get("bull_zigzag_tolerance", config.BULL_ZIGZAG_TOLERANCE))
        min_move = float(metadata.get("bull_min_rise", config.BULL_MIN_RISE))
        min_bars = int(metadata.get("bull_min_bars", config.BULL_MIN_BARS))
    else:
        tolerance = float(metadata.get("bear_zigzag_tolerance", config.BEAR_ZIGZAG_TOLERANCE))
        min_move = float(metadata.get("bear_min_drop", config.BEAR_MIN_DROP))
        min_bars = int(metadata.get("bear_min_bars", config.BEAR_MIN_BARS))
    return RegimeArchive(
        archive_path=path,
        archive_sha256=_file_sha256(path),
        rank=int(rank),
        mode=mode,
        horizon=horizons[0],
        top_fraction=float(metadata.get("trade_top_fraction", config.TRADE_TOP_FRACTION)),
        individual=CryptoIndividual(
            features=[str(value) for value in row["features"]],
            generation=int(row.get("generation", 0) or 0),
            score=float(row.get("score", np.nan)),
            metrics=dict(row.get("metrics", {})),
        ),
        metadata=dict(metadata),
        tolerance=tolerance,
        min_move=min_move,
        min_bars=min_bars,
    )


def _archive_label_frame(raw_df: pd.DataFrame, archive: RegimeArchive) -> pd.DataFrame:
    labels = config._confirmed_zigzag_body_labels(
        raw_df,
        tolerance=archive.tolerance,
        min_move=archive.min_move,
        min_bars=archive.min_bars,
        start_kind="trough" if archive.mode == "bull" else "peak",
    )
    result = raw_df.copy()
    result["label_h1"] = labels
    result["future_return_h1"] = pd.Series(0.0, index=result.index).where(
        labels.notna()
    )
    return result


def _base_prediction(
    archive: RegimeArchive,
    feature_space: CryptoFeatureSpace,
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
) -> pd.Series:
    train = train_df.dropna(subset=["label_h1"])
    if train.empty or predict_df.empty:
        raise ValueError(f"{archive.mode} OOF train/prediction frame is empty.")
    X_train = feature_space.matrix(archive.individual.features, train.index)
    X_predict = feature_space.matrix(archive.individual.features, predict_df.index)
    y_train = train["label_h1"].astype(int)
    if y_train.nunique() < 2:
        raise ValueError(f"{archive.mode} OOF training label is constant.")
    evaluator = CryptoFitnessEvaluator(horizons=[1], precision_only=True)
    booster = evaluator._train_booster(
        X_train,
        y_train,
        X_train.iloc[:0],
        y_train.iloc[:0],
    )
    prediction = pd.Series(
        booster.predict(X_predict), index=predict_df.index, dtype="float64"
    )
    booster.free_dataset()
    return prediction


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _log_target_counts(name: str, frame: pd.DataFrame) -> None:
    eligible = int(frame["label_h1"].notna().sum())
    positive = int(frame["label_h1"].eq(1.0).sum())
    logger.info(
        "%s: rows=%d | exclusive regime=%d | exit labels=%d (%.2f%%)",
        name,
        len(frame),
        eligible,
        positive,
        100.0 * positive / max(eligible, 1),
    )
