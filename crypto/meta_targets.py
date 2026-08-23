"""OOF target construction for the dynamic-TP meta learner modes.

The base quantile-MFE archive is retrained on every original walk-forward
train fold. Its prediction on that fold's validation period defines a dynamic
TP, but is deliberately not added to the model feature matrix. Depending on
the selected mode, the binary target asks whether the path hits that TP, the
final close clears a threshold, or the executable TP-or-close payoff clears a
threshold.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crypto import config
from crypto.data import CryptoFold
from crypto.evolution import CryptoIndividual
from crypto.expression import CryptoFeatureSpace
from crypto.quantile_fitness import QuantileFitnessEvaluator


logger = logging.getLogger(__name__)

_WINDOW_SUFFIX_RE = re.compile(r"_(\d+)(?=\b|[^0-9])")
_WINDOW_ARG_RE = re.compile(r",\s*(\d+)\s*\)")


@dataclass(frozen=True)
class MetaLearnerBase:
    archive_path: Path
    archive_sha256: str
    rank: int
    horizon: int
    quantile: float
    individual: CryptoIndividual
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MetaLearnerData:
    folds: list[CryptoFold]
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    base: MetaLearnerBase


@dataclass(frozen=True)
class MetaFeatureAlignment:
    target_to_feature: pd.Series
    target_interval: pd.Timedelta
    feature_interval: pd.Timedelta
    lookahead_bars: int = 0

    @property
    def source_index(self) -> pd.DatetimeIndex:
        values = self.target_to_feature.dropna().unique()
        return pd.DatetimeIndex(values).sort_values()

    def source_index_for_targets(self, target_index: pd.Index) -> pd.DatetimeIndex:
        values = self.target_to_feature.reindex(target_index).dropna().unique()
        return pd.DatetimeIndex(values).sort_values()


def build_meta_feature_alignment(
    target_index: pd.Index,
    feature_index: pd.Index,
    *,
    include_h1: bool = False,
    lookahead_bars: int = 0,
    horizon: int = 1,
) -> MetaFeatureAlignment:
    """Map targets to the last source candle available at decision time."""
    targets = pd.DatetimeIndex(target_index).sort_values()
    features = pd.DatetimeIndex(feature_index).sort_values()
    target_interval = _infer_regular_interval(targets, "target")
    feature_interval = _infer_regular_interval(features, "feature")
    slower_features = feature_interval > target_interval
    if not slower_features and target_interval.value % feature_interval.value != 0:
        raise ValueError(
            "target candle interval must be an integer multiple of the meta "
            f"feature interval: feature={feature_interval}, target={target_interval}."
        )
    holding_horizon = int(horizon)
    if holding_horizon < 1:
        raise ValueError("meta feature alignment horizon must be positive.")
    observed_bars = int(lookahead_bars)
    if observed_bars < 0:
        raise ValueError("meta feature lookahead bars must be non-negative.")
    if slower_features and (include_h1 or observed_bars):
        raise ValueError(
            "slower meta feature candles only support completed-candle features "
            "with include_h1=False and lookahead_bars=0."
        )
    bars_per_target = (
        0 if slower_features else int(target_interval.value // feature_interval.value)
    )
    if include_h1:
        if observed_bars:
            raise ValueError(
                "include_h1 and lookahead_bars cannot both be enabled."
            )
        observed_bars = bars_per_target
    max_observed_bars = holding_horizon * bars_per_target
    if not slower_features and observed_bars > max_observed_bars:
        raise ValueError(
            "meta feature lookahead cannot extend beyond the final horizon: "
            f"lookahead={observed_bars}, horizon={holding_horizon}, "
            f"bars_per_target={bars_per_target}, maximum={max_observed_bars}."
        )

    # CSV timestamps identify candle opens. Normally a 5m signal candle opened
    # at 00:00 maps to the 1m candle opened at 00:04. With one observed H1
    # candle, the same row maps to 00:09; longer lookahead may continue into
    # later holding candles but never beyond the configured final horizon.
    cutoffs = (
        targets
        + target_interval
        - feature_interval
        + observed_bars * feature_interval
    )
    # Use the latest feature candle whose close is available by the cutoff.
    # This also permits, for example, completed 15m context for a 5m target.
    positions = features.searchsorted(cutoffs, side="right") - 1
    mapped = pd.Series(pd.NaT, index=targets, dtype="datetime64[ns]")
    valid = positions >= 0
    if valid.any():
        mapped.iloc[np.flatnonzero(valid)] = features.take(positions[valid]).to_numpy()
    coverage = float(mapped.notna().mean()) if len(mapped) else 0.0
    if coverage <= 0.0:
        raise ValueError("No target candles can be aligned to meta feature candles.")
    logger.info(
        "Meta feature alignment: target=%s feature=%s | matched=%d/%d (%.2f%%)",
        target_interval,
        feature_interval,
        int(mapped.notna().sum()),
        len(mapped),
        100.0 * coverage,
    )
    return MetaFeatureAlignment(
        target_to_feature=mapped,
        target_interval=target_interval,
        feature_interval=feature_interval,
        lookahead_bars=observed_bars,
    )


def build_post_observation_mfe(
    target_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    alignment: MetaFeatureAlignment,
    *,
    horizon: int,
) -> pd.Series:
    """Return MFE after the observed H1 micro-candles, anchored to open H1."""
    h = int(horizon)
    observed_bars = int(alignment.lookahead_bars)
    if h < 1:
        raise ValueError("meta learner horizon must be positive.")
    if observed_bars < 1:
        raise ValueError("post-observation MFE requires at least one lookahead bar.")
    if "open" not in target_frame or "high" not in feature_frame:
        raise ValueError("Target open and lower-timeframe high columns are required.")

    ratio = int(alignment.target_interval.value // alignment.feature_interval.value)
    path_bars = h * ratio - observed_bars
    if path_bars < 1:
        raise ValueError(
            "No target path remains after the observed lower-timeframe candles."
        )

    targets = pd.DatetimeIndex(target_frame.index)
    features = pd.DatetimeIndex(feature_frame.index)
    path_start = (
        targets
        + alignment.target_interval
        + observed_bars * alignment.feature_interval
    )
    start_positions = features.get_indexer(path_start)
    end_positions = start_positions + path_bars - 1
    expected_end = (
        targets
        + (h + 1) * alignment.target_interval
        - alignment.feature_interval
    )
    valid = (
        (start_positions >= 0)
        & (end_positions >= 0)
        & (end_positions < len(features))
    )
    safe_end = np.clip(end_positions, 0, max(len(features) - 1, 0))
    if len(features):
        valid &= features.take(safe_end).to_numpy() == expected_end.to_numpy()

    highs = pd.to_numeric(feature_frame["high"], errors="coerce").to_numpy()
    path_high = np.full(len(targets), -np.inf, dtype="float64")
    complete = valid.copy()
    safe_start = np.clip(start_positions, 0, max(len(features) - 1, 0))
    for offset in range(path_bars):
        values = highs[np.clip(safe_start + offset, 0, max(len(highs) - 1, 0))]
        complete &= np.isfinite(values)
        path_high = np.maximum(path_high, values)

    entry_open = pd.to_numeric(target_frame["open"].shift(-1), errors="coerce")
    entry = entry_open.to_numpy(dtype="float64")
    complete &= np.isfinite(entry) & (entry > 0.0)
    mfe = np.full(len(targets), np.nan, dtype="float64")
    mfe[complete] = path_high[complete] / entry[complete] - 1.0
    return pd.Series(mfe, index=target_frame.index, dtype="float64")


def build_observed_mfe(
    target_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    alignment: MetaFeatureAlignment,
) -> pd.Series:
    """Return MFE inside the lower-timeframe candles already observed in H1."""
    observed_bars = int(alignment.lookahead_bars)
    if observed_bars < 1:
        raise ValueError("observed MFE requires at least one lookahead bar.")
    if "open" not in target_frame or "high" not in feature_frame:
        raise ValueError("Target open and lower-timeframe high columns are required.")

    targets = pd.DatetimeIndex(target_frame.index)
    features = pd.DatetimeIndex(feature_frame.index)
    observed_start = targets + alignment.target_interval
    start_positions = features.get_indexer(observed_start)
    end_positions = start_positions + observed_bars - 1
    valid = (
        (start_positions >= 0)
        & (end_positions >= 0)
        & (end_positions < len(features))
    )
    expected_end = (
        observed_start + (observed_bars - 1) * alignment.feature_interval
    )
    safe_end = np.clip(end_positions, 0, max(len(features) - 1, 0))
    if len(features):
        valid &= features.take(safe_end).to_numpy() == expected_end.to_numpy()

    highs = pd.to_numeric(feature_frame["high"], errors="coerce").to_numpy()
    observed_high = np.full(len(targets), -np.inf, dtype="float64")
    complete = valid.copy()
    safe_start = np.clip(start_positions, 0, max(len(features) - 1, 0))
    for offset in range(observed_bars):
        values = highs[np.clip(safe_start + offset, 0, max(len(highs) - 1, 0))]
        complete &= np.isfinite(values)
        observed_high = np.maximum(observed_high, values)

    entry_open = pd.to_numeric(target_frame["open"].shift(-1), errors="coerce")
    entry = entry_open.to_numpy(dtype="float64")
    complete &= np.isfinite(entry) & (entry > 0.0)
    mfe = np.full(len(targets), np.nan, dtype="float64")
    mfe[complete] = observed_high[complete] / entry[complete] - 1.0
    return pd.Series(mfe, index=target_frame.index, dtype="float64")


def align_meta_feature_frame(
    sampled_feature_frame: pd.DataFrame,
    alignment: MetaFeatureAlignment,
) -> pd.DataFrame:
    """Reindex sampled lower-timeframe features onto target candle opens."""
    mapping = alignment.target_to_feature
    valid_mapping = mapping.dropna()
    source_index = pd.DatetimeIndex(valid_mapping.to_numpy())
    aligned = sampled_feature_frame.reindex(source_index)
    aligned.index = valid_mapping.index
    if len(valid_mapping) == len(mapping):
        return aligned
    return aligned.reindex(mapping.index)


def _infer_regular_interval(index: pd.DatetimeIndex, name: str) -> pd.Timedelta:
    if len(index) < 2:
        raise ValueError(f"Cannot infer {name} candle interval from fewer than 2 rows.")
    deltas = index.to_series().diff().dropna()
    interval = pd.Timedelta(deltas.median())
    if interval <= pd.Timedelta(0):
        raise ValueError(f"Unable to infer a positive {name} candle interval.")
    return interval


def required_feature_windows(individual: CryptoIndividual) -> list[int]:
    """Return rolling windows referenced by one archived feature set."""
    windows: set[int] = set()
    for formula in individual.features:
        windows.update(int(match.group(1)) for match in _WINDOW_SUFFIX_RE.finditer(formula))
        windows.update(int(match.group(1)) for match in _WINDOW_ARG_RE.finditer(formula))
    return sorted(window for window in windows if window > 1)


def load_meta_base(
    archive_path: str | Path,
    rank: int,
    horizons: list[int] | tuple[int, ...],
) -> MetaLearnerBase:
    """Load and validate the quantile-MFE individual used to create OOF TPs."""
    path = Path(archive_path)
    if not path.exists():
        raise FileNotFoundError(f"Meta learner base archive not found: {path}")
    if int(rank) < 1:
        raise ValueError("Meta learner base rank must be positive.")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed meta learner base archive: {path}")
    metadata = payload.get("metadata")
    entries = payload.get("entries")
    if not isinstance(metadata, dict) or not isinstance(entries, list):
        raise ValueError(f"Malformed meta learner base archive: {path}")
    if config.canonical_label_mode(metadata.get("label_mode")) != "quantile_trade":
        raise ValueError("meta_learner base archive must use label_mode=quantile_trade.")
    if config.canonical_quantile_target(metadata.get("quantile_target")) != "mfe":
        raise ValueError("meta_learner base archive must use quantile_target=mfe.")

    base_horizons = [int(value) for value in metadata.get("horizons", [])]
    requested_horizons = [int(value) for value in horizons]
    if len(base_horizons) != 1:
        raise ValueError(
            "meta_learner currently requires a base archive with exactly one horizon."
        )
    if requested_horizons != base_horizons:
        raise ValueError(
            "meta_learner --horizons must exactly match the base archive: "
            f"requested={requested_horizons}, base={base_horizons}."
        )

    row = next(
        (item for item in entries if int(item.get("rank", 0) or 0) == int(rank)),
        None,
    )
    if row is None and int(rank) <= len(entries):
        row = entries[int(rank) - 1]
    if not isinstance(row, dict):
        raise ValueError(f"Rank {rank} not found in meta learner base archive {path}.")
    features = [str(value).strip() for value in row.get("features", []) if str(value).strip()]
    if not features:
        raise ValueError(f"Rank {rank} in {path} has no features.")

    quantile = config.validate_quantile_alpha(metadata.get("quantile_alpha"))
    individual = CryptoIndividual(
        features=features,
        generation=int(row.get("generation", 0) or 0),
        score=float(row.get("score", np.nan)),
        metrics=dict(row.get("metrics", {})),
    )
    return MetaLearnerBase(
        archive_path=path,
        archive_sha256=_file_sha256(path),
        rank=int(rank),
        horizon=base_horizons[0],
        quantile=quantile,
        individual=individual,
        metadata=dict(metadata),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def attach_meta_targets(
    frame: pd.DataFrame,
    prediction: pd.Series,
    *,
    horizon: int,
    min_prediction: float,
    target_mode: str = "meta_learner",
    label_threshold: float = 0.0,
    tp_offset: float = 0.0,
    stop_loss: float = config.META_STRATEGY_STOP_LOSS,
    target_start_step: int = 1,
    path_mfe_column: str | None = None,
    observed_mfe_column: str | None = None,
) -> pd.DataFrame:
    """Attach one OOF dynamic-TP meta target while preserving row indices."""
    h = int(horizon)
    selected_mode = config.canonical_label_mode(target_mode)
    if selected_mode not in config.META_LEARNER_LABEL_MODES:
        raise ValueError(
            f"target_mode must be a meta label mode, got {selected_mode!r}."
        )
    threshold = float(label_threshold)
    if not np.isfinite(threshold):
        raise ValueError("meta label_threshold must be finite.")
    if not np.isfinite(float(stop_loss)) or float(stop_loss) < 0.0:
        raise ValueError("meta strategy stop_loss must be finite and non-negative.")
    stop_enabled = selected_mode == "meta_strategy_profit" and float(stop_loss) > 0.0
    close_col = f"quantile_close_return_h{h}"
    adverse_col = f"quantile_down_mfe_h{h}"
    start_step = int(target_start_step)
    if not 1 <= start_step <= h:
        raise ValueError(
            f"target_start_step must be in [1, {h}], got {start_step}."
        )
    if path_mfe_column is not None:
        path_columns = [str(path_mfe_column)]
    elif start_step == 1:
        path_columns = [f"quantile_up_mfe_h{h}"]
    else:
        path_columns = [
            f"quantile_up_s{step}" for step in range(start_step, h + 1)
        ]
    required_columns = [*path_columns, close_col]
    if stop_enabled:
        required_columns.append(adverse_col)
    if observed_mfe_column is not None:
        required_columns.append(str(observed_mfe_column))
    missing = [column for column in required_columns if column not in frame]
    if missing:
        raise ValueError(f"Missing meta learner path targets: {missing}")

    result = frame.copy()
    base_prediction = pd.to_numeric(
        prediction.reindex(result.index), errors="coerce"
    )
    tp = base_prediction + float(tp_offset)
    if path_mfe_column is not None or start_step == 1:
        mfe = pd.to_numeric(result[path_columns[0]], errors="coerce")
    else:
        mfe = (
            result[path_columns]
            .apply(pd.to_numeric, errors="coerce")
            .max(axis=1, skipna=False)
        )
    close_return = pd.to_numeric(result[close_col], errors="coerce")
    adverse = (
        pd.to_numeric(result[adverse_col], errors="coerce")
        if stop_enabled
        else pd.Series(0.0, index=result.index)
    )
    eligible = (
        base_prediction.notna()
        & np.isfinite(base_prediction)
        & base_prediction.gt(float(min_prediction))
        & tp.notna()
        & np.isfinite(tp)
        & mfe.notna()
        & close_return.notna()
        & adverse.notna()
    )
    if observed_mfe_column is not None:
        observed_mfe = pd.to_numeric(
            result[str(observed_mfe_column)], errors="coerce"
        )
        # TP hits during the observation candle are already resolved before
        # the meta decision, so they must not become positive or negative rows.
        eligible &= observed_mfe.notna() & observed_mfe.lt(tp)
    hit = mfe.ge(tp) & eligible
    sl_hit = pd.Series(False, index=result.index)
    if stop_enabled:
        sl_hit = adverse.ge(float(stop_loss)) & eligible

    label = pd.Series(np.nan, index=result.index, dtype="float64")
    future_return = pd.Series(np.nan, index=result.index, dtype="float64")
    future_return.loc[eligible] = close_return.loc[eligible]
    future_return.loc[hit] = tp.loc[hit]
    if stop_enabled:
        # Stop-first policy: any SL touch overrides TP, including both-touch rows.
        future_return.loc[sl_hit] = -float(stop_loss)
    if selected_mode == "meta_learner":
        positive = hit
    elif selected_mode == "meta_close_exit":
        positive = close_return.gt(threshold) & eligible
    else:
        positive = future_return.gt(threshold) & eligible & ~sl_hit
    label.loc[eligible] = positive.loc[eligible].astype(float)

    result[f"meta_dynamic_tp_h{h}"] = tp.where(eligible)
    result[f"label_h{h}"] = label
    result[f"future_return_h{h}"] = future_return
    return result


def make_meta_fold(
    name: str,
    oof_frame: pd.DataFrame,
    *,
    val_fraction: float,
    purge_bars: int,
) -> CryptoFold:
    """Chronologically split one original WF-Val into meta train and meta val."""
    if oof_frame.empty:
        raise ValueError(f"Cannot build {name} from an empty OOF frame.")
    fraction = float(val_fraction)
    if not 0.0 < fraction < 0.5:
        raise ValueError("Meta validation fraction must be in (0, 0.5).")

    data = oof_frame.sort_index()
    split_pos = int(np.floor(len(data) * (1.0 - fraction)))
    train_end_pos = split_pos - max(int(purge_bars), 0)
    if train_end_pos <= 0 or split_pos >= len(data):
        raise ValueError(f"Not enough rows to split meta fold {name}.")
    train = data.iloc[:train_end_pos].copy()
    val = data.iloc[split_pos:].copy()
    if train.empty or val.empty:
        raise ValueError(f"Meta fold {name} produced an empty train or val split.")
    return CryptoFold(
        name=f"meta_{name}",
        train_df=train,
        val_df=val,
        train_start=pd.Timestamp(train.index[0]),
        train_end=pd.Timestamp(data.index[train_end_pos]),
        val_start=pd.Timestamp(val.index[0]),
        val_end=pd.Timestamp(val.index[-1]) + pd.Timedelta(nanoseconds=1),
    )


def build_meta_learner_data(
    *,
    base_labeled_df: pd.DataFrame,
    original_folds: list[CryptoFold],
    final_train_df: pd.DataFrame,
    final_val_df: pd.DataFrame,
    final_test_df: pd.DataFrame,
    feature_space: CryptoFeatureSpace,
    base: MetaLearnerBase,
    min_prediction: float,
    target_mode: str = "meta_learner",
    label_threshold: float = 0.0,
    meta_val_fraction: float,
    target_start_step: int,
    purge_bars: int,
    test_start: str | pd.Timestamp,
    path_mfe_column: str | None = None,
    observed_mfe_column: str | None = None,
    tp_offset: float = 0.0,
    stop_loss: float = config.META_STRATEGY_STOP_LOSS,
) -> MetaLearnerData:
    """Create six OOF meta folds plus leakage-safe final Val/Test targets."""
    evaluator = QuantileFitnessEvaluator(
        horizons=[base.horizon],
        target="mfe",
        quantile=base.quantile,
    )
    oof_frames: list[pd.DataFrame] = []
    meta_folds: list[CryptoFold] = []
    for fold in original_folds:
        prediction = _base_prediction(
            evaluator,
            base.individual,
            feature_space,
            train_df=fold.train_df,
            predict_df=fold.val_df,
            horizon=base.horizon,
        )
        targeted = attach_meta_targets(
            fold.val_df,
            prediction,
            horizon=base.horizon,
            min_prediction=min_prediction,
            target_mode=target_mode,
            label_threshold=label_threshold,
            tp_offset=tp_offset,
            stop_loss=stop_loss,
            target_start_step=target_start_step,
            path_mfe_column=path_mfe_column,
            observed_mfe_column=observed_mfe_column,
        )
        meta_fold = make_meta_fold(
            fold.name,
            targeted,
            val_fraction=meta_val_fraction,
            purge_bars=purge_bars,
        )
        oof_frames.append(targeted)
        meta_folds.append(meta_fold)
        logger.info(
            "%s: OOF=%d eligible=%d positive=%d (%.2f%%) | "
            "meta train=%d val=%d",
            meta_fold.name,
            len(targeted),
            int(targeted[f"label_h{base.horizon}"].notna().sum()),
            int(targeted[f"label_h{base.horizon}"].eq(1.0).sum()),
            100.0
            * float(targeted[f"label_h{base.horizon}"].mean(skipna=True)),
            len(meta_fold.train_df),
            len(meta_fold.val_df),
        )

    if not oof_frames:
        raise ValueError("No original walk-forward folds available for meta mode.")
    meta_train = (
        pd.concat(oof_frames)
        .sort_index()
        .loc[lambda value: ~value.index.duplicated(keep="last")]
    )

    final_val_prediction = _base_prediction(
        evaluator,
        base.individual,
        feature_space,
        train_df=final_train_df,
        predict_df=final_val_df,
        horizon=base.horizon,
    )
    meta_val = attach_meta_targets(
        final_val_df,
        final_val_prediction,
        horizon=base.horizon,
        min_prediction=min_prediction,
        target_mode=target_mode,
        label_threshold=label_threshold,
        tp_offset=tp_offset,
        stop_loss=stop_loss,
        target_start_step=target_start_step,
        path_mfe_column=path_mfe_column,
        observed_mfe_column=observed_mfe_column,
    )

    pretest_train = _purged_rows_before(
        base_labeled_df,
        boundary=pd.Timestamp(test_start),
        purge_bars=purge_bars,
    )
    final_test_prediction = _base_prediction(
        evaluator,
        base.individual,
        feature_space,
        train_df=pretest_train,
        predict_df=final_test_df,
        horizon=base.horizon,
    )
    meta_test = attach_meta_targets(
        final_test_df,
        final_test_prediction,
        horizon=base.horizon,
        min_prediction=min_prediction,
        target_mode=target_mode,
        label_threshold=label_threshold,
        tp_offset=tp_offset,
        stop_loss=stop_loss,
        target_start_step=target_start_step,
        path_mfe_column=path_mfe_column,
        observed_mfe_column=observed_mfe_column,
    )
    return MetaLearnerData(
        folds=meta_folds,
        train_df=meta_train,
        val_df=meta_val,
        test_df=meta_test,
        base=base,
    )


def _base_prediction(
    evaluator: QuantileFitnessEvaluator,
    individual: CryptoIndividual,
    feature_space: CryptoFeatureSpace,
    *,
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    train = evaluator._valid_frame(train_df, horizon)
    predict = evaluator._valid_frame(predict_df, horizon)
    if train.empty or predict.empty:
        raise ValueError("Base MFE train/prediction frame is empty.")
    _, prediction, _ = evaluator._fit_predict(
        individual,
        feature_space,
        horizon,
        train,
        predict,
    )
    return prediction.reindex(predict_df.index)


def _purged_rows_before(
    frame: pd.DataFrame,
    *,
    boundary: pd.Timestamp,
    purge_bars: int,
) -> pd.DataFrame:
    data = frame.sort_index()
    position = int(data.index.searchsorted(pd.Timestamp(boundary), side="left"))
    end = max(position - max(int(purge_bars), 0), 0)
    result = data.iloc[:end].copy()
    if result.empty:
        raise ValueError("Meta learner pre-test base training frame is empty.")
    return result
