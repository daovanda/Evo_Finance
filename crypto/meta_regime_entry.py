"""Leakage-safe episode-entry targets built from OOF Bull/Bear signals."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from crypto.data import CryptoFold
from crypto.expression import CryptoFeatureSpace
from crypto.meta_regime_exit import (
    REGIME_ACTION_PRICE_COLUMN,
    REGIME_ACTION_TIME_COLUMN,
    REGIME_BEAR_COLUMN,
    REGIME_BULL_COLUMN,
    REGIME_CLOSE_PRICE_COLUMN,
    REGIME_CLOSE_TIME_COLUMN,
    REGIME_SIDE_COLUMN,
    MetaRegimeExitBase,
    _archive_label_frame,
    _base_prediction,
    top_fraction_cutoff,
)
from crypto.meta_targets import make_meta_fold


logger = logging.getLogger(__name__)

REGIME_ACTION_HIGH_COLUMN = "meta_regime_action_high"
REGIME_ACTION_LOW_COLUMN = "meta_regime_action_low"
REGIME_ENTRY_CANDIDATE_COLUMN = "meta_regime_entry_candidate"
REGIME_ENTRY_GROSS_COLUMN = "meta_regime_entry_gross_return"
REGIME_ENTRY_NET_COLUMN = "meta_regime_entry_net_return"
REGIME_ENTRY_EXIT_TIME_COLUMN = "meta_regime_entry_exit_time"
REGIME_ENTRY_SEGMENT_COLUMN = "meta_regime_entry_segment"


@dataclass(frozen=True)
class MetaRegimeEntryData:
    folds: list[CryptoFold]
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    base: MetaRegimeExitBase


@dataclass(frozen=True)
class RegimeEntrySimulation:
    trades: pd.DataFrame
    candidate_episodes: int
    selected_episodes: int
    rejected_episodes: int
    stopped_trades: int
    locked_rows: int


def build_meta_regime_entry_data(
    *,
    raw_df: pd.DataFrame,
    original_folds: list[CryptoFold],
    final_train_df: pd.DataFrame,
    final_val_df: pd.DataFrame,
    final_test_df: pd.DataFrame,
    base_feature_space: CryptoFeatureSpace,
    base: MetaRegimeExitBase,
    stop_loss: float,
    trade_cost: float,
    meta_val_fraction: float,
    purge_bars: int,
) -> MetaRegimeEntryData:
    """Create episode-level OOF labels from the executable base strategy."""
    selected_stop = _validate_stop_loss(stop_loss)
    selected_cost = float(trade_cost)
    if not np.isfinite(selected_cost) or selected_cost < 0.0:
        raise ValueError("meta regime entry trade cost must be finite and non-negative.")

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
        targeted = attach_meta_regime_entry_targets(
            fold.val_df,
            raw_df=raw_df,
            bull_prediction=bull_prediction,
            bear_prediction=bear_prediction,
            top_fraction=base.top_fraction,
            stop_loss=selected_stop,
            trade_cost=selected_cost,
        )
        targeted[REGIME_ENTRY_SEGMENT_COLUMN] = fold.name
        meta_fold = make_meta_fold(
            fold.name,
            targeted,
            val_fraction=meta_val_fraction,
            purge_bars=purge_bars,
        )
        meta_fold = _censor_cross_boundary_train_targets(meta_fold)
        meta_folds.append(meta_fold)
        oof_frames.append(targeted)
        _log_target_counts(meta_fold.name, targeted)

    if not oof_frames:
        raise ValueError("No original walk-forward folds available for meta_regime_entry.")
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
    meta_val = attach_meta_regime_entry_targets(
        final_val_df,
        raw_df=raw_df,
        bull_prediction=final_bull_val,
        bear_prediction=final_bear_val,
        top_fraction=base.top_fraction,
        stop_loss=selected_stop,
        trade_cost=selected_cost,
    )
    meta_val[REGIME_ENTRY_SEGMENT_COLUMN] = "final_val"

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
    meta_test = attach_meta_regime_entry_targets(
        final_test_df,
        raw_df=raw_df,
        bull_prediction=final_bull_test,
        bear_prediction=final_bear_test,
        top_fraction=base.top_fraction,
        stop_loss=selected_stop,
        trade_cost=selected_cost,
        bull_cutoff=float(meta_val.attrs["bull_cutoff"]),
        bear_cutoff=float(meta_val.attrs["bear_cutoff"]),
    )
    meta_test[REGIME_ENTRY_SEGMENT_COLUMN] = "final_test"
    return MetaRegimeEntryData(
        folds=meta_folds,
        train_df=meta_train,
        val_df=meta_val,
        test_df=meta_test,
        base=base,
    )


def attach_meta_regime_entry_targets(
    frame: pd.DataFrame,
    *,
    raw_df: pd.DataFrame,
    bull_prediction: pd.Series,
    bear_prediction: pd.Series,
    top_fraction: float,
    stop_loss: float,
    trade_cost: float,
    bull_cutoff: float | None = None,
    bear_cutoff: float | None = None,
) -> pd.DataFrame:
    """Attach exclusive signals and one net-win label per executable episode."""
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

    result["meta_regime_bull_prediction"] = bull_pred
    result["meta_regime_bear_prediction"] = bear_pred
    result[REGIME_BULL_COLUMN] = bull.astype(bool)
    result[REGIME_BEAR_COLUMN] = bear.astype(bool)
    result[REGIME_SIDE_COLUMN] = side
    result[REGIME_ACTION_TIME_COLUMN] = pd.to_datetime(
        raw_df.index.to_series().shift(-1).reindex(result.index)
    )
    result[REGIME_ACTION_PRICE_COLUMN] = pd.to_numeric(
        raw_df["open"].shift(-1), errors="coerce"
    ).reindex(result.index)
    result[REGIME_ACTION_HIGH_COLUMN] = pd.to_numeric(
        raw_df["high"].shift(-1), errors="coerce"
    ).reindex(result.index)
    result[REGIME_ACTION_LOW_COLUMN] = pd.to_numeric(
        raw_df["low"].shift(-1), errors="coerce"
    ).reindex(result.index)
    result[REGIME_CLOSE_TIME_COLUMN] = result[REGIME_ACTION_TIME_COLUMN]
    result[REGIME_CLOSE_PRICE_COLUMN] = pd.to_numeric(
        raw_df["close"].shift(-1), errors="coerce"
    ).reindex(result.index)
    result[REGIME_ENTRY_CANDIDATE_COLUMN] = False
    result[REGIME_ENTRY_GROSS_COLUMN] = np.nan
    result[REGIME_ENTRY_NET_COLUMN] = np.nan
    result[REGIME_ENTRY_EXIT_TIME_COLUMN] = pd.NaT
    result["label_h1"] = np.nan
    result["future_return_h1"] = np.nan

    baseline = simulate_regime_entry(
        result,
        None,
        prediction_cutoff=float("-inf"),
        stop_loss=stop_loss,
        trade_cost=trade_cost,
        candidate_only=False,
    )
    if not baseline.trades.empty:
        complete = baseline.trades[baseline.trades["exit_reason"] != "split_end"]
        for trade in complete.itertuples(index=False):
            signal_time = pd.Timestamp(trade.entry_signal_time)
            result.at[signal_time, REGIME_ENTRY_CANDIDATE_COLUMN] = True
            result.at[signal_time, REGIME_ENTRY_GROSS_COLUMN] = float(
                trade.gross_return
            )
            result.at[signal_time, REGIME_ENTRY_NET_COLUMN] = float(trade.net_return)
            result.at[signal_time, REGIME_ENTRY_EXIT_TIME_COLUMN] = pd.Timestamp(
                trade.exit_time
            )
            result.at[signal_time, "future_return_h1"] = float(trade.net_return)
            result.at[signal_time, "label_h1"] = float(trade.net_return > 0.0)

    result.attrs["bull_cutoff"] = selected_bull_cutoff
    result.attrs["bear_cutoff"] = selected_bear_cutoff
    return result


def simulate_regime_entry(
    frame: pd.DataFrame,
    prediction: pd.Series | None,
    *,
    prediction_cutoff: float,
    stop_loss: float,
    trade_cost: float,
    candidate_only: bool = True,
) -> RegimeEntrySimulation:
    """Select at most one trade per signal episode and enforce episode locks."""
    data = frame.sort_index()
    if REGIME_ENTRY_SEGMENT_COLUMN in data:
        segment = data[REGIME_ENTRY_SEGMENT_COLUMN].astype("string")
        if segment.nunique(dropna=False) > 1:
            simulations = [
                simulate_regime_entry(
                    group,
                    prediction,
                    prediction_cutoff=prediction_cutoff,
                    stop_loss=stop_loss,
                    trade_cost=trade_cost,
                    candidate_only=candidate_only,
                )
                for _, group in data.groupby(REGIME_ENTRY_SEGMENT_COLUMN, sort=False)
            ]
            trade_frames = [item.trades for item in simulations if not item.trades.empty]
            return RegimeEntrySimulation(
                trades=(
                    pd.concat(trade_frames, ignore_index=True)
                    if trade_frames
                    else pd.DataFrame()
                ),
                candidate_episodes=sum(item.candidate_episodes for item in simulations),
                selected_episodes=sum(item.selected_episodes for item in simulations),
                rejected_episodes=sum(item.rejected_episodes for item in simulations),
                stopped_trades=sum(item.stopped_trades for item in simulations),
                locked_rows=sum(item.locked_rows for item in simulations),
            )
    pred = (
        pd.to_numeric(prediction.reindex(data.index), errors="coerce")
        if prediction is not None
        else None
    )
    selected_stop = _validate_stop_loss(stop_loss)
    rows: list[dict[str, object]] = []
    current_side = 0
    entry_signal_time = pd.NaT
    entry_time = pd.NaT
    entry_price = float("nan")
    has_candidate_flags = candidate_only and REGIME_ENTRY_CANDIDATE_COLUMN in data
    first_side = int(data[REGIME_SIDE_COLUMN].iloc[0]) if len(data) else 0
    first_candidate = bool(data[REGIME_ENTRY_CANDIDATE_COLUMN].iloc[0]) if (
        len(data) and has_candidate_flags
    ) else False
    episode_side = first_side
    # A non-flat first run began before this split and is therefore censored.
    episode_consumed = first_side != 0 and not first_candidate
    candidate_episodes = 0
    selected_episodes = 0
    rejected_episodes = 0
    stopped_trades = 0
    locked_rows = 0

    def close_trade(exit_time: object, exit_price: float, reason: str) -> None:
        nonlocal current_side, entry_signal_time, entry_time, entry_price
        gross = current_side * (float(exit_price) / float(entry_price) - 1.0)
        rows.append(
            {
                "side": "long" if current_side > 0 else "short",
                "entry_signal_time": entry_signal_time,
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
        entry_signal_time = pd.NaT
        entry_time = pd.NaT
        entry_price = float("nan")

    for timestamp, row in data.iterrows():
        wanted = int(row.get(REGIME_SIDE_COLUMN, 0) or 0)
        eligible_candidate = (
            bool(row.get(REGIME_ENTRY_CANDIDATE_COLUMN, False))
            if has_candidate_flags
            else True
        )
        if wanted != episode_side:
            episode_side = wanted
            episode_consumed = wanted != 0 and not eligible_candidate

        action_time = row.get(REGIME_ACTION_TIME_COLUMN)
        action_price = float(row.get(REGIME_ACTION_PRICE_COLUMN, np.nan))
        executable = pd.notna(action_time) and np.isfinite(action_price)
        closed_this_open = False
        if current_side != 0 and wanted != current_side and executable:
            close_trade(action_time, action_price, "signal_end")
            closed_this_open = True

        if (
            current_side == 0
            and wanted != 0
            and not episode_consumed
            and not closed_this_open
            and executable
            and eligible_candidate
        ):
            candidate_episodes += 1
            episode_consumed = True
            score = (
                float(pred.get(timestamp, np.nan))
                if pred is not None
                else float("nan")
            )
            accepted = prediction is None or (
                np.isfinite(score) and score >= float(prediction_cutoff)
            )
            if accepted:
                current_side = wanted
                entry_signal_time = pd.Timestamp(timestamp)
                entry_time = pd.Timestamp(action_time)
                entry_price = action_price
                selected_episodes += 1
            else:
                rejected_episodes += 1

        if current_side == 0:
            if wanted != 0 and episode_consumed:
                locked_rows += 1
            continue
        if wanted != current_side or selected_stop <= 0.0:
            continue

        action_low = float(row.get(REGIME_ACTION_LOW_COLUMN, np.nan))
        action_high = float(row.get(REGIME_ACTION_HIGH_COLUMN, np.nan))
        if current_side > 0:
            stop_price = entry_price * (1.0 - selected_stop)
            stop_hit = np.isfinite(action_low) and action_low <= stop_price
        else:
            stop_price = entry_price * (1.0 + selected_stop)
            stop_hit = np.isfinite(action_high) and action_high >= stop_price
        if stop_hit:
            close_trade(action_time, stop_price, "stop_loss")
            stopped_trades += 1

    if current_side != 0:
        valid = data.dropna(subset=[REGIME_CLOSE_TIME_COLUMN, REGIME_CLOSE_PRICE_COLUMN])
        if not valid.empty:
            final = valid.iloc[-1]
            close_trade(
                final[REGIME_CLOSE_TIME_COLUMN],
                float(final[REGIME_CLOSE_PRICE_COLUMN]),
                "split_end",
            )
    return RegimeEntrySimulation(
        trades=pd.DataFrame(rows),
        candidate_episodes=int(candidate_episodes),
        selected_episodes=int(selected_episodes),
        rejected_episodes=int(rejected_episodes),
        stopped_trades=int(stopped_trades),
        locked_rows=int(locked_rows),
    )


def _validate_stop_loss(stop_loss: float) -> float:
    selected = float(stop_loss)
    if not np.isfinite(selected) or selected < 0.0 or selected >= 1.0:
        raise ValueError("meta regime entry stop loss must be in [0, 1).")
    return selected


def _censor_cross_boundary_train_targets(fold: CryptoFold) -> CryptoFold:
    """Drop train labels whose episode outcome is not observable in train."""
    train = fold.train_df.copy()
    action_times = pd.to_datetime(
        train[REGIME_ACTION_TIME_COLUMN], errors="coerce"
    ).dropna()
    if action_times.empty:
        return fold
    last_action_time = pd.Timestamp(action_times.max())
    exit_times = pd.to_datetime(
        train[REGIME_ENTRY_EXIT_TIME_COLUMN], errors="coerce"
    )
    crosses = train[REGIME_ENTRY_CANDIDATE_COLUMN].astype(bool) & (
        exit_times.isna() | exit_times.gt(last_action_time)
    )
    if not crosses.any():
        return fold
    train.loc[crosses, REGIME_ENTRY_CANDIDATE_COLUMN] = False
    train.loc[
        crosses,
        [
            REGIME_ENTRY_GROSS_COLUMN,
            REGIME_ENTRY_NET_COLUMN,
            "label_h1",
            "future_return_h1",
        ],
    ] = np.nan
    train.loc[crosses, REGIME_ENTRY_EXIT_TIME_COLUMN] = pd.NaT
    return replace(fold, train_df=train)


def _log_target_counts(name: str, frame: pd.DataFrame) -> None:
    eligible = int(frame["label_h1"].notna().sum())
    positive = int(frame["label_h1"].eq(1.0).sum())
    logger.info(
        "%s: rows=%d | entry episodes=%d | net winners=%d (%.2f%%)",
        name,
        len(frame),
        eligible,
        positive,
        100.0 * positive / max(eligible, 1),
    )
