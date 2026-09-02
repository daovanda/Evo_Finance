"""Episode-entry fitness for ``meta_regime_entry``."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from crypto import config
from crypto.data import CryptoFold
from crypto.evolution import CryptoIndividual
from crypto.expression import CryptoFeatureSpace
from crypto.fitness import CryptoFitnessEvaluator, _binary_auc
from crypto.meta_regime_entry import simulate_regime_entry
from crypto.meta_regime_exit import top_fraction_cutoff


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeEntrySplitMetrics:
    auc: float
    precision: float
    base_rate: float
    precision_excess: float
    strategy_net_mean: float
    baseline_net_mean: float
    strategy_delta: float
    return_score: float
    delta_score: float
    n_samples: int
    n_selected: int
    n_trades: int
    n_baseline_trades: int
    n_stopped: int
    n_rejected: int
    locked_rows: int


class RegimeEntryFitnessEvaluator(CryptoFitnessEvaluator):
    """Optimize net-winning episode selection and executable strategy return."""

    def __init__(self) -> None:
        super().__init__(horizons=[1], precision_only=False)

    def evaluate_walk_forward(
        self,
        individual: CryptoIndividual,
        folds: list[CryptoFold],
        feature_data: CryptoFeatureSpace | pd.DataFrame,
    ) -> float:
        if not isinstance(feature_data, CryptoFeatureSpace):
            raise ValueError("meta_regime_entry requires CryptoFeatureSpace.")
        rows: list[dict[str, float]] = []
        for fold in folds:
            row = self._evaluate_fold(individual, fold, feature_data)
            if row is not None:
                rows.append(row)
        if not rows:
            raise ValueError("No valid meta_regime_entry fold metrics were produced.")

        metrics_df = pd.DataFrame(rows)
        val_auc = metrics_df["val_auc"].astype(float)
        auc_edge = float((val_auc - 0.5).mean())
        precision_excess = float(metrics_df["val_precision_excess"].mean())
        return_score = float(metrics_df["val_trade_return_score"].mean())
        delta_score = float(metrics_df["val_strategy_delta_score"].mean())
        auc_std = float(val_auc.std(ddof=0)) if len(val_auc) > 1 else 0.0
        overfit_gap = float(metrics_df["overfit_gap"].mean())
        bad_ratio = float(metrics_df["bad_fold"].mean())
        weights = config.META_REGIME_ENTRY_FITNESS_WEIGHTS
        score = (
            weights["auc_edge"] * auc_edge
            + weights["precision_excess"] * precision_excess
            + weights["trade_return_score"] * return_score
            + weights["strategy_delta_score"] * delta_score
            + weights["auc_std"] * auc_std
            + weights["overfit_gap"] * overfit_gap
            + weights["bad_fold_ratio"] * bad_ratio
        )
        individual.score = float(score)
        individual.metrics = {
            "score": float(score),
            "mean_auc": float(val_auc.mean()),
            "auc_edge": auc_edge,
            "precision_at_trade": float(metrics_df["val_precision"].mean()),
            "meta_entry_precision": float(metrics_df["val_precision"].mean()),
            "base_rate": float(metrics_df["val_base_rate"].mean()),
            "precision_excess": precision_excess,
            "trade_return_mean": float(metrics_df["val_strategy_net_mean"].mean()),
            "trade_return_score": return_score,
            "baseline_trade_return_mean": float(
                metrics_df["val_baseline_net_mean"].mean()
            ),
            "strategy_delta_mean": float(metrics_df["val_strategy_delta"].mean()),
            "strategy_delta_score": delta_score,
            "selected_entries": float(metrics_df["val_selected"].mean()),
            "strategy_trades": float(metrics_df["val_strategy_trades"].mean()),
            "baseline_trades": float(metrics_df["val_baseline_trades"].mean()),
            "stopped_trades": float(metrics_df["val_stopped"].mean()),
            "rejected_episodes": float(metrics_df["val_rejected"].mean()),
            "locked_rows": float(metrics_df["val_locked_rows"].mean()),
            "auc_std": auc_std,
            "overfit_gap": overfit_gap,
            "bad_fold_ratio": bad_ratio,
            "n_fold_horizon_scores": float(len(metrics_df)),
            "n_horizons": 1.0,
        }
        logger.info(
            "Regime-entry WF: score=%.4f | AUC=%.4f | precision_excess=%.4f | "
            "strategy_net=%.4f%% | baseline_net=%.4f%% | delta=%.4f%%",
            score,
            individual.metrics["mean_auc"],
            precision_excess,
            100.0 * individual.metrics["trade_return_mean"],
            100.0 * individual.metrics["baseline_trade_return_mean"],
            100.0 * individual.metrics["strategy_delta_mean"],
        )
        return float(score)

    def evaluate_final(
        self,
        individual: CryptoIndividual,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_data: CryptoFeatureSpace | pd.DataFrame,
    ) -> dict[str, float]:
        if not isinstance(feature_data, CryptoFeatureSpace):
            raise ValueError("meta_regime_entry requires CryptoFeatureSpace.")
        train = _valid_candidates(train_df)
        val = _valid_candidates(val_df)
        test = _valid_candidates(test_df)
        if train.empty or val.empty or test.empty:
            raise ValueError("meta_regime_entry final split contains no candidates.")

        X_train = feature_data.matrix(individual.features, train.index)
        X_val = feature_data.matrix(individual.features, val.index)
        X_test = feature_data.matrix(individual.features, test.index)
        y_train = train["label_h1"].astype(int)
        y_val = val["label_h1"].astype(int)
        y_test = test["label_h1"].astype(int)
        if y_train.nunique() < 2:
            raise ValueError("meta_regime_entry final training label is constant.")

        booster = self._train_booster_final(X_train, y_train, X_val, y_val)
        train_pred = pd.Series(booster.predict(X_train), index=train.index)
        val_pred = pd.Series(booster.predict(X_val), index=val.index)
        test_pred = pd.Series(booster.predict(X_test), index=test.index)
        booster.free_dataset()
        cutoff = top_fraction_cutoff(val_pred, config.TRADE_TOP_FRACTION)
        train_cutoff = top_fraction_cutoff(train_pred, config.TRADE_TOP_FRACTION)
        train_metrics = _split_metrics(train_df, y_train, train_pred, train_cutoff)
        val_metrics = _split_metrics(val_df, y_val, val_pred, cutoff)
        test_metrics = _split_metrics(test_df, y_test, test_pred, cutoff)

        metrics: dict[str, float] = {
            "final_n_horizon_scores": 1.0,
            "final_train_rows": float(train_metrics.n_samples),
            "final_val_rows": float(val_metrics.n_samples),
            "final_test_rows": float(test_metrics.n_samples),
            **_final_split_fields("final_val", val_metrics),
            **_final_split_fields("final_test", test_metrics),
            "final_val_auc_std": 0.0,
            "final_test_auc_std": 0.0,
            "final_val_overfit_gap": max(0.0, train_metrics.auc - val_metrics.auc),
            "final_test_overfit_gap": max(0.0, train_metrics.auc - test_metrics.auc),
            "final_val_bad_ratio": float(_is_bad(val_metrics)),
            "final_test_bad_ratio": float(_is_bad(test_metrics)),
            "final_meta_entry_cutoff": float(cutoff),
        }
        individual.metrics.update(metrics)
        logger.info(
            "Regime-entry final: val net=%.4f%% (delta %.4f%%) | "
            "test net=%.4f%% (delta %.4f%%) | cutoff=%.6f",
            100.0 * val_metrics.strategy_net_mean,
            100.0 * val_metrics.strategy_delta,
            100.0 * test_metrics.strategy_net_mean,
            100.0 * test_metrics.strategy_delta,
            cutoff,
        )
        return metrics

    def _evaluate_fold(
        self,
        individual: CryptoIndividual,
        fold: CryptoFold,
        feature_space: CryptoFeatureSpace,
    ) -> dict[str, float] | None:
        train = _valid_candidates(fold.train_df)
        val = _valid_candidates(fold.val_df)
        if train.empty or val.empty:
            return None
        X_train = feature_space.matrix(individual.features, train.index)
        X_val = feature_space.matrix(individual.features, val.index)
        y_train = train["label_h1"].astype(int)
        y_val = val["label_h1"].astype(int)
        if y_train.nunique() < 2 or y_val.nunique() < 2:
            return None
        booster = self._train_booster(X_train, y_train, X_val, y_val)
        train_pred = pd.Series(booster.predict(X_train), index=train.index)
        val_pred = pd.Series(booster.predict(X_val), index=val.index)
        booster.free_dataset()
        cutoff = top_fraction_cutoff(train_pred, config.TRADE_TOP_FRACTION)
        train_metrics = _split_metrics(fold.train_df, y_train, train_pred, cutoff)
        val_metrics = _split_metrics(fold.val_df, y_val, val_pred, cutoff)
        return {
            "train_auc": train_metrics.auc,
            "val_auc": val_metrics.auc,
            "val_precision": val_metrics.precision,
            "val_base_rate": val_metrics.base_rate,
            "val_precision_excess": val_metrics.precision_excess,
            "val_strategy_net_mean": val_metrics.strategy_net_mean,
            "val_baseline_net_mean": val_metrics.baseline_net_mean,
            "val_strategy_delta": val_metrics.strategy_delta,
            "val_trade_return_score": val_metrics.return_score,
            "val_strategy_delta_score": val_metrics.delta_score,
            "val_selected": float(val_metrics.n_selected),
            "val_strategy_trades": float(val_metrics.n_trades),
            "val_baseline_trades": float(val_metrics.n_baseline_trades),
            "val_stopped": float(val_metrics.n_stopped),
            "val_rejected": float(val_metrics.n_rejected),
            "val_locked_rows": float(val_metrics.locked_rows),
            "overfit_gap": max(0.0, train_metrics.auc - val_metrics.auc),
            "bad_fold": float(_is_bad(val_metrics)),
        }


def _valid_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if "label_h1" not in frame:
        raise ValueError("meta_regime_entry frame is missing label_h1.")
    return frame.dropna(subset=["label_h1"]).copy()


def _split_metrics(
    full_frame: pd.DataFrame,
    y_true: pd.Series,
    prediction: pd.Series,
    cutoff: float,
) -> RegimeEntrySplitMetrics:
    aligned = pd.DataFrame({"y": y_true, "pred": prediction}).dropna()
    y = aligned["y"].astype(int)
    selected = aligned[aligned["pred"] >= float(cutoff)]
    auc = float(_binary_auc(y, aligned["pred"]))
    base_rate = float(y.mean())
    precision = float(selected["y"].mean()) if len(selected) else 0.0
    strategy = simulate_regime_entry(
        full_frame,
        prediction,
        prediction_cutoff=float(cutoff),
        stop_loss=float(config.META_REGIME_ENTRY_STOP_LOSS),
        trade_cost=float(config.TRADE_COST),
    )
    baseline = simulate_regime_entry(
        full_frame,
        None,
        prediction_cutoff=float("-inf"),
        stop_loss=float(config.META_REGIME_ENTRY_STOP_LOSS),
        trade_cost=float(config.TRADE_COST),
    )
    strategy_net = _trade_net_mean(strategy.trades)
    baseline_net = _trade_net_mean(baseline.trades)
    delta = strategy_net - baseline_net
    scale = float(config.META_REGIME_ENTRY_RETURN_SCORE_SCALE)
    return RegimeEntrySplitMetrics(
        auc=auc,
        precision=precision,
        base_rate=base_rate,
        precision_excess=precision - base_rate,
        strategy_net_mean=strategy_net,
        baseline_net_mean=baseline_net,
        strategy_delta=delta,
        return_score=float(np.clip(strategy_net / scale, -1.0, 1.0)),
        delta_score=float(np.clip(delta / scale, -1.0, 1.0)),
        n_samples=len(aligned),
        n_selected=len(selected),
        n_trades=len(strategy.trades),
        n_baseline_trades=len(baseline.trades),
        n_stopped=strategy.stopped_trades,
        n_rejected=strategy.rejected_episodes,
        locked_rows=strategy.locked_rows,
    )


def _trade_net_mean(trades: pd.DataFrame) -> float:
    if trades.empty or "net_return" not in trades:
        return 0.0
    values = pd.to_numeric(trades["net_return"], errors="coerce").dropna()
    return float(values.mean()) if len(values) else 0.0


def _is_bad(metrics: RegimeEntrySplitMetrics) -> bool:
    return bool(
        metrics.auc <= config.BAD_AUC_THRESHOLD
        or metrics.precision_excess <= 0.0
        or metrics.strategy_net_mean <= 0.0
        or metrics.strategy_delta <= 0.0
        or metrics.n_trades <= 0
    )


def _final_split_fields(
    prefix: str,
    metrics: RegimeEntrySplitMetrics,
) -> dict[str, float]:
    return {
        f"{prefix}_mean_auc": metrics.auc,
        f"{prefix}_auc_edge": metrics.auc - 0.5,
        f"{prefix}_precision_at_trade": metrics.precision,
        f"{prefix}_meta_entry_precision": metrics.precision,
        f"{prefix}_base_rate": metrics.base_rate,
        f"{prefix}_precision_excess": metrics.precision_excess,
        f"{prefix}_trade_return_mean": metrics.strategy_net_mean,
        f"{prefix}_trade_return_score": metrics.return_score,
        f"{prefix}_baseline_trade_return_mean": metrics.baseline_net_mean,
        f"{prefix}_strategy_delta_mean": metrics.strategy_delta,
        f"{prefix}_strategy_delta_score": metrics.delta_score,
        f"{prefix}_selected_entries": float(metrics.n_selected),
        f"{prefix}_selected_rate": metrics.n_selected / max(metrics.n_samples, 1),
        f"{prefix}_strategy_trades": float(metrics.n_trades),
        f"{prefix}_baseline_trades": float(metrics.n_baseline_trades),
        f"{prefix}_stopped_trades": float(metrics.n_stopped),
        f"{prefix}_rejected_episodes": float(metrics.n_rejected),
        f"{prefix}_locked_rows": float(metrics.locked_rows),
    }
