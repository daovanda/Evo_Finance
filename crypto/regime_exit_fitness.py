"""State-machine fitness for ``meta_regime_exit``."""

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
from crypto.meta_regime_exit import simulate_regime_exit, top_fraction_cutoff


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeSplitMetrics:
    auc: float
    precision: float
    base_rate: float
    precision_excess: float
    strategy_net_mean: float
    baseline_net_mean: float
    strategy_delta: float
    return_score: float
    n_samples: int
    n_selected_exits: int
    n_trades: int
    n_early_exits: int
    locked_rows: int


class RegimeExitFitnessEvaluator(CryptoFitnessEvaluator):
    """Optimize exit classification and the executable episode strategy."""

    def __init__(self) -> None:
        super().__init__(horizons=[1], precision_only=False)

    def evaluate_walk_forward(
        self,
        individual: CryptoIndividual,
        folds: list[CryptoFold],
        feature_data: CryptoFeatureSpace | pd.DataFrame,
    ) -> float:
        if not isinstance(feature_data, CryptoFeatureSpace):
            raise ValueError("meta_regime_exit requires CryptoFeatureSpace.")
        rows: list[dict[str, float]] = []
        for fold in folds:
            row = self._evaluate_fold(individual, fold, feature_data)
            if row is not None:
                rows.append(row)
        if not rows:
            raise ValueError("No valid meta_regime_exit fold metrics were produced.")

        metrics_df = pd.DataFrame(rows)
        val_auc = metrics_df["val_auc"].astype(float)
        auc_edge = float((val_auc - 0.5).mean())
        precision_excess = float(metrics_df["val_precision_excess"].mean())
        return_score = float(metrics_df["val_trade_return_score"].mean())
        auc_std = float(val_auc.std(ddof=0)) if len(val_auc) > 1 else 0.0
        overfit_gap = float(metrics_df["overfit_gap"].mean())
        bad_ratio = float(metrics_df["bad_fold"].mean())
        weights = config.FITNESS_WEIGHTS
        score = (
            weights["auc_edge"] * auc_edge
            + weights["precision_excess"] * precision_excess
            + weights["trade_return_score"] * return_score
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
            "meta_exit_precision": float(metrics_df["val_precision"].mean()),
            "base_rate": float(metrics_df["val_base_rate"].mean()),
            "precision_excess": precision_excess,
            "trade_return_mean": float(metrics_df["val_strategy_net_mean"].mean()),
            "trade_return_score": return_score,
            "baseline_trade_return_mean": float(
                metrics_df["val_baseline_net_mean"].mean()
            ),
            "strategy_delta_mean": float(metrics_df["val_strategy_delta"].mean()),
            "early_exits": float(metrics_df["val_early_exits"].mean()),
            "locked_rows": float(metrics_df["val_locked_rows"].mean()),
            "strategy_trades": float(metrics_df["val_strategy_trades"].mean()),
            "selected_exit_decisions": float(
                metrics_df["val_selected_exits"].mean()
            ),
            "auc_std": auc_std,
            "overfit_gap": overfit_gap,
            "bad_fold_ratio": bad_ratio,
            "n_fold_horizon_scores": float(len(metrics_df)),
            "n_horizons": 1.0,
        }
        logger.info(
            "Regime-exit WF: score=%.4f | AUC=%.4f | precision_excess=%.4f | "
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
            raise ValueError("meta_regime_exit requires CryptoFeatureSpace.")
        train = _valid_candidates(train_df)
        val = _valid_candidates(val_df)
        test = _valid_candidates(test_df)
        if train.empty or val.empty or test.empty:
            raise ValueError("meta_regime_exit final split contains no candidates.")

        X_train = feature_data.matrix(individual.features, train.index)
        X_val = feature_data.matrix(individual.features, val.index)
        X_test = feature_data.matrix(individual.features, test.index)
        y_train = train["label_h1"].astype(int)
        y_val = val["label_h1"].astype(int)
        y_test = test["label_h1"].astype(int)
        if y_train.nunique() < 2:
            raise ValueError("meta_regime_exit final training label is constant.")

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
            "final_val_mean_auc": val_metrics.auc,
            "final_val_auc_edge": val_metrics.auc - 0.5,
            "final_val_precision_at_trade": val_metrics.precision,
            "final_val_meta_exit_precision": val_metrics.precision,
            "final_val_base_rate": val_metrics.base_rate,
            "final_val_precision_excess": val_metrics.precision_excess,
            "final_val_trade_return_mean": val_metrics.strategy_net_mean,
            "final_val_trade_return_score": val_metrics.return_score,
            "final_val_baseline_trade_return_mean": val_metrics.baseline_net_mean,
            "final_val_strategy_delta_mean": val_metrics.strategy_delta,
            "final_val_early_exits": float(val_metrics.n_early_exits),
            "final_val_locked_rows": float(val_metrics.locked_rows),
            "final_val_strategy_trades": float(val_metrics.n_trades),
            "final_val_selected_exit_decisions": float(
                val_metrics.n_selected_exits
            ),
            "final_val_selected_rate": (
                val_metrics.n_selected_exits / max(val_metrics.n_samples, 1)
            ),
            "final_val_auc_std": 0.0,
            "final_val_bad_ratio": float(
                val_metrics.auc <= config.BAD_AUC_THRESHOLD
                or val_metrics.precision_excess <= 0.0
                or val_metrics.return_score <= 0.0
            ),
            "final_val_overfit_gap": max(0.0, train_metrics.auc - val_metrics.auc),
            "final_test_mean_auc": test_metrics.auc,
            "final_test_auc_edge": test_metrics.auc - 0.5,
            "final_test_precision_at_trade": test_metrics.precision,
            "final_test_meta_exit_precision": test_metrics.precision,
            "final_test_base_rate": test_metrics.base_rate,
            "final_test_precision_excess": test_metrics.precision_excess,
            "final_test_trade_return_mean": test_metrics.strategy_net_mean,
            "final_test_trade_return_score": test_metrics.return_score,
            "final_test_baseline_trade_return_mean": test_metrics.baseline_net_mean,
            "final_test_strategy_delta_mean": test_metrics.strategy_delta,
            "final_test_early_exits": float(test_metrics.n_early_exits),
            "final_test_locked_rows": float(test_metrics.locked_rows),
            "final_test_strategy_trades": float(test_metrics.n_trades),
            "final_test_selected_exit_decisions": float(
                test_metrics.n_selected_exits
            ),
            "final_test_selected_rate": (
                test_metrics.n_selected_exits / max(test_metrics.n_samples, 1)
            ),
            "final_test_auc_std": 0.0,
            "final_test_bad_ratio": float(
                test_metrics.auc <= config.BAD_AUC_THRESHOLD
                or test_metrics.precision_excess <= 0.0
                or test_metrics.return_score <= 0.0
            ),
            "final_test_overfit_gap": max(0.0, train_metrics.auc - test_metrics.auc),
            "final_meta_exit_cutoff": float(cutoff),
        }
        individual.metrics.update(metrics)
        logger.info(
            "Regime-exit final: val net=%.4f%% (delta %.4f%%) | "
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
        bad = float(
            val_metrics.auc <= config.BAD_AUC_THRESHOLD
            or val_metrics.precision_excess <= 0.0
            or val_metrics.return_score <= 0.0
            or val_metrics.n_trades <= 0
        )
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
            "val_early_exits": float(val_metrics.n_early_exits),
            "val_locked_rows": float(val_metrics.locked_rows),
            "val_strategy_trades": float(val_metrics.n_trades),
            "val_selected_exits": float(val_metrics.n_selected_exits),
            "overfit_gap": max(0.0, train_metrics.auc - val_metrics.auc),
            "bad_fold": bad,
        }


def _valid_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if "label_h1" not in frame:
        raise ValueError("meta_regime_exit frame is missing label_h1.")
    return frame.dropna(subset=["label_h1"]).copy()


def _split_metrics(
    full_frame: pd.DataFrame,
    y_true: pd.Series,
    prediction: pd.Series,
    cutoff: float,
) -> RegimeSplitMetrics:
    aligned = pd.DataFrame({"y": y_true, "pred": prediction}).dropna()
    y = aligned["y"].astype(int)
    selected = aligned[aligned["pred"] >= float(cutoff)]
    auc = float(_binary_auc(y, aligned["pred"]))
    base_rate = float(y.mean())
    precision = float(selected["y"].mean()) if len(selected) else 0.0
    strategy = simulate_regime_exit(
        full_frame,
        prediction,
        prediction_cutoff=float(cutoff),
        trade_cost=float(config.TRADE_COST),
    )
    baseline = simulate_regime_exit(
        full_frame,
        None,
        prediction_cutoff=float("inf"),
        trade_cost=float(config.TRADE_COST),
    )
    strategy_net = _trade_net_mean(strategy.trades)
    baseline_net = _trade_net_mean(baseline.trades)
    return RegimeSplitMetrics(
        auc=auc,
        precision=precision,
        base_rate=base_rate,
        precision_excess=precision - base_rate,
        strategy_net_mean=strategy_net,
        baseline_net_mean=baseline_net,
        strategy_delta=strategy_net - baseline_net,
        return_score=float(
            np.clip(
                strategy_net / float(config.RETURN_SCORE_SCALE),
                -1.0,
                1.0,
            )
        ),
        n_samples=len(aligned),
        n_selected_exits=len(selected),
        n_trades=len(strategy.trades),
        n_early_exits=strategy.early_exits,
        locked_rows=strategy.locked_rows,
    )


def _trade_net_mean(trades: pd.DataFrame) -> float:
    if trades.empty or "net_return" not in trades:
        return 0.0
    values = pd.to_numeric(trades["net_return"], errors="coerce").dropna()
    return float(values.mean()) if len(values) else 0.0
