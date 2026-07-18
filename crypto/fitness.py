"""Binary walk-forward fitness for crypto evolution."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from crypto import config
from crypto.data import CryptoFold
from crypto.evolution import CryptoIndividual
from crypto.expression import CryptoFeatureSpace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitMetrics:
    auc: float
    precision_at_trade: float
    base_rate: float
    precision_excess: float
    trade_return_mean: float
    trade_return_score: float
    n_samples: int
    n_trades: int


class CryptoFitnessEvaluator:
    def __init__(
        self,
        horizons: list[int] | tuple[int, ...] = tuple(config.HOLDING_HORIZONS),
        lgbm_params: dict | None = None,
        num_boost_round: int = config.LGBM_NUM_BOOST_ROUND,
        early_stopping_rounds: int = config.LGBM_EARLY_STOPPING,
    ):
        self.horizons = [int(h) for h in horizons]
        self.lgbm_params = dict(lgbm_params or config.LGBM_PARAMS)
        self.num_boost_round = int(num_boost_round)
        self.early_stopping_rounds = int(early_stopping_rounds)

    def evaluate_walk_forward(
        self,
        individual: CryptoIndividual,
        folds: list[CryptoFold],
        feature_data: CryptoFeatureSpace | pd.DataFrame,
    ) -> float:
        feature_space = _feature_space_for(individual, feature_data)
        if config.FITNESS_HORIZON_MODE == "ensemble" and len(self.horizons) > 1:
            return self._evaluate_walk_forward_ensemble(
                individual=individual,
                folds=folds,
                feature_space=feature_space,
            )

        fold_rows: list[dict[str, float]] = []

        for horizon in self.horizons:
            label_col = f"label_h{horizon}"
            ret_col = f"future_return_h{horizon}"
            for fold in folds:
                row = self._evaluate_one_fold(
                    individual=individual,
                    fold=fold,
                    feature_space=feature_space,
                    label_col=label_col,
                    ret_col=ret_col,
                    horizon=horizon,
                )
                if row is not None:
                    fold_rows.append(row)

        if not fold_rows:
            raise ValueError("No valid crypto fold metrics were produced.")

        metrics_df = pd.DataFrame(fold_rows)
        val_auc = metrics_df["val_auc"].astype(float)
        auc_edge = float((val_auc - 0.5).mean())
        precision_excess = float(metrics_df["val_precision_excess"].mean())
        trade_return_score = float(metrics_df["val_trade_return_score"].mean())
        auc_std = float(val_auc.std(ddof=0)) if len(val_auc) > 1 else 0.0
        overfit_gap = float(metrics_df["overfit_gap"].mean())
        bad_fold_ratio = float(metrics_df["bad_fold"].mean())

        w = config.FITNESS_WEIGHTS
        score = (
            w["auc_edge"] * auc_edge
            + w["precision_excess"] * precision_excess
            + w["trade_return_score"] * trade_return_score
            + w["auc_std"] * auc_std
            + w["overfit_gap"] * overfit_gap
            + w["bad_fold_ratio"] * bad_fold_ratio
        )

        metrics = {
            "score": float(score),
            "mean_auc": float(val_auc.mean()),
            "auc_edge": auc_edge,
            "precision_at_trade": float(metrics_df["val_precision_at_trade"].mean()),
            "base_rate": float(metrics_df["val_base_rate"].mean()),
            "precision_excess": precision_excess,
            "trade_return_mean": float(metrics_df["val_trade_return_mean"].mean()),
            "trade_return_score": trade_return_score,
            "auc_std": auc_std,
            "overfit_gap": overfit_gap,
            "bad_fold_ratio": bad_fold_ratio,
            "n_fold_horizon_scores": float(len(metrics_df)),
            "n_horizons": float(len(self.horizons)),
        }
        for horizon in self.horizons:
            subset = metrics_df[metrics_df["horizon"] == float(horizon)]
            if subset.empty:
                continue
            metrics[f"h{horizon}_auc"] = float(subset["val_auc"].mean())
            metrics[f"h{horizon}_precision_excess"] = float(
                subset["val_precision_excess"].mean()
            )
            metrics[f"h{horizon}_trade_return_score"] = float(
                subset["val_trade_return_score"].mean()
            )

        individual.score = float(score)
        individual.metrics = metrics
        logger.info(
            "Crypto WF fitness: score=%.4f | AUC=%.4f | precision_excess=%.4f | "
            "ret_score=%.4f | std=%.4f | gap=%.4f | bad=%.2f | parts=%d",
            score,
            metrics["mean_auc"],
            precision_excess,
            trade_return_score,
            auc_std,
            overfit_gap,
            bad_fold_ratio,
            len(metrics_df),
        )
        return float(score)

    def _evaluate_walk_forward_ensemble(
        self,
        individual: CryptoIndividual,
        folds: list[CryptoFold],
        feature_space: CryptoFeatureSpace,
    ) -> float:
        exit_horizon = int(max(self.horizons))
        fold_rows: list[dict[str, float]] = []

        for fold in folds:
            row = self._evaluate_one_ensemble_fold(
                individual=individual,
                fold=fold,
                feature_space=feature_space,
                exit_horizon=exit_horizon,
            )
            if row is not None:
                fold_rows.append(row)

        if not fold_rows:
            raise ValueError("No valid crypto ensemble fold metrics were produced.")

        metrics_df = pd.DataFrame(fold_rows)
        val_auc = metrics_df["val_auc"].astype(float)
        auc_edge = float((val_auc - 0.5).mean())
        precision_excess = float(metrics_df["val_precision_excess"].mean())
        trade_return_score = float(metrics_df["val_trade_return_score"].mean())
        auc_std = float(val_auc.std(ddof=0)) if len(val_auc) > 1 else 0.0
        overfit_gap = float(metrics_df["overfit_gap"].mean())
        bad_fold_ratio = float(metrics_df["bad_fold"].mean())

        w = config.FITNESS_WEIGHTS
        score = (
            w["auc_edge"] * auc_edge
            + w["precision_excess"] * precision_excess
            + w["trade_return_score"] * trade_return_score
            + w["auc_std"] * auc_std
            + w["overfit_gap"] * overfit_gap
            + w["bad_fold_ratio"] * bad_fold_ratio
        )

        individual.score = float(score)
        individual.metrics = {
            "score": float(score),
            "fitness_horizon_mode": "ensemble",
            "ensemble_exit_horizon": float(exit_horizon),
            "mean_auc": float(val_auc.mean()),
            "auc_edge": auc_edge,
            "precision_at_trade": float(metrics_df["val_precision_at_trade"].mean()),
            "base_rate": float(metrics_df["val_base_rate"].mean()),
            "precision_excess": precision_excess,
            "trade_return_mean": float(metrics_df["val_trade_return_mean"].mean()),
            "trade_return_score": trade_return_score,
            "auc_std": auc_std,
            "overfit_gap": overfit_gap,
            "bad_fold_ratio": bad_fold_ratio,
            "n_fold_horizon_scores": float(len(metrics_df)),
            "n_horizons": float(len(self.horizons)),
            "ensemble_selected_rate": float(metrics_df["val_selected_rate"].mean()),
        }
        logger.info(
            "Crypto WF ensemble fitness: score=%.4f | AUC=%.4f | "
            "precision_excess=%.4f | ret_score=%.4f | std=%.4f | "
            "gap=%.4f | bad=%.2f | folds=%d",
            score,
            individual.metrics["mean_auc"],
            precision_excess,
            trade_return_score,
            auc_std,
            overfit_gap,
            bad_fold_ratio,
            len(metrics_df),
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
        """
        Train one final model per horizon and append final val/test metrics.

        The archive score is intentionally kept as the walk-forward score. Final
        metrics are diagnostic fields saved into the JSON after evolution ends.
        """
        feature_space = _feature_space_for(individual, feature_data)
        if config.FITNESS_HORIZON_MODE == "ensemble" and len(self.horizons) > 1:
            metrics = self._evaluate_final_ensemble(
                individual=individual,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_space=feature_space,
            )
            individual.metrics.update(metrics)
            logger.info(
                "Crypto final ensemble: WF score=%.4f | val_auc=%.4f | "
                "test_auc=%.4f | test_precision_excess=%.4f | "
                "test_ret_score=%.4f | test_gap=%.4f",
                float(individual.score) if individual.score is not None else float("nan"),
                metrics["final_val_mean_auc"],
                metrics["final_test_mean_auc"],
                metrics["final_test_precision_excess"],
                metrics["final_test_trade_return_score"],
                metrics["final_test_overfit_gap"],
            )
            return metrics

        rows: list[dict[str, float]] = []

        for horizon in self.horizons:
            label_col = f"label_h{horizon}"
            ret_col = f"future_return_h{horizon}"
            row = self._evaluate_one_final_horizon(
                individual=individual,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_space=feature_space,
                label_col=label_col,
                ret_col=ret_col,
                horizon=horizon,
            )
            if row is not None:
                rows.append(row)

        if not rows:
            raise ValueError("No valid crypto final metrics were produced.")

        final_df = pd.DataFrame(rows)
        metrics: dict[str, float] = {
            "final_n_horizon_scores": float(len(final_df)),
            "final_train_rows": float(final_df["train_n_samples"].mean()),
            "final_val_rows": float(final_df["val_n_samples"].mean()),
            "final_test_rows": float(final_df["test_n_samples"].mean()),
        }
        for split in ("val", "test"):
            auc = final_df[f"{split}_auc"].astype(float)
            metrics[f"final_{split}_mean_auc"] = float(auc.mean())
            metrics[f"final_{split}_auc_edge"] = float((auc - 0.5).mean())
            metrics[f"final_{split}_precision_at_trade"] = float(
                final_df[f"{split}_precision_at_trade"].mean()
            )
            metrics[f"final_{split}_base_rate"] = float(
                final_df[f"{split}_base_rate"].mean()
            )
            metrics[f"final_{split}_precision_excess"] = float(
                final_df[f"{split}_precision_excess"].mean()
            )
            metrics[f"final_{split}_trade_return_mean"] = float(
                final_df[f"{split}_trade_return_mean"].mean()
            )
            metrics[f"final_{split}_trade_return_score"] = float(
                final_df[f"{split}_trade_return_score"].mean()
            )
            metrics[f"final_{split}_auc_std"] = (
                float(auc.std(ddof=0)) if len(auc) > 1 else 0.0
            )
            metrics[f"final_{split}_bad_ratio"] = float(
                final_df[f"{split}_bad"].mean()
            )

        metrics["final_test_overfit_gap"] = float(
            final_df["test_overfit_gap"].mean()
        )
        metrics["final_val_overfit_gap"] = float(
            final_df["val_overfit_gap"].mean()
        )

        for horizon in self.horizons:
            subset = final_df[final_df["horizon"] == float(horizon)]
            if subset.empty:
                continue
            for split in ("val", "test"):
                metrics[f"final_h{horizon}_{split}_auc"] = float(
                    subset[f"{split}_auc"].mean()
                )
                metrics[f"final_h{horizon}_{split}_precision_excess"] = float(
                    subset[f"{split}_precision_excess"].mean()
                )
                metrics[f"final_h{horizon}_{split}_trade_return_score"] = float(
                    subset[f"{split}_trade_return_score"].mean()
                )
                metrics[f"final_h{horizon}_{split}_trade_return_mean"] = float(
                    subset[f"{split}_trade_return_mean"].mean()
                )

        individual.metrics.update(metrics)
        logger.info(
            "Crypto final: WF score=%.4f | val_auc=%.4f | test_auc=%.4f | "
            "test_precision_excess=%.4f | test_ret_score=%.4f | test_gap=%.4f",
            float(individual.score) if individual.score is not None else float("nan"),
            metrics["final_val_mean_auc"],
            metrics["final_test_mean_auc"],
            metrics["final_test_precision_excess"],
            metrics["final_test_trade_return_score"],
            metrics["final_test_overfit_gap"],
        )
        return metrics

    def _evaluate_final_ensemble(
        self,
        individual: CryptoIndividual,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_space: CryptoFeatureSpace,
    ) -> dict[str, float]:
        exit_horizon = int(max(self.horizons))
        train_preds: list[pd.Series] = []
        val_preds: list[pd.Series] = []
        test_preds: list[pd.Series] = []
        common_train_index: pd.Index | None = None
        common_val_index: pd.Index | None = None
        common_test_index: pd.Index | None = None

        for horizon in self.horizons:
            label_col = f"label_h{horizon}"
            ret_col = f"future_return_h{horizon}"
            train = _valid_labeled_frame(train_df, label_col, ret_col)
            val = _valid_labeled_frame(val_df, label_col, ret_col)
            test = _valid_labeled_frame(test_df, label_col, ret_col)
            if train.empty or val.empty or test.empty:
                continue

            X_train = feature_space.matrix(individual.features, train.index)
            X_val = feature_space.matrix(individual.features, val.index)
            X_test = feature_space.matrix(individual.features, test.index)
            y_train = train[label_col].astype(int)
            y_val = val[label_col].astype(int)

            if y_train.nunique() < 2 or y_val.nunique() < 2:
                continue

            booster = self._train_booster(X_train, y_train, X_val, y_val)
            train_pred = pd.Series(
                booster.predict(X_train),
                index=train.index,
                name=f"pred_h{horizon}",
            )
            val_pred = pd.Series(
                booster.predict(X_val),
                index=val.index,
                name=f"pred_h{horizon}",
            )
            test_pred = pd.Series(
                booster.predict(X_test),
                index=test.index,
                name=f"pred_h{horizon}",
            )
            booster.free_dataset()
            train_preds.append(train_pred)
            val_preds.append(val_pred)
            test_preds.append(test_pred)
            common_train_index = (
                train.index
                if common_train_index is None
                else common_train_index.intersection(train.index)
            )
            common_val_index = (
                val.index
                if common_val_index is None
                else common_val_index.intersection(val.index)
            )
            common_test_index = (
                test.index
                if common_test_index is None
                else common_test_index.intersection(test.index)
            )

        if len(train_preds) != len(self.horizons):
            raise ValueError("Could not train all horizon models for final ensemble.")

        exit_label_col = f"label_h{exit_horizon}"
        exit_ret_col = f"future_return_h{exit_horizon}"
        train_exit = _valid_labeled_frame(train_df, exit_label_col, exit_ret_col)
        val_exit = _valid_labeled_frame(val_df, exit_label_col, exit_ret_col)
        test_exit = _valid_labeled_frame(test_df, exit_label_col, exit_ret_col)

        train_pred_df, train_exit = _align_ensemble_predictions(
            train_preds,
            train_exit,
            common_train_index,
            exit_label_col,
            exit_ret_col,
        )
        val_pred_df, val_exit = _align_ensemble_predictions(
            val_preds,
            val_exit,
            common_val_index,
            exit_label_col,
            exit_ret_col,
        )
        test_pred_df, test_exit = _align_ensemble_predictions(
            test_preds,
            test_exit,
            common_test_index,
            exit_label_col,
            exit_ret_col,
        )
        if train_pred_df.empty or val_pred_df.empty or test_pred_df.empty:
            raise ValueError("No aligned rows for final ensemble metrics.")

        train_metrics = _ensemble_trade_metrics(
            y_true=train_exit[exit_label_col],
            pred_df=train_pred_df,
            future_return=train_exit[exit_ret_col],
        )
        val_thresholds = _ensemble_thresholds(val_pred_df)
        val_metrics = _ensemble_trade_metrics(
            y_true=val_exit[exit_label_col],
            pred_df=val_pred_df,
            future_return=val_exit[exit_ret_col],
            thresholds=val_thresholds,
        )
        test_metrics = _ensemble_trade_metrics(
            y_true=test_exit[exit_label_col],
            pred_df=test_pred_df,
            future_return=test_exit[exit_ret_col],
            thresholds=val_thresholds,
        )

        return {
            "final_horizon_mode": "ensemble",
            "final_ensemble_exit_horizon": float(exit_horizon),
            "final_n_horizon_scores": float(len(self.horizons)),
            "final_train_rows": float(train_metrics.n_samples),
            "final_val_rows": float(val_metrics.n_samples),
            "final_test_rows": float(test_metrics.n_samples),
            "final_val_mean_auc": float(val_metrics.auc),
            "final_val_auc_edge": float(val_metrics.auc - 0.5),
            "final_val_precision_at_trade": float(val_metrics.precision_at_trade),
            "final_val_base_rate": float(val_metrics.base_rate),
            "final_val_precision_excess": float(val_metrics.precision_excess),
            "final_val_trade_return_mean": float(val_metrics.trade_return_mean),
            "final_val_trade_return_score": float(val_metrics.trade_return_score),
            "final_val_auc_std": 0.0,
            "final_val_bad_ratio": float(
                val_metrics.auc <= config.BAD_AUC_THRESHOLD
                or val_metrics.precision_excess <= 0.0
                or val_metrics.trade_return_score <= 0.0
            ),
            "final_test_mean_auc": float(test_metrics.auc),
            "final_test_auc_edge": float(test_metrics.auc - 0.5),
            "final_test_precision_at_trade": float(test_metrics.precision_at_trade),
            "final_test_base_rate": float(test_metrics.base_rate),
            "final_test_precision_excess": float(test_metrics.precision_excess),
            "final_test_trade_return_mean": float(test_metrics.trade_return_mean),
            "final_test_trade_return_score": float(test_metrics.trade_return_score),
            "final_test_auc_std": 0.0,
            "final_test_bad_ratio": float(
                test_metrics.auc <= config.BAD_AUC_THRESHOLD
                or test_metrics.precision_excess <= 0.0
                or test_metrics.trade_return_score <= 0.0
            ),
            "final_val_overfit_gap": float(max(0.0, train_metrics.auc - val_metrics.auc)),
            "final_test_overfit_gap": float(max(0.0, train_metrics.auc - test_metrics.auc)),
            "final_val_selected_rate": (
                float(val_metrics.n_trades / val_metrics.n_samples)
                if val_metrics.n_samples
                else 0.0
            ),
            "final_test_selected_rate": (
                float(test_metrics.n_trades / test_metrics.n_samples)
                if test_metrics.n_samples
                else 0.0
            ),
        }

    def _evaluate_one_fold(
        self,
        individual: CryptoIndividual,
        fold: CryptoFold,
        feature_space: CryptoFeatureSpace,
        label_col: str,
        ret_col: str,
        horizon: int,
    ) -> dict[str, float] | None:
        train = _valid_labeled_frame(fold.train_df, label_col, ret_col)
        val = _valid_labeled_frame(fold.val_df, label_col, ret_col)
        if train.empty or val.empty:
            return None

        X_train = feature_space.matrix(individual.features, train.index)
        X_val = feature_space.matrix(individual.features, val.index)
        y_train = train[label_col].astype(int)
        y_val = val[label_col].astype(int)

        if y_train.nunique() < 2 or y_val.nunique() < 2:
            train_metrics = _neutral_metrics(y_train, train[ret_col])
            val_metrics = _neutral_metrics(y_val, val[ret_col])
        else:
            booster = self._train_booster(X_train, y_train, X_val, y_val)
            train_pred = pd.Series(booster.predict(X_train), index=train.index)
            val_pred = pd.Series(booster.predict(X_val), index=val.index)
            booster.free_dataset()
            train_metrics = _classification_trade_metrics(
                y_true=y_train,
                pred=train_pred,
                future_return=train[ret_col],
            )
            val_metrics = _classification_trade_metrics(
                y_true=y_val,
                pred=val_pred,
                future_return=val[ret_col],
            )

        overfit_gap = max(0.0, train_metrics.auc - val_metrics.auc)
        bad_fold = float(
            val_metrics.auc <= config.BAD_AUC_THRESHOLD
            or val_metrics.precision_excess <= 0.0
            or val_metrics.trade_return_score <= 0.0
        )
        return {
            "horizon": float(horizon),
            "train_auc": train_metrics.auc,
            "val_auc": val_metrics.auc,
            "val_precision_at_trade": val_metrics.precision_at_trade,
            "val_base_rate": val_metrics.base_rate,
            "val_precision_excess": val_metrics.precision_excess,
            "val_trade_return_mean": val_metrics.trade_return_mean,
            "val_trade_return_score": val_metrics.trade_return_score,
            "overfit_gap": overfit_gap,
            "bad_fold": bad_fold,
            "val_n_samples": float(val_metrics.n_samples),
            "val_n_trades": float(val_metrics.n_trades),
        }

    def _evaluate_one_ensemble_fold(
        self,
        individual: CryptoIndividual,
        fold: CryptoFold,
        feature_space: CryptoFeatureSpace,
        exit_horizon: int,
    ) -> dict[str, float] | None:
        train_preds: list[pd.Series] = []
        val_preds: list[pd.Series] = []
        common_train_index: pd.Index | None = None
        common_val_index: pd.Index | None = None

        for horizon in self.horizons:
            label_col = f"label_h{horizon}"
            ret_col = f"future_return_h{horizon}"
            train = _valid_labeled_frame(fold.train_df, label_col, ret_col)
            val = _valid_labeled_frame(fold.val_df, label_col, ret_col)
            if train.empty or val.empty:
                return None

            X_train = feature_space.matrix(individual.features, train.index)
            X_val = feature_space.matrix(individual.features, val.index)
            y_train = train[label_col].astype(int)
            y_val = val[label_col].astype(int)

            if y_train.nunique() < 2 or y_val.nunique() < 2:
                return None

            booster = self._train_booster(X_train, y_train, X_val, y_val)
            train_pred = pd.Series(
                booster.predict(X_train),
                index=train.index,
                name=f"pred_h{horizon}",
            )
            val_pred = pd.Series(
                booster.predict(X_val),
                index=val.index,
                name=f"pred_h{horizon}",
            )
            booster.free_dataset()
            train_preds.append(train_pred)
            val_preds.append(val_pred)
            common_train_index = (
                train.index
                if common_train_index is None
                else common_train_index.intersection(train.index)
            )
            common_val_index = (
                val.index
                if common_val_index is None
                else common_val_index.intersection(val.index)
            )

        exit_label_col = f"label_h{exit_horizon}"
        exit_ret_col = f"future_return_h{exit_horizon}"
        train_exit = _valid_labeled_frame(fold.train_df, exit_label_col, exit_ret_col)
        val_exit = _valid_labeled_frame(fold.val_df, exit_label_col, exit_ret_col)
        common_train_index = common_train_index.intersection(train_exit.index) if common_train_index is not None else train_exit.index
        common_val_index = common_val_index.intersection(val_exit.index) if common_val_index is not None else val_exit.index
        if len(common_train_index) == 0 or len(common_val_index) == 0:
            return None

        train_pred_df = pd.concat(train_preds, axis=1).reindex(common_train_index).dropna()
        val_pred_df = pd.concat(val_preds, axis=1).reindex(common_val_index).dropna()
        train_exit = train_exit.reindex(train_pred_df.index).dropna(subset=[exit_label_col, exit_ret_col])
        val_exit = val_exit.reindex(val_pred_df.index).dropna(subset=[exit_label_col, exit_ret_col])
        train_pred_df = train_pred_df.reindex(train_exit.index)
        val_pred_df = val_pred_df.reindex(val_exit.index)
        if train_pred_df.empty or val_pred_df.empty:
            return None

        train_metrics = _ensemble_trade_metrics(
            y_true=train_exit[exit_label_col],
            pred_df=train_pred_df,
            future_return=train_exit[exit_ret_col],
        )
        val_metrics = _ensemble_trade_metrics(
            y_true=val_exit[exit_label_col],
            pred_df=val_pred_df,
            future_return=val_exit[exit_ret_col],
        )

        overfit_gap = max(0.0, train_metrics.auc - val_metrics.auc)
        bad_fold = float(
            val_metrics.auc <= config.BAD_AUC_THRESHOLD
            or val_metrics.precision_excess <= 0.0
            or val_metrics.trade_return_score <= 0.0
            or val_metrics.n_trades <= 0
        )
        return {
            "horizon": float(exit_horizon),
            "train_auc": train_metrics.auc,
            "val_auc": val_metrics.auc,
            "val_precision_at_trade": val_metrics.precision_at_trade,
            "val_base_rate": val_metrics.base_rate,
            "val_precision_excess": val_metrics.precision_excess,
            "val_trade_return_mean": val_metrics.trade_return_mean,
            "val_trade_return_score": val_metrics.trade_return_score,
            "overfit_gap": overfit_gap,
            "bad_fold": bad_fold,
            "val_n_samples": float(val_metrics.n_samples),
            "val_n_trades": float(val_metrics.n_trades),
            "val_selected_rate": (
                float(val_metrics.n_trades / val_metrics.n_samples)
                if val_metrics.n_samples
                else 0.0
            ),
        }

    def _evaluate_one_final_horizon(
        self,
        individual: CryptoIndividual,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_space: CryptoFeatureSpace,
        label_col: str,
        ret_col: str,
        horizon: int,
    ) -> dict[str, float] | None:
        train = _valid_labeled_frame(train_df, label_col, ret_col)
        val = _valid_labeled_frame(val_df, label_col, ret_col)
        test = _valid_labeled_frame(test_df, label_col, ret_col)
        if train.empty or val.empty or test.empty:
            return None

        X_train = feature_space.matrix(individual.features, train.index)
        X_val = feature_space.matrix(individual.features, val.index)
        X_test = feature_space.matrix(individual.features, test.index)
        y_train = train[label_col].astype(int)
        y_val = val[label_col].astype(int)
        y_test = test[label_col].astype(int)

        if y_train.nunique() < 2:
            train_metrics = _neutral_metrics(y_train, train[ret_col])
            val_metrics = _neutral_metrics(y_val, val[ret_col])
            test_metrics = _neutral_metrics(y_test, test[ret_col])
        else:
            booster = self._train_booster_final(X_train, y_train, X_val, y_val)
            train_pred = pd.Series(booster.predict(X_train), index=train.index)
            val_pred = pd.Series(booster.predict(X_val), index=val.index)
            test_pred = pd.Series(booster.predict(X_test), index=test.index)
            booster.free_dataset()
            train_metrics = _classification_trade_metrics(
                y_true=y_train,
                pred=train_pred,
                future_return=train[ret_col],
            )
            val_metrics = _classification_trade_metrics(
                y_true=y_val,
                pred=val_pred,
                future_return=val[ret_col],
            )
            test_metrics = _classification_trade_metrics(
                y_true=y_test,
                pred=test_pred,
                future_return=test[ret_col],
            )

        val_bad = float(
            val_metrics.auc <= config.BAD_AUC_THRESHOLD
            or val_metrics.precision_excess <= 0.0
            or val_metrics.trade_return_score <= 0.0
        )
        test_bad = float(
            test_metrics.auc <= config.BAD_AUC_THRESHOLD
            or test_metrics.precision_excess <= 0.0
            or test_metrics.trade_return_score <= 0.0
        )
        return {
            "horizon": float(horizon),
            "train_auc": train_metrics.auc,
            "train_n_samples": float(train_metrics.n_samples),
            "val_auc": val_metrics.auc,
            "val_precision_at_trade": val_metrics.precision_at_trade,
            "val_base_rate": val_metrics.base_rate,
            "val_precision_excess": val_metrics.precision_excess,
            "val_trade_return_mean": val_metrics.trade_return_mean,
            "val_trade_return_score": val_metrics.trade_return_score,
            "val_n_samples": float(val_metrics.n_samples),
            "val_n_trades": float(val_metrics.n_trades),
            "val_bad": val_bad,
            "val_overfit_gap": max(0.0, train_metrics.auc - val_metrics.auc),
            "test_auc": test_metrics.auc,
            "test_precision_at_trade": test_metrics.precision_at_trade,
            "test_base_rate": test_metrics.base_rate,
            "test_precision_excess": test_metrics.precision_excess,
            "test_trade_return_mean": test_metrics.trade_return_mean,
            "test_trade_return_score": test_metrics.trade_return_score,
            "test_n_samples": float(test_metrics.n_samples),
            "test_n_trades": float(test_metrics.n_trades),
            "test_bad": test_bad,
            "test_overfit_gap": max(0.0, train_metrics.auc - test_metrics.auc),
        }

    def _train_booster(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> lgb.Booster:
        split = _internal_early_stop_split(X_train, y_train)
        callbacks = [lgb.log_evaluation(period=-1)]
        valid_sets = None
        if split is None or self.early_stopping_rounds <= 0:
            train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
        else:
            X_fit, y_fit, X_stop, y_stop = split
            train_set = lgb.Dataset(X_fit, label=y_fit, free_raw_data=True)
            stop_set = lgb.Dataset(
                X_stop,
                label=y_stop,
                reference=train_set,
                free_raw_data=True,
            )
            valid_sets = [stop_set]
            callbacks.insert(
                0,
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
            )
        return lgb.train(
            params=dict(self.lgbm_params),
            train_set=train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )

    def _train_booster_final(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> lgb.Booster:
        callbacks = [lgb.log_evaluation(period=-1)]
        valid_sets = None
        train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
        if (
            self.early_stopping_rounds > 0
            and len(X_val) > 0
            and y_val.nunique() >= 2
        ):
            val_set = lgb.Dataset(
                X_val,
                label=y_val,
                reference=train_set,
                free_raw_data=True,
            )
            valid_sets = [val_set]
            callbacks.insert(
                0,
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
            )
        return lgb.train(
            params=dict(self.lgbm_params),
            train_set=train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )


def _valid_labeled_frame(df: pd.DataFrame, label_col: str, ret_col: str) -> pd.DataFrame:
    if label_col not in df.columns or ret_col not in df.columns:
        raise ValueError(f"Missing label/return columns: {label_col}, {ret_col}")
    return df.dropna(subset=[label_col, ret_col]).copy()


def _feature_space_for(
    individual: CryptoIndividual,
    feature_data: CryptoFeatureSpace | pd.DataFrame,
) -> CryptoFeatureSpace:
    if isinstance(feature_data, CryptoFeatureSpace):
        return feature_data
    base_features = [feature for feature in individual.features if feature in feature_data.columns]
    if len(base_features) != len(individual.features):
        missing = sorted(set(individual.features) - set(base_features))
        raise ValueError(
            "Generated crypto expressions require CryptoFeatureSpace; "
            f"unknown columns: {missing[:5]}"
        )
    return CryptoFeatureSpace(feature_data, base_features)


def _align_ensemble_predictions(
    pred_series: list[pd.Series],
    exit_frame: pd.DataFrame,
    common_index: pd.Index | None,
    label_col: str,
    ret_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if common_index is None or len(pred_series) == 0:
        return pd.DataFrame(), pd.DataFrame()
    pred_df = pd.concat(pred_series, axis=1).reindex(common_index).dropna()
    aligned_exit = exit_frame.reindex(pred_df.index).dropna(subset=[label_col, ret_col])
    pred_df = pred_df.reindex(aligned_exit.index).dropna()
    aligned_exit = aligned_exit.reindex(pred_df.index).dropna(subset=[label_col, ret_col])
    return pred_df, aligned_exit


def _classification_trade_metrics(
    y_true: pd.Series,
    pred: pd.Series,
    future_return: pd.Series,
) -> SplitMetrics:
    data = (
        pd.DataFrame({"y": y_true, "pred": pred, "ret": future_return})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if data.empty:
        return SplitMetrics(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    y = data["y"].astype(int)
    auc = _binary_auc(y, data["pred"])
    base_rate = float(y.mean())
    n_trades = min(
        len(data),
        max(int(config.MIN_TRADES_PER_SPLIT), int(np.ceil(len(data) * config.TRADE_TOP_FRACTION))),
    )
    traded = data.nlargest(n_trades, "pred")
    precision = float(traded["y"].mean()) if n_trades else 0.0
    net_return = traded["ret"].astype(float) - float(config.TRADE_COST)
    trade_return_mean = float(net_return.mean()) if len(net_return) else 0.0
    trade_return_score = float(
        np.clip(trade_return_mean / float(config.RETURN_SCORE_SCALE), -1.0, 1.0)
    )
    return SplitMetrics(
        auc=float(auc),
        precision_at_trade=precision,
        base_rate=base_rate,
        precision_excess=precision - base_rate,
        trade_return_mean=trade_return_mean,
        trade_return_score=trade_return_score,
        n_samples=int(len(data)),
        n_trades=int(n_trades),
    )


def _ensemble_trade_metrics(
    y_true: pd.Series,
    pred_df: pd.DataFrame,
    future_return: pd.Series,
    thresholds: dict[str, float] | None = None,
) -> SplitMetrics:
    data = (
        pd.concat(
            [
                pd.Series(y_true, name="y"),
                pd.Series(future_return, name="ret"),
                pred_df,
            ],
            axis=1,
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if data.empty:
        return SplitMetrics(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    pred_cols = [col for col in pred_df.columns if col in data.columns]
    if not pred_cols:
        return SplitMetrics(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, len(data), 0)

    y = data["y"].astype(int)
    score = data[pred_cols].astype(float).mean(axis=1)
    auc = _binary_auc(y, score)
    base_rate = float(y.mean())
    selected_mask = pd.Series(True, index=data.index)
    selected_thresholds = thresholds or _ensemble_thresholds(data[pred_cols])
    for col in pred_cols:
        threshold = selected_thresholds.get(col)
        if threshold is None:
            selected_mask &= False
        else:
            selected_mask &= data[col].astype(float) >= float(threshold)

    traded = data[selected_mask]
    precision = float(traded["y"].mean()) if len(traded) else 0.0
    net_return = traded["ret"].astype(float) - float(config.TRADE_COST)
    trade_return_mean = float(net_return.mean()) if len(net_return) else 0.0
    trade_return_score = float(
        np.clip(trade_return_mean / float(config.RETURN_SCORE_SCALE), -1.0, 1.0)
    )
    return SplitMetrics(
        auc=float(auc),
        precision_at_trade=precision,
        base_rate=base_rate,
        precision_excess=precision - base_rate,
        trade_return_mean=trade_return_mean,
        trade_return_score=trade_return_score,
        n_samples=int(len(data)),
        n_trades=int(len(traded)),
    )


def _ensemble_thresholds(pred_df: pd.DataFrame) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for col in pred_df.columns:
        pred = (
            pd.to_numeric(pred_df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if pred.empty:
            continue
        n_select = min(
            len(pred),
            max(
                int(config.MIN_TRADES_PER_SPLIT),
                int(np.ceil(len(pred) * config.TRADE_TOP_FRACTION)),
            ),
        )
        thresholds[str(col)] = float(pred.nlargest(n_select).min())
    return thresholds


def _internal_early_stop_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series] | None:
    """
    Split train chronologically for early stopping without touching fold val.

    Returning None intentionally disables early stopping for tiny/degenerate
    train windows instead of leaking validation labels into model selection.
    """
    n = len(X_train)
    if n < 4:
        return None
    frac = float(config.EARLY_STOP_VALID_FRACTION)
    if frac <= 0.0:
        return None
    n_stop = max(int(config.EARLY_STOP_MIN_VALID_SAMPLES), int(np.ceil(n * frac)))
    n_stop = min(n_stop, n // 2)
    if n_stop < 1:
        return None
    split_pos = n - n_stop
    X_fit = X_train.iloc[:split_pos]
    y_fit = y_train.iloc[:split_pos]
    X_stop = X_train.iloc[split_pos:]
    y_stop = y_train.iloc[split_pos:]
    if len(X_fit) == 0 or len(X_stop) == 0:
        return None
    if y_fit.nunique() < 2 or y_stop.nunique() < 2:
        return None
    return X_fit, y_fit, X_stop, y_stop


def _neutral_metrics(y_true: pd.Series, future_return: pd.Series) -> SplitMetrics:
    base_rate = float(pd.to_numeric(y_true, errors="coerce").dropna().mean() or 0.0)
    mean_return = float(
        pd.to_numeric(future_return, errors="coerce").dropna().mean() or 0.0
    ) - float(config.TRADE_COST)
    return SplitMetrics(
        auc=0.5,
        precision_at_trade=base_rate,
        base_rate=base_rate,
        precision_excess=0.0,
        trade_return_mean=mean_return,
        trade_return_score=float(np.clip(mean_return / config.RETURN_SCORE_SCALE, -1.0, 1.0)),
        n_samples=int(len(y_true)),
        n_trades=0,
    )


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
