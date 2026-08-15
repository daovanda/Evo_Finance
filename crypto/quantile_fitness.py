"""Prediction-only walk-forward fitness for ``LABEL_MODE=quantile_trade``.

The mode trains one LightGBM quantile regressor per horizon. The requested
target is upward MFE, downward MAE, or signed final-close return. Fitness measures
quantile accuracy only; it never constructs a TP, SL, EV, or trade signal.
"""

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
from crypto.fitness import _feature_space_for

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuantileAccuracyMetrics:
    pinball_loss: float
    pinball_score: float
    baseline_pinball_loss: float
    baseline_pinball_score: float
    pinball_skill: float
    coverage: float
    coverage_error: float
    spearman_ic: float
    direction_accuracy: float
    direction_baseline: float
    direction_edge: float
    mae: float
    rmse: float
    prediction_mean: float
    target_quantile: float
    target_mean: float
    n_samples: int


class QuantileFitnessEvaluator:
    """Quantile-accuracy evaluator with the standard crypto evaluator contract."""

    def __init__(
        self,
        horizons: list[int] | tuple[int, ...] = tuple(config.HOLDING_HORIZONS),
        target: str = config.QUANTILE_TARGET,
        quantile: float = config.QUANTILE_ALPHA,
        lgbm_params: dict | None = None,
        num_boost_round: int = config.LGBM_NUM_BOOST_ROUND,
        early_stopping_rounds: int = config.LGBM_EARLY_STOPPING,
    ) -> None:
        self.horizons = [int(h) for h in horizons]
        if not self.horizons or any(h < 1 for h in self.horizons):
            raise ValueError("quantile_trade horizons must be positive integers.")
        self.target = config.canonical_quantile_target(target)
        self.quantile = config.validate_quantile_alpha(quantile)
        self.lgbm_params = dict(lgbm_params or config.QUANTILE_TRADE_LGBM_PARAMS)
        self.num_boost_round = int(num_boost_round)
        self.early_stopping_rounds = int(early_stopping_rounds)

    @property
    def target_column_prefix(self) -> str:
        source = {
            "mfe": "up_mfe",
            "mae": "down_mfe",
            "close": "close_return",
        }[self.target]
        return f"quantile_{source}_h"

    def target_column(self, horizon: int) -> str:
        return f"{self.target_column_prefix}{int(horizon)}"

    def evaluate_walk_forward(
        self,
        individual: CryptoIndividual,
        folds: list[CryptoFold],
        feature_data: CryptoFeatureSpace | pd.DataFrame,
    ) -> float:
        feature_space = _feature_space_for(individual, feature_data)
        rows: list[dict[str, float]] = []
        for fold in folds:
            for horizon in self.horizons:
                row = self._evaluate_one_fold(individual, fold, feature_space, horizon)
                if row is not None:
                    rows.append(row)
        if not rows:
            raise ValueError("No valid quantile_trade fold metrics were produced.")

        metrics_df = pd.DataFrame(rows)
        pinball_mean = float(metrics_df["val_pinball_score"].mean())
        pinball_skill = float(metrics_df["val_pinball_skill"].mean())
        pinball_skill_std = float(metrics_df["val_pinball_skill"].std(ddof=0))
        coverage_error = float(metrics_df["val_coverage_error"].mean())
        spearman_ic = float(metrics_df["val_spearman_ic"].mean())
        direction_accuracy = float(metrics_df["val_direction_accuracy"].mean())
        direction_baseline = float(metrics_df["val_direction_baseline"].mean())
        direction_edge = float(metrics_df["val_direction_edge"].mean())
        overfit_gap = float(metrics_df["overfit_gap"].mean())
        bad_fold_ratio = float(metrics_df["bad_fold"].mean())
        w = config.quantile_trade_fitness_weights(self.target)
        score = (
            w["quantile_pinball_skill"] * pinball_skill
            + w["quantile_coverage_error"] * coverage_error
            + w["quantile_spearman_ic"] * spearman_ic
            + w["quantile_pinball_skill_std"] * pinball_skill_std
            + w["overfit_gap"] * overfit_gap
            + w["bad_fold_ratio"] * bad_fold_ratio
        )
        metrics: dict[str, float | str] = {
            "score": float(score),
            "fitness_type": "quantile_accuracy",
            "fitness_horizon_mode": "mean",
            "quantile_target": self.target,
            "quantile_alpha": self.quantile,
            "quantile_pinball_loss": float(metrics_df["val_pinball_loss"].mean()),
            "quantile_pinball_score": pinball_mean,
            "quantile_baseline_pinball_loss": float(
                metrics_df["val_baseline_pinball_loss"].mean()
            ),
            "quantile_baseline_pinball_score": float(
                metrics_df["val_baseline_pinball_score"].mean()
            ),
            "quantile_pinball_skill": pinball_skill,
            "quantile_pinball_skill_std": pinball_skill_std,
            "quantile_coverage": float(metrics_df["val_coverage"].mean()),
            "quantile_coverage_error": coverage_error,
            "quantile_spearman_ic": spearman_ic,
            "quantile_mae": float(metrics_df["val_mae"].mean()),
            "quantile_rmse": float(metrics_df["val_rmse"].mean()),
            "quantile_prediction_mean": float(metrics_df["val_prediction_mean"].mean()),
            "quantile_target_quantile": float(
                metrics_df["val_target_quantile"].mean()
            ),
            "quantile_target_mean": float(metrics_df["val_target_mean"].mean()),
            "overfit_gap": overfit_gap,
            "bad_fold_ratio": bad_fold_ratio,
            "n_fold_horizon_scores": float(len(metrics_df)),
            "n_horizons": float(len(self.horizons)),
        }
        if self.target == "close":
            metrics["quantile_direction_accuracy"] = direction_accuracy
            metrics["quantile_direction_baseline"] = direction_baseline
            metrics["quantile_direction_edge"] = direction_edge
        for horizon in self.horizons:
            subset = metrics_df[metrics_df["horizon"] == float(horizon)]
            if subset.empty:
                continue
            metrics[f"h{horizon}_quantile_pinball_score"] = float(
                subset["val_pinball_score"].mean()
            )
            metrics[f"h{horizon}_quantile_pinball_skill"] = float(
                subset["val_pinball_skill"].mean()
            )
            metrics[f"h{horizon}_quantile_coverage_error"] = float(
                subset["val_coverage_error"].mean()
            )
            metrics[f"h{horizon}_quantile_spearman_ic"] = float(
                subset["val_spearman_ic"].mean()
            )
            if self.target == "close":
                metrics[f"h{horizon}_quantile_direction_accuracy"] = float(
                    subset["val_direction_accuracy"].mean()
                )
                metrics[f"h{horizon}_quantile_direction_baseline"] = float(
                    subset["val_direction_baseline"].mean()
                )
                metrics[f"h{horizon}_quantile_direction_edge"] = float(
                    subset["val_direction_edge"].mean()
                )

        individual.score = float(score)
        individual.metrics = metrics
        logger.info(
            "Crypto quantile WF fitness: score=%.4f | target=%s Q%.0f | "
            "pinball=%.5f | baseline=%.5f | skill=%+.2f%% | "
            "coverage=%.2f%% (error=%.2f%%) | IC=%.4f%s | MAE=%.4f%% | "
            "RMSE=%.4f%% | pred_mean=%+.4f%% | actual_q%.0f=%+.4f%% | "
            "skill_std=%.2f%% | gap=%.2f%% | bad=%.2f | parts=%d",
            score,
            self.target.upper(),
            100.0 * self.quantile,
            pinball_mean,
            float(metrics["quantile_baseline_pinball_score"]),
            100.0 * pinball_skill,
            100.0 * float(metrics["quantile_coverage"]),
            100.0 * coverage_error,
            spearman_ic,
            (
                f" | direction_acc={100.0 * direction_accuracy:.2f}%"
                f" baseline={100.0 * direction_baseline:.2f}%"
                f" edge={100.0 * direction_edge:+.2f}%"
                if self.target == "close"
                else ""
            ),
            100.0 * float(metrics["quantile_mae"]),
            100.0 * float(metrics["quantile_rmse"]),
            100.0 * float(metrics["quantile_prediction_mean"]),
            100.0 * self.quantile,
            100.0 * float(metrics["quantile_target_quantile"]),
            100.0 * pinball_skill_std,
            100.0 * overfit_gap,
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
    ) -> dict[str, float | str]:
        feature_space = _feature_space_for(individual, feature_data)
        rows = [
            self._evaluate_final_horizon(
                individual, train_df, val_df, test_df, feature_space, horizon
            )
            for horizon in self.horizons
        ]
        rows = [row for row in rows if row is not None]
        if not rows:
            raise ValueError("No valid quantile_trade final metrics were produced.")
        final_df = pd.DataFrame(rows)
        metrics: dict[str, float | str] = {
            "final_fitness_type": "quantile_accuracy",
            "final_horizon_mode": "mean",
            "final_quantile_target": self.target,
            "final_quantile_alpha": self.quantile,
            "final_n_horizon_scores": float(len(final_df)),
            "final_train_rows": float(final_df["train_n_samples"].mean()),
            "final_val_rows": float(final_df["val_n_samples"].mean()),
            "final_test_rows": float(final_df["test_n_samples"].mean()),
        }
        names = (
            "pinball_loss",
            "pinball_score",
            "baseline_pinball_loss",
            "baseline_pinball_score",
            "pinball_skill",
            "coverage",
            "coverage_error",
            "spearman_ic",
            "mae",
            "rmse",
            "prediction_mean",
            "target_quantile",
            "target_mean",
        )
        if self.target == "close":
            names += (
                "direction_accuracy",
                "direction_baseline",
                "direction_edge",
            )
        for split in ("val", "test"):
            for name in names:
                metrics[f"final_{split}_quantile_{name}"] = float(
                    final_df[f"{split}_{name}"].mean()
                )
            metrics[f"final_{split}_overfit_gap"] = float(
                final_df[f"{split}_overfit_gap"].mean()
            )
            metrics[f"final_{split}_bad_ratio"] = float(final_df[f"{split}_bad"].mean())
        for horizon in self.horizons:
            subset = final_df[final_df["horizon"] == float(horizon)]
            if subset.empty:
                continue
            for split in ("val", "test"):
                for name in (
                    "pinball_score",
                    "pinball_skill",
                    "coverage_error",
                    "spearman_ic",
                    *(
                        (
                            "direction_accuracy",
                            "direction_baseline",
                            "direction_edge",
                        )
                        if self.target == "close"
                        else ()
                    ),
                ):
                    metrics[f"final_h{horizon}_{split}_quantile_{name}"] = float(
                        subset[f"{split}_{name}"].mean()
                    )
        individual.metrics.update(metrics)
        logger.info(
            "Crypto quantile final: WF score=%.4f | val pinball=%.5f skill=%+.2f%% "
            "coverage=%.2f%% IC=%.4f%s pred_mean=%+.4f%% actual_q%.0f=%+.4f%% | "
            "test pinball=%.5f skill=%+.2f%% coverage=%.2f%% IC=%.4f "
            "pred_mean=%+.4f%% actual_q%.0f=%+.4f%%%s",
            float(individual.score) if individual.score is not None else float("nan"),
            float(metrics["final_val_quantile_pinball_score"]),
            100.0 * float(metrics["final_val_quantile_pinball_skill"]),
            100.0 * float(metrics["final_val_quantile_coverage"]),
            float(metrics["final_val_quantile_spearman_ic"]),
            (
                f" direction_acc={100.0 * float(metrics['final_val_quantile_direction_accuracy']):.2f}%"
                f" baseline={100.0 * float(metrics['final_val_quantile_direction_baseline']):.2f}%"
                f" edge={100.0 * float(metrics['final_val_quantile_direction_edge']):+.2f}%"
                if self.target == "close"
                else ""
            ),
            100.0 * float(metrics["final_val_quantile_prediction_mean"]),
            100.0 * self.quantile,
            100.0 * float(metrics["final_val_quantile_target_quantile"]),
            float(metrics["final_test_quantile_pinball_score"]),
            100.0 * float(metrics["final_test_quantile_pinball_skill"]),
            100.0 * float(metrics["final_test_quantile_coverage"]),
            float(metrics["final_test_quantile_spearman_ic"]),
            100.0 * float(metrics["final_test_quantile_prediction_mean"]),
            100.0 * self.quantile,
            100.0 * float(metrics["final_test_quantile_target_quantile"]),
            (
                f" | direction_acc={100.0 * float(metrics['final_test_quantile_direction_accuracy']):.2f}%"
                f" baseline={100.0 * float(metrics['final_test_quantile_direction_baseline']):.2f}%"
                f" edge={100.0 * float(metrics['final_test_quantile_direction_edge']):+.2f}%"
                if self.target == "close"
                else ""
            ),
        )
        return metrics

    def _evaluate_one_fold(
        self,
        individual: CryptoIndividual,
        fold: CryptoFold,
        feature_space: CryptoFeatureSpace,
        horizon: int,
    ) -> dict[str, float] | None:
        train = self._valid_frame(fold.train_df, horizon)
        val = self._valid_frame(fold.val_df, horizon)
        if train.empty or val.empty:
            return None
        train_pred, val_pred, _ = self._fit_predict(
            individual, feature_space, horizon, train, val
        )
        train_target = train[self.target_column(horizon)]
        baseline_value = float(train_target.quantile(self.quantile))
        train_metrics = self._accuracy(train_target, train_pred, baseline_value)
        val_metrics = self._accuracy(
            val[self.target_column(horizon)], val_pred, baseline_value
        )
        return _fold_row(horizon, train_metrics, val_metrics)

    def _evaluate_final_horizon(
        self,
        individual: CryptoIndividual,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_space: CryptoFeatureSpace,
        horizon: int,
    ) -> dict[str, float] | None:
        train = self._valid_frame(train_df, horizon)
        val = self._valid_frame(val_df, horizon)
        test = self._valid_frame(test_df, horizon)
        if train.empty or val.empty or test.empty:
            return None
        train_pred, val_pred, test_pred = self._fit_predict(
            individual, feature_space, horizon, train, val, test
        )
        assert test_pred is not None
        train_target = train[self.target_column(horizon)]
        baseline_value = float(train_target.quantile(self.quantile))
        return _final_row(
            horizon,
            self._accuracy(train_target, train_pred, baseline_value),
            self._accuracy(val[self.target_column(horizon)], val_pred, baseline_value),
            self._accuracy(
                test[self.target_column(horizon)], test_pred, baseline_value
            ),
        )

    def _fit_predict(
        self,
        individual: CryptoIndividual,
        feature_space: CryptoFeatureSpace,
        horizon: int,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame | None = None,
    ) -> tuple[pd.Series, pd.Series, pd.Series | None]:
        X_train = feature_space.matrix(individual.features, train.index)
        X_val = feature_space.matrix(individual.features, val.index)
        X_test = (
            feature_space.matrix(individual.features, test.index)
            if test is not None
            else None
        )
        y_train = train[self.target_column(horizon)].astype(float)
        booster = self._train_booster(X_train, y_train, horizon=horizon)
        train_pred = pd.Series(booster.predict(X_train), index=train.index)
        val_pred = pd.Series(booster.predict(X_val), index=val.index)
        test_pred = (
            pd.Series(booster.predict(X_test), index=test.index)
            if X_test is not None and test is not None
            else None
        )
        booster.free_dataset()
        return train_pred, val_pred, test_pred

    def _train_booster(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        horizon: int,
    ) -> lgb.Booster:
        params = dict(self.lgbm_params)
        params.update(objective="quantile", metric="quantile", alpha=self.quantile)
        callbacks = [lgb.log_evaluation(period=-1)]
        valid_sets = None
        split = _internal_quantile_stop_split(
            X_train,
            y_train,
            purge_bars=max(int(horizon), 0),
        )
        if split is None or self.early_stopping_rounds <= 0:
            train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
        else:
            X_fit, y_fit, X_stop, y_stop = split
            train_set = lgb.Dataset(X_fit, label=y_fit, free_raw_data=True)
            stop_set = lgb.Dataset(
                X_stop, label=y_stop, reference=train_set, free_raw_data=True
            )
            valid_sets = [stop_set]
            callbacks.insert(
                0, lgb.early_stopping(self.early_stopping_rounds, verbose=False)
            )
        return lgb.train(
            params=params,
            train_set=train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )

    def _valid_frame(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        column = self.target_column(horizon)
        if column not in df.columns:
            raise ValueError(f"Missing quantile_trade target column: {column}")
        return df.dropna(subset=[column]).copy()

    def _accuracy(
        self,
        actual: pd.Series,
        prediction: pd.Series,
        baseline_value: float,
    ) -> QuantileAccuracyMetrics:
        actual_values = actual.to_numpy(float)
        predicted_values = prediction.reindex(actual.index).to_numpy(float)
        error = actual_values - predicted_values
        pinball = np.maximum(self.quantile * error, (self.quantile - 1.0) * error)
        baseline_error = actual_values - float(baseline_value)
        baseline_pinball = np.maximum(
            self.quantile * baseline_error,
            (self.quantile - 1.0) * baseline_error,
        )
        coverage = float(np.mean(actual_values <= predicted_values))
        rank_ic = (
            pd.Series(predicted_values)
            .rank(method="average")
            .corr(pd.Series(actual_values).rank(method="average"))
        )
        scale = max(float(config.RETURN_SCORE_SCALE), np.finfo(float).eps)
        pinball_loss = float(np.mean(pinball))
        baseline_pinball_loss = float(np.mean(baseline_pinball))
        skill_denominator = max(baseline_pinball_loss, np.finfo(float).eps)
        actual_nonnegative = actual_values >= 0.0
        direction_accuracy = float(
            np.mean((predicted_values >= 0.0) == actual_nonnegative)
        )
        actual_up_rate = float(np.mean(actual_nonnegative))
        direction_baseline = max(actual_up_rate, 1.0 - actual_up_rate)
        return QuantileAccuracyMetrics(
            pinball_loss=pinball_loss,
            pinball_score=float(pinball_loss / scale),
            baseline_pinball_loss=baseline_pinball_loss,
            baseline_pinball_score=float(baseline_pinball_loss / scale),
            pinball_skill=float(1.0 - pinball_loss / skill_denominator),
            coverage=coverage,
            coverage_error=abs(coverage - self.quantile),
            spearman_ic=float(rank_ic) if np.isfinite(rank_ic) else 0.0,
            direction_accuracy=direction_accuracy,
            direction_baseline=direction_baseline,
            direction_edge=direction_accuracy - direction_baseline,
            mae=float(np.mean(np.abs(error))),
            rmse=float(np.sqrt(np.mean(np.square(error)))),
            prediction_mean=float(np.mean(predicted_values)),
            target_quantile=float(np.quantile(actual_values, self.quantile)),
            target_mean=float(np.mean(actual_values)),
            n_samples=len(actual_values),
        )


# Backward-compatible import name used by older local callers.
QuantileTradeFitnessEvaluator = QuantileFitnessEvaluator


def _fold_row(
    horizon: int,
    train: QuantileAccuracyMetrics,
    val: QuantileAccuracyMetrics,
) -> dict[str, float]:
    overfit_gap = max(0.0, train.pinball_skill - val.pinball_skill)
    bad = float(
        val.pinball_skill <= 0.0
        or val.coverage_error > float(config.QUANTILE_BAD_COVERAGE_ERROR)
        or val.spearman_ic <= float(config.QUANTILE_BAD_SPEARMAN_IC)
    )
    row = {"horizon": float(horizon), "overfit_gap": overfit_gap, "bad_fold": bad}
    row.update(_metric_values("train", train))
    row.update(_metric_values("val", val))
    return row


def _final_row(
    horizon: int,
    train: QuantileAccuracyMetrics,
    val: QuantileAccuracyMetrics,
    test: QuantileAccuracyMetrics,
) -> dict[str, float]:
    row: dict[str, float] = {"horizon": float(horizon)}
    row.update(_metric_values("train", train))
    row.update(_metric_values("val", val))
    row.update(_metric_values("test", test))
    for split, metrics in (("val", val), ("test", test)):
        row[f"{split}_overfit_gap"] = max(
            0.0, train.pinball_skill - metrics.pinball_skill
        )
        row[f"{split}_bad"] = float(
            metrics.pinball_skill <= 0.0
            or metrics.coverage_error > float(config.QUANTILE_BAD_COVERAGE_ERROR)
            or metrics.spearman_ic <= float(config.QUANTILE_BAD_SPEARMAN_IC)
        )
    return row


def _metric_values(prefix: str, metrics: QuantileAccuracyMetrics) -> dict[str, float]:
    return {
        f"{prefix}_pinball_loss": metrics.pinball_loss,
        f"{prefix}_pinball_score": metrics.pinball_score,
        f"{prefix}_baseline_pinball_loss": metrics.baseline_pinball_loss,
        f"{prefix}_baseline_pinball_score": metrics.baseline_pinball_score,
        f"{prefix}_pinball_skill": metrics.pinball_skill,
        f"{prefix}_coverage": metrics.coverage,
        f"{prefix}_coverage_error": metrics.coverage_error,
        f"{prefix}_spearman_ic": metrics.spearman_ic,
        f"{prefix}_direction_accuracy": metrics.direction_accuracy,
        f"{prefix}_direction_baseline": metrics.direction_baseline,
        f"{prefix}_direction_edge": metrics.direction_edge,
        f"{prefix}_mae": metrics.mae,
        f"{prefix}_rmse": metrics.rmse,
        f"{prefix}_prediction_mean": metrics.prediction_mean,
        f"{prefix}_target_quantile": metrics.target_quantile,
        f"{prefix}_target_mean": metrics.target_mean,
        f"{prefix}_n_samples": float(metrics.n_samples),
    }


def _internal_quantile_stop_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    purge_bars: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series] | None:
    n = len(X_train)
    if n < 4 or config.EARLY_STOP_VALID_FRACTION <= 0.0:
        return None
    n_stop = max(
        int(config.EARLY_STOP_MIN_VALID_SAMPLES),
        int(np.ceil(n * config.EARLY_STOP_VALID_FRACTION)),
    )
    n_stop = min(n_stop, n // 2)
    if n_stop < 1 or n - n_stop < 1:
        return None
    split_pos = n - n_stop
    # A target at row t consumes candles t+1..t+h. Leave those rows out at
    # the fit/stop boundary so fitting labels cannot contain stop-set prices.
    fit_end = split_pos - max(int(purge_bars), 0)
    if fit_end < 1:
        return None
    return (
        X_train.iloc[:fit_end],
        y_train.iloc[:fit_end],
        X_train.iloc[split_pos:],
        y_train.iloc[split_pos:],
    )
