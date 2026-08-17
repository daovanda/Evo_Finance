"""Walk-forward fitness for quantile-based close-horizon selection.

One close-return quantile regressor is trained for every candidate horizon.
At each row, the strategy selects the horizon with the highest predicted
quantile and trades only when that value exceeds the configured minimum.
Fitness is computed from the realized return at the selected close, with an
oracle-regret penalty and no access to validation targets during selection.
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
from crypto.quantile_fitness import _internal_quantile_stop_split


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuantileExitMetrics:
    realized_net_return_mean: float
    realized_return_score: float
    trade_net_return_mean: float
    return_std: float
    return_std_score: float
    oracle_net_return_mean: float
    regret_mean: float
    regret_score: float
    selected_rate: float
    oracle_selected_rate: float
    win_rate: float
    exact_action_accuracy: float
    within_one_horizon_accuracy: float
    mean_exit_horizon: float
    mean_best_prediction: float
    n_selected: int
    n_samples: int
    selected_horizon_rates: dict[int, float]


class QuantileExitFitnessEvaluator:
    """Evaluate a close-quantile argmax exit policy across walk-forward folds."""

    def __init__(
        self,
        horizons: list[int] | tuple[int, ...],
        quantile: float = config.QUANTILE_ALPHA,
        direction: str = config.LABEL_DIRECTION,
        min_return: float | None = None,
        trade_cost: float = config.TRADE_COST,
        lgbm_params: dict | None = None,
        num_boost_round: int = config.LGBM_NUM_BOOST_ROUND,
        early_stopping_rounds: int = config.LGBM_EARLY_STOPPING,
    ) -> None:
        self.horizons = sorted({int(h) for h in horizons})
        if not self.horizons or any(h < 1 for h in self.horizons):
            raise ValueError("quantile_exit horizons must be positive integers.")
        self.quantile = config.validate_quantile_alpha(quantile)
        self.direction = config.canonical_label_direction(direction)
        self.min_return = config.quantile_exit_min_return(min_return)
        self.trade_cost = float(trade_cost)
        if not np.isfinite(self.trade_cost) or self.trade_cost < 0.0:
            raise ValueError("quantile_exit trade_cost must be finite and non-negative.")
        self.lgbm_params = dict(lgbm_params or config.QUANTILE_TRADE_LGBM_PARAMS)
        self.num_boost_round = int(num_boost_round)
        self.early_stopping_rounds = int(early_stopping_rounds)

    @staticmethod
    def target_column(horizon: int) -> str:
        return f"quantile_close_return_h{int(horizon)}"

    def evaluate_walk_forward(
        self,
        individual: CryptoIndividual,
        folds: list[CryptoFold],
        feature_data: CryptoFeatureSpace | pd.DataFrame,
    ) -> float:
        feature_space = _feature_space_for(individual, feature_data)
        rows: list[dict[str, float]] = []
        for fold in folds:
            train = self._valid_frame(fold.train_df)
            val = self._valid_frame(fold.val_df)
            if train.empty or val.empty:
                continue
            train_pred, val_pred, _ = self._fit_predict(
                individual, feature_space, train, val
            )
            train_metrics = self._selection_metrics(train, train_pred)
            val_metrics = self._selection_metrics(val, val_pred)
            rows.append(self._fold_row(train_metrics, val_metrics))
        if not rows:
            raise ValueError("No valid quantile_exit fold metrics were produced.")

        metrics_df = pd.DataFrame(rows)
        means = {
            name: float(metrics_df[f"val_{name}"].mean())
            for name in _EXIT_METRIC_NAMES
        }
        overfit_gap = float(metrics_df["overfit_gap"].mean())
        bad_fold_ratio = float(metrics_df["bad_fold"].mean())
        weights = config.QUANTILE_EXIT_FITNESS_WEIGHTS
        score = (
            weights["realized_return_score"] * means["realized_return_score"]
            + weights["regret_score"] * means["regret_score"]
            + weights["return_std_score"] * means["return_std_score"]
            + weights["overfit_gap"] * overfit_gap
            + weights["bad_fold_ratio"] * bad_fold_ratio
        )
        metrics: dict[str, float | str] = {
            "score": float(score),
            "fitness_type": "quantile_exit",
            "fitness_horizon_mode": "exit_selection",
            "quantile_target": "close",
            "quantile_alpha": self.quantile,
            "quantile_exit_min_return": self.min_return,
            "trade_cost": self.trade_cost,
            **means,
            "overfit_gap": overfit_gap,
            "bad_fold_ratio": bad_fold_ratio,
            "n_fold_scores": float(len(metrics_df)),
            "n_horizons": float(len(self.horizons)),
            "max_exit_horizon": float(max(self.horizons)),
        }
        for horizon in self.horizons:
            metrics[f"h{horizon}_selected_rate"] = float(
                metrics_df[f"val_h{horizon}_selected_rate"].mean()
            )
        individual.score = float(score)
        individual.metrics = metrics
        logger.info(
            "Crypto quantile-exit WF: score=%.4f | Q%.0f | net=%+.4f%% | "
            "trade_net=%+.4f%% | selected=%.2f%% | regret=%.4f%% | "
            "std=%.4f%% | exact=%.2f%% | within1=%.2f%% | gap=%.4f | "
            "bad=%.2f | folds=%d",
            score,
            100.0 * self.quantile,
            100.0 * means["realized_net_return_mean"],
            100.0 * means["trade_net_return_mean"],
            100.0 * means["selected_rate"],
            100.0 * means["regret_mean"],
            100.0 * means["return_std"],
            100.0 * means["exact_action_accuracy"],
            100.0 * means["within_one_horizon_accuracy"],
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
    ) -> dict[str, float | str]:
        feature_space = _feature_space_for(individual, feature_data)
        train = self._valid_frame(train_df)
        val = self._valid_frame(val_df)
        test = self._valid_frame(test_df)
        if train.empty or val.empty or test.empty:
            raise ValueError("quantile_exit final split contains no complete rows.")
        train_pred, val_pred, test_pred = self._fit_predict(
            individual, feature_space, train, val, test
        )
        assert test_pred is not None
        train_metrics = self._selection_metrics(train, train_pred)
        val_metrics = self._selection_metrics(val, val_pred)
        test_metrics = self._selection_metrics(test, test_pred)
        metrics: dict[str, float | str] = {
            "final_fitness_type": "quantile_exit",
            "final_horizon_mode": "exit_selection",
            "final_quantile_alpha": self.quantile,
            "final_quantile_exit_min_return": self.min_return,
            "final_train_rows": float(train_metrics.n_samples),
            "final_val_rows": float(val_metrics.n_samples),
            "final_test_rows": float(test_metrics.n_samples),
            "final_n_horizon_scores": float(len(self.horizons)),
        }
        for split, split_metrics in (("val", val_metrics), ("test", test_metrics)):
            for name in _EXIT_METRIC_NAMES:
                metrics[f"final_{split}_{name}"] = float(
                    getattr(split_metrics, name)
                )
            metrics[f"final_{split}_overfit_gap"] = max(
                0.0,
                train_metrics.realized_return_score
                - split_metrics.realized_return_score,
            )
            metrics[f"final_{split}_bad_ratio"] = float(
                split_metrics.realized_net_return_mean <= 0.0
                or split_metrics.n_selected < int(config.MIN_TRADES_PER_SPLIT)
            )
            for horizon in self.horizons:
                metrics[f"final_h{horizon}_{split}_selected_rate"] = float(
                    split_metrics.selected_horizon_rates[horizon]
                )
        individual.metrics.update(metrics)
        logger.info(
            "Crypto quantile-exit final: WF score=%.4f | "
            "val net=%+.4f%% selected=%.2f%% regret=%.4f%% | "
            "test net=%+.4f%% selected=%.2f%% regret=%.4f%%",
            float(individual.score) if individual.score is not None else float("nan"),
            100.0 * val_metrics.realized_net_return_mean,
            100.0 * val_metrics.selected_rate,
            100.0 * val_metrics.regret_mean,
            100.0 * test_metrics.realized_net_return_mean,
            100.0 * test_metrics.selected_rate,
            100.0 * test_metrics.regret_mean,
        )
        return metrics

    def _valid_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        columns = [self.target_column(horizon) for horizon in self.horizons]
        missing = [column for column in columns if column not in frame]
        if missing:
            raise ValueError(f"Missing quantile_exit target columns: {missing}")
        return frame.dropna(subset=columns).copy()

    def _directional_actual(self, frame: pd.DataFrame) -> pd.DataFrame:
        multiplier = -1.0 if self.direction == "short" else 1.0
        return pd.DataFrame(
            {
                horizon: pd.to_numeric(
                    frame[self.target_column(horizon)], errors="coerce"
                )
                * multiplier
                for horizon in self.horizons
            },
            index=frame.index,
        )

    def _fit_predict(
        self,
        individual: CryptoIndividual,
        feature_space: CryptoFeatureSpace,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        X_train = feature_space.matrix(individual.features, train.index)
        X_val = feature_space.matrix(individual.features, val.index)
        X_test = (
            feature_space.matrix(individual.features, test.index)
            if test is not None
            else None
        )
        train_actual = self._directional_actual(train)
        train_predictions: dict[int, np.ndarray] = {}
        val_predictions: dict[int, np.ndarray] = {}
        test_predictions: dict[int, np.ndarray] = {}
        for horizon in self.horizons:
            booster = self._train_booster(
                X_train,
                train_actual[horizon],
                horizon=horizon,
            )
            train_predictions[horizon] = booster.predict(X_train)
            val_predictions[horizon] = booster.predict(X_val)
            if X_test is not None:
                test_predictions[horizon] = booster.predict(X_test)
            booster.free_dataset()
        train_pred = pd.DataFrame(train_predictions, index=train.index)
        val_pred = pd.DataFrame(val_predictions, index=val.index)
        test_pred = (
            pd.DataFrame(test_predictions, index=test.index)
            if test is not None and X_test is not None
            else None
        )
        return train_pred, val_pred, test_pred

    def _train_booster(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        horizon: int,
    ) -> lgb.Booster:
        params = dict(self.lgbm_params)
        params.update(objective="quantile", metric="quantile", alpha=self.quantile)
        callbacks = [lgb.log_evaluation(period=-1)]
        valid_sets = None
        split = _internal_quantile_stop_split(
            X_train,
            y_train,
            purge_bars=int(horizon),
        )
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
            params=params,
            train_set=train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )

    def _selection_metrics(
        self,
        frame: pd.DataFrame,
        prediction: pd.DataFrame,
    ) -> QuantileExitMetrics:
        return quantile_exit_metrics(
            actual=self._directional_actual(frame),
            prediction=prediction,
            horizons=self.horizons,
            min_return=self.min_return,
            trade_cost=self.trade_cost,
        )

    def _fold_row(
        self,
        train: QuantileExitMetrics,
        val: QuantileExitMetrics,
    ) -> dict[str, float]:
        row: dict[str, float] = {
            "overfit_gap": max(
                0.0, train.realized_return_score - val.realized_return_score
            ),
            "bad_fold": float(
                val.realized_net_return_mean <= 0.0
                or val.n_selected < int(config.MIN_TRADES_PER_SPLIT)
            ),
        }
        row.update(_exit_metric_values("train", train))
        row.update(_exit_metric_values("val", val))
        return row


def quantile_exit_metrics(
    *,
    actual: pd.DataFrame,
    prediction: pd.DataFrame,
    horizons: list[int] | tuple[int, ...],
    min_return: float,
    trade_cost: float,
) -> QuantileExitMetrics:
    """Evaluate horizon choices without using actual returns in the decision."""
    ordered_horizons = sorted({int(h) for h in horizons})
    common_index = actual.index.intersection(prediction.index)
    actual_values = actual.reindex(common_index, columns=ordered_horizons).to_numpy(float)
    predicted_values = prediction.reindex(
        common_index, columns=ordered_horizons
    ).to_numpy(float)
    complete = np.isfinite(actual_values).all(axis=1) & np.isfinite(
        predicted_values
    ).all(axis=1)
    actual_values = actual_values[complete]
    predicted_values = predicted_values[complete]
    n_samples = int(len(actual_values))
    if n_samples == 0:
        raise ValueError("quantile_exit metrics require at least one complete row.")

    horizon_values = np.asarray(ordered_horizons, dtype=int)
    row_index = np.arange(n_samples)
    predicted_best_pos = np.argmax(predicted_values, axis=1)
    predicted_best = predicted_values[row_index, predicted_best_pos]
    selected = predicted_best > float(min_return)
    selected_actions = np.where(selected, horizon_values[predicted_best_pos], 0)

    chosen_gross = actual_values[row_index, predicted_best_pos]
    realized_net = np.zeros(n_samples, dtype=float)
    realized_net[selected] = chosen_gross[selected] - float(trade_cost)

    oracle_best_pos = np.argmax(actual_values, axis=1)
    oracle_best = actual_values[row_index, oracle_best_pos]
    oracle_selected = oracle_best > float(trade_cost)
    oracle_actions = np.where(oracle_selected, horizon_values[oracle_best_pos], 0)
    oracle_net = np.where(oracle_selected, oracle_best - float(trade_cost), 0.0)
    regret = oracle_net - realized_net

    n_selected = int(selected.sum())
    selected_returns = realized_net[selected]
    scale = max(float(config.RETURN_SCORE_SCALE), np.finfo(float).eps)
    realized_mean = float(realized_net.mean())
    return_std = float(realized_net.std(ddof=0))
    regret_mean = float(regret.mean())
    both_exit = (selected_actions > 0) & (oracle_actions > 0)
    within_one = (selected_actions == oracle_actions) | (
        both_exit & (np.abs(selected_actions - oracle_actions) <= 1)
    )
    selected_horizon_rates = {
        horizon: float(np.mean(selected_actions == horizon))
        for horizon in ordered_horizons
    }
    return QuantileExitMetrics(
        realized_net_return_mean=realized_mean,
        realized_return_score=float(np.clip(realized_mean / scale, -1.0, 1.0)),
        trade_net_return_mean=(
            float(selected_returns.mean()) if n_selected else 0.0
        ),
        return_std=return_std,
        return_std_score=float(np.clip(return_std / scale, 0.0, 1.0)),
        oracle_net_return_mean=float(oracle_net.mean()),
        regret_mean=regret_mean,
        regret_score=float(np.clip(regret_mean / scale, 0.0, 1.0)),
        selected_rate=float(selected.mean()),
        oracle_selected_rate=float(oracle_selected.mean()),
        win_rate=(
            float(np.mean(selected_returns > 0.0)) if n_selected else 0.0
        ),
        exact_action_accuracy=float(np.mean(selected_actions == oracle_actions)),
        within_one_horizon_accuracy=float(np.mean(within_one)),
        mean_exit_horizon=(
            float(selected_actions[selected].mean()) if n_selected else 0.0
        ),
        mean_best_prediction=(
            float(predicted_best[selected].mean()) if n_selected else 0.0
        ),
        n_selected=n_selected,
        n_samples=n_samples,
        selected_horizon_rates=selected_horizon_rates,
    )


_EXIT_METRIC_NAMES = (
    "realized_net_return_mean",
    "realized_return_score",
    "trade_net_return_mean",
    "return_std",
    "return_std_score",
    "oracle_net_return_mean",
    "regret_mean",
    "regret_score",
    "selected_rate",
    "oracle_selected_rate",
    "win_rate",
    "exact_action_accuracy",
    "within_one_horizon_accuracy",
    "mean_exit_horizon",
    "mean_best_prediction",
    "n_selected",
    "n_samples",
)


def _exit_metric_values(
    prefix: str,
    metrics: QuantileExitMetrics,
) -> dict[str, float]:
    values = {
        f"{prefix}_{name}": float(getattr(metrics, name))
        for name in _EXIT_METRIC_NAMES
    }
    values.update(
        {
            f"{prefix}_h{horizon}_selected_rate": rate
            for horizon, rate in metrics.selected_horizon_rates.items()
        }
    )
    return values
