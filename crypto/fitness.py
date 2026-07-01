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
            train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        else:
            X_fit, y_fit, X_stop, y_stop = split
            train_set = lgb.Dataset(X_fit, label=y_fit, free_raw_data=False)
            stop_set = lgb.Dataset(
                X_stop,
                label=y_stop,
                reference=train_set,
                free_raw_data=False,
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
    mean_return = float(pd.to_numeric(future_return, errors="coerce").dropna().mean() or 0.0)
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
