"""
Fitness metrics for Evo_Finance.

The evolutionary loop uses walk-forward fitness by default. The single-split
evaluator is kept for final validation/test reporting and backwards
compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config.settings import FITNESS_WEIGHTS, HIT_RATE_TOP_K
from mutator.gene import Individual

logger = logging.getLogger(__name__)


def _ic_per_date(
    pred: pd.Series,
    label: pd.Series,
    df_index: pd.Index,
) -> pd.Series:
    """Compute Spearman IC for each date."""
    data = _clean_metric_data(pred, label)

    if isinstance(data.index, pd.MultiIndex):
        dates_arr = data.index.get_level_values("date")
    else:
        dates_arr = data.index

    ics = {}
    for date in np.unique(dates_arr):
        mask = dates_arr == date
        grp = data[mask]
        if len(grp) < 5:
            continue
        if (
            grp["pred"].nunique(dropna=True) < 2
            or grp["label"].nunique(dropna=True) < 2
        ):
            ics[date] = 0.0
            continue
        corr, _ = spearmanr(grp["pred"], grp["label"])
        ics[date] = float(corr) if not np.isnan(corr) else 0.0

    return pd.Series(ics)


def _hit_rate(
    pred: pd.Series,
    label: pd.Series,
    df_index: pd.Index,
    top_k: int = HIT_RATE_TOP_K,
) -> float:
    data = _clean_metric_data(pred, label)

    if isinstance(data.index, pd.MultiIndex):
        dates_arr = data.index.get_level_values("date")
    else:
        dates_arr = data.index

    hits = []
    for date in np.unique(dates_arr):
        mask = dates_arr == date
        grp = data[mask]
        k = min(int(top_k), len(grp))
        if k <= 0:
            continue
        if (
            grp["pred"].nunique(dropna=True) < 2
            or grp["label"].nunique(dropna=True) < 2
        ):
            hits.append(float(k) / float(len(grp)))
            continue
        top_pred = set(grp.nlargest(k, "pred").index)
        top_label = set(grp.nlargest(k, "label").index)
        hits.append(len(top_pred & top_label) / k)

    return float(np.mean(hits)) if hits else 0.0


def _random_hit_baseline(df: pd.DataFrame, top_k: int = HIT_RATE_TOP_K) -> float:
    if len(df) == 0:
        return 0.0
    if isinstance(df.index, pd.MultiIndex):
        per_date = df.groupby(level="date").size()
        daily = [min(1.0, float(top_k) / float(n)) for n in per_date if n > 0]
        return float(np.mean(daily)) if daily else 0.0
    n_tickers = len(df)
    return min(1.0, float(top_k) / float(n_tickers)) if n_tickers > 0 else 0.0


def _clean_metric_data(pred: pd.Series, label: pd.Series) -> pd.DataFrame:
    return (
        pd.DataFrame({"pred": pred, "label": label})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def _random_hit_baseline_for_predictions(
    pred: pd.Series,
    label: pd.Series,
    top_k: int = HIT_RATE_TOP_K,
) -> float:
    """
    Random top-k overlap baseline on the same valid rows used by hit-rate.

    This keeps hit_excess honest when a metric call receives non-finite
    predictions or labels: both sides use the identical effective universe.
    """
    return _random_hit_baseline(_clean_metric_data(pred, label), top_k=top_k)


@dataclass
class FoldPrediction:
    name: str
    train_pred: pd.Series
    val_pred: pd.Series
    train_labels: pd.Series
    val_labels: pd.Series
    train_df: pd.DataFrame
    val_df: pd.DataFrame


@dataclass
class FitnessResult:
    score: float
    val_mean_ic: float
    val_icir: float
    val_icir_scaled: float
    hit_rate: float
    overfit_gap: float
    train_mean_ic: float
    val_ic_series: pd.Series = field(default_factory=pd.Series)
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        out = {
            "score": self.score,
            "val_mean_ic": self.val_mean_ic,
            "val_icir": self.val_icir,
            "val_icir_scaled": self.val_icir_scaled,
            "hit_rate": self.hit_rate,
            "overfit_gap": self.overfit_gap,
            "train_mean_ic": self.train_mean_ic,
        }
        out.update(self.extra)
        return out


class FitnessEvaluator:
    def evaluate(
        self,
        individual: Individual,
        train_pred: pd.Series,
        val_pred: pd.Series,
        train_labels: pd.Series,
        val_labels: pd.Series,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> FitnessResult:
        """Evaluate one train/validation split."""
        train_ic_series = _ic_per_date(train_pred, train_labels, train_df.index)
        val_ic_series = _ic_per_date(val_pred, val_labels, val_df.index)

        train_mean_ic = _safe_mean(train_ic_series)
        val_mean_ic = _safe_mean(val_ic_series)
        val_ic_std = float(val_ic_series.std()) if len(val_ic_series) > 1 else 0.0
        val_icir = val_mean_ic / (val_ic_std + 1e-9)
        val_icir_scaled = float(np.clip(val_icir, -3.0, 3.0)) / 3.0
        hit_rate = _hit_rate(val_pred, val_labels, val_df.index)
        baseline = _random_hit_baseline_for_predictions(val_pred, val_labels)
        hit_excess = hit_rate - baseline
        overfit_gap = max(0.0, train_mean_ic - val_mean_ic)

        w = FITNESS_WEIGHTS
        score = (
            w.get("val_mean_ic", w.get("wf_mean_ic", 0.0)) * val_mean_ic
            + w.get("val_icir", w.get("wf_icir", 0.0)) * val_icir_scaled
            + w.get("hit_rate", 0.0) * hit_rate
            + w.get("wf_hit_excess", 0.0) * hit_excess
            + w.get("overfit_gap", w.get("wf_overfit_gap", 0.0)) * overfit_gap
        )

        result = FitnessResult(
            score=score,
            val_mean_ic=val_mean_ic,
            val_icir=val_icir,
            val_icir_scaled=val_icir_scaled,
            hit_rate=hit_rate,
            overfit_gap=overfit_gap,
            train_mean_ic=train_mean_ic,
            val_ic_series=val_ic_series,
            extra={"hit_excess": hit_excess},
        )
        individual.score = score
        individual.metrics = result.as_dict()
        return result

    def evaluate_walk_forward(
        self,
        individual: Individual,
        folds: Iterable[FoldPrediction],
    ) -> FitnessResult:
        """Aggregate fold metrics into the configured walk-forward score."""
        fold_metrics: List[Dict[str, float]] = []
        val_ic_series_parts: List[pd.Series] = []

        for fold in folds:
            train_ic = _ic_per_date(
                fold.train_pred,
                fold.train_labels,
                fold.train_df.index,
            )
            val_ic = _ic_per_date(
                fold.val_pred,
                fold.val_labels,
                fold.val_df.index,
            )
            train_mean_ic = _safe_mean(train_ic)
            val_mean_ic = _safe_mean(val_ic)
            hit_rate = _hit_rate(fold.val_pred, fold.val_labels, fold.val_df.index)
            baseline = _random_hit_baseline_for_predictions(
                fold.val_pred,
                fold.val_labels,
            )
            fold_metrics.append(
                {
                    "train_mean_ic": train_mean_ic,
                    "val_mean_ic": val_mean_ic,
                    "hit_rate": hit_rate,
                    "hit_excess": hit_rate - baseline,
                    "overfit_gap": max(0.0, train_mean_ic - val_mean_ic),
                }
            )
            if len(val_ic):
                val_ic_series_parts.append(val_ic.rename(fold.name))

        if not fold_metrics:
            raise ValueError("No fold metrics available for walk-forward fitness.")

        fold_df = pd.DataFrame(fold_metrics)
        all_val_ic = (
            pd.concat(val_ic_series_parts)
            if val_ic_series_parts
            else pd.Series(dtype=float)
        )

        wf_mean_ic = float(fold_df["val_mean_ic"].mean())
        wf_ic_std = float(fold_df["val_mean_ic"].std(ddof=0))
        wf_hit_rate = float(fold_df["hit_rate"].mean())
        wf_hit_excess = float(fold_df["hit_excess"].mean())
        wf_overfit_gap = float(fold_df["overfit_gap"].mean())
        train_mean_ic = float(fold_df["train_mean_ic"].mean())
        bad_fold_ratio = float(
            (
                (fold_df["val_mean_ic"] <= 0.0)
                | (fold_df["hit_excess"] < 0.0)
            ).mean()
        )

        if len(all_val_ic) > 1:
            wf_icir = float(all_val_ic.mean()) / (float(all_val_ic.std()) + 1e-9)
        else:
            wf_icir = 0.0
        wf_icir_scaled = float(np.clip(wf_icir, -3.0, 3.0)) / 3.0

        w = FITNESS_WEIGHTS
        score = (
            w["wf_mean_ic"] * wf_mean_ic
            + w["wf_icir"] * wf_icir_scaled
            + w["wf_hit_excess"] * wf_hit_excess
            + w["wf_ic_std"] * wf_ic_std
            + w["bad_fold_ratio"] * bad_fold_ratio
            + w["wf_overfit_gap"] * wf_overfit_gap
        )

        extra = {
            "wf_mean_ic": wf_mean_ic,
            "wf_icir": wf_icir,
            "wf_icir_scaled": wf_icir_scaled,
            "wf_hit_rate": wf_hit_rate,
            "wf_hit_excess": wf_hit_excess,
            "wf_ic_std": wf_ic_std,
            "bad_fold_ratio": bad_fold_ratio,
            "wf_overfit_gap": wf_overfit_gap,
            "wf_train_mean_ic": train_mean_ic,
            "n_folds": float(len(fold_metrics)),
        }
        result = FitnessResult(
            score=score,
            val_mean_ic=wf_mean_ic,
            val_icir=wf_icir,
            val_icir_scaled=wf_icir_scaled,
            hit_rate=wf_hit_rate,
            overfit_gap=wf_overfit_gap,
            train_mean_ic=train_mean_ic,
            val_ic_series=all_val_ic,
            extra=extra,
        )

        individual.score = score
        individual.metrics = result.as_dict()
        logger.info(
            "WF fitness: score=%.4f | IC=%.4f | ICIR=%.4f | hit_excess=%.4f | "
            "std=%.4f | bad=%.2f | gap=%.4f | folds=%d",
            score,
            wf_mean_ic,
            wf_icir,
            wf_hit_excess,
            wf_ic_std,
            bad_fold_ratio,
            wf_overfit_gap,
            len(fold_metrics),
        )
        return result


def _safe_mean(series: pd.Series) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.mean()) if len(clean) else 0.0
