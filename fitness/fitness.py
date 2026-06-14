"""
Evo_Finance — Fitness
──────────────────────
Computes the fitness score for an Individual given model predictions.

score = 0.45 * val_mean_ic
      + 0.25 * val_icir_scaled
      + 0.20 * hit_rate
      - 0.10 * overfit_gap

Definitions
-----------
IC (Information Coefficient)
    Spearman rank correlation between predicted score and realised label,
    computed **per date** (cross-sectional), then averaged.

ICIR (IC Information Ratio)
    mean(IC) / std(IC)  — risk-adjusted signal quality.
    Clipped to [-3, 3] then divided by 3 → [-1, 1].

Hit Rate
    On each date in the val set:
        top-K predicted ∩ top-K actual label
        ─────────────────────────────────────  × 1/K
                         K
    Then averaged across dates.  K = HIT_RATE_TOP_K (default 10).

Overfit Gap
    max(0, IC_mean_train − IC_mean_val)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config.settings import FITNESS_WEIGHTS, HIT_RATE_TOP_K
from mutator.gene import Individual

logger = logging.getLogger(__name__)


# ─── Metric helpers ───────────────────────────────────────────────────────────

def _ic_per_date(
    pred: pd.Series,
    label: pd.Series,
    df_index: pd.Index,
) -> pd.Series:
    """
    Compute Spearman IC for each date.
    pred and label must have the same MultiIndex (date, ticker).
    """
    data = pd.DataFrame({"pred": pred, "label": label}).dropna()

    # Extract date from the MultiIndex levels (avoid column name collision)
    if isinstance(data.index, pd.MultiIndex):
        dates_arr = data.index.get_level_values("date")
    else:
        dates_arr = data.index

    ics = {}
    for date in np.unique(dates_arr):
        mask = dates_arr == date
        grp  = data[mask]
        if len(grp) < 5:
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
    data = pd.DataFrame({"pred": pred, "label": label}).dropna()

    if isinstance(data.index, pd.MultiIndex):
        dates_arr = data.index.get_level_values("date")
    else:
        dates_arr = data.index

    hits = []
    for date in np.unique(dates_arr):
        mask = dates_arr == date
        grp  = data[mask]
        if len(grp) < top_k:
            continue
        top_pred  = set(grp.nlargest(top_k, "pred").index)
        top_label = set(grp.nlargest(top_k, "label").index)
        hits.append(len(top_pred & top_label) / top_k)

    return float(np.mean(hits)) if hits else 0.0


# ─── Fitness dataclass ────────────────────────────────────────────────────────

@dataclass
class FitnessResult:
    score:          float
    val_mean_ic:    float
    val_icir:       float
    val_icir_scaled:float
    hit_rate:       float
    overfit_gap:    float
    train_mean_ic:  float
    val_ic_series:  pd.Series = field(default_factory=pd.Series)

    def as_dict(self) -> Dict[str, float]:
        return {
            "score":           self.score,
            "val_mean_ic":     self.val_mean_ic,
            "val_icir":        self.val_icir,
            "val_icir_scaled": self.val_icir_scaled,
            "hit_rate":        self.hit_rate,
            "overfit_gap":     self.overfit_gap,
            "train_mean_ic":   self.train_mean_ic,
        }


# ─── Fitness evaluator ────────────────────────────────────────────────────────

class FitnessEvaluator:

    def evaluate(
        self,
        individual:    Individual,
        train_pred:    pd.Series,
        val_pred:      pd.Series,
        train_labels:  pd.Series,
        val_labels:    pd.Series,
        train_df:      pd.DataFrame,
        val_df:        pd.DataFrame,
    ) -> FitnessResult:
        """
        Compute and return the FitnessResult.  Also writes metrics into
        individual.metrics and sets individual.score.
        """
        # ── IC series ─────────────────────────────────────────────────────────
        train_ic_series = _ic_per_date(train_pred, train_labels, train_df.index)
        val_ic_series   = _ic_per_date(val_pred,   val_labels,   val_df.index)

        train_mean_ic = float(train_ic_series.mean()) if len(train_ic_series) else 0.0
        val_mean_ic   = float(val_ic_series.mean())   if len(val_ic_series)   else 0.0

        # ── ICIR ──────────────────────────────────────────────────────────────
        val_ic_std    = float(val_ic_series.std()) if len(val_ic_series) > 1 else 1e-9
        val_icir      = val_mean_ic / (val_ic_std + 1e-9)
        val_icir_scaled = float(np.clip(val_icir, -3.0, 3.0)) / 3.0

        # ── Hit rate ──────────────────────────────────────────────────────────
        hit_rate = _hit_rate(val_pred, val_labels, val_df.index)

        # ── Overfit gap ───────────────────────────────────────────────────────
        overfit_gap = max(0.0, train_mean_ic - val_mean_ic)

        # ── Aggregate ─────────────────────────────────────────────────────────
        w = FITNESS_WEIGHTS
        score = (
            w["val_mean_ic"] * val_mean_ic
            + w["val_icir"]  * val_icir_scaled
            + w["hit_rate"]  * hit_rate
            + w["overfit_gap"] * overfit_gap   # weight is negative in config
        )

        result = FitnessResult(
            score           = score,
            val_mean_ic     = val_mean_ic,
            val_icir        = val_icir,
            val_icir_scaled = val_icir_scaled,
            hit_rate        = hit_rate,
            overfit_gap     = overfit_gap,
            train_mean_ic   = train_mean_ic,
            val_ic_series   = val_ic_series,
        )

        # write back into individual
        individual.score   = score
        individual.metrics = result.as_dict()

        logger.info(
            "Fitness: score=%.4f | val_IC=%.4f | ICIR=%.4f | hit=%.4f | gap=%.4f",
            score, val_mean_ic, val_icir, hit_rate, overfit_gap,
        )
        return result