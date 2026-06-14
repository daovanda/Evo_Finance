"""
Evo_Finance — Model Trainer
─────────────────────────────
Builds feature matrices from an Individual's genes, trains a LightGBM
lambdarank model, and returns raw predictions for fitness evaluation.

Data layout assumption
  df : pd.DataFrame with MultiIndex (date, ticker) or flat, containing
       columns [open, high, close, low, volume] plus a 'label' column
       (forward return) and a 'group_date' column for LightGBM group sizes.

The caller (main loop) is responsible for supplying pre-split DataFrames:
  train_df, val_df, test_df — each already has 'label' computed.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

from config.settings import (
    LGBM_PARAMS, LGBM_NUM_BOOST_ROUND, LGBM_EARLY_STOPPING,
)
from mutator.gene import Individual
from mutator.evaluator import evaluate

logger = logging.getLogger(__name__)

# Number of integer relevance grades for lambdarank (0 = worst, N-1 = best)
N_RELEVANCE_BINS: int = 5


# ─── Label binner ─────────────────────────────────────────────────────────────

def _bin_labels(labels: pd.Series, ref_df: pd.DataFrame) -> pd.Series:
    """
    Convert continuous return labels to integer relevance grades [0, N-1].
    Binning is done cross-sectionally per date so relative rank is preserved.
    """
    if isinstance(ref_df.index, pd.MultiIndex):
        date_vals = ref_df.index.get_level_values("date")
    else:
        date_vals = ref_df.get("date", ref_df.index)

    result = pd.Series(np.zeros(len(labels), dtype=np.int32), index=labels.index)
    for date in np.unique(date_vals):
        mask = date_vals == date
        grp  = labels[mask]
        if len(grp) < N_RELEVANCE_BINS:
            result[mask] = 0
            continue
        # qcut into N bins; labels 0..N-1
        try:
            binned = pd.qcut(grp, N_RELEVANCE_BINS, labels=False, duplicates="drop")
            result[mask] = binned.fillna(0).astype(np.int32)
        except Exception:
            result[mask] = 0
    return result


# ─── Feature matrix builder ───────────────────────────────────────────────────

def build_feature_matrix(
    individual: Individual,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate each gene formula on df and return a DataFrame of features.
    Columns are named by their formula string.
    """
    cols = {}
    for gene in individual.genes:
        try:
            series = evaluate(gene.formula, df)
            cols[gene.formula] = series
        except Exception as exc:
            logger.warning("Gene eval failed: %r — %s", gene.formula, exc)
    feat_df = pd.DataFrame(cols, index=df.index)
    return feat_df


def _group_sizes(df: pd.DataFrame) -> List[int]:
    """
    LightGBM needs the number of items per query (date).
    Assumes df has a MultiIndex (date, ticker) or a 'date' column.
    """
    if isinstance(df.index, pd.MultiIndex):
        return df.groupby(level="date").size().tolist()
    elif "date" in df.columns:
        return df.groupby("date").size().tolist()
    else:
        # fallback: all in one group
        return [len(df)]


# ─── Trainer ──────────────────────────────────────────────────────────────────

class Trainer:
    """Stateless helper; instantiate per evolutionary run or share across runs."""

    def train(
        self,
        individual: Individual,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> Tuple["lgb.Booster", pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Train LightGBM on *individual*'s features.

        Returns
        -------
        booster        : trained lgb.Booster
        train_pred     : raw score predictions on train set
        val_pred       : raw score predictions on val set
        train_labels   : ground-truth labels (train)
        val_labels     : ground-truth labels (val)
        """
        # ── Build feature matrices ────────────────────────────────────────────
        X_train = build_feature_matrix(individual, train_df)
        X_val   = build_feature_matrix(individual, val_df)

        train_labels = train_df["label"]
        val_labels   = val_df["label"]

        # drop rows where any feature or label is NaN
        train_mask = X_train.notna().all(axis=1) & train_labels.notna()
        val_mask   = X_val.notna().all(axis=1)   & val_labels.notna()

        X_train, y_train = X_train[train_mask], train_labels[train_mask]
        X_val,   y_val   = X_val[val_mask],     val_labels[val_mask]

        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("Empty feature matrix after NaN drop.")

        # ── LightGBM datasets ─────────────────────────────────────────────────
        # lambdarank requires integer relevance labels (0, 1, 2, …)
        # We bin continuous returns into N_RELEVANCE_BINS grades per date.
        y_train_int = _bin_labels(y_train, train_df[train_mask])
        y_val_int   = _bin_labels(y_val,   val_df[val_mask])

        train_groups = _group_sizes(train_df[train_mask])
        val_groups   = _group_sizes(val_df[val_mask])

        lgb_train = lgb.Dataset(
            X_train, label=y_train_int, group=train_groups, free_raw_data=False
        )
        lgb_val = lgb.Dataset(
            X_val, label=y_val_int, group=val_groups,
            reference=lgb_train, free_raw_data=False,
        )

        # ── Train ─────────────────────────────────────────────────────────────
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=LGBM_EARLY_STOPPING,
                verbose=False,
            ),
            lgb.log_evaluation(period=-1),   # silence per-round output
        ]

        booster = lgb.train(
            params            = LGBM_PARAMS,
            train_set         = lgb_train,
            num_boost_round   = LGBM_NUM_BOOST_ROUND,
            valid_sets        = [lgb_val],
            callbacks         = callbacks,
        )

        train_pred = pd.Series(
            booster.predict(X_train), index=X_train.index, name="pred"
        )
        val_pred = pd.Series(
            booster.predict(X_val), index=X_val.index, name="pred"
        )

        return booster, train_pred, val_pred, y_train, y_val

    def predict(
        self,
        booster: "lgb.Booster",
        individual: Individual,
        df: pd.DataFrame,
    ) -> pd.Series:
        """Run inference on an arbitrary split (e.g. test set)."""
        X = build_feature_matrix(individual, df)
        mask = X.notna().all(axis=1)
        preds = booster.predict(X[mask])
        out = pd.Series(np.nan, index=df.index, name="pred")
        out[mask] = preds
        return out