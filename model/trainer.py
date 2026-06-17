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
from typing import List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

from config.settings import (
    LGBM_FINAL_EARLY_STOPPING,
    LGBM_FINAL_NUM_BOOST_ROUND,
    LGBM_FINAL_PARAMS,
    LGBM_WF_EARLY_STOPPING,
    LGBM_WF_NUM_BOOST_ROUND,
    LGBM_WF_PARAMS,
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

def _sanitize_col_name(formula: str) -> str:
    """
    LightGBM không chấp nhận ký tự đặc biệt trong feature name.
    Thay thế tất cả ký tự không phải chữ/số/gạch dưới bằng '_'.
    """
    import re
    return re.sub(r'[^a-zA-Z0-9_]', '_', formula)


def build_feature_matrix(
    individual: Individual,
    df: pd.DataFrame,
    target_index: pd.Index | None = None,
) -> pd.DataFrame:
    """
    Evaluate each gene formula on df and return a DataFrame of features.
    Column names are sanitized for LightGBM compatibility.
    """
    cols = {}
    col_map = {}   # sanitized_name → formula (for debug)
    for gene in individual.genes:
        try:
            series = evaluate(gene.formula, df)
            safe_name = _sanitize_col_name(gene.formula)
            # handle rare collision after sanitize
            if safe_name in cols:
                safe_name = safe_name + f"_{len(cols)}"
            cols[safe_name] = series
            col_map[safe_name] = gene.formula
        except Exception as exc:
            logger.warning("Gene eval failed: %r — %s", gene.formula, exc)
    feat_df = pd.DataFrame(cols, index=df.index)
    if target_index is not None:
        feat_df = feat_df.loc[target_index]
    return feat_df


def clean_feature_matrix(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Keep LightGBM-compatible missing values without dropping rows."""
    return feat_df.replace([np.inf, -np.inf], np.nan)


def _feature_context(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Combine splits only as past/current feature context; labels are unused."""
    frames = [df for df in dfs if df is not None and not df.empty]
    if not frames:
        raise ValueError("No data available for feature context.")
    return pd.concat(frames).sort_index()


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


def _training_config(mode: str) -> tuple[dict, int, int]:
    """Return LightGBM params for walk-forward or final retraining."""
    if mode == "wf":
        return LGBM_WF_PARAMS, LGBM_WF_NUM_BOOST_ROUND, LGBM_WF_EARLY_STOPPING
    if mode == "final":
        return (
            LGBM_FINAL_PARAMS,
            LGBM_FINAL_NUM_BOOST_ROUND,
            LGBM_FINAL_EARLY_STOPPING,
        )
    raise ValueError(f"Unknown training mode: {mode!r}. Use 'wf' or 'final'.")


# ─── Trainer ──────────────────────────────────────────────────────────────────

class Trainer:
    """Stateless helper; instantiate per evolutionary run or share across runs."""

    def train(
        self,
        individual: Individual,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_df: pd.DataFrame | None = None,
        mode: str = "final",
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
        params, num_boost_round, early_stopping_rounds = _training_config(mode)

        context_df = feature_df if feature_df is not None else _feature_context(
            train_df, val_df
        )
        X_all = clean_feature_matrix(build_feature_matrix(individual, context_df))
        X_train = X_all.loc[train_df.index]
        X_val   = X_all.loc[val_df.index]

        if X_train.shape[1] == 0 or X_val.shape[1] == 0:
            raise ValueError("Empty feature matrix: no valid gene columns.")

        train_labels = train_df["label"]
        val_labels   = val_df["label"]

        # LightGBM handles NaN features natively. Only drop rows without labels
        # so each date keeps the full stock universe whenever labels exist.
        train_mask = train_labels.notna()
        val_mask   = val_labels.notna()

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
                stopping_rounds=early_stopping_rounds,
                verbose=False,
            ),
            lgb.log_evaluation(period=-1),   # silence per-round output
        ]

        booster = lgb.train(
            params            = dict(params),
            train_set         = lgb_train,
            num_boost_round   = num_boost_round,
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
        feature_df: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Run inference on an arbitrary split (e.g. test set)."""
        context_df = feature_df if feature_df is not None else df
        X = clean_feature_matrix(
            build_feature_matrix(individual, context_df, target_index=df.index)
        )
        if X.shape[1] == 0:
            raise ValueError("Empty feature matrix: no valid gene columns.")
        return pd.Series(booster.predict(X), index=df.index, name="pred")
