"""
Evo_Finance — Central Configuration
All tunable knobs live here.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Callable
import numpy as np

# ─── Data ─────────────────────────────────────────────────────────────────────

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15   # remainder

# ─── Holding horizon ──────────────────────────────────────────────────────────

HOLDING_HORIZON: int = 10          # h days forward

def default_label(df, h: int = HOLDING_HORIZON):
    """
    label(t) = (close(t+h) - open(t+1)) / open(t+1)
    df must have columns: open, close  (already shifted by caller).
    Returns a Series aligned to df.index.
    """
    future_close = df["close"].shift(-h)
    next_open    = df["open"].shift(-1)
    return (future_close - next_open) / next_open

LABEL_FN: Callable = default_label

# ─── Feature limits ───────────────────────────────────────────────────────────

FEATURE_MIN: int = 5
FEATURE_MAX: int = 30

# ─── Correlation threshold ────────────────────────────────────────────────────

CORR_THRESHOLD: float = 0.70       # used for domain & individual dedup

# ─── Window whitelist ─────────────────────────────────────────────────────────

WINDOWS: List[int] = [3, 5, 10, 14, 20, 30, 60, 120]

DEFAULT_WINDOW: int = 1            # initial {O,H,C,L,V} window

# ─── Mutator probabilities ────────────────────────────────────────────────────

MUTATOR_PROBS = {
    "c1": 0.40,   # add/remove a feature
    "c2": 0.35,   # change window
    "c3": 0.25,   # transform a gene
}
# must sum to 1.0 — validated at runtime

MAX_RETRY: int = 5                 # max retries before fallback in c1/c2/c3

# ─── Restart ──────────────────────────────────────────────────────────────────

RESTART_PROB: float = 0.0001       # 0.01 % — restart from raw OHLCV

# ─── Archive ──────────────────────────────────────────────────────────────────

ARCHIVE_SIZE: int = 50             # max individuals kept

# ─── LightGBM ─────────────────────────────────────────────────────────────────

LGBM_PARAMS: dict = {
    "objective":        "lambdarank",
    "metric":           "ndcg",
    "eval_at":          [10],
    "learning_rate":    0.01,
    "num_leaves":       31,
    "max_depth":        6,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "min_data_in_leaf": 50,
    "lambda_l1":        1.0,
    "lambda_l2":        1.0,
    "verbose":          -1,
    "seed":             42,
}

LGBM_NUM_BOOST_ROUND: int  = 500
LGBM_EARLY_STOPPING:  int  = 30    # stop if val NDCG@10 doesn't improve

# ─── Time budget ──────────────────────────────────────────────────────────────

TIME_BUDGET_SECONDS: float = 3600.0   # 1 hour default

# ─── Hit-rate ─────────────────────────────────────────────────────────────────

HIT_RATE_TOP_K: int = 10

# ─── Fitness weights ──────────────────────────────────────────────────────────

FITNESS_WEIGHTS = {
    "val_mean_ic":    0.35,
    "val_icir":       0.25,
    "hit_rate":       0.20,
    "overfit_gap":   -0.20,
}