"""
Evo_Finance — Archive
──────────────────────
Stores the top-N individuals seen so far, sorted by fitness score descending.

Admission rule
  A new individual is admitted if:
    - The archive is not yet full, OR
    - Its score > the worst score in the archive.
  The worst individual is then evicted.

Each stored entry also keeps the trained booster so we can run test-set
evaluation at the end without retraining.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import lightgbm as lgb

from config.settings import ARCHIVE_SIZE
from mutator.gene import Individual

logger = logging.getLogger(__name__)


# ─── Archive entry ────────────────────────────────────────────────────────────

@dataclass
class ArchiveEntry:
    individual: Individual
    booster:    lgb.Booster
    score:      float
    metrics:    Dict[str, float] = field(default_factory=dict)
    final_val_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)  # filled later

    def __repr__(self):
        genes = ", ".join(self.individual.formulas)
        return (
            f"ArchiveEntry(score={self.score:.4f}, "
            f"genes=[{genes}])"
        )


# ─── Archive ──────────────────────────────────────────────────────────────────

class Archive:
    def __init__(self, max_size: int = ARCHIVE_SIZE):
        self.max_size = max_size
        self._entries: List[ArchiveEntry] = []

    # ── Admission ─────────────────────────────────────────────────────────────

    def try_add(
        self,
        individual: Individual,
        booster: lgb.Booster,
    ) -> bool:
        """
        Attempt to add *individual* to the archive.

        Returns True if admitted, False if rejected.
        individual.score must already be set.
        """
        if individual.score is None:
            raise ValueError("Individual must be evaluated before archiving.")

        score = individual.score

        if len(self._entries) < self.max_size:
            self._admit(individual, booster, score)
            return True

        worst = self._entries[-1]
        if score > worst.score:
            self._entries.pop()   # evict worst
            self._admit(individual, booster, score)
            logger.info(
                "Archive: admitted score=%.4f (evicted %.4f), size=%d",
                score, worst.score, len(self._entries),
            )
            return True

        logger.debug(
            "Archive: rejected score=%.4f (worst=%.4f)", score, worst.score
        )
        return False

    # ── Random selection (for mutation parent) ────────────────────────────────

    def random_individual(self, rng) -> Optional[Individual]:
        """
        Draw a random individual from the archive (uniform).
        Returns None if archive is empty.
        """
        if not self._entries:
            return None
        idx = rng.integers(len(self._entries))
        return self._entries[idx].individual

    # ── Accessors ─────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self._entries)

    def is_empty(self) -> bool:
        return len(self._entries) == 0

    @property
    def best(self) -> Optional[ArchiveEntry]:
        return self._entries[0] if self._entries else None

    @property
    def worst_score(self) -> float:
        return self._entries[-1].score if self._entries else float("-inf")

    @property
    def entries(self) -> List[ArchiveEntry]:
        return list(self._entries)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> List[Dict]:
        rows = []
        for rank, e in enumerate(self._entries, start=1):
            rows.append(
                {
                    "rank":         rank,
                    "score":        e.score,
                    "n_genes":      len(e.individual),
                    "generation":   e.individual.generation,
                    "val_mean_ic":  e.metrics.get("val_mean_ic"),
                    "val_icir":     e.metrics.get("val_icir"),
                    "hit_rate":     e.metrics.get("hit_rate"),
                    "overfit_gap":  e.metrics.get("overfit_gap"),
                    "wf_mean_ic":   e.metrics.get("wf_mean_ic"),
                    "wf_icir":      e.metrics.get("wf_icir"),
                    "wf_hit_rate":  e.metrics.get("wf_hit_rate"),
                    "wf_hit_excess": e.metrics.get("wf_hit_excess"),
                    "wf_ic_std":    e.metrics.get("wf_ic_std"),
                    "bad_fold_ratio": e.metrics.get("bad_fold_ratio"),
                    "wf_overfit_gap": e.metrics.get("wf_overfit_gap"),
                    "n_folds":      e.metrics.get("n_folds"),
                    "final_val_metrics": e.final_val_metrics,
                    "test_metrics": e.test_metrics,
                    "genes":        e.individual.formulas,
                }
            )
        return rows

    # ── Internal ──────────────────────────────────────────────────────────────

    def _admit(self, individual: Individual, booster: lgb.Booster, score: float):
        entry = ArchiveEntry(
            individual = individual,
            booster    = booster,
            score      = score,
            metrics    = dict(individual.metrics),
        )
        self._entries.append(entry)
        self._entries.sort(key=lambda e: e.score, reverse=True)
        logger.debug(
            "Archive admitted score=%.4f, size=%d", score, len(self._entries)
        )
