"""
Evo_Finance — Domain
─────────────────────
The Domain is the ever-growing pool of formula strings that the Mutator
can draw from.  A new formula is admitted only if its Pearson correlation
with every existing formula in the domain is < CORR_THRESHOLD (computed on
the training set).

Thread-safety: single-process assumed; no locking needed.
"""

from __future__ import annotations
import logging
from typing import List, Optional
import numpy as np
import pandas as pd

from config.settings import CORR_THRESHOLD
from mutator.gene import Gene
from mutator.evaluator import evaluate

logger = logging.getLogger(__name__)

# Raw base columns (order preserved)
_RAW_BASES = ["open", "high", "close", "low", "volume"]


class Domain:
    """
    Stores formula strings (not computed arrays).
    Computed arrays are materialised on-demand for correlation checks.
    """

    def __init__(self):
        self._formulas: List[str] = []
        # Cache: formula → computed Series on train set
        self._cache: dict[str, pd.Series] = {}

    # ── Initialisation ────────────────────────────────────────────────────────

    def seed(self, window: int = 1) -> List[Gene]:
        """Populate with raw OHLCV genes; return them for the first individual."""
        genes = [Gene.raw(b, window) for b in _RAW_BASES]
        for g in genes:
            self._formulas.append(g.formula)
        logger.info("Domain seeded with %d raw features.", len(genes))
        return genes

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def formulas(self) -> List[str]:
        return list(self._formulas)

    def __len__(self):
        return len(self._formulas)

    def random_gene(self, rng: np.random.Generator) -> Gene:
        """Draw a uniformly random Gene from the domain."""
        formula = rng.choice(self._formulas)
        return Gene(formula=str(formula))

    def try_add(
        self,
        gene: Gene,
        train_df: pd.DataFrame,
        force: bool = False,
    ) -> bool:
        """
        Attempt to add *gene* to the domain.

        A gene is added only if:
          - Its formula is not already in the domain, AND
          - Its correlation with every existing domain member < CORR_THRESHOLD
            (evaluated on train_df).

        Parameters
        ----------
        gene      : Gene to test.
        train_df  : Training DataFrame (features must be computable).
        force     : If True, skip correlation check (used for seed genes).

        Returns True if the gene was added.
        """
        if gene.formula in self._formulas:
            return False

        if not force:
            try:
                new_series = self._compute(gene.formula, train_df)
            except Exception as exc:
                logger.warning("Domain.try_add: cannot evaluate %r — %s",
                               gene.formula, exc)
                return False

            for existing_formula in self._formulas:
                ex_series = self._cache.get(existing_formula)
                if ex_series is None:
                    continue
                corr = _safe_corr(new_series, ex_series)
                if corr >= CORR_THRESHOLD:
                    return False

            # passed — cache it
            self._cache[gene.formula] = new_series

        self._formulas.append(gene.formula)
        logger.debug("Domain ← %r  (size=%d)", gene.formula, len(self._formulas))
        return True

    def precompute(self, train_df: pd.DataFrame) -> None:
        """Materialise all domain formulas on train_df (call once after seeding)."""
        for f in self._formulas:
            if f not in self._cache:
                try:
                    self._cache[f] = self._compute(f, train_df)
                except Exception as exc:
                    logger.warning("Domain.precompute: %r failed — %s", f, exc)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute(self, formula: str, df: pd.DataFrame) -> pd.Series:
        if formula in self._cache:
            return self._cache[formula]
        series = evaluate(formula, df).dropna()
        self._cache[formula] = series
        return series


# ─── Utility ──────────────────────────────────────────────────────────────────

def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    """
    Pearson correlation; trả về 1.0 (reject) nếu:
      - Không đủ dữ liệu
      - Một trong hai series là hằng số (std ≈ 0) → gene vô nghĩa như x/x
    Trả về 0.0 nếu có exception khác.
    """
    try:
        aligned = pd.concat([a, b], axis=1).dropna()
        if len(aligned) < 10:
            return 1.0  # quá ít data → coi như duplicate, reject
        # Hằng số → corr = NaN → nếu không chặn sẽ pass threshold
        if aligned.iloc[:, 0].std() < 1e-8 or aligned.iloc[:, 1].std() < 1e-8:
            return 1.0  # reject: gene không có thông tin
        return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    except Exception:
        return 0.0


def individual_corr_check(
    new_gene: Gene,
    existing_genes: List[Gene],
    train_df: pd.DataFrame,
    threshold: float = CORR_THRESHOLD,
) -> bool:
    """
    Returns True nếu new_gene có corr < threshold với TẤT CẢ existing_genes
    VÀ new_gene không phải hằng số (std > 1e-8).
    """
    try:
        new_series = evaluate(new_gene.formula, train_df).dropna()
    except Exception as exc:
        logger.warning("corr_check eval failed for %r: %s", new_gene.formula, exc)
        return False

    # Loại gene hằng số ngay tại đây
    if new_series.std() < 1e-8:
        logger.debug("corr_check: reject constant gene %r", new_gene.formula)
        return False

    for g in existing_genes:
        try:
            ex_series = evaluate(g.formula, train_df).dropna()
        except Exception:
            continue
        corr = _safe_corr(new_series, ex_series)
        if abs(corr) >= threshold:
            return False
    return True