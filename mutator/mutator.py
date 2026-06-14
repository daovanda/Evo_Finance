"""
Evo_Finance — Mutator
──────────────────────
Implements the three mutation strategies (C1, C2, C3) and routes between them
according to configured probabilities.

After each successful gene transformation the gene is offered to the Domain
(domain.try_add) so the pool grows over time.
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from config.settings import (
    MUTATOR_PROBS, WINDOWS, FEATURE_MIN, FEATURE_MAX,
    MAX_RETRY, CORR_THRESHOLD,
)
from mutator.gene import Gene, Individual
from mutator.domain import Domain, individual_corr_check

logger = logging.getLogger(__name__)

_UNARY_OPS  = ("rank", "zscore", "abs", "signed_log")
_BINARY_OPS = ("+", "-", "*", "/")


class Mutator:
    def __init__(self, domain: Domain, seed: int = 42):
        self.domain = domain
        self.rng    = np.random.default_rng(seed)

    # ── Public entry ──────────────────────────────────────────────────────────

    def mutate(self, individual: Individual, train_df: pd.DataFrame) -> Individual:
        """
        Return a *new* Individual (clone) after one mutation step.
        The original individual is never modified.
        """
        child = individual.clone()

        # weighted random choice of strategy
        probs   = [MUTATOR_PROBS["c1"], MUTATOR_PROBS["c2"], MUTATOR_PROBS["c3"]]
        choices = ["c1", "c2", "c3"]
        strategy = self.rng.choice(choices, p=probs)

        if strategy == "c1":
            self._c1(child, train_df)
        elif strategy == "c2":
            self._c2(child, train_df)
        else:
            self._c3(child, train_df)

        return child

    # ── C1: Add / Remove a feature ────────────────────────────────────────────

    def _c1(self, ind: Individual, train_df: pd.DataFrame) -> None:
        action = self.rng.choice(["add", "remove"])

        # boundary enforcement
        if len(ind) >= FEATURE_MAX:
            action = "remove"
        if len(ind) <= FEATURE_MIN:
            action = "add"

        if action == "remove":
            gene = Gene(self.rng.choice(ind.formulas))
            ind.remove_gene(gene)
            logger.debug("C1 remove: %r", gene.formula)
            return

        # add: try up to MAX_RETRY domain genes
        for attempt in range(MAX_RETRY):
            candidate = self.domain.random_gene(self.rng)
            if candidate.formula in ind.formulas:
                continue
            others = [g for g in ind.genes if g.formula != candidate.formula]
            if individual_corr_check(candidate, others, train_df):
                ind.add_gene(candidate)
                logger.debug("C1 add: %r (attempt %d)", candidate.formula, attempt)
                return

        # fallback → C3
        logger.debug("C1 add exhausted retries → fallback C3")
        self._c3(ind, train_df)

    # ── C2: Change window of a gene ───────────────────────────────────────────

    def _c2(self, ind: Individual, train_df: pd.DataFrame) -> None:
        for attempt in range(MAX_RETRY):
            old_gene = Gene(str(self.rng.choice(ind.formulas)))
            new_window = int(self.rng.choice(WINDOWS))
            new_gene = Gene.change_window(old_gene, new_window)

            if new_gene.formula == old_gene.formula:
                continue
            if new_gene.formula in ind.formulas:
                continue

            others = [g for g in ind.genes if g.formula != old_gene.formula]
            if individual_corr_check(new_gene, others, train_df):
                ind.replace_gene(old_gene, new_gene)
                self.domain.try_add(new_gene, train_df)
                logger.debug("C2 window: %r → %r", old_gene.formula, new_gene.formula)
                return

        # fallback → C3
        logger.debug("C2 exhausted retries → fallback C3")
        self._c3(ind, train_df)

    # ── C3: Transform a gene ──────────────────────────────────────────────────

    def _c3(self, ind: Individual, train_df: pd.DataFrame) -> None:
        old_gene = Gene(str(self.rng.choice(ind.formulas)))
        mode = self.rng.choice(["unary", "binary"])

        if mode == "unary":
            success = self._c3_unary(ind, old_gene, train_df)
            if not success:
                self._c3_binary(ind, old_gene, train_df)
        else:
            self._c3_binary(ind, old_gene, train_df)

    def _c3_unary(
        self, ind: Individual, old_gene: Gene, train_df: pd.DataFrame
    ) -> bool:
        for attempt in range(MAX_RETRY):
            op       = str(self.rng.choice(_UNARY_OPS))
            new_gene = Gene.transform(old_gene, op)

            if new_gene.formula in ind.formulas:
                continue
            others = [g for g in ind.genes if g.formula != old_gene.formula]
            if individual_corr_check(new_gene, others, train_df):
                ind.replace_gene(old_gene, new_gene)
                self.domain.try_add(new_gene, train_df)
                logger.debug("C3 unary: %r → %r", old_gene.formula, new_gene.formula)
                return True
        return False

    def _c3_binary(
        self, ind: Individual, old_gene: Gene, train_df: pd.DataFrame
    ) -> None:
        """
        Combine old_gene with a random domain gene via a random binary op.
        Tries up to MAX_RETRY * 2 times; silently skips on exhaustion.
        """
        max_tries = MAX_RETRY * 2
        for attempt in range(max_tries):
            domain_gene = self.domain.random_gene(self.rng)
            op          = str(self.rng.choice(_BINARY_OPS))
            new_gene    = Gene.combine(old_gene, domain_gene, op)

            if new_gene.formula in ind.formulas:
                continue
            others = [g for g in ind.genes if g.formula != old_gene.formula]
            if individual_corr_check(new_gene, others, train_df):
                ind.replace_gene(old_gene, new_gene)
                self.domain.try_add(new_gene, train_df)
                logger.debug(
                    "C3 binary: %r %s %r → %r",
                    old_gene.formula, op, domain_gene.formula, new_gene.formula,
                )
                return

        logger.debug("C3 binary exhausted %d retries — skipping.", max_tries)