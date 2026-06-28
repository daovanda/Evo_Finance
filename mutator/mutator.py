"""
Evo_Finance — Mutator (v2)
───────────────────────────
C1 / C2 / C3 mutation strategies.

C3 unary now includes time-series rolling ops (std, max, min, shift) in
addition to the original cross-sectional ops (rank, zscore, abs, signed_log).
Rolling ops randomly sample a window from WINDOWS.
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from config.settings import (
    MUTATOR_PROBS, WINDOWS, FEATURE_MIN, FEATURE_MAX, MAX_RETRY,
)
from mutator.gene import (
    Gene, Individual, COMPARISON_OPS, CONSTANT_FORMULAS, CONSTANT_VALUES,
    CUMULATIVE_TS_OPS, CS_UNARY_OPS, TS_ROLLING_OPS, FINANCE_TS_OPS,
    PAIR_TS_OPS, BINARY_OPS,
)
from mutator.domain import Domain, individual_corr_check
from mutator.formula_guard import const_threshold_violation, raw_scale_violation

logger = logging.getLogger(__name__)
CONSTANT_BINARY_PROB = 0.25
CONTINUOUS_CONSTANT_PROB = 0.40


def _feature_signature(individual: Individual) -> tuple[str, ...]:
    return tuple(sorted(str(formula) for formula in individual.formulas))


def _is_selectable_formula(formula: str) -> bool:
    return (
        const_threshold_violation(formula) is None
        and raw_scale_violation(formula) is None
    )


class Mutator:
    def __init__(self, domain: Domain, seed: int = 42):
        self.domain = domain
        self.rng    = np.random.default_rng(seed)

    # ── Public entry ──────────────────────────────────────────────────────────

    def mutate(self, individual: Individual, train_df: pd.DataFrame) -> Individual:
        parent_signature = _feature_signature(individual)
        child = individual.clone()
        probs = [MUTATOR_PROBS["c1"], MUTATOR_PROBS["c2"], MUTATOR_PROBS["c3"]]
        strategy = self.rng.choice(["c1", "c2", "c3"], p=probs)

        if strategy == "c1":
            self._c1(child, train_df)
        elif strategy == "c2":
            self._c2(child, train_df)
        else:
            self._c3(child, train_df)

        if _feature_signature(child) == parent_signature:
            raise ValueError(f"{strategy} produced unchanged individual")

        return child

    # ── C1: Add / Remove ─────────────────────────────────────────────────────

    def _c1(self, ind: Individual, train_df: pd.DataFrame) -> None:
        action = self.rng.choice(["add", "remove"])
        if len(ind) >= FEATURE_MAX: action = "remove"
        if len(ind) <= FEATURE_MIN: action = "add"

        if action == "remove":
            gene = Gene(str(self.rng.choice(ind.formulas)))
            ind.remove_gene(gene)
            logger.debug("C1 remove: %r", gene.formula)
            return

        for attempt in range(MAX_RETRY):
            candidate = self._random_selectable_domain_gene()
            if candidate.formula in ind.formulas:
                continue
            others = [g for g in ind.genes if g.formula != candidate.formula]
            if individual_corr_check(candidate, others, train_df):
                ind.add_gene(candidate)
                logger.debug("C1 add: %r (attempt %d)", candidate.formula, attempt)
                return

        logger.debug("C1 add exhausted → fallback C3")
        self._c3(ind, train_df)

    # ── C2: Change window ─────────────────────────────────────────────────────

    def _c2(self, ind: Individual, train_df: pd.DataFrame) -> None:
        for attempt in range(MAX_RETRY):
            old_gene   = Gene(str(self.rng.choice(ind.formulas)))
            new_window = int(self.rng.choice(WINDOWS))
            new_gene   = Gene.change_window(old_gene, new_window)

            if new_gene.formula == old_gene.formula: continue
            if new_gene.formula in ind.formulas:     continue

            if self._try_replace(ind, old_gene, new_gene, train_df, "C2"):

                logger.debug("C2: %r → %r", old_gene.formula, new_gene.formula)
                return

        logger.debug("C2 exhausted → fallback C3")
        self._c3(ind, train_df)

    # ── C3: Transform ─────────────────────────────────────────────────────────

    def _c3(self, ind: Individual, train_df: pd.DataFrame) -> None:
        old_gene = Gene(str(self.rng.choice(ind.formulas)))
        mode     = self.rng.choice(
            ["unary", "binary", "compare", "pair_ts", "where", "rule", "constant"],
            p=[0.23, 0.23, 0.16, 0.10, 0.11, 0.10, 0.07],
        )

        if mode == "unary":
            if not self._c3_unary(ind, old_gene, train_df):
                self._c3_binary(ind, old_gene, train_df)
        elif mode == "compare":
            if not self._c3_compare(ind, old_gene, train_df):
                self._c3_binary(ind, old_gene, train_df)
        elif mode == "pair_ts":
            if not self._c3_pair_ts(ind, old_gene, train_df):
                self._c3_binary(ind, old_gene, train_df)
        elif mode == "where":
            if not self._c3_where(ind, old_gene, train_df):
                self._c3_binary(ind, old_gene, train_df)
        elif mode == "rule":
            if not self._c3_rule(ind, old_gene, train_df):
                self._c3_binary(ind, old_gene, train_df)
        elif mode == "constant":
            if not self._c3_mutate_constant(ind, old_gene, train_df):
                self._c3_binary(ind, old_gene, train_df)
        else:
            self._c3_binary(ind, old_gene, train_df)

    def _c3_unary(self, ind: Individual, old_gene: Gene, train_df: pd.DataFrame) -> bool:
        """
        Try up to MAX_RETRY times to apply a unary transform.

        Pool of ops:
          CS (no window):  rank, zscore, abs, signed_log
          TS (needs win):  std, max, min, shift  → window sampled from WINDOWS
        """
        all_ops = (
            list(CS_UNARY_OPS)
            + list(TS_ROLLING_OPS)
            + list(FINANCE_TS_OPS)
            + list(CUMULATIVE_TS_OPS)
        )

        for _ in range(MAX_RETRY):
            op = str(self.rng.choice(all_ops))

            if op in TS_ROLLING_OPS or op in FINANCE_TS_OPS:
                window   = int(self.rng.choice(WINDOWS))
                new_gene = Gene.transform(old_gene, op, window=window)
            else:
                new_gene = Gene.transform(old_gene, op)

            if self._try_replace(ind, old_gene, new_gene, train_df, "C3 unary"):
                return True

        return False

    def _c3_compare(self, ind: Individual, old_gene: Gene, train_df: pd.DataFrame) -> bool:
        for _ in range(MAX_RETRY * 2):
            op = str(self.rng.choice(list(COMPARISON_OPS)))
            if op in ("cross_above", "cross_below") and self.rng.random() < 0.75:
                other_gene = self._random_selectable_domain_gene()
            elif self.rng.random() < 0.70:
                other_gene = self._random_constant_gene()
            else:
                other_gene = self._random_selectable_domain_gene()

            new_gene = Gene.compare(old_gene, other_gene, op)
            if self._try_replace(ind, old_gene, new_gene, train_df, "C3 compare"):
                return True
        return False

    def _c3_pair_ts(self, ind: Individual, old_gene: Gene, train_df: pd.DataFrame) -> bool:
        for _ in range(MAX_RETRY * 2):
            op = str(self.rng.choice(list(PAIR_TS_OPS)))
            other_gene = self._random_selectable_domain_gene()
            window = int(self.rng.choice(WINDOWS))
            new_gene = Gene.pair_ts(old_gene, other_gene, op, window)
            if self._try_replace(ind, old_gene, new_gene, train_df, "C3 pair_ts"):
                return True
        return False

    def _c3_where(self, ind: Individual, old_gene: Gene, train_df: pd.DataFrame) -> bool:
        for _ in range(MAX_RETRY * 2):
            op = str(self.rng.choice(["gt", "lt"]))
            threshold = self._random_constant_gene()
            cond = Gene.compare(old_gene, threshold, op)
            true_value = Gene.constant(-1.0 if op == "gt" else 1.0)
            false_value = Gene.constant(0.0)
            new_gene = Gene.where(cond, true_value, false_value)
            if self._try_replace(ind, old_gene, new_gene, train_df, "C3 where"):
                return True
        return False

    def _c3_rule(self, ind: Individual, old_gene: Gene, train_df: pd.DataFrame) -> bool:
        threshold_pairs = [
            (30.0, 70.0),
            (20.0, 80.0),
            (-0.02, 0.02),
            (-0.05, 0.05),
            (0.2, 0.8),
            (-1.0, 1.0),
        ]
        for _ in range(MAX_RETRY * 2):
            low, high = threshold_pairs[int(self.rng.integers(len(threshold_pairs)))]
            if self.rng.random() < CONTINUOUS_CONSTANT_PROB:
                low = self._jitter_constant(low)
                high = self._jitter_constant(high)
                if low > high:
                    low, high = high, low
            new_gene = Gene.rule_signal(old_gene, Gene.constant(low), Gene.constant(high))
            if self._try_replace(ind, old_gene, new_gene, train_df, "C3 rule"):
                return True
        return False

    def _c3_mutate_constant(
        self,
        ind: Individual,
        old_gene: Gene,
        train_df: pd.DataFrame,
    ) -> bool:
        for _ in range(MAX_RETRY * 2):
            new_gene = Gene.mutate_constant(old_gene, self.rng)
            if new_gene is None:
                return False
            if self._try_replace(ind, old_gene, new_gene, train_df, "C3 const"):
                return True
        return False

    def _try_replace(
        self,
        ind: Individual,
        old_gene: Gene,
        new_gene: Gene,
        train_df: pd.DataFrame,
        tag: str,
    ) -> bool:
        if new_gene.formula in ind.formulas:
            return False

        others = [g for g in ind.genes if g.formula != old_gene.formula]
        if individual_corr_check(new_gene, others, train_df):
            if (
                new_gene.formula not in self.domain.formulas
                and not self.domain.try_add(new_gene, train_df)
            ):
                return False
            ind.replace_gene(old_gene, new_gene)
            logger.debug("%s: %r", tag, new_gene.formula)
            return True
        return False

    def _random_selectable_domain_gene(self) -> Gene:
        """Draw a domain gene that can legally appear as a selected feature."""
        max_attempts = max(MAX_RETRY * 10, 20)
        for _ in range(max_attempts):
            gene = self.domain.random_gene(self.rng)
            if _is_selectable_formula(gene.formula):
                return gene

        selectable = [
            formula for formula in self.domain.formulas
            if _is_selectable_formula(formula)
        ]
        if not selectable:
            raise ValueError("Domain has no guard-safe selectable formulas.")
        return Gene(str(self.rng.choice(selectable)))

    def _random_constant_gene(self) -> Gene:
        if self.rng.random() < CONTINUOUS_CONSTANT_PROB:
            base = float(self.rng.choice(CONSTANT_VALUES))
            return Gene.constant(self._jitter_constant(base))
        return Gene(str(self.rng.choice(CONSTANT_FORMULAS)))

    def _jitter_constant(self, value: float) -> float:
        scale = max(abs(value) * 0.15, 0.005 if abs(value) < 0.1 else 0.05)
        new_value = value + float(self.rng.normal(0.0, scale))
        if abs(new_value) < 1e-9:
            return 0.0
        return new_value

    def _c3_binary(self, ind: Individual, old_gene: Gene, train_df: pd.DataFrame) -> None:
        """Combine old_gene with a random domain gene via a random binary op."""
        for _ in range(MAX_RETRY * 2):
            if self.rng.random() < CONSTANT_BINARY_PROB:
                domain_gene = self._random_constant_gene()
            else:
                domain_gene = self._random_selectable_domain_gene()
            op          = str(self.rng.choice(list(BINARY_OPS)))
            new_gene    = Gene.combine(old_gene, domain_gene, op)

            if self._try_replace(ind, old_gene, new_gene, train_df, "C3 binary"):
                return

        logger.debug("C3 binary exhausted — skipping.")
