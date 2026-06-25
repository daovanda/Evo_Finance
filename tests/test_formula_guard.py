import unittest

import pandas as pd

from mutator.domain import Domain, individual_corr_check
from mutator.formula_guard import (
    const_threshold_violation,
    is_const_threshold_safe,
    is_normalized_for_const_threshold,
)
from mutator.gene import Gene
from main import _archive_row_to_individual


class FormulaGuardTests(unittest.TestCase):
    def test_rejects_absolute_const_thresholds(self):
        unsafe = [
            "lt(low_60, const(48.1738))",
            "gt(close_1, const(70))",
            "where(lt(low_60, const(48.1738)), const(1), const(0))",
            "rule_signal(close_1, const(30), const(70))",
            "gt((low_60 - const(48)), const(0))",
            "cross_above(ema(close_1, 12), const(50))",
            "lt(neutralize(close_1), const(1))",
        ]

        for formula in unsafe:
            with self.subTest(formula=formula):
                self.assertFalse(is_const_threshold_safe(formula))
                self.assertIsNotNone(const_threshold_violation(formula))

    def test_allows_const_thresholds_on_normalized_features(self):
        safe = [
            "gt(rsi(close_1, 14), const(70))",
            "lt(rank(low_60), const(0.3))",
            "lt(zscore(low_60), const(-1.5))",
            "gt((rsi(close_1, 14) - const(50)), const(0))",
            "where(lt(ret(close_1, 20), const(-0.05)), const(1), const(0))",
            "rule_signal(rsi(close_1, 14), const(30), const(70))",
            "cross_above(ma_ratio(close_1, 20), const(0))",
            "gt(ts_zscore(close_1, 20), const(1.5))",
            "gt(sector_zscore(volume_5), const(1.2))",
            "gt(advance_ratio(), const(0.55))",
        ]

        for formula in safe:
            with self.subTest(formula=formula):
                self.assertTrue(is_const_threshold_safe(formula))
                self.assertIsNone(const_threshold_violation(formula))

    def test_arithmetic_constants_are_still_allowed(self):
        safe = [
            "(atr(14) * const(1.5))",
            "((ema(close_1, 12) / ema(close_1, 26)) - const(1))",
            "(ret(close_1, 20) / const(2))",
            "(const(0) - rsi(close_1, 14))",
        ]

        for formula in safe:
            with self.subTest(formula=formula):
                self.assertTrue(is_const_threshold_safe(formula))

    def test_normalized_classifier(self):
        self.assertTrue(is_normalized_for_const_threshold("rsi(close_1, 14)"))
        self.assertTrue(is_normalized_for_const_threshold("rank(low_60)"))
        self.assertTrue(is_normalized_for_const_threshold("ret(close_1, 20)"))
        self.assertTrue(is_normalized_for_const_threshold("(ema(close_1, 12) / ema(close_1, 26))"))
        self.assertFalse(is_normalized_for_const_threshold("low_60"))
        self.assertFalse(is_normalized_for_const_threshold("days_since_low(high_1)"))
        self.assertFalse(is_normalized_for_const_threshold("neutralize(close_1)"))
        self.assertTrue(is_normalized_for_const_threshold("neutralize(ret(close_1, 1))"))

    def test_domain_and_individual_checks_apply_guard(self):
        empty_df = pd.DataFrame()
        unsafe_gene = Gene("where(lt(low_60, const(48.1738)), const(1), const(0))")

        domain = Domain()
        self.assertFalse(domain.try_add(unsafe_gene, empty_df))
        self.assertNotIn(unsafe_gene.formula, domain.formulas)
        self.assertFalse(individual_corr_check(unsafe_gene, [], empty_df))

    def test_seed_formulas_are_guard_safe(self):
        domain = Domain()
        domain.seed()

        unsafe = [
            (formula, const_threshold_violation(formula))
            for formula in domain.formulas
            if const_threshold_violation(formula) is not None
        ]
        self.assertEqual([], unsafe)

    def test_resume_archive_rows_reject_unsafe_genes(self):
        with self.assertRaises(ValueError):
            _archive_row_to_individual(
                {
                    "score": 1.0,
                    "genes": ["where(lt(low_60, const(48.1738)), const(1), const(0))"],
                }
            )

        individual = _archive_row_to_individual(
            {
                "score": 1.0,
                "genes": ["gt(rsi(close_1, 14), const(70))"],
            }
        )
        self.assertEqual(individual.formulas, ["gt(rsi(close_1, 14), const(70))"])


if __name__ == "__main__":
    unittest.main()
