import unittest

import pandas as pd

from mutator.domain import Domain, individual_corr_check
from mutator.formula_guard import (
    const_threshold_violation,
    is_const_threshold_safe,
    is_normalized_for_const_threshold,
    is_raw_scale_safe,
    raw_scale_violation,
)
from mutator.gene import Gene, Individual
from main import (
    _archive_row_to_individual,
    _safe_parent_for_evolution,
    _legacy_guard_violations,
    _safe_seed_individual,
)


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
            "gt(amihud(20), const(0.1))",
        ]

        for formula in unsafe:
            with self.subTest(formula=formula):
                self.assertFalse(is_const_threshold_safe(formula))
                self.assertIsNotNone(const_threshold_violation(formula))

    def test_allows_const_thresholds_on_normalized_features(self):
        safe = [
            "gt(rsi(close_1, 14), const(70))",
            "lt(rank(ret(close_1, 20)), const(0.3))",
            "lt(zscore(ma_ratio(close_1, 20)), const(-1.5))",
            "gt((rsi(close_1, 14) - const(50)), const(0))",
            "where(lt(ret(close_1, 20), const(-0.05)), const(1), const(0))",
            "rule_signal(rsi(close_1, 14), const(30), const(70))",
            "cross_above(ma_ratio(close_1, 20), const(0))",
            "gt(ts_zscore(close_1, 20), const(1.5))",
            "gt(sector_zscore(volume_ratio(20)), const(1.2))",
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

    def test_rejects_raw_scale_selected_features(self):
        unsafe = [
            "open_3",
            "high_1",
            "volume_14",
            "market_close_5",
            "market_close_30",
            "market_close_120",
            "vwap(20)",
            "sum(volume_1, 20)",
            "cum_sum(volume_1)",
            "cum_obv()",
            "dollar_volume()",
            "money_flow()",
            "body()",
            "range()",
            "gap()",
            "upper_wick()",
            "lower_wick()",
            "amihud(20)",
            "rank(open_3)",
            "zscore(volume_14)",
            "rank(market_close_30)",
            "sector_rank(close_1)",
            "sector_zscore(volume_14)",
            "rank(amihud(20))",
            "rank(low_1 * ts_corr(ret(close_1, 1), volume_ratio(20), 20))",
            "(ret(close_1, 20) + zscore(open_3))",
            "ts_corr(open_1, high_1, 20)",
            "ts_corr(open_1, market_close_1, 20)",
            "ts_corr(ret(close_1, 1), market_close_1, 20)",
            "gt(close_1, market_close_1)",
            "lt(open_3, market_close_30)",
            "cross_above(close_1, market_close_30)",
            "gt(volume_14, market_volume_20)",
            "gt(close_1, volume_1)",
            "where(gt(close_1, market_close_1), const(1), const(0))",
            "rule_signal(close_1, market_low_1, market_high_1)",
            "advance_count()",
            "decline_count()",
            "unchanged_count()",
            "sector_code()",
            "sector_size()",
            "sector_advance_count()",
            "sector_decline_count()",
            "sector_unchanged_count()",
            "advance_decline_spread()",
            "sector_advance_decline_spread()",
            "sign(sector_code())",
            "(advance_count() + const(1))",
            "(advance_decline_spread() + const(1))",
            "rank(advance_count())",
            "rank(sector_advance_decline_spread())",
            "sign(close_1 - const(50))",
            "where(sign(close_1 - const(50)), const(1), const(0))",
            "sign(close_1 - market_close_1)",
            "(low_1 * ts_corr(ret(close_1, 1), volume_ratio(20), 20))",
            "((low_1 * ts_corr(ret(close_1, 1), volume_ratio(20), 20)))_w120",
            "cummin((low_1 * ts_corr(ret(close_1, 1), volume_ratio(20), 20)))",
        ]

        for formula in unsafe:
            with self.subTest(formula=formula):
                self.assertFalse(is_raw_scale_safe(formula))
                self.assertIsNotNone(raw_scale_violation(formula))

    def test_allows_normalized_or_relative_selected_features(self):
        safe = [
            "ret(close_1, 20)",
            "logret(close_1, 20)",
            "ma_ratio(close_1, 20)",
            "volume_ratio(20)",
            "rsi(close_1, 14)",
            "rank(ret(close_1, 20))",
            "zscore(volume_ratio(20))",
            "ts_zscore(close_1, 20)",
            "sector_rank(ma_ratio(close_1, 20))",
            "sector_zscore(volume_ratio(20))",
            "market_ret(20)",
            "market_vol(20)",
            "market_drawdown(20)",
            "market_ma_ratio(20)",
            "market_rsi(14)",
            "ts_corr(ret(close_1, 1), volume_ratio(20), 20)",
            "ts_corr(ret(close_1, 1), ret(market_close_1, 1), 20)",
            "ts_cov(ret(close_1, 1), volume_ratio(20), 20)",
            "gt(close_1, open_1)",
            "cross_above(ema(close_1, 12), ema(close_1, 26))",
            "gt(ret(close_1, 20), market_ret(20))",
            "gt(rsi(close_1, 14), market_rsi(14))",
            "sign(close_1 - open_1)",
            "sign(ret(close_1, 20) - const(0.05))",
            "advance_ratio()",
            "decline_ratio()",
            "advance_decline_ratio()",
            "advance_decline_net_pct()",
            "sector_advance_ratio()",
            "sector_decline_ratio()",
            "sector_advance_decline_ratio()",
            "sector_advance_decline_net_pct()",
            "where(gt(rsi(close_1, 14), const(70)), const(-1), const(0))",
            "rule_signal(rsi(close_1, 14), const(30), const(70))",
        ]

        for formula in safe:
            with self.subTest(formula=formula):
                self.assertTrue(is_raw_scale_safe(formula))
                self.assertIsNone(raw_scale_violation(formula))

    def test_normalized_classifier(self):
        self.assertTrue(is_normalized_for_const_threshold("rsi(close_1, 14)"))
        self.assertTrue(is_normalized_for_const_threshold("rank(ret(close_1, 20))"))
        self.assertFalse(is_normalized_for_const_threshold("rank(low_60)"))
        self.assertFalse(is_normalized_for_const_threshold("zscore(volume_14)"))
        self.assertTrue(is_normalized_for_const_threshold("ret(close_1, 20)"))
        self.assertTrue(is_normalized_for_const_threshold("(ema(close_1, 12) / ema(close_1, 26))"))
        self.assertFalse(is_normalized_for_const_threshold("low_60"))
        self.assertFalse(is_normalized_for_const_threshold("days_since_low(high_1)"))
        self.assertFalse(is_normalized_for_const_threshold("neutralize(close_1)"))
        self.assertTrue(is_normalized_for_const_threshold("neutralize(ret(close_1, 1))"))

    def test_domain_and_individual_checks_apply_guard(self):
        empty_df = pd.DataFrame()
        unsafe_gene = Gene("where(lt(low_60, const(48.1738)), const(1), const(0))")
        raw_scale_gene = Gene("market_close_30")
        raw_rank_gene = Gene("rank(close_3)")
        cross_system_gene = Gene("gt(close_1, market_close_1)")
        id_count_gene = Gene("sector_code()")
        breadth_count_gene = Gene("advance_count()")
        breadth_spread_gene = Gene("advance_decline_spread()")

        domain = Domain()
        self.assertFalse(domain.try_add(unsafe_gene, empty_df))
        self.assertFalse(domain.try_add(raw_scale_gene, empty_df))
        self.assertFalse(domain.try_add(raw_rank_gene, empty_df))
        self.assertFalse(domain.try_add(cross_system_gene, empty_df))
        self.assertFalse(domain.try_add(id_count_gene, empty_df))
        self.assertFalse(domain.try_add(breadth_count_gene, empty_df))
        self.assertFalse(domain.try_add(breadth_spread_gene, empty_df))
        self.assertNotIn(unsafe_gene.formula, domain.formulas)
        self.assertFalse(individual_corr_check(unsafe_gene, [], empty_df))
        self.assertFalse(individual_corr_check(raw_scale_gene, [], empty_df))
        self.assertFalse(individual_corr_check(raw_rank_gene, [], empty_df))
        self.assertFalse(individual_corr_check(cross_system_gene, [], empty_df))
        self.assertFalse(individual_corr_check(id_count_gene, [], empty_df))
        self.assertFalse(individual_corr_check(breadth_count_gene, [], empty_df))
        self.assertFalse(individual_corr_check(breadth_spread_gene, [], empty_df))

    def test_seed_formulas_are_guard_safe(self):
        domain = Domain()
        domain.seed()

        unsafe = [
            (formula, const_threshold_violation(formula))
            for formula in domain.formulas
            if const_threshold_violation(formula) is not None
        ]
        self.assertEqual([], unsafe)

        unsafe_cs_normalizers = [
            (formula, raw_scale_violation(formula))
            for formula in domain.formulas
            if (
                raw_scale_violation(formula) is not None
                and "cross-sectional rank/zscore" in raw_scale_violation(formula)
            )
        ]
        self.assertEqual([], unsafe_cs_normalizers)

        unsafe_id_counts = [
            (formula, raw_scale_violation(formula))
            for formula in domain.formulas
            if (
                raw_scale_violation(formula) is not None
                and "ID/count primitive" in raw_scale_violation(formula)
            )
        ]
        self.assertEqual([], unsafe_id_counts)

        raw_scale_seed_violations = [
            (formula, raw_scale_violation(formula))
            for formula in _safe_seed_individual().formulas
            if raw_scale_violation(formula) is not None
        ]
        self.assertEqual([], raw_scale_seed_violations)

    def test_resume_archive_rows_keep_legacy_genes(self):
        legacy = _archive_row_to_individual(
            {
                "score": 1.0,
                "genes": [
                    "where(lt(low_60, const(48.1738)), const(1), const(0))",
                    "market_close_30",
                    "rank(close_3)",
                    "sector_code()",
                    "advance_count()",
                    "advance_decline_spread()",
                ],
            }
        )
        self.assertEqual(
            legacy.formulas,
            [
                "where(lt(low_60, const(48.1738)), const(1), const(0))",
                "market_close_30",
                "rank(close_3)",
                "sector_code()",
                "advance_count()",
                "advance_decline_spread()",
            ],
        )
        self.assertGreaterEqual(len(_legacy_guard_violations(legacy)), 6)

        parent = _safe_parent_for_evolution(legacy)
        self.assertIn("ret(close_1, 20)", parent.formulas)
        self.assertNotIn("market_close_30", parent.formulas)
        self.assertNotIn("rank(close_3)", parent.formulas)
        self.assertNotIn("sector_code()", parent.formulas)
        self.assertEqual(_legacy_guard_violations(parent), [])
        self.assertIn("market_close_30", legacy.formulas)

        full_safe = _safe_parent_for_evolution(
            Individual(
                genes=[Gene(formula=f"ret(close_1, {i})") for i in range(1, 40)],
                generation=3,
            )
        )
        self.assertLessEqual(len(full_safe.formulas), 30)

        individual = _archive_row_to_individual(
            {
                "score": 1.0,
                "genes": ["gt(rsi(close_1, 14), const(70))"],
            }
        )
        self.assertEqual(individual.formulas, ["gt(rsi(close_1, 14), const(70))"])
        self.assertEqual(_legacy_guard_violations(individual), [])


if __name__ == "__main__":
    unittest.main()
