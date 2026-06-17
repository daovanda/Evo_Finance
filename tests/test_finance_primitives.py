import unittest

import numpy as np
import pandas as pd

from mutator.domain import Domain
from mutator.evaluator import evaluate, has_division_by_zero
from mutator.gene import (
    BREADTH_NOARG_OPS,
    BREADTH_WINDOW_OPS,
    CS_UNARY_OPS,
    CUMULATIVE_NOARG_OPS,
    CUMULATIVE_TS_OPS,
    FINANCE_NOARG_OPS,
    FINANCE_TS_OPS,
    FINANCE_TWO_WINDOW_OPS,
    FINANCE_WINDOW_OPS,
    Gene,
    Individual,
    MARKET_WINDOW_OPS,
    PAIR_TS_OPS,
    SECTOR_CS_OPS,
    SECTOR_NOARG_OPS,
    SECTOR_WINDOW_OPS,
    TS_ROLLING_OPS,
)
from mutator.mutator import Mutator


FORMULAS = [
    "const(0.5)",
    "const(1)",
    "const(2)",
    "const(-1)",
    "const(-2)",
    "const(-0.5)",
    "const(0)",
    "const(0.01)",
    "const(0.2)",
    "const(1.2)",
    "const(1.5)",
    "const(2.5)",
    "const(30)",
    "const(70)",
    "ema(close_1, 12)",
    "ema(volume_1, 20)",
    "sum(ret(close_1, 1), 20)",
    "sign(ret(close_1, 1))",
    "clip(ret(close_1, 1))",
    "pos_part(ret(close_1, 1))",
    "neg_part(ret(close_1, 1))",
    "ret(close_1, 10)",
    "logret(close_1, 10)",
    "delta(close_1, 10)",
    "vol(close_1, 20)",
    "drawdown(close_1, 20)",
    "breakout(close_1, 20)",
    "ma_ratio(close_1, 20)",
    "bb_pos(close_1, 20)",
    "bb_width(close_1, 20)",
    "rsi(close_1, 14)",
    "pos(20)",
    "volume_ratio(20)",
    "liquidity(20)",
    "atr(14)",
    "stoch(14)",
    "stoch_d(14, 3)",
    "signed_volume()",
    "obv(20)",
    "typical_price()",
    "money_flow()",
    "mfi(14)",
    "cmf(20)",
    "adx(14)",
    "cci(20)",
    "willr(14)",
    "keltner_pos(20)",
    "keltner_width(20)",
    "donchian_width(20)",
    "body()",
    "range()",
    "gap()",
    "upper_wick()",
    "lower_wick()",
    "dollar_volume()",
    "cummax(close_1)",
    "cummin(close_1)",
    "cumret(close_1)",
    "expanding_drawdown(close_1)",
    "expanding_runup(close_1)",
    "days_since_high(close_1)",
    "days_since_low(close_1)",
    "cum_sum(close_1)",
    "cum_sum(volume_1)",
    "cum_obv()",
    "gt(rsi(close_1, 14), const(70))",
    "lt(rsi(close_1, 14), const(30))",
    "cross_above(ema(close_1, 12), ema(close_1, 26))",
    "cross_below(ema(close_1, 12), ema(close_1, 26))",
    "where(gt(rsi(close_1, 14), const(70)), const(-1), const(0))",
    "where(lt(rsi(close_1, 14), const(30)), const(1), const(0))",
    (
        "where(lt(rsi(close_1, 14), const(30)), const(1), "
        "where(gt(rsi(close_1, 14), const(70)), const(-1), const(0)))"
    ),
    "rule_signal(rsi(close_1, 14), const(30), const(70))",
    "winsorize(ret(close_1, 10))",
    "neutralize(ret(close_1, 10))",
    "vol_scale(ret(close_1, 1), 20)",
    "median(close_1, 20)",
    "q25(close_1, 20)",
    "q75(close_1, 20)",
    "iqr(close_1, 20)",
    "skew(ret(close_1, 1), 20)",
    "kurt(ret(close_1, 1), 20)",
    "ts_rank(close_1, 20)",
    "ts_zscore(ret(close_1, 1), 20)",
    "decay_linear(ret(close_1, 1), 20)",
    "slope(close_1, 20)",
    "ts_corr(ret(close_1, 1), volume_ratio(20), 20)",
    "ts_cov(ret(close_1, 1), volume_ratio(20), 20)",
    "ts_beta(ret(close_1, 1), volume_ratio(20), 20)",
    "vwap(20)",
    "vwap_pos(20)",
    "amihud(20)",
    "parkinson_vol(20)",
    "gk_vol(20)",
    "rs_vol(20)",
    "efficiency_ratio(close_1, 20)",
    "ulcer_index(close_1, 20)",
    "days_since_rolling_high(close_1, 20)",
    "days_since_rolling_low(close_1, 20)",
    "aroon_up(20)",
    "aroon_down(20)",
    "aroon_osc(20)",
    "choppiness(20)",
    "up_streak(close_1)",
    "down_streak(close_1)",
    "cum_adl()",
    "cum_pvt()",
    "market_close_1",
    "market_ret(20)",
    "market_vol(20)",
    "market_drawdown(20)",
    "market_ma_ratio(20)",
    "market_rsi(14)",
    "market_pos(20)",
    "market_volume_ratio(20)",
    "rel_ret(20)",
    "rel_strength(20)",
    "market_corr(20)",
    "market_beta(20)",
    "market_alpha(20)",
    "idiosyncratic_vol(20)",
    "up_capture(20)",
    "down_capture(20)",
    "advance_count()",
    "decline_count()",
    "unchanged_count()",
    "advance_ratio()",
    "decline_ratio()",
    "advance_decline_ratio()",
    "advance_decline_spread()",
    "advance_decline_net_pct()",
    "cs_dispersion()",
    "pct_above_ma(20)",
    "pct_above_ma(60)",
    "breadth_momentum(20)",
    "sector_code()",
    "sector_size()",
    "sector_advance_count()",
    "sector_decline_count()",
    "sector_unchanged_count()",
    "sector_advance_ratio()",
    "sector_decline_ratio()",
    "sector_advance_decline_ratio()",
    "sector_advance_decline_spread()",
    "sector_advance_decline_net_pct()",
    "sector_dispersion()",
    "sector_ret(20)",
    "sector_vol(20)",
    "sector_drawdown(20)",
    "sector_ma_ratio(20)",
    "sector_rsi(14)",
    "sector_pos(20)",
    "sector_volume_ratio(20)",
    "rel_sector_ret(20)",
    "sector_rel_strength(20)",
    "sector_corr(20)",
    "sector_beta(20)",
    "sector_alpha(20)",
    "sector_idiosyncratic_vol(20)",
    "sector_up_capture(20)",
    "sector_down_capture(20)",
    "sector_pct_above_ma(20)",
    "sector_breadth_momentum(20)",
    "sector_rank(ret(close_1, 1))",
    "sector_zscore(ret(close_1, 1))",
    "sector_neutralize(ret(close_1, 1))",
    "ts_corr(ret(close_1, 1), ret(market_close_1, 1), 20)",
    "ts_beta(ret(close_1, 1), ret(market_close_1, 1), 20)",
    "(body())_w20",
    "(ret(close_1, 10) / vol(close_1, 20))",
    "(rsi(close_1, 14) - const(70))",
    "(atr(14) * const(1.5))",
    "(ema(close_1, 12) - ema(close_1, 26))",
    "((ema(close_1, 12) / ema(close_1, 26)) - const(1))",
]


def _sample_df(periods=80):
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=periods), ["AAA", "BBB"]],
        names=["date", "ticker"],
    )
    row_num = np.arange(len(idx), dtype=float)
    df = pd.DataFrame(index=idx)
    df["open"] = 10.0 + row_num * 0.03
    df["high"] = df["open"] + 0.8
    df["low"] = df["open"] - 0.6
    df["close"] = df["open"] + 0.2
    df["volume"] = 1000.0 + row_num * 3.0
    dates = df.index.get_level_values("date")
    day_num = pd.Series(pd.factorize(dates)[0], index=df.index, dtype=float)
    market_swing = np.where((day_num.astype(int) % 4) < 2, 2.0, -2.0)
    df["market_open"] = 1000.0 + day_num
    df["market_close"] = df["market_open"] + market_swing
    df["market_high"] = pd.concat([df["market_open"], df["market_close"]], axis=1).max(axis=1) + 6.0
    df["market_low"] = pd.concat([df["market_open"], df["market_close"]], axis=1).min(axis=1) - 6.0
    df["market_volume"] = 1000000.0 + day_num * 1000.0
    return df


class FinancePrimitiveTests(unittest.TestCase):
    def test_finance_primitives_evaluate(self):
        df = _sample_df()

        for formula in FORMULAS:
            with self.subTest(formula=formula):
                series = evaluate(formula, df)
                self.assertEqual(len(series), len(df))
                self.assertFalse(np.isinf(series.dropna()).any())
                self.assertFalse(has_division_by_zero(formula, df))

    def test_registered_ops_have_evaluator_examples(self):
        df = _sample_df()
        examples = []

        examples.extend(f"{op}(ret(close_1, 1))" for op in CS_UNARY_OPS)
        examples.extend(f"{op}(close_1, 20)" for op in TS_ROLLING_OPS)
        examples.extend(f"{op}(close_1, 20)" for op in FINANCE_TS_OPS)
        examples.extend(f"{op}(20)" for op in FINANCE_WINDOW_OPS)
        examples.extend(f"{op}(20)" for op in MARKET_WINDOW_OPS)
        examples.extend(f"{op}()" for op in BREADTH_NOARG_OPS)
        examples.extend(f"{op}(20)" for op in BREADTH_WINDOW_OPS)
        examples.extend(f"{op}(ret(close_1, 1))" for op in SECTOR_CS_OPS)
        examples.extend(f"{op}()" for op in SECTOR_NOARG_OPS)
        examples.extend(f"{op}(20)" for op in SECTOR_WINDOW_OPS)
        examples.extend(f"{op}(14, 3)" for op in FINANCE_TWO_WINDOW_OPS)
        examples.extend(f"{op}()" for op in FINANCE_NOARG_OPS)
        examples.extend(f"{op}(close_1)" for op in CUMULATIVE_TS_OPS)
        examples.extend(f"{op}()" for op in CUMULATIVE_NOARG_OPS)
        examples.extend(
            f"{op}(ret(close_1, 1), volume_ratio(20), 20)"
            for op in PAIR_TS_OPS
        )
        examples.extend(
            [
                "gt(rsi(close_1, 14), const(70))",
                "lt(rsi(close_1, 14), const(30))",
                "cross_above(ema(close_1, 12), ema(close_1, 26))",
                "cross_below(ema(close_1, 12), ema(close_1, 26))",
                "where(gt(rsi(close_1, 14), const(70)), const(-1), const(0))",
                "rule_signal(rsi(close_1, 14), const(30), const(70))",
            ]
        )

        for formula in dict.fromkeys(examples):
            with self.subTest(formula=formula):
                series = evaluate(formula, df)
                self.assertEqual(len(series), len(df))
                self.assertFalse(np.isinf(series.dropna()).any())

    def test_mutator_c3_modes_generate_evaluable_formulas(self):
        df = _sample_df(periods=140)
        domain = Domain()
        domain.seed()
        mutator = Mutator(domain, seed=17)
        individual = Individual.seed()
        captured = []

        def capture(ind_arg, old_gene, new_gene, train_df, tag):
            captured.append((tag, new_gene.formula))
            return True

        mutator._try_replace = capture
        self.assertTrue(mutator._c3_unary(individual, Gene("close_1"), df))
        self.assertTrue(mutator._c3_compare(individual, Gene("rsi(close_1, 14)"), df))
        self.assertTrue(mutator._c3_pair_ts(individual, Gene("ret(close_1, 1)"), df))
        self.assertTrue(mutator._c3_where(individual, Gene("rsi(close_1, 14)"), df))
        self.assertTrue(mutator._c3_rule(individual, Gene("rsi(close_1, 14)"), df))
        self.assertTrue(
            mutator._c3_mutate_constant(
                individual,
                Gene("(atr(14) * const(1.5))"),
                df,
            )
        )

        self.assertEqual(len(captured), 6)
        for tag, formula in captured:
            with self.subTest(tag=tag, formula=formula):
                series = evaluate(formula, df)
                self.assertEqual(len(series), len(df))
                self.assertFalse(np.isinf(series.dropna()).any())

    def test_zero_denominators_are_detected(self):
        df = _sample_df()
        df.loc[df.index[:3], "close"] = 0.0

        self.assertTrue(has_division_by_zero("ret(close_1, 5)", df))
        self.assertTrue(has_division_by_zero("gap()", df))
        self.assertTrue(has_division_by_zero("(high_1 / close_1)", df))
        self.assertTrue(has_division_by_zero("(high_1 / const(0))", df))
        self.assertTrue(has_division_by_zero("cumret(close_1)", df))
        self.assertTrue(has_division_by_zero("expanding_drawdown(close_1)", df))

        flat_close = _sample_df()
        flat_close["close"] = 10.0
        self.assertTrue(has_division_by_zero("efficiency_ratio(close_1, 20)", flat_close))

        flat_range = _sample_df()
        flat_range["high"] = flat_range["close"]
        flat_range["low"] = flat_range["close"]
        self.assertTrue(has_division_by_zero("choppiness(20)", flat_range))

    def test_window_tag_parser_is_not_greedy(self):
        df = _sample_df()
        formula = "((close_1 - open_1)_w20 - (close_1 - open_1)_w30)"

        series = evaluate(formula, df)
        self.assertEqual(len(series), len(df))
        self.assertFalse(has_division_by_zero(formula, df))

    def test_path_dependent_primitives_are_per_ticker(self):
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=5), ["AAA"]],
            names=["date", "ticker"],
        )
        df = pd.DataFrame(index=idx)
        close = pd.Series([1.0, 3.0, 2.0, 4.0, 4.0], index=idx)
        df["close"] = close
        df["open"] = close
        df["high"] = close + 0.1
        df["low"] = close - 0.1
        df["volume"] = 1000.0

        expected_high_age = pd.Series([0.0, 0.0, 1.0, 0.0, 0.0], index=idx)
        expected_low_age = pd.Series([0.0, 1.0, 2.0, 1.0, 2.0], index=idx)
        expected_up_streak = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0], index=idx)
        expected_down_streak = pd.Series([0.0, 0.0, 1.0, 0.0, 0.0], index=idx)

        pd.testing.assert_series_equal(
            evaluate("days_since_rolling_high(close_1, 3)", df),
            expected_high_age,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            evaluate("days_since_rolling_low(close_1, 3)", df),
            expected_low_age,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            evaluate("up_streak(close_1)", df),
            expected_up_streak,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            evaluate("down_streak(close_1)", df),
            expected_down_streak,
            check_names=False,
        )

    def test_sector_primitives_use_settings_mapping(self):
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=4), ["ACB", "VCB", "HPG"]],
            names=["date", "ticker"],
        )
        df = pd.DataFrame(index=idx)
        close_values = [
            10.0, 20.0, 30.0,
            11.0, 19.0, 33.0,
            12.0, 18.0, 36.0,
            11.0, 19.0, 39.0,
        ]
        df["close"] = close_values
        df["open"] = df["close"]
        df["high"] = df["close"] + 1.0
        df["low"] = df["close"] - 1.0
        df["volume"] = 1000.0

        size = evaluate("sector_size()", df)
        self.assertEqual(size.loc[(pd.Timestamp("2024-01-02"), "ACB")], 2.0)
        self.assertEqual(size.loc[(pd.Timestamp("2024-01-02"), "VCB")], 2.0)
        self.assertEqual(size.loc[(pd.Timestamp("2024-01-02"), "HPG")], 1.0)

        adv = evaluate("sector_advance_count()", df)
        dec = evaluate("sector_decline_count()", df)
        self.assertEqual(adv.loc[(pd.Timestamp("2024-01-02"), "ACB")], 1.0)
        self.assertEqual(dec.loc[(pd.Timestamp("2024-01-02"), "ACB")], 1.0)
        self.assertEqual(adv.loc[(pd.Timestamp("2024-01-02"), "HPG")], 1.0)
        self.assertEqual(dec.loc[(pd.Timestamp("2024-01-02"), "HPG")], 0.0)

        neutral = evaluate("sector_neutralize(close_1)", df)
        banking_day = neutral.loc[pd.Timestamp("2024-01-02")]
        self.assertAlmostEqual(float(banking_day.loc[["ACB", "VCB"]].sum()), 0.0)
        self.assertAlmostEqual(float(banking_day.loc["HPG"]), 0.0)

    def test_finance_primitives_do_not_use_future_rows(self):
        base = _sample_df()
        mutated_future = base.copy()

        cutoff = pd.Timestamp("2024-02-15")
        future_mask = mutated_future.index.get_level_values("date") > cutoff
        past_mask = base.index.get_level_values("date") <= cutoff
        mutated_future.loc[
            future_mask,
            [
                "open", "high", "low", "close", "volume",
                "market_open", "market_high", "market_low",
                "market_close", "market_volume",
            ],
        ] *= 100.0

        for formula in FORMULAS:
            with self.subTest(formula=formula):
                original = evaluate(formula, base).loc[past_mask]
                changed = evaluate(formula, mutated_future).loc[past_mask]
                pd.testing.assert_series_equal(original, changed)

    def test_gene_and_domain_wiring(self):
        self.assertEqual(
            Gene.transform(Gene("close_1"), "ret", window=20).formula,
            "ret(close_1, 20)",
        )
        self.assertEqual(
            Gene.transform(Gene("close_1"), "ema", window=20).formula,
            "ema(close_1, 20)",
        )
        self.assertEqual(
            Gene.transform(Gene("close_1"), "bb_pos", window=20).formula,
            "bb_pos(close_1, 20)",
        )
        self.assertEqual(
            Gene.transform(Gene("close_1"), "cummax").formula,
            "cummax(close_1)",
        )
        self.assertEqual(
            Gene.transform(Gene("ret(close_1, 1)"), "vol_scale", window=20).formula,
            "vol_scale(ret(close_1, 1), 20)",
        )
        self.assertEqual(
            Gene.transform(Gene("close_1"), "ts_rank", window=20).formula,
            "ts_rank(close_1, 20)",
        )
        self.assertEqual(
            Gene.transform(Gene("close_1"), "efficiency_ratio", window=20).formula,
            "efficiency_ratio(close_1, 20)",
        )
        self.assertEqual(
            Gene.transform(Gene("close_1"), "up_streak").formula,
            "up_streak(close_1)",
        )
        self.assertEqual(
            Gene.transform(Gene("ret(close_1, 1)"), "sector_rank").formula,
            "sector_rank(ret(close_1, 1))",
        )
        self.assertEqual(
            Gene.pair_ts(
                Gene("ret(close_1, 1)"),
                Gene("volume_ratio(20)"),
                "ts_corr",
                20,
            ).formula,
            "ts_corr(ret(close_1, 1), volume_ratio(20), 20)",
        )
        self.assertEqual(
            Gene.compare(Gene("rsi(close_1, 14)"), Gene("const(70)"), "gt").formula,
            "gt(rsi(close_1, 14), const(70))",
        )
        self.assertEqual(
            Gene.where(Gene("gt(close_1, const(1))"), Gene("const(1)"), Gene("const(0)")).formula,
            "where(gt(close_1, const(1)), const(1), const(0))",
        )
        self.assertEqual(
            Gene.rule_signal(Gene("rsi(close_1, 14)"), Gene("const(30)"), Gene("const(70)")).formula,
            "rule_signal(rsi(close_1, 14), const(30), const(70))",
        )
        self.assertIsNotNone(
            Gene.mutate_constant(
                Gene("(atr(14) * const(1.5))"),
                np.random.default_rng(7),
            )
        )
        self.assertEqual(
            Gene.change_window(Gene("ret(close_1, 20)"), 60).formula,
            "ret(close_1, 60)",
        )
        self.assertEqual(
            Gene.change_window(Gene("pos(20)"), 60).formula,
            "pos(60)",
        )
        self.assertEqual(
            Gene.change_window(Gene("sector_ret(20)"), 60).formula,
            "sector_ret(60)",
        )
        self.assertEqual(
            Gene.change_window(Gene("stoch_d(14, 3)"), 60).formula,
            "stoch_d(60, 3)",
        )
        self.assertEqual(
            Gene.change_window(
                Gene("ts_corr(ret(close_1, 1), volume_ratio(20), 20)"),
                60,
            ).formula,
            "ts_corr(ret(close_1, 1), volume_ratio(20), 60)",
        )

        domain = Domain()
        domain.seed()
        for op in TS_ROLLING_OPS:
            self.assertIn(f"{op}(close_1, 20)", domain.formulas)
            self.assertIn(f"{op}(volume_1, 20)", domain.formulas)
        for op in FINANCE_TS_OPS:
            self.assertIn(f"{op}(close_1, 20)", domain.formulas)
        for op in FINANCE_WINDOW_OPS:
            self.assertIn(f"{op}(20)", domain.formulas)
        for op in MARKET_WINDOW_OPS:
            self.assertIn(f"{op}(20)", domain.formulas)
        for op in BREADTH_NOARG_OPS:
            self.assertIn(f"{op}()", domain.formulas)
        for op in BREADTH_WINDOW_OPS:
            self.assertIn(f"{op}(20)", domain.formulas)
        for op in SECTOR_CS_OPS:
            self.assertIn(f"{op}(ret(close_1, 1))", domain.formulas)
        for op in SECTOR_NOARG_OPS:
            self.assertIn(f"{op}()", domain.formulas)
        for op in SECTOR_WINDOW_OPS:
            self.assertIn(f"{op}(20)", domain.formulas)
        for op in FINANCE_TWO_WINDOW_OPS:
            self.assertIn(f"{op}(20, 3)", domain.formulas)
        for op in FINANCE_NOARG_OPS:
            self.assertIn(f"{op}()", domain.formulas)
        for op in CUMULATIVE_TS_OPS:
            self.assertIn(f"{op}(close_1)", domain.formulas)
        for op in CUMULATIVE_NOARG_OPS:
            self.assertIn(f"{op}()", domain.formulas)
        for op in PAIR_TS_OPS:
            self.assertIn(
                f"{op}(ret(close_1, 1), volume_ratio(20), 20)",
                domain.formulas,
            )
            self.assertIn(
                f"{op}(ret(close_1, 1), ret(market_close_1, 1), 20)",
                domain.formulas,
            )

        for formula in [
            "const(0.5)", "ema(close_1, 20)", "bb_pos(close_1, 20)",
            "rsi(close_1, 14)", "stoch_d(14, 3)", "obv(20)", "mfi(14)",
            "cmf(20)", "adx(14)", "cci(20)", "willr(14)",
            "keltner_pos(20)", "donchian_width(20)",
            "ret(close_1, 20)", "pos(20)", "atr(14)", "body()",
            "const(70)", "const(0.01)", "cummax(close_1)",
            "cummin(close_1)", "cumret(close_1)",
            "expanding_drawdown(close_1)", "expanding_runup(close_1)",
            "days_since_high(close_1)", "days_since_low(close_1)",
            "cum_sum(close_1)", "cum_sum(volume_1)", "cum_obv()",
            "gt(rsi(close_1, 14), const(70))",
            "lt(rsi(close_1, 14), const(30))",
            "cross_above(ema(close_1, 12), ema(close_1, 26))",
            "cross_below(ema(close_1, 12), ema(close_1, 26))",
            "rule_signal(rsi(close_1, 14), const(30), const(70))",
            "winsorize(ret(close_1, 10))",
            "neutralize(ret(close_1, 10))",
            "vol_scale(ret(close_1, 1), 20)",
            "ts_rank(close_1, 20)", "ts_zscore(ret(close_1, 1), 20)",
            "decay_linear(ret(close_1, 1), 20)", "slope(close_1, 20)",
            "iqr(ret(close_1, 1), 20)",
            "ts_corr(ret(close_1, 1), volume_ratio(20), 20)",
            "ts_beta(ret(close_1, 1), volume_ratio(20), 20)",
            "vwap(20)", "vwap_pos(20)", "amihud(20)",
            "parkinson_vol(20)", "gk_vol(20)", "rs_vol(20)",
            "efficiency_ratio(close_1, 20)", "ulcer_index(close_1, 20)",
            "days_since_rolling_high(close_1, 20)",
            "days_since_rolling_low(close_1, 20)",
            "aroon_up(20)", "aroon_down(20)", "aroon_osc(20)",
            "choppiness(20)", "up_streak(close_1)", "down_streak(close_1)",
            "cum_adl()", "cum_pvt()",
            "market_close_1", "market_ret(20)", "market_vol(20)",
            "market_drawdown(20)", "market_ma_ratio(20)", "market_rsi(14)",
            "market_pos(20)", "market_volume_ratio(20)", "rel_ret(20)",
            "rel_strength(20)", "market_corr(20)", "market_beta(20)",
            "market_alpha(20)", "idiosyncratic_vol(20)",
            "up_capture(20)", "down_capture(20)",
            "advance_count()", "decline_count()", "unchanged_count()",
            "advance_ratio()", "decline_ratio()", "advance_decline_ratio()",
            "advance_decline_spread()", "advance_decline_net_pct()",
            "cs_dispersion()", "pct_above_ma(20)", "pct_above_ma(60)",
            "breadth_momentum(20)",
            "sector_code()", "sector_size()", "sector_advance_count()",
            "sector_decline_count()", "sector_unchanged_count()",
            "sector_advance_ratio()", "sector_decline_ratio()",
            "sector_advance_decline_ratio()",
            "sector_advance_decline_spread()",
            "sector_advance_decline_net_pct()", "sector_dispersion()",
            "sector_ret(20)", "sector_vol(20)", "sector_drawdown(20)",
            "sector_ma_ratio(20)", "sector_rsi(14)", "sector_pos(20)",
            "sector_volume_ratio(20)", "rel_sector_ret(20)",
            "sector_rel_strength(20)", "sector_corr(20)", "sector_beta(20)",
            "sector_alpha(20)", "sector_idiosyncratic_vol(20)",
            "sector_up_capture(20)", "sector_down_capture(20)",
            "sector_pct_above_ma(20)", "sector_breadth_momentum(20)",
            "sector_rank(ret(close_1, 1))",
            "sector_zscore(ret(close_1, 1))",
            "sector_neutralize(ret(close_1, 1))",
        ]:
            self.assertIn(formula, domain.formulas)


if __name__ == "__main__":
    unittest.main()
