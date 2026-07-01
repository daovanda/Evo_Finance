import unittest
import warnings

import pandas as pd
from scipy.stats import ConstantInputWarning

from fitness.fitness import (
    FitnessEvaluator,
    FoldPrediction,
    _ic_per_date,
    _hit_rate,
    _random_hit_baseline,
    _random_hit_baseline_for_predictions,
)
from mutator.gene import Individual


class FitnessMetricTests(unittest.TestCase):
    def test_ic_per_date_handles_constant_inputs_without_warning(self):
        dates = pd.date_range("2024-01-01", periods=2)
        tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
        idx = pd.MultiIndex.from_product(
            [dates, tickers],
            names=["date", "ticker"],
        )
        pred = pd.Series([1.0] * 6 + list(range(6)), index=idx)
        label = pd.Series(list(range(6)) + [1.0] * 6, index=idx)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ic = _ic_per_date(pred, label, idx)

        constant_warnings = [
            warning for warning in caught
            if isinstance(warning.message, ConstantInputWarning)
        ]
        self.assertEqual(constant_warnings, [])
        self.assertEqual(ic.loc[dates[0]], 0.0)
        self.assertEqual(ic.loc[dates[1]], 0.0)

    def test_hit_rate_uses_available_universe_when_top_k_is_too_large(self):
        dates = pd.date_range("2024-01-01", periods=2)
        tickers = ["AAA", "BBB", "CCC"]
        idx = pd.MultiIndex.from_product(
            [dates, tickers],
            names=["date", "ticker"],
        )
        pred = pd.Series([3.0, 2.0, 1.0, 1.0, 3.0, 2.0], index=idx)
        label = pd.Series([3.0, 2.0, 1.0, 1.0, 3.0, 2.0], index=idx)
        df = pd.DataFrame(index=idx)

        self.assertEqual(_hit_rate(pred, label, idx, top_k=10), 1.0)
        self.assertEqual(_random_hit_baseline(df, top_k=10), 1.0)

    def test_hit_rate_constant_predictions_equal_random_baseline(self):
        dates = pd.date_range("2024-01-01", periods=1)
        tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        idx = pd.MultiIndex.from_product(
            [dates, tickers],
            names=["date", "ticker"],
        )
        pred = pd.Series(1.0, index=idx)
        label = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=idx)
        df = pd.DataFrame(index=idx)

        self.assertEqual(
            _hit_rate(pred, label, idx, top_k=2),
            _random_hit_baseline(df, top_k=2),
        )

    def test_hit_excess_baseline_uses_only_valid_prediction_rows(self):
        date = pd.Timestamp("2024-01-01")
        tickers = [f"T{i:02d}" for i in range(20)]
        idx = pd.MultiIndex.from_product([[date], tickers], names=["date", "ticker"])
        df = pd.DataFrame(index=idx)
        labels = pd.Series(range(20), index=idx, dtype=float)
        pred = pd.Series(1.0, index=idx)
        pred.iloc[:10] = float("nan")

        self.assertEqual(_hit_rate(pred, labels, idx), 1.0)
        self.assertEqual(_random_hit_baseline(df), 0.5)
        self.assertEqual(
            _random_hit_baseline_for_predictions(pred, labels),
            1.0,
        )

        individual = Individual.seed()
        result = FitnessEvaluator().evaluate(
            individual,
            train_pred=pred,
            val_pred=pred,
            train_labels=labels,
            val_labels=labels,
            train_df=df,
            val_df=df,
        )

        self.assertEqual(result.extra["hit_excess"], 0.0)

    def test_walk_forward_fitness_penalizes_fold_train_val_gap(self):
        dates = pd.date_range("2024-01-01", periods=4)
        tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        df = pd.DataFrame(index=idx)
        labels = pd.Series(list(range(6)) * 4, index=idx, dtype=float)

        good_pred = labels.copy()
        bad_val_pred = -labels.copy()
        folds = [
            FoldPrediction(
                name="wf_01",
                train_pred=good_pred.loc[idx[:12]],
                val_pred=good_pred.loc[idx[12:24]],
                train_labels=labels.loc[idx[:12]],
                val_labels=labels.loc[idx[12:24]],
                train_df=df.loc[idx[:12]],
                val_df=df.loc[idx[12:24]],
            ),
            FoldPrediction(
                name="wf_02",
                train_pred=good_pred.loc[idx[:12]],
                val_pred=bad_val_pred.loc[idx[12:24]],
                train_labels=labels.loc[idx[:12]],
                val_labels=labels.loc[idx[12:24]],
                train_df=df.loc[idx[:12]],
                val_df=df.loc[idx[12:24]],
            ),
        ]

        ind = Individual.seed()
        result = FitnessEvaluator().evaluate_walk_forward(ind, folds)

        self.assertGreater(result.extra["wf_overfit_gap"], 0.0)
        self.assertGreater(result.extra["bad_fold_ratio"], 0.0)
        self.assertIn("wf_mean_ic", ind.metrics)

    def test_bad_fold_ratio_flags_negative_hit_excess_even_with_positive_ic(self):
        date = pd.Timestamp("2024-01-01")
        tickers = [f"T{i:02d}" for i in range(20)]
        idx = pd.MultiIndex.from_product([[date], tickers], names=["date", "ticker"])
        df = pd.DataFrame(index=idx)
        labels = pd.Series(range(20), index=idx, dtype=float)
        pred = pd.Series(
            [10, 12, 17, 8, 0, 1, 4, 18, 14, 19, 7, 2, 5, 6, 3, 13, 16, 15, 9, 11],
            index=idx,
            dtype=float,
        )

        folds = [
            FoldPrediction(
                name="wf_01",
                train_pred=labels,
                val_pred=pred,
                train_labels=labels,
                val_labels=labels,
                train_df=df,
                val_df=df,
            )
        ]

        result = FitnessEvaluator().evaluate_walk_forward(Individual.seed(), folds)

        self.assertGreater(result.extra["wf_mean_ic"], 0.0)
        self.assertLess(result.extra["wf_hit_excess"], 0.0)
        self.assertEqual(result.extra["bad_fold_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
