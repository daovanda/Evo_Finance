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


if __name__ == "__main__":
    unittest.main()
