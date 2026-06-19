import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from model import trainer as trainer_module
from model.trainer import Trainer
from mutator.gene import Gene, Individual


def _sample_df(periods=4, tickers=None):
    tickers = tickers or ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=periods, freq="B"), tickers],
        names=["date", "ticker"],
    )
    seq = pd.Series(range(len(idx)), index=idx, dtype=float)
    df = pd.DataFrame(index=idx)
    df["open"] = 10.0 + seq * 0.01
    df["high"] = df["open"] + 0.5
    df["low"] = df["open"] - 0.5
    df["close"] = df["open"] + 0.2
    df["volume"] = 1000.0 + seq
    df["label"] = seq.groupby(level="date").rank(method="first") / len(tickers)
    return df


class TrainerCacheTests(unittest.TestCase):
    def test_feature_cache_reuses_formula_only_within_same_context(self):
        df = _sample_df()
        other_df = df.copy()
        other_df["close"] = other_df["close"] + 100.0
        individual = Individual(
            genes=[Gene("close_1"), Gene("volume_1")]
        )
        cached_trainer = Trainer(enable_feature_cache=True)
        uncached_trainer = Trainer(enable_feature_cache=False)

        with patch(
            "model.trainer.evaluate",
            wraps=trainer_module.evaluate,
        ) as eval_mock:
            first = cached_trainer._build_feature_matrix(individual, df)
            second = cached_trainer._build_feature_matrix(individual, df)
            self.assertEqual(eval_mock.call_count, 2)
            pd.testing.assert_frame_equal(first, second)

            separate_context = cached_trainer._build_feature_matrix(
                individual, other_df
            )
            self.assertEqual(eval_mock.call_count, 4)

        expected = uncached_trainer._build_feature_matrix(individual, other_df)
        pd.testing.assert_frame_equal(separate_context, expected)

    def test_split_cache_reuses_label_bins_and_groups_per_split_only(self):
        df = _sample_df()
        dates = df.index.get_level_values("date")
        split_a = df[dates < dates.unique()[2]].copy()
        split_b = df[dates >= dates.unique()[2]].copy()
        trainer = Trainer(enable_split_cache=True)

        with (
            patch(
                "model.trainer._bin_labels",
                wraps=trainer_module._bin_labels,
            ) as bin_mock,
            patch(
                "model.trainer._group_sizes",
                wraps=trainer_module._group_sizes,
            ) as group_mock,
        ):
            labels_a = split_a["label"]
            mask_a = labels_a.notna()
            y_a_1, groups_a_1 = trainer._labels_and_groups(
                split_a, labels_a, mask_a
            )
            y_a_2, groups_a_2 = trainer._labels_and_groups(
                split_a, labels_a, mask_a
            )

            self.assertEqual(bin_mock.call_count, 1)
            self.assertEqual(group_mock.call_count, 1)
            pd.testing.assert_series_equal(y_a_1, y_a_2)
            self.assertEqual(groups_a_1, groups_a_2)

            labels_b = split_b["label"]
            mask_b = labels_b.notna()
            trainer._labels_and_groups(split_b, labels_b, mask_b)

            self.assertEqual(bin_mock.call_count, 2)
            self.assertEqual(group_mock.call_count, 2)

    def test_train_path_uses_feature_and_split_caches(self):
        class DummyBooster:
            def predict(self, matrix):
                return np.arange(len(matrix), dtype=float)

        df = _sample_df(periods=6)
        dates = df.index.get_level_values("date")
        unique_dates = dates.unique()
        train_df = df[dates < unique_dates[4]].copy()
        val_df = df[dates >= unique_dates[4]].copy()
        individual = Individual(
            genes=[Gene("close_1"), Gene("volume_1")]
        )
        trainer = Trainer()

        with (
            patch("model.trainer.lgb.train", return_value=DummyBooster()),
            patch(
                "model.trainer.evaluate",
                wraps=trainer_module.evaluate,
            ) as eval_mock,
            patch(
                "model.trainer._bin_labels",
                wraps=trainer_module._bin_labels,
            ) as bin_mock,
            patch(
                "model.trainer._group_sizes",
                wraps=trainer_module._group_sizes,
            ) as group_mock,
        ):
            trainer.train(individual, train_df, val_df, feature_df=df, mode="wf")
            trainer.train(individual, train_df, val_df, feature_df=df, mode="wf")

        self.assertEqual(eval_mock.call_count, 2)
        self.assertEqual(bin_mock.call_count, 2)
        self.assertEqual(group_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
