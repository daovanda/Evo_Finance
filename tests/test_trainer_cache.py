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
    def test_sort_index_if_needed_keeps_sorted_frame_identity(self):
        df = _sample_df()
        sorted_df = trainer_module._sort_index_if_needed(df)
        self.assertIs(sorted_df, df)

        unsorted_df = df.iloc[::-1]
        resorted = trainer_module._sort_index_if_needed(unsorted_df)
        self.assertTrue(resorted.index.is_monotonic_increasing)
        self.assertIsNot(resorted, unsorted_df)

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

    def test_feature_matrix_fails_when_any_gene_cannot_evaluate(self):
        df = _sample_df()
        individual = Individual(
            genes=[Gene("close_1"), Gene("not_a_column_1")]
        )

        with self.assertRaisesRegex(ValueError, "not_a_column_1"):
            Trainer(enable_feature_cache=False)._build_feature_matrix(
                individual,
                df,
            )

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

    def test_split_cache_invalidates_when_label_values_change(self):
        df = _sample_df(periods=2)
        trainer = Trainer(enable_split_cache=True)

        with patch(
            "model.trainer._bin_labels",
            wraps=trainer_module._bin_labels,
        ) as bin_mock:
            labels = df["label"]
            mask = labels.notna()
            y_first, _ = trainer._labels_and_groups(df, labels, mask)

            reversed_by_date = labels.groupby(level="date").transform(
                lambda s: s.iloc[::-1].to_numpy()
            )
            df["label"] = reversed_by_date
            labels = df["label"]
            mask = labels.notna()
            y_second, _ = trainer._labels_and_groups(df, labels, mask)

        self.assertEqual(bin_mock.call_count, 2)
        self.assertFalse(y_first.equals(y_second))

    def test_group_alignment_validation_rejects_mismatched_group_total(self):
        labels = pd.Series([0, 1, 2], dtype=np.int32)

        with self.assertRaisesRegex(ValueError, "group sizes sum to 2"):
            trainer_module._validate_group_alignment(labels, [2], "train")

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

    def test_predict_sorts_feature_context_for_time_series_features(self):
        class EchoBooster:
            def predict(self, matrix):
                return matrix.iloc[:, 0].to_numpy()

        df = _sample_df(periods=4, tickers=["AAA"])
        unsorted_feature_df = df.iloc[::-1]
        individual = Individual(genes=[Gene("shift(close_1, 1)")])

        pred = Trainer(enable_feature_cache=False).predict(
            EchoBooster(),
            individual,
            df,
            feature_df=unsorted_feature_df,
        )
        expected = trainer_module.evaluate("shift(close_1, 1)", df).loc[df.index]

        pd.testing.assert_series_equal(
            pred,
            expected.rename("pred"),
            check_dtype=False,
        )


if __name__ == "__main__":
    unittest.main()
