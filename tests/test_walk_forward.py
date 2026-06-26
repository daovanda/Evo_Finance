import unittest

import pandas as pd

from model.data_utils import (
    label_dataframe,
    make_walk_forward_folds,
    split_labeled_by_dates,
    tickers_missing_sector,
)


def _sample_ohlcv(periods=2200):
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2018-06-04", periods=periods, freq="B"), ["AAA", "BBB"]],
        names=["date", "ticker"],
    )
    df = pd.DataFrame(index=idx)
    seq = pd.Series(range(len(idx)), index=idx, dtype=float)
    df["open"] = 10.0 + seq * 0.01
    df["high"] = df["open"] + 1.0
    df["low"] = df["open"] - 1.0
    df["close"] = df["open"] + 0.5
    df["volume"] = 1000.0 + seq
    return df


class WalkForwardSplitTests(unittest.TestCase):
    def test_make_walk_forward_folds_uses_expanding_train_and_purge(self):
        labeled = label_dataframe(_sample_ohlcv())
        folds = make_walk_forward_folds(
            labeled,
            wf_end="2024-07-29",
            min_train_months=24,
            val_months=6,
            step_months=6,
            purge_days=10,
        )

        self.assertGreaterEqual(len(folds), 6)
        for fold in folds:
            train_dates = fold.train_df.index.get_level_values("date")
            val_dates = fold.val_df.index.get_level_values("date")
            self.assertLess(train_dates.max(), val_dates.min())
            gap = (
                labeled.index.get_level_values("date").unique().sort_values()
                .get_loc(val_dates.min())
                - labeled.index.get_level_values("date").unique().sort_values()
                .get_loc(train_dates.max())
            )
            self.assertGreaterEqual(gap, 10)
        self.assertLess(folds[-1].val_end, pd.Timestamp("2024-07-30"))

    def test_walk_forward_skips_incomplete_last_validation_window(self):
        labeled = label_dataframe(_sample_ohlcv())
        folds = make_walk_forward_folds(
            labeled,
            wf_end="2024-07-29",
            min_train_months=48,
            val_months=12,
            step_months=12,
            purge_days=10,
        )

        self.assertEqual(len(folds), 2)
        self.assertEqual(folds[-1].val_start, pd.Timestamp("2023-06-06"))
        self.assertEqual(folds[-1].val_end, pd.Timestamp("2024-06-06"))

    def test_final_split_still_uses_val_and_test_start(self):
        labeled = label_dataframe(_sample_ohlcv())
        train, val, test = split_labeled_by_dates(
            labeled,
            val_start="2023-05-12",
            test_start="2024-07-29",
            test_end=None,
        )

        self.assertLess(train.index.get_level_values("date").max(), pd.Timestamp("2023-05-12"))
        self.assertGreaterEqual(val.index.get_level_values("date").min(), pd.Timestamp("2023-05-12"))
        self.assertLess(val.index.get_level_values("date").max(), pd.Timestamp("2024-07-29"))
        self.assertGreaterEqual(test.index.get_level_values("date").min(), pd.Timestamp("2024-07-29"))

    def test_final_split_can_purge_boundary_labels(self):
        labeled = label_dataframe(_sample_ohlcv())
        train, val, test = split_labeled_by_dates(
            labeled,
            val_start="2023-05-12",
            test_start="2024-07-29",
            test_end=None,
            purge_days=10,
        )

        unique_dates = labeled.index.get_level_values("date").unique().sort_values()
        val_pos = unique_dates.get_loc(pd.Timestamp("2023-05-12"))
        test_pos = unique_dates.get_loc(pd.Timestamp("2024-07-29"))

        self.assertLessEqual(
            train.index.get_level_values("date").max(),
            unique_dates[val_pos - 11],
        )
        self.assertLessEqual(
            val.index.get_level_values("date").max(),
            unique_dates[test_pos - 11],
        )
        self.assertGreaterEqual(
            test.index.get_level_values("date").min(),
            pd.Timestamp("2024-07-29"),
        )

    def test_tickers_missing_sector_reports_unmapped_tickers(self):
        self.assertEqual(tickers_missing_sector(_sample_ohlcv(periods=5)), ["AAA", "BBB"])


if __name__ == "__main__":
    unittest.main()
