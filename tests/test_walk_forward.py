import unittest
from unittest.mock import patch

import pandas as pd

from config import settings
from model.data_utils import (
    label_dataframe,
    make_walk_forward_folds,
    sector_member_counts,
    small_sectors_in_universe,
    split_labeled_by_dates,
    tickers_missing_sector,
    validate_balanced_panel,
    validate_full_universe_panel,
    validate_market_ohlcv,
    validate_ohlcv,
    validate_temporal_splits,
)


def _sample_ohlcv(periods=2200, start="2018-06-04"):
    idx = pd.MultiIndex.from_product(
        [pd.date_range(start, periods=periods, freq="B"), ["AAA", "BBB"]],
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
    def test_label_dataframe_sorts_by_date_before_forward_label(self):
        idx = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-03"), "AAA"),
                (pd.Timestamp("2024-01-01"), "AAA"),
                (pd.Timestamp("2024-01-02"), "AAA"),
            ],
            names=["date", "ticker"],
        )
        df = pd.DataFrame(index=idx)
        df["open"] = [30.0, 10.0, 20.0]
        df["high"] = df["open"] + 1.0
        df["low"] = df["open"] - 1.0
        df["close"] = [33.0, 11.0, 22.0]
        df["volume"] = 1000.0

        labeled = label_dataframe(df, holding_horizon=1)

        self.assertTrue(labeled.index.is_monotonic_increasing)
        first_label = labeled.loc[(pd.Timestamp("2024-01-01"), "AAA"), "label"]
        self.assertAlmostEqual(first_label, 0.10)

    def test_ohlcv_validation_rejects_invalid_raw_values(self):
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=3), ["AAA"]],
            names=["date", "ticker"],
        )
        df = pd.DataFrame(
            {
                "open": [10.0, 11.0, 12.0],
                "high": [11.0, 10.0, 13.0],
                "low": [9.0, 12.0, 11.0],
                "close": [10.5, 11.5, 12.5],
                "volume": [1000.0, 1000.0, 1000.0],
            },
            index=idx,
        )

        with self.assertRaisesRegex(ValueError, "high >= low"):
            validate_ohlcv(df)

        df.loc[(pd.Timestamp("2024-01-02"), "AAA"), ["high", "low"]] = [12.0, 10.0]
        df.loc[(pd.Timestamp("2024-01-03"), "AAA"), "volume"] = 0.0
        with self.assertRaisesRegex(ValueError, "Volume must be positive"):
            validate_ohlcv(df)

    def test_balanced_panel_validation_requires_full_universe_each_date(self):
        dates = pd.date_range("2024-01-01", periods=3)
        idx = pd.MultiIndex.from_tuples(
            [
                (dates[0], "AAA"),
                (dates[0], "BBB"),
                (dates[1], "AAA"),
                (dates[2], "AAA"),
                (dates[2], "BBB"),
            ],
            names=["date", "ticker"],
        )
        df = pd.DataFrame(index=idx)

        with self.assertRaisesRegex(ValueError, "full ticker universe"):
            validate_balanced_panel(df)

    def test_labeled_split_validation_rejects_partial_label_drop(self):
        labeled = label_dataframe(_sample_ohlcv(periods=80, start="2024-01-01"))
        expected_tickers = labeled.index.get_level_values("ticker").unique()
        bad_date = pd.Timestamp("2024-01-22")
        labeled.loc[(bad_date, "BBB"), "label"] = pd.NA

        train, _, _ = split_labeled_by_dates(
            labeled,
            val_start="2024-02-05",
            test_start="2024-03-04",
            test_end=None,
            purge_days=0,
        )

        with self.assertRaisesRegex(ValueError, "full ticker universe"):
            validate_full_universe_panel(train, expected_tickers, "train split")

    def test_index_validation_rejects_duplicate_date_ticker_rows(self):
        idx = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-01"), "AAA"),
                (pd.Timestamp("2024-01-01"), "AAA"),
            ],
            names=["date", "ticker"],
        )
        df = pd.DataFrame(
            {
                "open": [10.0, 10.0],
                "high": [11.0, 11.0],
                "low": [9.0, 9.0],
                "close": [10.5, 10.5],
                "volume": [1000.0, 1000.0],
            },
            index=idx,
        )

        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            validate_ohlcv(df)

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

    def test_walk_forward_keeps_fold_ending_exactly_at_wf_end(self):
        labeled = label_dataframe(
            _sample_ohlcv(periods=80, start="2020-01-01"),
            holding_horizon=1,
        )
        folds = make_walk_forward_folds(
            labeled,
            wf_end="2020-03-03",
            min_train_months=1,
            val_months=1,
            step_months=1,
            purge_days=0,
        )

        self.assertEqual(len(folds), 1)
        self.assertEqual(folds[0].val_start, pd.Timestamp("2020-02-03"))
        self.assertEqual(folds[0].val_end, pd.Timestamp("2020-03-03"))
        val_dates = folds[0].val_df.index.get_level_values("date")
        self.assertLess(val_dates.max(), pd.Timestamp("2020-03-03"))

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

    def test_final_split_purges_test_tail_when_test_end_is_closed(self):
        horizon = 5
        labeled = label_dataframe(_sample_ohlcv(periods=80, start="2024-01-01"), holding_horizon=horizon)
        _, _, test = split_labeled_by_dates(
            labeled,
            val_start="2024-01-15",
            test_start="2024-02-05",
            test_end="2024-03-15",
            purge_days=horizon,
        )

        unique_dates = labeled.index.get_level_values("date").unique().sort_values()
        end_pos_after = int(unique_dates.searchsorted(pd.Timestamp("2024-03-15"), side="right"))
        expected_last = unique_dates[end_pos_after - horizon - 1]
        test_dates = test.index.get_level_values("date")

        self.assertLessEqual(test_dates.max(), expected_last)
        for date in test_dates.unique():
            target_pos = unique_dates.get_loc(date) + horizon
            self.assertLessEqual(unique_dates[target_pos], pd.Timestamp("2024-03-15"))

    def test_market_validation_rejects_inconsistent_daily_benchmark(self):
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=3), ["AAA", "BBB"]],
            names=["date", "ticker"],
        )
        df = pd.DataFrame(index=idx)
        df["open"] = 10.0
        df["high"] = 11.0
        df["low"] = 9.0
        df["close"] = 10.5
        df["volume"] = 1000.0
        for col, value in {
            "market_open": 1000.0,
            "market_high": 1010.0,
            "market_low": 990.0,
            "market_close": 1005.0,
            "market_volume": 1_000_000.0,
        }.items():
            df[col] = value
        df.loc[(pd.Timestamp("2024-01-02"), "BBB"), "market_close"] = 1006.0

        with self.assertRaisesRegex(ValueError, "identical within each date"):
            validate_market_ohlcv(df)

    def test_market_validation_rejects_invalid_benchmark_values(self):
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=3), ["AAA", "BBB"]],
            names=["date", "ticker"],
        )
        df = pd.DataFrame(index=idx)
        df["open"] = 10.0
        df["high"] = 11.0
        df["low"] = 9.0
        df["close"] = 10.5
        df["volume"] = 1000.0
        df["market_open"] = 1000.0
        df["market_high"] = 990.0
        df["market_low"] = 1010.0
        df["market_close"] = 1005.0
        df["market_volume"] = 1_000_000.0

        with self.assertRaisesRegex(ValueError, "market_high >= market_low"):
            validate_market_ohlcv(df)

        df["market_high"] = 1010.0
        df["market_low"] = 990.0
        df["market_volume"] = 0.0
        with self.assertRaisesRegex(ValueError, "Market volume must be positive"):
            validate_market_ohlcv(df)

    def test_runtime_temporal_validation_rejects_leaky_overrides(self):
        with self.assertRaisesRegex(ValueError, "WF_PURGE_DAYS must be >= HOLDING_HORIZON"):
            validate_temporal_splits(
                val_start="2023-05-12",
                test_start="2024-07-29",
                wf_end="2024-07-29",
                wf_purge_days=9,
                holding_horizon=10,
            )

        with self.assertRaisesRegex(ValueError, "WF_END must be <= TEST_START"):
            validate_temporal_splits(
                val_start="2023-05-12",
                test_start="2024-07-29",
                wf_end="2024-08-01",
                wf_purge_days=10,
                holding_horizon=10,
            )

    def test_tickers_missing_sector_reports_unmapped_tickers(self):
        self.assertEqual(tickers_missing_sector(_sample_ohlcv(periods=5)), ["AAA", "BBB"])

    def test_sector_member_counts_use_loaded_universe(self):
        sectors = {
            "Sector_A": ["AAA", "CCC"],
            "Sector_B": ["BBB", "DDD"],
        }
        with patch.object(settings, "SECTORS", sectors):
            df = _sample_ohlcv(periods=5)
            self.assertEqual(
                sector_member_counts(df),
                {"Sector_A": 1, "Sector_B": 1},
            )
            self.assertEqual(
                small_sectors_in_universe(df, min_members=2),
                {"Sector_A": 1, "Sector_B": 1},
            )
            self.assertEqual(small_sectors_in_universe(df, min_members=1), {})


if __name__ == "__main__":
    unittest.main()
