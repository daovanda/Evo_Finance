import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd

import analyze
from analyze import _hitrate_per_date, _safe_mean, json_to_individual
from fitness.fitness import FitnessEvaluator, FoldPrediction, _ic_per_date, _random_hit_baseline
from mutator.gene import Gene, Individual


class AnalyzeMetricTests(unittest.TestCase):
    def test_filter_archive_entries_can_select_specific_ranks(self):
        entries = [
            {"rank": 1, "genes": ["ret(close_1, 5)"]},
            {"rank": 2, "genes": ["ret(close_1, 10)"]},
            {"rank": 7, "genes": ["ret(close_1, 20)"]},
        ]

        selected = analyze._filter_archive_entries(entries, ranks=[7])
        self.assertEqual([entry["rank"] for entry in selected], [7])

        selected = analyze._filter_archive_entries(entries, top=1, ranks=[2, 7])
        self.assertEqual([entry["rank"] for entry in selected], [2, 7])

        with self.assertRaisesRegex(ValueError, "No archive entries matched"):
            analyze._filter_archive_entries(entries, ranks=[99])

    def test_json_to_individual_strips_and_deduplicates_genes(self):
        individual = json_to_individual(
            {
                "generation": 9,
                "genes": [
                    " ret(close_1, 5) ",
                    "ret(close_1, 5)",
                    "",
                    " volume_ratio(20)",
                ],
            }
        )

        self.assertEqual(
            individual.formulas,
            ["ret(close_1, 5)", "volume_ratio(20)"],
        )
        self.assertEqual(individual.generation, 9)

        with self.assertRaisesRegex(ValueError, "no valid gene formulas"):
            json_to_individual({"genes": [" ", ""]})

    def test_market_shock_windows_include_surrounding_trading_sessions(self):
        dates = pd.bdate_range("2024-01-01", periods=7)
        tickers = ["AAA", "BBB"]
        idx = pd.MultiIndex.from_product(
            [dates, tickers],
            names=["date", "ticker"],
        )
        market_close_by_date = pd.Series(
            [100.0, 96.0, 97.0, 98.0, 99.0, 95.0, 96.0],
            index=dates,
        )
        df = pd.DataFrame(index=idx)
        df["market_close"] = [
            market_close_by_date.loc[date]
            for date, _ in idx
        ]

        windows = analyze._market_shock_windows(
            df,
            threshold=0.03,
            backward_days=2,
            forward_days=2,
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0][0], dates[0])
        self.assertEqual(windows[0][1] - pd.Timedelta(days=1), dates[6])

    def test_market_shock_windows_use_absolute_market_return(self):
        dates = pd.bdate_range("2024-01-01", periods=6)
        idx = pd.MultiIndex.from_product(
            [dates, ["AAA", "BBB"]],
            names=["date", "ticker"],
        )
        market_close_by_date = pd.Series(
            [100.0, 104.0, 104.0, 100.0, 100.0, 103.5],
            index=dates,
        )
        df = pd.DataFrame(index=idx)
        df["market_close"] = [
            market_close_by_date.loc[date]
            for date, _ in idx
        ]

        windows = analyze._market_shock_windows(
            df,
            threshold=0.03,
            backward_days=0,
            forward_days=0,
        )

        shock_dates = [start for start, _ in windows]
        self.assertEqual(shock_dates, [dates[1], dates[3], dates[5]])

    def test_hitrate_per_date_uses_available_names_when_top_k_is_too_large(self):
        dates = pd.date_range("2024-01-01", periods=2)
        tickers = ["AAA", "BBB", "CCC"]
        idx = pd.MultiIndex.from_product(
            [dates, tickers],
            names=["date", "ticker"],
        )
        pred = pd.Series([3.0, 2.0, 1.0, 1.0, 3.0, 2.0], index=idx)
        label = pd.Series([3.0, 2.0, 1.0, 1.0, 3.0, 2.0], index=idx)

        hitrate = _hitrate_per_date(pred, label, top_k=10)

        self.assertEqual(hitrate.tolist(), [1.0, 1.0])

    def test_hitrate_per_date_ignores_non_finite_values(self):
        dates = pd.date_range("2024-01-01", periods=1)
        tickers = ["AAA", "BBB", "CCC"]
        idx = pd.MultiIndex.from_product(
            [dates, tickers],
            names=["date", "ticker"],
        )
        pred = pd.Series([np.inf, 2.0, 1.0], index=idx)
        label = pd.Series([3.0, 2.0, 1.0], index=idx)

        hitrate = _hitrate_per_date(pred, label, top_k=2)

        self.assertEqual(hitrate.tolist(), [1.0])

    def test_hitrate_per_date_constant_predictions_use_baseline(self):
        dates = pd.date_range("2024-01-01", periods=1)
        tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        idx = pd.MultiIndex.from_product(
            [dates, tickers],
            names=["date", "ticker"],
        )
        pred = pd.Series(1.0, index=idx)
        label = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=idx)

        hitrate = _hitrate_per_date(pred, label, top_k=2)

        self.assertEqual(hitrate.tolist(), [0.4])

    def test_safe_mean_matches_fitness_empty_series_behavior(self):
        self.assertEqual(_safe_mean(pd.Series(dtype=float)), 0.0)
        self.assertEqual(_safe_mean(pd.Series([np.inf, -np.inf])), 0.0)

    def test_missing_sector_mapping_raises_by_default(self):
        with patch.object(analyze, "REQUIRE_SECTOR_MAPPING", True):
            with self.assertRaisesRegex(ValueError, "SECTORS mapping is missing"):
                analyze._handle_missing_sector_mapping(["AAA"])

    def test_missing_sector_mapping_can_warn_when_disabled(self):
        with patch.object(analyze, "REQUIRE_SECTOR_MAPPING", False):
            with self.assertLogs("evo_finance.analyze", level="WARNING") as logs:
                analyze._handle_missing_sector_mapping(["AAA"])

        self.assertIn("Unknown", "\n".join(logs.output))

    def test_small_sector_universe_raises_by_default(self):
        with patch.object(analyze, "REQUIRE_SECTOR_MAPPING", True):
            with self.assertRaisesRegex(ValueError, "too few loaded tickers"):
                analyze._handle_small_sector_universe({"Banking": 1})

    def test_small_sector_universe_can_warn_when_disabled(self):
        with patch.object(analyze, "REQUIRE_SECTOR_MAPPING", False):
            with self.assertLogs("evo_finance.analyze", level="WARNING") as logs:
                analyze._handle_small_sector_universe({"Banking": 1})

        self.assertIn("noisy", "\n".join(logs.output))

    def test_walk_forward_recompute_does_not_score_partial_folds(self):
        tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
        dates = pd.date_range("2024-01-01", periods=4)
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        df = pd.DataFrame(index=idx)
        df["label"] = np.tile(np.arange(len(tickers), dtype=float), len(dates))

        fold_1 = SimpleNamespace(
            name="wf_01",
            train_df=df.loc[dates[:2]],
            val_df=df.loc[dates[2:3]],
            train_start=dates[0],
            train_end=dates[1],
            val_start=dates[2],
            val_end=dates[3],
        )
        fold_2 = SimpleNamespace(
            name="wf_02",
            train_df=df.loc[dates[:3]],
            val_df=df.loc[dates[3:]],
            train_start=dates[0],
            train_end=dates[2],
            val_start=dates[3],
            val_end=dates[3] + pd.Timedelta(days=1),
        )

        class PartiallyFailingTrainer:
            def __init__(self):
                self.calls = 0

            def train(self, individual, train_df, val_df, feature_df=None, mode="wf"):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("boom")
                return (
                    object(),
                    train_df["label"],
                    val_df["label"],
                    train_df["label"],
                    val_df["label"],
                )

        with patch.object(analyze, "Trainer", PartiallyFailingTrainer):
            metrics, ic_series, hr_series, wf_dates = analyze.compute_walk_forward_metrics(
                Individual(genes=[Gene("ret(close_1, 1)")]),
                [fold_1, fold_2],
                df,
            )

        self.assertTrue(metrics.empty)
        self.assertTrue(ic_series.empty)
        self.assertTrue(hr_series.empty)
        self.assertEqual(wf_dates.tolist(), [dates[2], dates[3]])

    def test_analyze_wf_aggregate_matches_fitness_score(self):
        dates = pd.date_range("2024-01-01", periods=4)
        tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        df = pd.DataFrame(index=idx)
        labels = pd.Series(list(range(6)) * 4, index=idx, dtype=float)
        good_pred = labels.copy()
        mixed_pred = labels.copy()
        mixed_pred.loc[idx[18:24]] = -mixed_pred.loc[idx[18:24]]

        folds = [
            FoldPrediction(
                name="wf_01",
                train_pred=good_pred.loc[idx[:12]],
                val_pred=good_pred.loc[idx[12:18]],
                train_labels=labels.loc[idx[:12]],
                val_labels=labels.loc[idx[12:18]],
                train_df=df.loc[idx[:12]],
                val_df=df.loc[idx[12:18]],
            ),
            FoldPrediction(
                name="wf_02",
                train_pred=good_pred.loc[idx[:18]],
                val_pred=mixed_pred.loc[idx[18:24]],
                train_labels=labels.loc[idx[:18]],
                val_labels=labels.loc[idx[18:24]],
                train_df=df.loc[idx[:18]],
                val_df=df.loc[idx[18:24]],
            ),
        ]

        individual = Individual(genes=[Gene("ret(close_1, 1)")])
        fitness_result = FitnessEvaluator().evaluate_walk_forward(individual, folds)

        rows = []
        ic_parts = []
        for fold in folds:
            train_ic = _ic_per_date(fold.train_pred, fold.train_labels, fold.train_df.index)
            val_ic = _ic_per_date(fold.val_pred, fold.val_labels, fold.val_df.index)
            hit_series = _hitrate_per_date(fold.val_pred, fold.val_labels)
            baseline = _random_hit_baseline(fold.val_df)
            rows.append(
                {
                    "train_mean_ic": _safe_mean(train_ic),
                    "val_mean_ic": _safe_mean(val_ic),
                    "hit_rate": _safe_mean(hit_series),
                    "hit_excess": _safe_mean(hit_series) - baseline,
                    "overfit_gap": max(0.0, _safe_mean(train_ic) - _safe_mean(val_ic)),
                }
            )
            ic_parts.append(val_ic)

        aggregate = analyze._aggregate_wf_metrics(
            pd.DataFrame(rows),
            pd.concat(ic_parts),
        )

        self.assertAlmostEqual(aggregate["score"], fitness_result.score)
        for key in (
            "wf_mean_ic",
            "wf_icir",
            "wf_hit_rate",
            "wf_hit_excess",
            "wf_ic_std",
            "bad_fold_ratio",
            "wf_overfit_gap",
        ):
            self.assertAlmostEqual(aggregate[key], fitness_result.extra[key])

    def test_analyze_bad_fold_ratio_flags_negative_hit_excess(self):
        aggregate = analyze._aggregate_wf_metrics(
            pd.DataFrame(
                [
                    {
                        "train_mean_ic": 0.20,
                        "val_mean_ic": 0.05,
                        "hit_rate": 0.40,
                        "hit_excess": -0.10,
                        "overfit_gap": 0.15,
                    },
                    {
                        "train_mean_ic": 0.15,
                        "val_mean_ic": 0.04,
                        "hit_rate": 0.60,
                        "hit_excess": 0.10,
                        "overfit_gap": 0.11,
                    },
                ]
            ),
            pd.Series([0.05, 0.04]),
        )

        self.assertEqual(aggregate["bad_fold_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
