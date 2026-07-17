import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from crypto import config
from crypto.backtest import (
    ModelSpec,
    SplitSignals,
    _backtest_name,
    _dynamic_tp_arrays,
    _parse_spec,
    _simulate_dynamic_tp_arrays,
    _summarize_split,
    _sweep_table,
)
from crypto.main import _validate_resume_metadata
from crypto.data import CryptoFold, add_binary_labels, split_labeled_by_dates
from crypto.evolution import CryptoArchive, CryptoIndividual, CryptoMutator
from crypto.expression import CryptoFeatureSpace
from crypto.features import RAW_SCALE_COLUMNS, build_feature_frame, selectable_features
from crypto.fitness import CryptoFitnessEvaluator, _internal_early_stop_split


class CryptoPipelineTests(unittest.TestCase):
    def test_backtest_name_stays_within_windows_safe_length(self):
        base_specs = [
            ModelSpec(
                archive_path=Path(f"crypto_btc_archive_with_a_long_name_{i}.json"),
                rank=1,
                label_mode="mfe",
                label_threshold=0.003,
                top_fraction=0.15,
            )
            for i in range(5)
        ]
        exit1 = ModelSpec(
            Path("crypto_btc_exit_after_h1_h5_tp04_seed1_12h.json"),
            1,
            "exit_after_h1",
            0.004,
            0.20,
        )
        exit2 = ModelSpec(
            Path("crypto_btc_exit_after_h2_h5_tp04_seed1_12h.json"),
            1,
            "exit_after_h2",
            0.004,
            0.10,
        )

        name = _backtest_name(base_specs, exit1, exit2, "and", 0.005)

        self.assertLessEqual(len(name), 150)
        self.assertTrue(name.startswith("strategy_e1top_20p000pct_e2top_10p000pct"))
        self.assertIn("_bases5_", name)

    def test_dynamic_tp_simulator_routes_all_five_terminal_branches(self):
        frame = pd.DataFrame(
            {
                "high_h1": [0.006, 0.0, 0.0, 0.0, 0.0],
                "high_h2": [0.0, 0.0, 0.005, 0.0, 0.0],
                "max_high_h2_plus": [0.0, 0.005, 0.005, 0.005, 0.004],
                "max_high_h3_plus": [0.0, 0.0, 0.0, 0.005, 0.004],
                "close_final": [-0.01] * 5,
                "exit1_selected": [False, True, False, False, False],
                "exit2_selected": [False, False, False, True, False],
            }
        )
        thresholds = (0.005, 0.0045, 0.004, 0.004, 0.003)

        metrics = _simulate_dynamic_tp_arrays(
            _dynamic_tp_arrays(frame),
            thresholds,
        )

        expected_gross = sum(thresholds) / len(thresholds)
        self.assertAlmostEqual(
            metrics["e_net"],
            expected_gross - config.TRADE_COST,
        )
        self.assertEqual(metrics["hit_rate"], 1.0)
        self.assertEqual(metrics["n_trades"], 5.0)

    def test_backtest_part1_counts_h1_low_drawdown_rates(self):
        idx = pd.date_range("2024-01-01", periods=8, freq="15min")
        selected = pd.Index(idx[:3])
        base_split = SplitSignals(
            split="val",
            data=pd.DataFrame(index=idx),
            selected_index=selected,
            pred_threshold=0.5,
            top_fraction=0.25,
        )
        empty_exit = SplitSignals(
            split="val",
            data=pd.DataFrame(index=idx),
            selected_index=pd.Index([]),
            pred_threshold=0.5,
            top_fraction=0.20,
        )
        path_returns = pd.DataFrame(index=idx)
        for horizon in range(1, 6):
            path_returns[f"high_h{horizon}"] = 0.0
            path_returns[f"close_h{horizon}"] = 0.0
        path_returns["low_h1"] = [-0.0005, -0.001, -0.003] + [0.0] * 5
        path_returns["close_h2"] = [-0.001, -0.002, -0.003] + [0.0] * 5

        summary, tp_rows = _summarize_split(
            split="val",
            base_split=base_split,
            exit1_split=empty_exit,
            exit2_split=empty_exit,
            path_returns=path_returns,
            raw_index=pd.DatetimeIndex(idx),
            tp_threshold=0.004,
            tp_sweep_thresholds=[0.004],
        )

        self.assertEqual(summary["base_low_h1_le_neg01"], 2)
        self.assertAlmostEqual(summary["base_low_h1_le_neg01_rate"], 2 / 3)
        self.assertEqual(summary["base_low_h1_le_neg005"], 3)
        self.assertAlmostEqual(summary["base_low_h1_le_neg005_rate"], 1.0)
        tp_sweep = pd.DataFrame(tp_rows)
        exit2_no_selected = tp_sweep[
            tp_sweep["group"] == "p2_exit_after_h2_no_selected"
        ]
        self.assertAlmostEqual(
            float(exit2_no_selected["close_h2_return_mean"].iloc[0]),
            -0.002,
        )
        rendered = _sweep_table(
            tp_sweep,
            group_name="p2_exit_after_h2_no_selected",
            include_close_h2=True,
        )
        self.assertIn("val mean close H2", rendered["split"].tolist())

    def test_backtest_base_specs_accept_independent_top_fractions(self):
        mfe_spec = _parse_spec(
            "mfe.json#1#mfe#0.003#0.15",
            default_top_fraction=0.25,
        )
        close_spec = _parse_spec(
            "close.json#1#close_exit#-0.001#0.25",
            default_top_fraction=0.15,
        )

        self.assertEqual(mfe_spec.top_fraction, 0.15)
        self.assertEqual(close_spec.top_fraction, 0.25)

    def test_config_allows_finite_negative_label_threshold(self):
        old_threshold = config.LABEL_THRESHOLD
        try:
            config.LABEL_THRESHOLD = -0.001
            config.validate_config()
        finally:
            config.LABEL_THRESHOLD = old_threshold

    def test_binary_label_uses_next_open_and_future_close(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="15min")
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [101.0, 102.0, 103.0, 104.0, 105.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "close": [100.5, 102.5, 103.5, 104.5, 105.5],
                "volume": [10.0] * 5,
                "trade_count": [10] * 5,
                "taker_buy_base_volume": [5.0] * 5,
                "taker_buy_quote_volume": [500.0] * 5,
            },
            index=idx,
        )

        labeled = add_binary_labels(
            df,
            horizons=[2],
            threshold=0.001,
            return_fn=config.close_exit_future_return,
        )

        expected = (df["close"].shift(-2) - df["open"].shift(-1)) / df["open"].shift(-1)
        pd.testing.assert_series_equal(
            labeled["future_return_h2"],
            expected.rename("future_return_h2"),
        )
        self.assertEqual(labeled["label_h2"].iloc[0], 1.0)
        self.assertTrue(pd.isna(labeled["label_h2"].iloc[-1]))

    def test_mfe_label_uses_next_open_and_future_max_high(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="15min")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [100.0, 102.0, 101.0, 104.0, 103.0],
                "low": [99.0] * 5,
                "close": [100.0] * 5,
                "volume": [10.0] * 5,
                "trade_count": [10] * 5,
                "taker_buy_base_volume": [5.0] * 5,
                "taker_buy_quote_volume": [500.0] * 5,
            },
            index=idx,
        )

        labeled = add_binary_labels(
            df,
            horizons=[3],
            threshold=0.03,
            return_fn=config.mfe_future_return,
        )

        expected = pd.Series(
            [0.04, 0.04, np.nan, np.nan, np.nan],
            index=idx,
            name="future_return_h3",
        )
        pd.testing.assert_series_equal(labeled["future_return_h3"], expected)
        self.assertEqual(labeled["label_h3"].iloc[0], 1.0)
        self.assertTrue(pd.isna(labeled["label_h3"].iloc[-1]))

    def test_close_path_mean_label_uses_mean_future_closes_from_next_open(self):
        idx = pd.date_range("2024-01-01", periods=6, freq="15min")
        df = pd.DataFrame(
            {
                "open": [100.0] * 6,
                "high": [111.0] * 6,
                "low": [89.0] * 6,
                "close": [100.0, 99.0, 101.0, 103.0, 90.0, 110.0],
                "volume": [10.0] * 6,
                "trade_count": [10] * 6,
                "taker_buy_base_volume": [5.0] * 6,
                "taker_buy_quote_volume": [500.0] * 6,
            },
            index=idx,
        )

        labeled = add_binary_labels(
            df,
            horizons=[3],
            label_mode="close_path_mean",
            threshold=0.0,
        )

        expected = pd.Series(
            [0.01, -0.02, 0.01, np.nan, np.nan, np.nan],
            index=idx,
            name="future_return_h3",
        )
        pd.testing.assert_series_equal(labeled["future_return_h3"], expected)
        self.assertEqual(labeled["label_h3"].iloc[0], 1.0)
        self.assertEqual(labeled["label_h3"].iloc[1], 0.0)
        self.assertTrue(labeled["label_h3"].iloc[2] == 1.0)
        self.assertTrue(labeled["label_h3"].iloc[3:].isna().all())

    def test_close_path_mean_is_registered_as_default_mode(self):
        self.assertEqual(config.LABEL_MODE, "close_path_mean")
        self.assertEqual(config.default_label_threshold("close_path_mean"), 0.0)
        self.assertIs(
            config.get_label_return_fn("close_path_mean"),
            config.close_path_mean_future_return,
        )

    def test_payoff_label_uses_tp_or_future_close_and_preserves_tail_nan(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="15min")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [100.0, 100.5, 100.0, 100.0, 100.0],
                "low": [99.0] * 5,
                "close": [100.0, 100.0, 100.0, 100.0, 99.0],
                "volume": [10.0] * 5,
                "trade_count": [10] * 5,
                "taker_buy_base_volume": [5.0] * 5,
                "taker_buy_quote_volume": [500.0] * 5,
            },
            index=idx,
        )

        old_tp = config.PAYOFF_TP
        try:
            config.PAYOFF_TP = 0.003
            labeled = add_binary_labels(df, horizons=[3], label_mode="payoff")
        finally:
            config.PAYOFF_TP = old_tp

        expected = pd.Series(
            [0.003, -0.01, np.nan, np.nan, np.nan],
            index=idx,
            name="future_return_h3",
        )
        pd.testing.assert_series_equal(labeled["future_return_h3"], expected)
        self.assertEqual(labeled["label_h3"].iloc[0], 1.0)
        self.assertEqual(labeled["label_h3"].iloc[1], 0.0)
        self.assertTrue(pd.isna(labeled["label_h3"].iloc[-1]))

    def test_payoff_default_label_threshold_uses_trade_cost(self):
        self.assertEqual(
            config.default_label_threshold("payoff"),
            config.TRADE_COST,
        )
        self.assertEqual(
            config.default_label_threshold("mfe"),
            config.LABEL_THRESHOLD,
        )

    def test_crypto_archive_preserves_metadata_on_load(self):
        archive = CryptoArchive(metadata={"label_mode": "mfe", "label_threshold": 0.003})
        individual = CryptoIndividual(features=["ret_close_3"], score=0.1)
        self.assertTrue(archive.try_add(individual))

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "archive.json"
            archive.save(path, metadata=archive.metadata)
            loaded = CryptoArchive.load(path)

        self.assertEqual(loaded.metadata["label_mode"], "mfe")
        self.assertEqual(loaded.metadata["label_threshold"], 0.003)
        self.assertEqual(len(loaded), 1)

    def test_resume_metadata_mismatch_is_rejected(self):
        archive = CryptoArchive(
            metadata={
                "horizons": [3, 5],
                "label_mode": "mfe",
                "label_threshold": 0.003,
                "trade_top_fraction": config.TRADE_TOP_FRACTION,
                "trade_cost": config.TRADE_COST,
            }
        )

        with self.assertRaisesRegex(ValueError, "label_mode"):
            _validate_resume_metadata(
                archive=archive,
                resume_path=Path("archive.json"),
                horizons=[3, 5],
                label_mode="payoff",
                label_threshold=config.TRADE_COST,
            )

    def test_label_mode_mfe_is_used_by_default_labeling(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="15min")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [100.0, 100.5, 101.5, 100.2, 100.1],
                "low": [99.0] * 5,
                "close": [100.0] * 5,
                "volume": [10.0] * 5,
                "trade_count": [10] * 5,
                "taker_buy_base_volume": [5.0] * 5,
                "taker_buy_quote_volume": [500.0] * 5,
            },
            index=idx,
        )

        old_mode = config.LABEL_MODE
        try:
            config.LABEL_MODE = "mfe"
            labeled = add_binary_labels(df, horizons=[2], threshold=0.01)
        finally:
            config.LABEL_MODE = old_mode

        self.assertEqual(labeled["label_h2"].iloc[0], 1.0)
        self.assertTrue(pd.isna(labeled["label_h2"].iloc[-1]))

    def test_feature_pool_excludes_raw_scale_columns(self):
        df = _synthetic_crypto_frame(700)
        features = build_feature_frame(df, windows=[3, 5, 10, 20], min_valid_ratio=0.5)
        pool = selectable_features(features)

        self.assertGreater(len(pool), 20)
        self.assertFalse(set(pool) & RAW_SCALE_COLUMNS)

    def test_advanced_volatility_and_imbalance_features_are_available(self):
        df = _synthetic_crypto_frame(900)
        features = build_feature_frame(df, windows=[10, 20], min_valid_ratio=0.5)
        pool = set(selectable_features(features))

        expected = {
            "realized_vol_20",
            "downside_realized_vol_20",
            "upside_realized_vol_20",
            "parkinson_vol_20",
            "garman_klass_vol_20",
            "rogers_satchell_vol_20",
            "vol_of_vol_20",
            "taker_delta_sum_ratio_20",
            "taker_delta_z_20",
            "buy_pressure_persistence_20",
            "imbalance_x_high_volume_20",
            "imbalance_x_high_volatility_20",
            "imbalance_return_corr_20",
        }
        self.assertTrue(expected <= pool, sorted(expected - pool))

    def test_seed_individual_samples_randomly_from_feature_pool(self):
        pool = [f"feature_{idx}" for idx in range(12)]
        idx = pd.date_range("2024-01-01", periods=30, freq="15min")
        feature_df = pd.DataFrame(
            {
                name: np.linspace(float(pos), float(pos + 1), len(idx))
                for pos, name in enumerate(pool)
            },
            index=idx,
        )
        seed = 123
        mutator = CryptoMutator(pool, feature_df, idx, seed=seed)

        individual = mutator.seed_individual()

        expected_idx = np.random.default_rng(seed).choice(
            len(pool),
            size=config.FEATURE_MIN,
            replace=False,
        )
        expected = [pool[int(pos)] for pos in expected_idx]
        self.assertEqual(individual.features, expected)
        self.assertEqual(len(individual.features), config.FEATURE_MIN)
        self.assertEqual(len(set(individual.features)), config.FEATURE_MIN)
        self.assertTrue(set(individual.features) <= set(pool))

    def test_crypto_mutator_c1_adds_from_domain(self):
        idx, feature_df = _synthetic_feature_space(
            [f"feature_{idx}" for idx in range(12)]
        )
        pool = list(feature_df.columns)
        mutator = CryptoMutator(pool, feature_df, idx, seed=7)
        child = mutator.seed_individual()

        changed = mutator._c1(child)

        self.assertTrue(changed)
        self.assertEqual(len(child.features), config.FEATURE_MIN + 1)
        self.assertEqual(len(set(child.features)), len(child.features))
        self.assertTrue(set(child.features) <= set(pool))

    def test_crypto_mutator_c2_changes_window(self):
        pool = [f"alpha_{window}" for window in config.WINDOWS]
        idx, feature_df = _synthetic_feature_space(pool, rows=2000)
        mutator = CryptoMutator(pool, feature_df, idx, seed=3)
        individual = mutator.seed_individual()
        individual.features = ["alpha_3"]

        changed = mutator._c2(individual)

        self.assertTrue(changed)
        self.assertEqual(len(individual.features), 1)
        self.assertNotEqual(individual.features[0], "alpha_3")
        self.assertRegex(individual.features[0], r"^alpha_\d+$")
        self.assertIn(individual.features[0], pool)

    def test_crypto_mutator_c1_c2_fallback_to_c3(self):
        pool = [f"feature_{idx}" for idx in range(8)]
        idx, feature_df = _synthetic_feature_space(pool)
        original_probs = dict(config.MUTATOR_PROBS)
        cases = [
            ("c1", {"c1": 1.0, "c2": 0.0, "c3": 0.0}),
            ("c2", {"c1": 0.0, "c2": 1.0, "c3": 0.0}),
        ]

        for strategy, probs in cases:
            mutator = CryptoMutator(pool, feature_df, idx, seed=5)
            calls: list[str] = []

            def fake_c3(individual):
                calls.append("c3")
                individual.features[0] = "feature_7"
                return True

            try:
                config.MUTATOR_PROBS.update(probs)
                if strategy == "c1":
                    mutator._c1 = lambda individual: False  # type: ignore[method-assign]
                else:
                    mutator._c2 = lambda individual: False  # type: ignore[method-assign]
                mutator._c3 = fake_c3  # type: ignore[method-assign]
                parent = CryptoIndividual(features=["feature_0", "feature_1"])
                child = mutator.mutate(parent)
            finally:
                config.MUTATOR_PROBS.clear()
                config.MUTATOR_PROBS.update(original_probs)

            self.assertEqual(calls, ["c3"])
            self.assertIn("feature_7", child.features)

    def test_feature_quality_filter_can_use_train_only_index(self):
        df = _synthetic_crypto_frame(120)
        train_only = df.index[:50]

        full_quality = build_feature_frame(
            df,
            windows=[80],
            min_valid_ratio=0.10,
        )
        train_quality = build_feature_frame(
            df,
            windows=[80],
            min_valid_ratio=0.10,
            quality_index=train_only,
        )

        self.assertIn("ret_close_80", full_quality.columns)
        self.assertNotIn("ret_close_80", train_quality.columns)

    def test_feature_values_are_causal_under_future_truncation(self):
        df = _synthetic_crypto_frame(300)
        quality_index = df.index[:180]
        full = build_feature_frame(
            df,
            windows=[3, 10, 20],
            min_valid_ratio=0.10,
            quality_index=quality_index,
        )
        truncated = build_feature_frame(
            df.iloc[:220],
            windows=[3, 10, 20],
            min_valid_ratio=0.10,
            quality_index=quality_index,
        )
        common = sorted(set(full.columns) & set(truncated.columns))

        pd.testing.assert_frame_equal(
            full.loc[truncated.index, common],
            truncated[common],
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_test_end_split_purges_tail_labels(self):
        df = _synthetic_crypto_frame(12)
        labeled = add_binary_labels(df, horizons=[2], threshold=0.0)
        _, _, test_df = split_labeled_by_dates(
            labeled,
            val_start=str(df.index[2]),
            test_start=str(df.index[4]),
            test_end=str(df.index[9]),
            purge_bars=3,
        )

        self.assertEqual(test_df.index[0], df.index[4])
        self.assertEqual(test_df.index[-1], df.index[6])
        self.assertNotIn(df.index[7], test_df.index)

    def test_crypto_fitness_runs_on_synthetic_binary_fold(self):
        df = _synthetic_crypto_frame(900)
        labeled = add_binary_labels(df, horizons=[3], threshold=0.001)
        features = build_feature_frame(df, windows=[3, 5, 10, 20], min_valid_ratio=0.5)
        pool = selectable_features(features)
        feature_space = CryptoFeatureSpace(features, pool)
        fold = CryptoFold(
            name="wf_01",
            train_df=labeled.iloc[:600],
            val_df=labeled.iloc[620:850],
            train_start=labeled.index[0],
            train_end=labeled.index[599],
            val_start=labeled.index[620],
            val_end=labeled.index[849],
        )
        mutator = CryptoMutator(pool, feature_space, fold.train_df.index, seed=11)
        individual = mutator.seed_individual()
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 7,
            "max_depth": 3,
            "min_data_in_leaf": 20,
            "force_col_wise": True,
            "verbose": -1,
            "seed": 11,
        }

        evaluator = CryptoFitnessEvaluator(
            horizons=[3],
            lgbm_params=params,
            num_boost_round=5,
            early_stopping_rounds=2,
        )
        score = evaluator.evaluate_walk_forward(individual, [fold], feature_space)

        self.assertTrue(np.isfinite(score))
        self.assertIn("mean_auc", individual.metrics)
        self.assertIn("precision_excess", individual.metrics)
        self.assertIn("trade_return_score", individual.metrics)

    def test_crypto_final_evaluation_appends_final_metrics(self):
        df = _synthetic_crypto_frame(900)
        labeled = add_binary_labels(df, horizons=[3], threshold=0.001)
        features = build_feature_frame(df, windows=[3, 5, 10, 20], min_valid_ratio=0.5)
        pool = selectable_features(features)
        feature_space = CryptoFeatureSpace(features, pool)
        mutator = CryptoMutator(pool, feature_space, labeled.index[:500], seed=13)
        individual = mutator.seed_individual()
        individual.score = 0.01
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 7,
            "max_depth": 3,
            "min_data_in_leaf": 20,
            "force_col_wise": True,
            "verbose": -1,
            "seed": 13,
        }
        evaluator = CryptoFitnessEvaluator(
            horizons=[3],
            lgbm_params=params,
            num_boost_round=5,
            early_stopping_rounds=2,
        )

        metrics = evaluator.evaluate_final(
            individual=individual,
            train_df=labeled.iloc[:500],
            val_df=labeled.iloc[520:700],
            test_df=labeled.iloc[720:880],
            feature_data=feature_space,
        )

        self.assertIn("final_val_mean_auc", metrics)
        self.assertIn("final_test_mean_auc", metrics)
        self.assertIn("final_test_precision_excess", individual.metrics)
        self.assertTrue(np.isfinite(individual.metrics["final_test_mean_auc"]))

        archive = CryptoArchive(max_size=5)
        archive.try_add(individual)
        row = archive.summary()[0]
        self.assertIn("final_val_metrics", row)
        self.assertIn("test_metrics", row)
        self.assertIn("mean_auc", row["final_val_metrics"])
        self.assertIn("mean_auc", row["test_metrics"])
        self.assertIn("h3", row["test_metrics"]["horizons"])
        self.assertNotIn("final_test_mean_auc", row["metrics"])

    def test_internal_early_stopping_split_uses_train_tail_only(self):
        idx = pd.date_range("2024-01-01", periods=200, freq="15min")
        X = pd.DataFrame({"x": np.arange(200)}, index=idx)
        y = pd.Series(([0, 1] * 100), index=idx)

        split = _internal_early_stop_split(X, y)

        self.assertIsNotNone(split)
        X_fit, y_fit, X_stop, y_stop = split
        self.assertLess(X_fit.index.max(), X_stop.index.min())
        self.assertEqual(len(X_fit) + len(X_stop), len(X))
        self.assertEqual(y_fit.nunique(), 2)
        self.assertEqual(y_stop.nunique(), 2)

    def test_expression_space_and_mutator_generate_safe_formulas(self):
        df = _synthetic_crypto_frame(900)
        features = build_feature_frame(df, windows=[3, 5, 10, 20], min_valid_ratio=0.5)
        pool = selectable_features(features)
        feature_space = CryptoFeatureSpace(features, pool)
        train_index = features.index[:650]
        mutator = CryptoMutator(pool, feature_space, train_index, seed=21)
        individual = mutator.seed_individual()

        generated = None
        for _ in range(80):
            child = mutator.mutate(individual)
            new_features = [feature for feature in child.features if feature not in pool]
            if new_features:
                generated = new_features[0]
                break

        self.assertIsNotNone(generated)
        quality = feature_space.quality(generated, train_index)
        self.assertTrue(quality.ok, quality.reason)
        matrix = feature_space.matrix([generated], train_index)
        self.assertEqual(len(matrix), len(train_index))
        self.assertFalse(set(child.features) & RAW_SCALE_COLUMNS)


def _synthetic_crypto_frame(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    idx = pd.date_range("2020-01-01", periods=n, freq="15min")
    returns = rng.normal(0.0001, 0.004, size=n)
    close = 10000.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]] * (1.0 + rng.normal(0.0, 0.0005, size=n))
    spread = np.abs(rng.normal(0.002, 0.001, size=n))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.lognormal(mean=4.5, sigma=0.3, size=n)
    trade_count = rng.integers(100, 800, size=n)
    buy_ratio = np.clip(0.5 + rng.normal(0.0, 0.08, size=n), 0.05, 0.95)
    taker_base = volume * buy_ratio
    taker_quote = close * taker_base
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "trade_count": trade_count,
            "taker_buy_base_volume": taker_base,
            "taker_buy_quote_volume": taker_quote,
        },
        index=idx,
    )


def _synthetic_feature_space(
    names: list[str],
    rows: int = 700,
) -> tuple[pd.Index, pd.DataFrame]:
    rng = np.random.default_rng(321)
    idx = pd.date_range("2024-01-01", periods=rows, freq="15min")
    data = {}
    for name in names:
        data[name] = rng.normal(0.0, 1.0, size=rows)
    return idx, pd.DataFrame(data, index=idx)


if __name__ == "__main__":
    unittest.main()
