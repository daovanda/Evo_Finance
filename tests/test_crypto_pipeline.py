import unittest

import numpy as np
import pandas as pd

from crypto.data import CryptoFold, add_binary_labels, split_labeled_by_dates
from crypto.evolution import CryptoMutator
from crypto.expression import CryptoFeatureSpace
from crypto.features import RAW_SCALE_COLUMNS, build_feature_frame, selectable_features
from crypto.fitness import CryptoFitnessEvaluator, _internal_early_stop_split


class CryptoPipelineTests(unittest.TestCase):
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

        labeled = add_binary_labels(df, horizons=[2], threshold=0.001)

        expected = (df["close"].shift(-2) - df["open"].shift(-1)) / df["open"].shift(-1)
        pd.testing.assert_series_equal(
            labeled["future_return_h2"],
            expected.rename("future_return_h2"),
        )
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


if __name__ == "__main__":
    unittest.main()
