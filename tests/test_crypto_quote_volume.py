import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from craw_btc import OUTPUT_COLUMNS, _klines_to_dataframe
from crypto import config
from crypto.data import load_ohlcv
from crypto.evolution import CryptoArchive, CryptoIndividual
from crypto.expression import CryptoFeatureSpace
from crypto.features import (
    build_feature_frame,
    formula_uses_legacy_quote_proxy,
    selectable_features,
)
from crypto.prod.live_backend import _read_existing_ohlcv
from crypto.main import _save_archive, _validate_resume_metadata


class CryptoQuoteVolumeTests(unittest.TestCase):
    def test_crawler_preserves_binance_quote_volume(self):
        rows = [
            [
                1_704_067_200_000,
                "100.0",
                "102.0",
                "99.0",
                "101.0",
                "10.0",
                1_704_067_499_999,
                "1005.0",
                20,
                "6.0",
                "603.0",
                "0",
            ]
        ]

        actual = _klines_to_dataframe(rows)

        self.assertEqual(actual.columns.tolist(), OUTPUT_COLUMNS)
        self.assertEqual(float(actual.loc[0, "quote_volume"]), 1005.0)

    def test_loader_requires_quote_volume_instead_of_silent_proxy(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.csv"
            self._raw_frame().drop(columns=["quote_volume"]).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "quote_volume"):
                load_ohlcv(path)

    def test_loader_accepts_valid_exact_quote_volume(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "exact.csv"
            self._raw_frame().to_csv(path, index=False)

            actual = load_ohlcv(path)

        self.assertEqual(len(actual), 3)
        self.assertAlmostEqual(float(actual.iloc[0]["quote_volume"]), 1005.0)

    def test_loader_rejects_impossible_taker_quote_volume(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.csv"
            frame = self._raw_frame()
            frame.loc[0, "taker_buy_quote_volume"] = 1006.0
            frame.to_csv(path, index=False)

            with self.assertRaisesRegex(
                ValueError, "taker_buy_quote_volume must not exceed quote_volume"
            ):
                load_ohlcv(path)

    def test_loader_rejects_quote_volume_outside_ohlc_range(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid-vwap.csv"
            frame = self._raw_frame()
            frame.loc[0, "quote_volume"] = 2000.0
            frame.loc[0, "taker_buy_quote_volume"] = 603.0
            frame.to_csv(path, index=False)

            with self.assertRaisesRegex(
                ValueError, "quote_volume / volume must lie within"
            ):
                load_ohlcv(path)

    def test_loader_rejects_quote_volume_when_base_volume_is_zero(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid-zero-volume.csv"
            frame = self._raw_frame()
            frame.loc[0, "volume"] = 0.0
            frame.loc[0, "quote_volume"] = 1.0
            frame.loc[0, "taker_buy_base_volume"] = 0.0
            frame.loc[0, "taker_buy_quote_volume"] = 0.0
            frame.to_csv(path, index=False)

            with self.assertRaisesRegex(
                ValueError, "quote_volume must be zero when volume is zero"
            ):
                load_ohlcv(path)

    def test_exact_features_use_exchange_quote_volume(self):
        raw = self._raw_frame().set_index("date").drop(columns=["is_trading_day"])

        features = build_feature_frame(raw, windows=[2], quality_filter=False)

        self.assertAlmostEqual(features.iloc[0]["vwap_vs_open"], 0.005)
        self.assertAlmostEqual(
            features.iloc[0]["taker_buy_quote_ratio_exact"], 603.0 / 1005.0
        )
        self.assertAlmostEqual(
            features.iloc[0]["trade_size_quote_exact"], 1005.0 / 20.0
        )
        self.assertAlmostEqual(features.iloc[0]["vwap_pos_in_range"], 0.5)
        # The legacy value is deliberately different and remains reproducible.
        self.assertAlmostEqual(features.iloc[0]["taker_buy_quote_ratio"], 603.0 / 1010.0)

    def test_new_evolution_excludes_proxy_but_archives_can_evaluate_it(self):
        raw = self._raw_frame().set_index("date").drop(columns=["is_trading_day"])
        features = build_feature_frame(raw, windows=[2], quality_filter=False)
        pool = selectable_features(features)

        self.assertIn("taker_buy_quote_ratio_exact", pool)
        self.assertIn("quote_volume_ratio_2", pool)
        self.assertIn("trade_size_quote_exact_ratio_2", pool)
        self.assertIn("taker_delta_quote_sum_ratio_exact_2", pool)
        self.assertNotIn("trade_size_quote_exact", pool)
        self.assertNotIn("taker_buy_quote_ratio", pool)
        self.assertNotIn("quote_volume_proxy_ratio_2", pool)
        self.assertNotIn("trade_size_quote_ratio_2", pool)

        space = CryptoFeatureSpace(features, pool)
        legacy = space.evaluate("quote_volume_proxy_ratio_2")
        pd.testing.assert_series_equal(
            legacy,
            features["quote_volume_proxy_ratio_2"],
            check_names=False,
        )
        self.assertTrue(
            formula_uses_legacy_quote_proxy("ema(quote_volume_proxy_ratio_2, 2)")
        )
        self.assertTrue(formula_uses_legacy_quote_proxy("trade_size_quote_ratio_2"))
        self.assertFalse(
            formula_uses_legacy_quote_proxy("trade_size_quote_exact_ratio_2")
        )

    def test_live_refresh_rejects_partially_upgradeable_history(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.csv"
            self._raw_frame().drop(columns=["quote_volume"]).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "Re-crawl the complete history"):
                _read_existing_ohlcv(path)

    def test_resume_rejects_legacy_proxy_genes(self):
        archive = CryptoArchive(
            metadata={
                "pipeline": "crypto",
                "horizons": [3],
                "label_mode": "mfe",
                "label_direction": "long",
                "label_threshold": 0.001,
            }
        )
        archive.try_add(
            CryptoIndividual(
                features=["ema(quote_volume_proxy_ratio_2, 2)"],
                score=0.1,
            )
        )

        with self.assertRaisesRegex(ValueError, r"legacy close\*volume"):
            _validate_resume_metadata(
                archive=archive,
                resume_path=Path("legacy.json"),
                horizons=[3],
                label_mode="mfe",
                label_direction="long",
                label_threshold=0.001,
            )

        archive.metadata = {}
        with self.assertRaisesRegex(ValueError, r"legacy close\*volume"):
            _validate_resume_metadata(
                archive=archive,
                resume_path=Path("legacy-no-metadata.json"),
                horizons=[3],
                label_mode="mfe",
                label_direction="long",
                label_threshold=0.001,
            )

    def test_saved_archive_records_exact_feature_schema(self):
        archive = CryptoArchive()
        archive.try_add(CryptoIndividual(features=["quote_volume_ratio_2"], score=0.1))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "archive.json"
            _save_archive(
                archive=archive,
                path=path,
                horizons=[3],
                label_threshold=0.001,
                label_mode="mfe",
                label_direction="long",
                exit_after_k=None,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["metadata"]["feature_schema_version"],
            config.CRYPTO_FEATURE_SCHEMA_VERSION,
        )

    @staticmethod
    def _raw_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3, freq="5min"),
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [10.0, 11.0, 12.0],
                "quote_volume": [1005.0, 1116.5, 1230.0],
                "trade_count": [20, 22, 24],
                "taker_buy_base_volume": [6.0, 5.5, 7.0],
                "taker_buy_quote_volume": [603.0, 558.25, 717.5],
                "is_trading_day": [1, 1, 1],
            }
        )


if __name__ == "__main__":
    unittest.main()
