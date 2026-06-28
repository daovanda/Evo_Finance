import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.loader import load_from_dir


def _write_ohlcv(path: Path, dates, base: float, trading_flags=None) -> None:
    if trading_flags is None:
        trading_flags = [1 for _ in dates]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": [base + i for i in range(len(dates))],
            "high": [base + i + 1.0 for i in range(len(dates))],
            "low": [base + i - 1.0 for i in range(len(dates))],
            "close": [base + i + 0.5 for i in range(len(dates))],
            "volume": [1000 + i for i in range(len(dates))],
            "is_trading_day": trading_flags,
        }
    )
    df.to_csv(path, index=False)


class LoaderVNIndexTests(unittest.TestCase):
    def test_load_from_dir_inner_joins_vnindex_as_market_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stock_dates = pd.date_range("2024-01-01", periods=12)
            market_dates = pd.date_range("2024-01-03", periods=10)
            _write_ohlcv(root / "AAA.csv", stock_dates, 10.0)
            _write_ohlcv(root / "BBB.csv", stock_dates, 20.0)
            _write_ohlcv(root / "VNINDEX.csv", market_dates, 1000.0)

            df = load_from_dir(root, min_rows=1)

            self.assertEqual(df.index.get_level_values("ticker").unique().tolist(), ["AAA", "BBB"])
            self.assertEqual(df.index.get_level_values("date").nunique(), 10)
            self.assertEqual(len(df), 20)
            for col in [
                "market_open", "market_high", "market_low",
                "market_close", "market_volume",
            ]:
                self.assertIn(col, df.columns)
                self.assertFalse(df[col].isna().any())
            self.assertNotIn("VNINDEX", df.index.get_level_values("ticker"))

    def test_load_from_dir_can_use_configurable_market_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stock_dates = pd.date_range("2024-01-01", periods=12)
            market_dates = pd.date_range("2024-01-03", periods=10)
            _write_ohlcv(root / "AAA.csv", stock_dates, 10.0)
            _write_ohlcv(root / "BBB.csv", stock_dates, 20.0)
            _write_ohlcv(root / "VNINDEX.csv", market_dates, 1000.0)
            _write_ohlcv(root / "VN30.csv", market_dates, 2000.0)

            df = load_from_dir(root, min_rows=1, market_index_ticker="VN30")

            tickers = df.index.get_level_values("ticker").unique().tolist()
            self.assertEqual(tickers, ["AAA", "BBB"])
            self.assertNotIn("VNINDEX", tickers)
            self.assertNotIn("VN30", tickers)
            self.assertAlmostEqual(float(df["market_close"].iloc[0]), 2000.5)

    def test_load_from_dir_normalizes_requested_tickers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dates = pd.date_range("2024-01-01", periods=12)
            _write_ohlcv(root / "AAA.csv", dates, 10.0)

            df = load_from_dir(
                root,
                tickers=["aaa"],
                min_rows=1,
                include_vnindex=False,
            )

            self.assertEqual(
                df.index.get_level_values("ticker").unique().tolist(),
                ["AAA"],
            )

    def test_load_from_dir_drops_invalid_numeric_ohlcv_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dates = pd.date_range("2024-01-01", periods=3)
            df = pd.DataFrame(
                {
                    "date": dates,
                    "open": ["10", "bad", "12"],
                    "high": ["11", "12", "13"],
                    "low": ["9", "10", "11"],
                    "close": ["10.5", "11.5", "12.5"],
                    "volume": ["1000", "1001", "1002"],
                }
            )
            df.to_csv(root / "AAA.csv", index=False)

            loaded = load_from_dir(root, min_rows=1, include_vnindex=False)

            self.assertEqual(loaded.index.get_level_values("date").nunique(), 2)
            self.assertNotIn(pd.Timestamp("2024-01-02"), loaded.index.get_level_values("date"))
            self.assertTrue(pd.api.types.is_numeric_dtype(loaded["open"]))

    def test_load_from_dir_requires_configured_market_index_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dates = pd.date_range("2024-01-01", periods=12)
            _write_ohlcv(root / "AAA.csv", dates, 10.0)

            with self.assertRaisesRegex(FileNotFoundError, "Market index VNINDEX.csv is required"):
                load_from_dir(root, min_rows=1)

            loaded = load_from_dir(root, min_rows=1, include_vnindex=False)
            self.assertNotIn("market_close", loaded.columns)

    def test_load_from_dir_respects_stock_is_trading_day_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dates = pd.date_range("2024-01-01", periods=6)
            non_trading_date = dates[2]
            _write_ohlcv(
                root / "AAA.csv",
                dates,
                10.0,
                trading_flags=[1, 1, 0, 1, 1, 1],
            )
            _write_ohlcv(root / "BBB.csv", dates, 20.0)

            loaded = load_from_dir(root, min_rows=1, include_vnindex=False)

            loaded_dates = loaded.index.get_level_values("date")
            self.assertNotIn(non_trading_date, loaded_dates)
            self.assertEqual(loaded_dates.nunique(), 5)
            self.assertTrue((loaded["is_trading_day"] == 1).all())

    def test_load_from_dir_respects_market_is_trading_day_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stock_dates = pd.date_range("2024-01-01", periods=14)
            market_dates = pd.date_range("2024-01-03", periods=12)
            non_trading_market_date = market_dates[4]
            _write_ohlcv(root / "AAA.csv", stock_dates, 10.0)
            _write_ohlcv(root / "BBB.csv", stock_dates, 20.0)
            _write_ohlcv(
                root / "VNINDEX.csv",
                market_dates,
                1000.0,
                trading_flags=[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            )

            loaded = load_from_dir(root, min_rows=1)

            loaded_dates = loaded.index.get_level_values("date")
            self.assertNotIn(non_trading_market_date, loaded_dates)
            self.assertEqual(loaded_dates.nunique(), 11)


if __name__ == "__main__":
    unittest.main()
