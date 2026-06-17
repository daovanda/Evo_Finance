import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.loader import load_from_dir


def _write_ohlcv(path: Path, dates, base: float) -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": [base + i for i in range(len(dates))],
            "high": [base + i + 1.0 for i in range(len(dates))],
            "low": [base + i - 1.0 for i in range(len(dates))],
            "close": [base + i + 0.5 for i in range(len(dates))],
            "volume": [1000 + i for i in range(len(dates))],
            "is_trading_day": [1 for _ in dates],
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


if __name__ == "__main__":
    unittest.main()
