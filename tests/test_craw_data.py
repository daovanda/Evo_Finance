import tempfile
import unittest
from pathlib import Path

import pandas as pd

from craw_data import append_atomic, prepare_append


class StockAppendTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "ACB.csv"
        self.original = (
            b"date,open,high,low,close,volume,is_trading_day\r\n"
            b"2026-06-10 07:00:00,20,22,18,20,100,1\r\n"
            b"2026-06-11 07:00:00,20,22,18,20,100,1\r\n"
            b"2026-06-12 07:00:00,20,22,18,20,100,1\r\n"
        )
        self.path.write_bytes(self.original)
        self.fresh = pd.DataFrame(
            {
                "time": ["2026-06-10", "2026-06-11", "2026-06-12", "2026-06-15", "2026-06-15", "2026-06-16"],
                "open": [10, 10, 10, 9, 10, 20],
                "high": [11, 11, 11, 11, 11, 20],
                "low": [9, 9, 9, 9, 9, 20],
                "close": [10, 10, 10, 9, 11, 20],
                "volume": [100, 100, 100, 100, 120, 120],
            }
        )

    def prepare(self):
        def fetch(ticker, start, end):
            self.assertEqual((ticker, start, end), ("ACB", "2026-06-13", "2026-06-15"))
            return self.fresh
        return prepare_append(self.path, fetch, pd.Timestamp("2026-06-15"))

    def test_appends_only_new_day_without_adjusting_prices(self):
        rows, _ = self.prepare()
        self.assertEqual(rows["date"].tolist(), ["2026-06-15 07:00:00"])
        self.assertEqual(rows["close"].tolist(), [11])
        self.assertEqual(rows["volume"].tolist(), [120])
        append_atomic(self.path, rows)
        self.assertTrue(self.path.read_bytes().startswith(self.original))
        self.assertTrue(self.prepare()[0].empty)

    def test_empty_response_keeps_original(self):
        self.fresh = pd.DataFrame()
        self.assertTrue(self.prepare()[0].empty)
        self.assertEqual(self.path.read_bytes(), self.original)


if __name__ == "__main__":
    unittest.main()
