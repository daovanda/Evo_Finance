import unittest
from unittest.mock import patch

import pandas as pd

from crypto.prod.long_mfe_7seed_monitor import (
    _band_metrics,
    _complete_minute_snapshot,
    _score_cutoffs,
    aggregate_rolling_5m,
)


class LongMfeMonitorBandTests(unittest.TestCase):
    def test_test_precision_uses_validation_cutoffs(self):
        val_prediction = pd.Series(range(1, 101), dtype=float)
        test_prediction = pd.Series([100.0, 95.0, 92.0, 1.0])
        test_success = pd.Series([True, False, True, False])

        cutoffs = _score_cutoffs(val_prediction)
        metrics = _band_metrics(test_prediction, test_success, cutoffs)

        self.assertEqual(cutoffs["q1"], 96.0)
        self.assertEqual(metrics["q1"]["samples"], 1)
        self.assertEqual(metrics["q1"]["precision"], 1.0)
        self.assertEqual(metrics["q2"]["samples"], 2)
        self.assertEqual(metrics["q2"]["precision"], 0.5)
        self.assertEqual(metrics["q20"]["samples"], 1)
        self.assertEqual(metrics["q20"]["precision"], 0.0)

    def test_rolling_5m_uses_complete_source_minutes_and_availability_time(self):
        index = pd.date_range("2026-01-01 10:02:00", periods=5, freq="1min")
        source = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "high": [102, 103, 104, 105, 106],
                "low": [99, 100, 101, 102, 103],
                "close": [101, 102, 103, 104, 105],
                "volume": [1, 2, 3, 4, 5],
                "quote_volume": [10, 20, 30, 40, 50],
                "trade_count": [2, 2, 2, 2, 2],
                "taker_buy_base_volume": [0.5] * 5,
                "taker_buy_quote_volume": [5.0] * 5,
                "is_trading_day": [1] * 5,
            },
            index=index,
        )

        bars = aggregate_rolling_5m(source, phase=2)

        self.assertEqual(list(bars.index), [pd.Timestamp("2026-01-01 10:07:00")])
        self.assertEqual(bars.iloc[0]["open"], 100)
        self.assertEqual(bars.iloc[0]["high"], 106)
        self.assertEqual(bars.iloc[0]["low"], 99)
        self.assertEqual(bars.iloc[0]["close"], 105)
        self.assertEqual(bars.iloc[0]["volume"], 15)
        self.assertEqual(bars.iloc[0]["quote_volume"], 150)

    @patch("crypto.prod.long_mfe_7seed_monitor.fetch_klines")
    def test_minute_snapshot_backfills_boundary_race_in_memory(self, fetch_mock):
        source = self._minute_frame("2026-01-01 10:00:00", periods=4)
        missing = self._minute_frame("2026-01-01 10:04:00", periods=2).reset_index()
        fetch_mock.return_value = missing

        snapshot, added = _complete_minute_snapshot(
            source,
            inference_time=pd.Timestamp("2026-01-01 10:05:00"),
            symbol="BTCUSDT",
            base_url="https://example.test",
        )

        self.assertEqual(added, 1)
        self.assertEqual(snapshot.index[-1], pd.Timestamp("2026-01-01 10:04:00"))
        self.assertNotIn(pd.Timestamp("2026-01-01 10:05:00"), snapshot.index)

    @staticmethod
    def _minute_frame(start: str, periods: int) -> pd.DataFrame:
        index = pd.date_range(start, periods=periods, freq="1min", name="date")
        return pd.DataFrame(
            {
                "open": [100.0] * periods,
                "high": [101.0] * periods,
                "low": [99.0] * periods,
                "close": [100.5] * periods,
                "volume": [1.0] * periods,
                "quote_volume": [100.0] * periods,
                "trade_count": [2] * periods,
                "taker_buy_base_volume": [0.5] * periods,
                "taker_buy_quote_volume": [50.0] * periods,
                "is_trading_day": [1] * periods,
            },
            index=index,
        )


if __name__ == "__main__":
    unittest.main()
