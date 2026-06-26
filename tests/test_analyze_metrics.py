import unittest

import pandas as pd

from analyze import _hitrate_per_date


class AnalyzeMetricTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
