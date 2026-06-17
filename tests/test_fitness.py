import unittest
import warnings

import pandas as pd
from scipy.stats import ConstantInputWarning

from fitness.fitness import _ic_per_date


class FitnessMetricTests(unittest.TestCase):
    def test_ic_per_date_handles_constant_inputs_without_warning(self):
        dates = pd.date_range("2024-01-01", periods=2)
        tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
        idx = pd.MultiIndex.from_product(
            [dates, tickers],
            names=["date", "ticker"],
        )
        pred = pd.Series([1.0] * 6 + list(range(6)), index=idx)
        label = pd.Series(list(range(6)) + [1.0] * 6, index=idx)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ic = _ic_per_date(pred, label, idx)

        constant_warnings = [
            warning for warning in caught
            if isinstance(warning.message, ConstantInputWarning)
        ]
        self.assertEqual(constant_warnings, [])
        self.assertEqual(ic.loc[dates[0]], 0.0)
        self.assertEqual(ic.loc[dates[1]], 0.0)


if __name__ == "__main__":
    unittest.main()
