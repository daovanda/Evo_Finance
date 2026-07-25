import unittest

import numpy as np
import pandas as pd

from crypto.backtest import SplitSignals
from crypto.backtest_long_short import _simulate_split


def _signals(
    index: pd.DatetimeIndex,
    selected: list[pd.Timestamp],
) -> SplitSignals:
    data = pd.DataFrame(
        {
            "label": np.zeros(len(index), dtype=int),
            "pred": np.linspace(0.9, 0.1, len(index)),
        },
        index=index,
    )
    return SplitSignals(
        split="val",
        data=data,
        selected_index=pd.DatetimeIndex(selected),
        pred_threshold=0.5,
        top_fraction=0.2,
    )


class LongShortSlopeBacktestTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2025-01-01", periods=8, freq="15min")
        price = np.arange(100.0, 108.0)
        self.raw = pd.DataFrame(
            {
                "open": price,
                "high": price + 0.2,
                "low": price - 0.2,
                "close": price + 0.1,
            },
            index=self.index,
        )

    def test_long_model_opens_short_and_unconfirmed_exits_after_h(self):
        initial = pd.Series(np.nan, index=self.index)
        initial.iloc[0] = 0.0003
        long_strength = pd.Series(np.nan, index=self.index)
        long_strength.iloc[0] = 0.0002

        trades = _simulate_split(
            raw_df=self.raw,
            split="val",
            split_index=self.index,
            long_bundle=_signals(self.index, [self.index[0]]),
            short_bundle=_signals(self.index, []),
            initial_slope=initial,
            long_strength=long_strength,
            short_strength=pd.Series(np.nan, index=self.index),
            horizon=3,
            long_threshold=0.0003,
            short_threshold=0.0003,
        )

        self.assertEqual(len(trades), 1)
        trade = trades.iloc[0]
        self.assertEqual(trade["position_side"], "short")
        self.assertEqual(trade["entry_time"], self.index[1])
        self.assertEqual(trade["review_time"], self.index[3])
        self.assertEqual(trade["exit_time"], self.index[4])
        self.assertEqual(trade["exit_reason"], "not_confirmed_h3")
        self.assertFalse(bool(trade["confirmed_at_h"]))
        self.assertAlmostEqual(trade["gross_return"], 1.0 - 104.0 / 101.0)

    def test_confirmed_short_exits_on_opposite_slope_without_model_signal(self):
        initial = pd.Series(np.nan, index=self.index)
        initial.iloc[0] = 0.0003
        initial.iloc[4] = -0.0003
        long_strength = pd.Series(np.nan, index=self.index)
        long_strength.iloc[0] = 0.0004

        trades = _simulate_split(
            raw_df=self.raw,
            split="test",
            split_index=self.index,
            long_bundle=_signals(self.index, [self.index[0]]),
            short_bundle=_signals(self.index, []),
            initial_slope=initial,
            long_strength=long_strength,
            short_strength=pd.Series(np.nan, index=self.index),
            horizon=3,
            long_threshold=0.0003,
            short_threshold=0.0003,
        )

        self.assertEqual(len(trades), 1)
        trade = trades.iloc[0]
        self.assertTrue(bool(trade["confirmed_at_h"]))
        self.assertEqual(trade["exit_signal_time"], self.index[4])
        self.assertEqual(trade["exit_time"], self.index[5])
        self.assertEqual(trade["exit_reason"], "opposite_short_slope")

    def test_short_model_opens_long_and_exits_on_opposite_slope_only(self):
        initial = pd.Series(np.nan, index=self.index)
        initial.iloc[0] = -0.0003
        initial.iloc[4] = 0.0003
        short_strength = pd.Series(np.nan, index=self.index)
        short_strength.iloc[0] = 0.0004

        trades = _simulate_split(
            raw_df=self.raw,
            split="test",
            split_index=self.index,
            long_bundle=_signals(self.index, []),
            short_bundle=_signals(self.index, [self.index[0]]),
            initial_slope=initial,
            long_strength=pd.Series(np.nan, index=self.index),
            short_strength=short_strength,
            horizon=3,
            long_threshold=0.0003,
            short_threshold=0.0003,
        )

        self.assertEqual(len(trades), 1)
        trade = trades.iloc[0]
        self.assertEqual(trade["position_side"], "long")
        self.assertEqual(trade["exit_reason"], "opposite_long_slope")
        self.assertEqual(trade["entry_time"], self.index[1])
        self.assertEqual(trade["exit_time"], self.index[5])
        self.assertAlmostEqual(trade["gross_return"], 105.0 / 101.0 - 1.0)


if __name__ == "__main__":
    unittest.main()
