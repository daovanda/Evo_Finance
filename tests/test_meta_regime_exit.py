from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from crypto import config
from crypto.main import _make_fitness_evaluator
from crypto.meta_regime_exit import (
    REGIME_ACTION_PRICE_COLUMN,
    REGIME_ACTION_TIME_COLUMN,
    REGIME_CLOSE_PRICE_COLUMN,
    REGIME_CLOSE_TIME_COLUMN,
    REGIME_EXIT_PRICE_COLUMN,
    REGIME_EXIT_TIME_COLUMN,
    REGIME_SIDE_COLUMN,
    attach_meta_regime_exit_targets,
    simulate_regime_exit,
)
from crypto.meta_targets import build_meta_feature_alignment
from crypto.regime_exit_fitness import RegimeExitFitnessEvaluator


class MetaRegimeExitTests(unittest.TestCase):
    def test_mode_registry_selects_state_machine_evaluator(self):
        self.assertEqual(config.canonical_label_mode("meta_regime_exit"), "meta_regime_exit")
        self.assertTrue(config.is_meta_regime_exit_label_mode("meta_regime_exit"))
        self.assertTrue(config.is_direction_neutral_label_mode("meta_regime_exit"))
        self.assertAlmostEqual(
            config.default_label_threshold("meta_regime_exit"),
            config.META_REGIME_EXIT_THRESHOLD,
        )
        self.assertIsInstance(
            _make_fitness_evaluator("meta_regime_exit", [1]),
            RegimeExitFitnessEvaluator,
        )

    def test_target_uses_entry_candle_and_exit_is_open_after_observation(self):
        target_index = pd.date_range("2025-01-01", periods=7, freq="5min")
        minute_index = pd.date_range("2025-01-01", periods=40, freq="1min")
        raw = pd.DataFrame(
            {
                "open": [100.0] * 7,
                "close": [100.0, 99.8, 100.1, 100.2, 100.0, 100.0, 100.0],
            },
            index=target_index,
        )
        minute = pd.DataFrame({"open": np.arange(40.0) + 100.0}, index=minute_index)
        alignment = build_meta_feature_alignment(
            target_index,
            minute_index,
            lookahead_bars=1,
            horizon=1,
        )
        frame = raw.iloc[:4]
        bull_prediction = pd.Series([0.9, 0.8, 0.2, 0.1], index=frame.index)
        bear_prediction = pd.Series([0.1, 0.2, 0.8, 0.9], index=frame.index)

        targeted = attach_meta_regime_exit_targets(
            frame,
            raw_df=raw,
            minute_df=minute,
            alignment=alignment,
            bull_prediction=bull_prediction,
            bear_prediction=bear_prediction,
            top_fraction=0.5,
            exit_threshold=0.001,
        )

        self.assertEqual(int(targeted.loc[target_index[0], REGIME_SIDE_COLUMN]), 1)
        self.assertEqual(float(targeted.loc[target_index[0], "label_h1"]), 1.0)
        self.assertEqual(
            targeted.loc[target_index[0], REGIME_ACTION_TIME_COLUMN],
            target_index[1],
        )
        self.assertEqual(
            targeted.loc[target_index[0], REGIME_EXIT_TIME_COLUMN],
            target_index[1] + pd.Timedelta(minutes=1),
        )
        self.assertEqual(
            targeted.loc[target_index[0], REGIME_EXIT_PRICE_COLUMN],
            minute.loc[target_index[1] + pd.Timedelta(minutes=1), "open"],
        )

    def test_early_exit_locks_same_episode_until_signal_turns_off(self):
        index = pd.date_range("2025-01-01", periods=6, freq="5min")
        frame = _execution_frame(index, [1, 1, 1, 0, 1, 1])
        prediction = pd.Series([0.9, 0.1, 0.1, np.nan, 0.1, 0.1], index=index)

        result = simulate_regime_exit(
            frame,
            prediction,
            prediction_cutoff=0.5,
            trade_cost=0.0,
        )

        self.assertEqual(len(result.trades), 2)
        self.assertEqual(result.early_exits, 1)
        self.assertGreaterEqual(result.locked_rows, 2)
        self.assertEqual(result.trades.iloc[0]["exit_reason"], "meta_early_exit")
        self.assertEqual(result.trades.iloc[0]["entry_time"], index[0] + pd.Timedelta(minutes=5))
        self.assertEqual(result.trades.iloc[1]["entry_time"], index[4] + pd.Timedelta(minutes=5))

    def test_direction_change_closes_without_reversing_same_open(self):
        index = pd.date_range("2025-01-01", periods=4, freq="5min")
        frame = _execution_frame(index, [-1, 1, 1, 0])

        result = simulate_regime_exit(
            frame,
            None,
            prediction_cutoff=float("inf"),
            trade_cost=0.0,
        )

        self.assertEqual(len(result.trades), 2)
        short_trade = result.trades.iloc[0]
        long_trade = result.trades.iloc[1]
        self.assertEqual(short_trade["side"], "short")
        self.assertEqual(long_trade["side"], "long")
        self.assertEqual(short_trade["exit_time"], index[1] + pd.Timedelta(minutes=5))
        self.assertEqual(long_trade["entry_time"], index[2] + pd.Timedelta(minutes=5))
        self.assertGreater(long_trade["entry_time"], short_trade["exit_time"])


def _execution_frame(index: pd.DatetimeIndex, sides: list[int]) -> pd.DataFrame:
    sequence = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            REGIME_SIDE_COLUMN: sides,
            REGIME_ACTION_TIME_COLUMN: index + pd.Timedelta(minutes=5),
            REGIME_ACTION_PRICE_COLUMN: 100.0 + sequence,
            REGIME_EXIT_TIME_COLUMN: index + pd.Timedelta(minutes=6),
            REGIME_EXIT_PRICE_COLUMN: 100.1 + sequence,
            REGIME_CLOSE_TIME_COLUMN: index + pd.Timedelta(minutes=10),
            REGIME_CLOSE_PRICE_COLUMN: 100.2 + sequence,
            "label_h1": [0.0 if side else np.nan for side in sides],
            "future_return_h1": [0.0 if side else np.nan for side in sides],
        },
        index=index,
    )


if __name__ == "__main__":
    unittest.main()
