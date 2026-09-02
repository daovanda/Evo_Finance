from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from crypto import config
from crypto.data import CryptoFold
from crypto.main import _make_fitness_evaluator
from crypto.meta_regime_entry import (
    REGIME_ACTION_HIGH_COLUMN,
    REGIME_ACTION_LOW_COLUMN,
    REGIME_ENTRY_CANDIDATE_COLUMN,
    REGIME_ENTRY_EXIT_TIME_COLUMN,
    REGIME_ENTRY_GROSS_COLUMN,
    REGIME_ENTRY_NET_COLUMN,
    _censor_cross_boundary_train_targets,
    attach_meta_regime_entry_targets,
    simulate_regime_entry,
)
from crypto.meta_regime_exit import (
    REGIME_ACTION_PRICE_COLUMN,
    REGIME_ACTION_TIME_COLUMN,
    REGIME_CLOSE_PRICE_COLUMN,
    REGIME_CLOSE_TIME_COLUMN,
    REGIME_SIDE_COLUMN,
)
from crypto.regime_entry_fitness import RegimeEntryFitnessEvaluator


class MetaRegimeEntryTests(unittest.TestCase):
    def test_mode_registry_selects_episode_entry_evaluator(self):
        self.assertEqual(
            config.canonical_label_mode("meta_regime_entry"),
            "meta_regime_entry",
        )
        self.assertTrue(config.is_meta_regime_entry_label_mode("meta_regime_entry"))
        self.assertTrue(config.is_direction_neutral_label_mode("meta_regime_entry"))
        self.assertEqual(config.default_label_threshold("meta_regime_entry"), 0.0)
        self.assertIsInstance(
            _make_fitness_evaluator("meta_regime_entry", [1]),
            RegimeEntryFitnessEvaluator,
        )

    def test_target_labels_only_completed_episode_entry_from_net_return(self):
        index = pd.date_range("2025-01-01", periods=7, freq="5min")
        raw = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 100.4, 100.5, 100.5, 100.5],
                "high": [100.2] * 7,
                "low": [99.9] * 7,
                "close": [100.0, 100.0, 100.2, 100.4, 100.5, 100.5, 100.5],
            },
            index=index,
        )
        frame = raw.iloc[:6]
        bull_prediction = pd.Series(
            [0.0, 0.9, 0.8, 0.7, 0.0, 0.0], index=frame.index
        )
        bear_prediction = pd.Series(0.0, index=frame.index)

        targeted = attach_meta_regime_entry_targets(
            frame,
            raw_df=raw,
            bull_prediction=bull_prediction,
            bear_prediction=bear_prediction,
            top_fraction=0.5,
            stop_loss=0.0015,
            trade_cost=0.0002,
            bull_cutoff=0.5,
            bear_cutoff=0.5,
        )

        candidate = targeted[REGIME_ENTRY_CANDIDATE_COLUMN]
        self.assertEqual(candidate.sum(), 1)
        self.assertTrue(candidate.loc[index[1]])
        self.assertTrue(pd.isna(targeted.loc[index[2], "label_h1"]))
        self.assertEqual(float(targeted.loc[index[1], "label_h1"]), 1.0)
        expected_net = 100.5 / 100.0 - 1.0 - 0.0002
        self.assertAlmostEqual(
            float(targeted.loc[index[1], "future_return_h1"]),
            expected_net,
        )

    def test_rejected_entry_locks_episode_until_signal_ends(self):
        index = pd.date_range("2025-01-01", periods=8, freq="5min")
        frame = _execution_frame(index, [0, 1, 1, 1, 0, 1, 1, 0])
        prediction = pd.Series([0.1, 0.9], index=[index[1], index[5]])

        result = simulate_regime_entry(
            frame,
            prediction,
            prediction_cutoff=0.5,
            stop_loss=0.0,
            trade_cost=0.0,
        )

        self.assertEqual(result.candidate_episodes, 2)
        self.assertEqual(result.rejected_episodes, 1)
        self.assertEqual(result.selected_episodes, 1)
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades.iloc[0]["entry_signal_time"], index[5])
        self.assertGreaterEqual(result.locked_rows, 2)

    def test_stop_loss_closes_at_stop_and_prevents_same_episode_reentry(self):
        index = pd.date_range("2025-01-01", periods=8, freq="5min")
        frame = _execution_frame(index, [0, 1, 1, 1, 0, 1, 1, 0])
        frame.loc[index[1], REGIME_ACTION_LOW_COLUMN] = 99.7

        result = simulate_regime_entry(
            frame,
            None,
            prediction_cutoff=float("-inf"),
            stop_loss=0.0015,
            trade_cost=0.0002,
        )

        self.assertEqual(result.candidate_episodes, 2)
        self.assertEqual(result.selected_episodes, 2)
        self.assertEqual(result.stopped_trades, 1)
        self.assertEqual(len(result.trades), 2)
        stopped = result.trades.iloc[0]
        self.assertEqual(stopped["exit_reason"], "stop_loss")
        self.assertAlmostEqual(float(stopped["gross_return"]), -0.0015)
        self.assertAlmostEqual(float(stopped["net_return"]), -0.0017)
        self.assertEqual(result.trades.iloc[1]["entry_signal_time"], index[5])

    def test_direction_change_does_not_reverse_on_same_open(self):
        index = pd.date_range("2025-01-01", periods=6, freq="5min")
        frame = _execution_frame(index, [0, 1, -1, -1, 0, 0])

        result = simulate_regime_entry(
            frame,
            None,
            prediction_cutoff=float("-inf"),
            stop_loss=0.0,
            trade_cost=0.0,
        )

        self.assertEqual(len(result.trades), 2)
        long_trade = result.trades.iloc[0]
        short_trade = result.trades.iloc[1]
        self.assertEqual(long_trade["side"], "long")
        self.assertEqual(short_trade["side"], "short")
        self.assertEqual(long_trade["exit_time"], index[3])
        self.assertEqual(short_trade["entry_time"], index[4])
        self.assertGreater(short_trade["entry_time"], long_trade["exit_time"])

    def test_train_target_crossing_meta_boundary_is_censored(self):
        index = pd.date_range("2025-01-01", periods=6, freq="5min")
        frame = _execution_frame(index, [0, 1, 1, 1, 0, 0])
        frame[REGIME_ENTRY_CANDIDATE_COLUMN] = False
        frame[REGIME_ENTRY_EXIT_TIME_COLUMN] = pd.NaT
        frame[REGIME_ENTRY_GROSS_COLUMN] = np.nan
        frame[REGIME_ENTRY_NET_COLUMN] = np.nan
        frame["label_h1"] = np.nan
        frame["future_return_h1"] = np.nan
        frame.loc[index[1], REGIME_ENTRY_CANDIDATE_COLUMN] = True
        frame.loc[index[1], REGIME_ENTRY_EXIT_TIME_COLUMN] = index[5]
        frame.loc[index[1], REGIME_ENTRY_GROSS_COLUMN] = 0.002
        frame.loc[index[1], REGIME_ENTRY_NET_COLUMN] = 0.0018
        frame.loc[index[1], "label_h1"] = 1.0
        frame.loc[index[1], "future_return_h1"] = 0.0018
        fold = CryptoFold(
            name="meta_wf_01",
            train_df=frame.iloc[:4],
            val_df=frame.iloc[5:],
            train_start=index[0],
            train_end=index[4],
            val_start=index[5],
            val_end=index[-1] + pd.Timedelta(nanoseconds=1),
        )

        censored = _censor_cross_boundary_train_targets(fold)

        self.assertFalse(
            bool(censored.train_df.loc[index[1], REGIME_ENTRY_CANDIDATE_COLUMN])
        )
        self.assertTrue(pd.isna(censored.train_df.loc[index[1], "label_h1"]))
        baseline = simulate_regime_entry(
            censored.train_df,
            None,
            prediction_cutoff=float("-inf"),
            stop_loss=0.0,
            trade_cost=0.0,
        )
        self.assertEqual(baseline.candidate_episodes, 0)
        self.assertTrue(baseline.trades.empty)


def _execution_frame(index: pd.DatetimeIndex, sides: list[int]) -> pd.DataFrame:
    sequence = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            REGIME_SIDE_COLUMN: sides,
            REGIME_ACTION_TIME_COLUMN: index + pd.Timedelta(minutes=5),
            REGIME_ACTION_PRICE_COLUMN: np.full(len(index), 100.0),
            REGIME_ACTION_HIGH_COLUMN: np.full(len(index), 100.1),
            REGIME_ACTION_LOW_COLUMN: np.full(len(index), 99.9),
            REGIME_CLOSE_TIME_COLUMN: index + pd.Timedelta(minutes=10),
            REGIME_CLOSE_PRICE_COLUMN: 100.0 + sequence * 0.1,
        },
        index=index,
    )


if __name__ == "__main__":
    unittest.main()
