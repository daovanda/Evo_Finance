import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from crypto.prod.backend_exness_5m import (
    _notify_signal_once,
    _telegram_message,
    _validated_entry,
)
from crypto.prod.train_model import _top_prediction_threshold


def _manifest_entry(direction: str) -> dict:
    return {
        "rank": 1,
        "label_mode": "mfe",
        "label_direction": direction,
        "label_threshold": 0.001,
        "features": ["ret_close_3"],
        "models": [
            {
                "horizon": 3,
                "trade_top_fraction": 0.40,
                "val_trade_threshold": 0.61,
            }
        ],
    }


class ExnessFiveMinuteBackendTests(unittest.TestCase):
    def test_top_prediction_threshold_accepts_explicit_fraction(self):
        predictions = pd.Series(range(1, 101), dtype=float)

        threshold = _top_prediction_threshold(
            predictions,
            trade_top_fraction=0.40,
        )

        self.assertEqual(threshold, 61.0)

    def test_manifest_entry_requires_matching_mfe_direction_and_h3(self):
        long_entry = _manifest_entry("long")

        selected = _validated_entry(
            {"entries": [long_entry]},
            expected_direction="long",
        )

        self.assertEqual(selected, long_entry)
        with self.assertRaises(ValueError):
            _validated_entry(
                {"entries": [long_entry]},
                expected_direction="short",
            )

    def test_no_signal_sends_status_once(self):
        payload = {
            "has_trade_signal": False,
            "symbol": "BTCUSDT",
            "interval": "5m",
            "signal_time": "2026-01-01 00:00:00",
            "signal": "NO_SIGNAL",
            "strategy": {},
            "models": {},
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "crypto.prod.backend_exness_5m.send_telegram_message"
            ) as send,
        ):
            send.return_value = {"ok": True}
            state_path = Path(tmp) / "state.json"
            first = _notify_signal_once(
                payload,
                state_path,
                enabled=True,
            )
            second = _notify_signal_once(
                payload,
                state_path,
                enabled=True,
            )

        self.assertTrue(first["ok"])
        self.assertEqual(second["reason"], "duplicate")
        send.assert_called_once()

    def test_signal_message_contains_manual_strategy_and_model_details(self):
        payload = {
            "has_trade_signal": True,
            "signal": "LONG",
            "symbol": "BTCUSDT",
            "interval": "5m",
            "signal_time": "2026-01-01 00:00:00",
            "entry_candle_time": "2026-01-01 00:05:00",
            "entry_open_binance": 100.0,
            "entry_open_exness": 20.0,
            "strategy": {
                "filter_first_1m": 0.00025,
                "long_trigger_price": 100.025,
                "long_trigger_price_exness": 20.025,
                "short_trigger_price": 99.975,
                "short_trigger_price_exness": 19.975,
                "take_profit": 0.01,
                "stop_loss": 0.0,
            },
            "models": {
                "long": {
                    "ensemble_signal": True,
                    "pred_mean": 0.72,
                    "trade_top_fraction": 0.40,
                    "label_threshold": 0.001,
                    "predictions": [
                        {
                            "horizon": 3,
                            "pred": 0.72,
                            "threshold": 0.61,
                            "is_signal": True,
                        }
                    ],
                },
                "short": {
                    "ensemble_signal": False,
                    "pred_mean": 0.52,
                    "trade_top_fraction": 0.40,
                    "label_threshold": 0.001,
                    "predictions": [
                        {
                            "horizon": 3,
                            "pred": 0.52,
                            "threshold": 0.62,
                            "is_signal": False,
                        }
                    ],
                },
            },
        }

        message = _telegram_message(payload)

        self.assertIn("First 1m trigger: 0.025%", message)
        self.assertIn("5M SIGNAL | LONG", message)
        self.assertIn("Binance open: 100.00 | 20.00", message)
        self.assertIn("Long trigger >= 100.03 | 20.02", message)
        self.assertIn("Short trigger <= 99.97 | 19.98", message)
        self.assertIn("LONG: signal=YES | score=0.720000 | thr=0.610000", message)
        self.assertIn("SHORT: signal=NO | score=0.520000 | thr=0.620000", message)
        self.assertNotIn("Exness", message)
        self.assertNotIn("MONITOR ONLY", message)
        self.assertNotIn("BTCUSDT 5m", message)
        self.assertNotIn("Strategy:", message)
        self.assertNotIn("label_thr", message)


if __name__ == "__main__":
    unittest.main()
