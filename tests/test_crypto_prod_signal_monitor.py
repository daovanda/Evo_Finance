import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from crypto.prod.live_backend import (
    _atomic_write_csv,
    _entry_score_band_index,
    _lst_strategy_signal,
    _prediction_score_band_index,
)
from crypto.prod.train_model import (
    _archive_horizons,
    _score_band_cutoffs,
)


class ProductionScoreBandTests(unittest.TestCase):
    def test_atomic_csv_write_replaces_complete_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prices.csv"
            path.write_text("old\n1\n", encoding="utf-8")
            expected = pd.DataFrame({"date": ["2026-01-01"], "close": [100.0]})

            _atomic_write_csv(path, expected)

            actual = pd.read_csv(path)
            pd.testing.assert_frame_equal(actual, expected)
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_score_band_cutoffs_match_disjoint_top_fractions(self):
        pred = pd.Series(range(1, 101), dtype=float)

        cutoffs = _score_band_cutoffs(pred)

        self.assertEqual(cutoffs["q1"], 96.0)
        self.assertEqual(cutoffs["q2"], 91.0)
        self.assertEqual(cutoffs["q6"], 71.0)
        self.assertEqual(_prediction_score_band_index(98.0, cutoffs), 1)
        self.assertEqual(_prediction_score_band_index(93.0, cutoffs), 2)
        self.assertEqual(_prediction_score_band_index(70.0, cutoffs), 7)

    def test_horizon_ensemble_uses_weakest_band(self):
        predictions = [
            {"score_band_index": 1},
            {"score_band_index": 3},
            {"score_band_index": 2},
        ]

        self.assertEqual(_entry_score_band_index(predictions), 3)

    def test_archive_horizons_come_from_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "archive.json"
            archive.write_text(
                json.dumps({"metadata": {"horizons": [24, 5, 10, 5]}}),
                encoding="utf-8",
            )

            self.assertEqual(_archive_horizons(archive), [5, 10, 24])


class LstDecisionFlowTests(unittest.TestCase):
    def test_ordered_three_model_flow(self):
        cases = {
            (1, 1, 1): "LONG_STRONG",
            (2, 3, 4): "BOTH",
            (2, 3, 7): "LONG",
            (3, 7, 4): "LONG",
            (7, 1, 1): "LONG",
            (7, 2, 3): "SHORT",
            (7, 4, 4): "BOTH_WEAK",
            (7, 1, 3): "NO_SIGNAL",
            (3, 7, 7): "LONG",
            (7, 7, 1): "LONG",
            (7, 7, 2): "SHORT",
            (7, 7, 3): "NO_SIGNAL",
            (7, 7, 7): "NO_SIGNAL",
        }
        for bands, expected in cases.items():
            with self.subTest(bands=bands):
                self.assertEqual(_lst_strategy_signal(*bands), expected)


if __name__ == "__main__":
    unittest.main()
