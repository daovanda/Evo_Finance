import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from archive.archive import Archive
from main import (
    _checkpoint_path,
    _load_archive_json_into_archive,
    _maybe_save_checkpoint,
    _prediction_metrics,
    _safe_float,
    _save_json,
)
from model.trainer import Trainer
from mutator.gene import Gene, Individual


def _archive_with_one_entry() -> Archive:
    archive = Archive()
    individual = Individual.seed()
    individual.score = 1.23
    individual.metrics = {
        "val_mean_ic": 0.1,
        "val_icir": 0.2,
        "hit_rate": 0.3,
        "overfit_gap": 0.0,
    }
    archive.try_add(individual, booster=object())
    return archive


class CheckpointTests(unittest.TestCase):
    def test_resume_archive_is_always_re_evaluated(self):
        rows = [
            {
                "generation": 7,
                "score": 999.0,
                "genes": ["ret(close_1, 5)", "volume_ratio(20)"],
                "wf_mean_ic": 999.0,
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "archive.json"
            path.write_text(json.dumps(rows))
            archive = Archive(max_size=10)

            with patch.object(
                archive, "add_loaded", wraps=archive.add_loaded
            ) as add_loaded_mock:
                with patch(
                    "main._evaluate_and_archive_wf", return_value=True
                ) as eval_mock:
                    _load_archive_json_into_archive(
                        path,
                        archive,
                        Trainer(),
                        object(),
                        [],
                        pd.DataFrame(),
                    )

        add_loaded_mock.assert_not_called()
        eval_mock.assert_called_once()
        individual = eval_mock.call_args.args[0]
        self.assertEqual(
            individual.formulas,
            ["ret(close_1, 5)", "volume_ratio(20)"],
        )
        self.assertEqual(len(archive), 0)

    def test_archive_add_loaded_preserves_metrics_for_resume(self):
        archive = Archive(max_size=1)
        individual = Individual.seed()

        added = archive.add_loaded(
            individual,
            score=0.42,
            metrics={"wf_mean_ic": 0.12},
            final_val_metrics={"final_val_mean_ic": 0.03},
            test_metrics={"test_mean_ic": -0.01},
        )

        self.assertTrue(added)
        self.assertEqual(len(archive), 1)
        entry = archive.entries[0]
        self.assertIsNone(entry.booster)
        self.assertEqual(entry.score, 0.42)
        self.assertEqual(entry.metrics["wf_mean_ic"], 0.12)
        self.assertEqual(entry.final_val_metrics["final_val_mean_ic"], 0.03)
        self.assertEqual(entry.test_metrics["test_mean_ic"], -0.01)

    def test_archive_keeps_best_duplicate_feature_set(self):
        archive = Archive(max_size=10)
        first = Individual(
            genes=[Gene("ret(close_1, 5)"), Gene("volume_ratio(20)")],
            score=0.10,
        )
        weaker_duplicate_reordered = Individual(
            genes=[Gene("volume_ratio(20)"), Gene("ret(close_1, 5)")],
            score=0.05,
        )
        stronger_duplicate_reordered = Individual(
            genes=[Gene("volume_ratio(20)"), Gene("ret(close_1, 5)")],
            score=0.99,
        )

        self.assertTrue(archive.try_add(first, booster=object()))
        self.assertFalse(archive.try_add(weaker_duplicate_reordered, booster=object()))
        self.assertEqual(len(archive), 1)
        self.assertEqual(archive.best.score, 0.10)

        self.assertTrue(archive.try_add(stronger_duplicate_reordered, booster=object()))
        self.assertEqual(len(archive), 1)
        self.assertEqual(archive.best.score, 0.99)

        loaded_duplicate = Individual(
            genes=[Gene("ret(close_1, 5)"), Gene("volume_ratio(20)")],
        )
        self.assertFalse(archive.add_loaded(loaded_duplicate, score=0.50))
        self.assertTrue(archive.add_loaded(loaded_duplicate, score=1.0))
        self.assertEqual(len(archive), 1)
        self.assertEqual(archive.best.score, 1.0)

    def test_archive_rejects_non_finite_scores(self):
        archive = Archive(max_size=10)
        bad_live = Individual.seed()
        bad_live.score = math.inf

        self.assertFalse(archive.try_add(bad_live, booster=object()))
        self.assertFalse(archive.add_loaded(Individual.seed(), score=float("nan")))
        self.assertFalse(archive.add_loaded(Individual.seed(), score=float("inf")))
        self.assertEqual(len(archive), 0)
        self.assertIsNone(_safe_float(float("inf")))
        self.assertIsNone(_safe_float(float("-inf")))

    def test_checkpoint_path_is_next_to_final_archive(self):
        self.assertEqual(
            _checkpoint_path(Path("results/archive.json")),
            Path("results/archive.checkpoint.json"),
        )
        self.assertEqual(
            _checkpoint_path(Path("results/archive")),
            Path("results/archive.checkpoint.json"),
        )
        self.assertIsNone(_checkpoint_path(None))

    def test_maybe_save_checkpoint_respects_interval(self):
        archive = _archive_with_one_entry()
        trainer = Trainer()
        trainer._feature_cache[("ctx", "close_1")] = object()
        trainer._split_cache[("split",)] = (object(), [1])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "archive.checkpoint.json"

            last = _maybe_save_checkpoint(
                archive,
                path,
                last_checkpoint=0.0,
                checkpoint_every=100.0,
                now=50.0,
                trainer=trainer,
            )
            self.assertEqual(last, 0.0)
            self.assertFalse(path.exists())
            self.assertEqual(len(trainer._feature_cache), 1)
            self.assertEqual(len(trainer._split_cache), 1)

            last = _maybe_save_checkpoint(
                archive,
                path,
                last_checkpoint=0.0,
                checkpoint_every=100.0,
                now=100.0,
                trainer=trainer,
            )
            self.assertEqual(last, 100.0)
            self.assertTrue(path.exists())
            self.assertEqual(trainer._feature_cache, {})
            self.assertEqual(trainer._split_cache, {})

            rows = json.loads(path.read_text())
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["score"], 1.23)

    def test_save_json_writes_strict_json_without_nan_literals(self):
        archive = Archive(max_size=10)
        individual = Individual.seed()
        archive.add_loaded(
            individual,
            score=0.42,
            metrics={"wf_mean_ic": float("nan")},
            final_val_metrics={"final_val_mean_ic": float("inf")},
            test_metrics={"test_mean_ic": float("-inf")},
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "archive.json"
            _save_json(archive, path)
            raw = path.read_text()
            self.assertNotIn("NaN", raw)
            self.assertNotIn("Infinity", raw)

            rows = json.loads(raw)
            self.assertIsNone(rows[0]["wf_mean_ic"])
            self.assertIsNone(rows[0]["final_val_metrics"]["final_val_mean_ic"])
            self.assertIsNone(rows[0]["test_metrics"]["test_mean_ic"])

    def test_prediction_metrics_include_hit_excess_on_valid_rows(self):
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], [f"T{i:02d}" for i in range(20)]],
            names=["date", "ticker"],
        )
        pred = pd.Series(range(20), index=index, dtype=float)
        labels = pd.Series(range(20), index=index, dtype=float)
        pred.iloc[:10] = float("nan")

        metrics = _prediction_metrics(pred, labels, index, prefix="test")

        self.assertEqual(metrics["test_hit_rate"], 1.0)
        self.assertEqual(metrics["test_hit_baseline"], 1.0)
        self.assertEqual(metrics["test_hit_excess"], 0.0)


if __name__ == "__main__":
    unittest.main()
