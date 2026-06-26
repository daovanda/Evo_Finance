import json
import tempfile
import unittest
from pathlib import Path

from archive.archive import Archive
from main import _checkpoint_path, _maybe_save_checkpoint
from model.trainer import Trainer
from mutator.gene import Individual


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


if __name__ == "__main__":
    unittest.main()
