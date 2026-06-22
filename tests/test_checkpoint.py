import json
import tempfile
import unittest
from pathlib import Path

from archive.archive import Archive
from main import _checkpoint_path, _maybe_save_checkpoint
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
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "archive.checkpoint.json"

            last = _maybe_save_checkpoint(
                archive,
                path,
                last_checkpoint=0.0,
                checkpoint_every=100.0,
                now=50.0,
            )
            self.assertEqual(last, 0.0)
            self.assertFalse(path.exists())

            last = _maybe_save_checkpoint(
                archive,
                path,
                last_checkpoint=0.0,
                checkpoint_every=100.0,
                now=100.0,
            )
            self.assertEqual(last, 100.0)
            self.assertTrue(path.exists())

            rows = json.loads(path.read_text())
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["score"], 1.23)


if __name__ == "__main__":
    unittest.main()
