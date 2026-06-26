import unittest
from unittest.mock import patch

import config.settings as settings


class SettingsValidationTests(unittest.TestCase):
    def test_current_settings_validate(self):
        settings.validate_config()

    def test_mutator_probabilities_must_sum_to_one(self):
        bad_probs = {"c1": 0.5, "c2": 0.5, "c3": 0.5}
        with patch.object(settings, "MUTATOR_PROBS", bad_probs):
            with self.assertRaisesRegex(ValueError, "sum to 1.0"):
                settings.validate_config()

    def test_feature_limits_must_be_ordered(self):
        with (
            patch.object(settings, "FEATURE_MIN", 10),
            patch.object(settings, "FEATURE_MAX", 3),
        ):
            with self.assertRaisesRegex(ValueError, "FEATURE_MIN"):
                settings.validate_config()

    def test_final_dates_must_be_ordered(self):
        with patch.object(settings, "VAL_START", "2025-01-01"):
            with self.assertRaisesRegex(ValueError, "VAL_START must be before TEST_START"):
                settings.validate_config()

        with patch.object(settings, "TEST_END", "2024-12-31"):
            with self.assertRaisesRegex(ValueError, "TEST_END must be >= TEST_START"):
                settings.validate_config()

    def test_walk_forward_must_not_enter_final_test(self):
        with patch.object(settings, "WF_END", "2025-02-01"):
            with self.assertRaisesRegex(ValueError, "WF_END must be <= TEST_START"):
                settings.validate_config()

    def test_purge_must_cover_holding_horizon(self):
        with patch.object(settings, "WF_PURGE_DAYS", settings.HOLDING_HORIZON - 1):
            with self.assertRaisesRegex(ValueError, "WF_PURGE_DAYS must be >= HOLDING_HORIZON"):
                settings.validate_config()


if __name__ == "__main__":
    unittest.main()
