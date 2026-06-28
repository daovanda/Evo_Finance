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

    def test_feature_min_valid_ratio_must_be_probability(self):
        with patch.object(settings, "FEATURE_MIN_VALID_RATIO", 1.5):
            with self.assertRaisesRegex(ValueError, "FEATURE_MIN_VALID_RATIO"):
                settings.validate_config()

    def test_feature_max_dominant_value_ratio_must_be_probability(self):
        with patch.object(settings, "FEATURE_MAX_DOMINANT_VALUE_RATIO", 1.5):
            with self.assertRaisesRegex(ValueError, "FEATURE_MAX_DOMINANT_VALUE_RATIO"):
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

    def test_sector_mapping_rejects_duplicate_ticker(self):
        sectors = {
            "Banking": ["AAA", "BBB"],
            "Real_Estate": ["BBB", "CCC"],
        }
        with patch.object(settings, "SECTORS", sectors):
            with self.assertRaisesRegex(ValueError, "multiple sectors"):
                settings.validate_config()

    def test_sector_mapping_rejects_empty_ticker(self):
        sectors = {"Banking": ["AAA", "  "]}
        with patch.object(settings, "SECTORS", sectors):
            with self.assertRaisesRegex(ValueError, "empty ticker"):
                settings.validate_config()

    def test_sector_mapping_rejects_non_list_values(self):
        sectors = {"Banking": "AAA"}
        with patch.object(settings, "SECTORS", sectors):
            with self.assertRaisesRegex(ValueError, "ticker list"):
                settings.validate_config()

    def test_sector_mapping_rejects_too_small_sector_when_strict(self):
        sectors = {
            "Solo": ["AAA"],
            "Pair": ["BBB", "CCC"],
        }
        with patch.object(settings, "SECTORS", sectors):
            with self.assertRaisesRegex(ValueError, "too small"):
                settings.validate_config()

    def test_sector_mapping_allows_small_sector_when_not_strict(self):
        sectors = {
            "Solo": ["AAA"],
            "Pair": ["BBB", "CCC"],
        }
        with (
            patch.object(settings, "SECTORS", sectors),
            patch.object(settings, "REQUIRE_SECTOR_MAPPING", False),
        ):
            settings.validate_config()

    def test_min_sector_members_must_be_positive(self):
        with patch.object(settings, "MIN_SECTOR_MEMBERS_IN_UNIVERSE", 0):
            with self.assertRaisesRegex(ValueError, "MIN_SECTOR_MEMBERS_IN_UNIVERSE"):
                settings.validate_config()


if __name__ == "__main__":
    unittest.main()
