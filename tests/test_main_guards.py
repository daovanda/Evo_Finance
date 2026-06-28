import unittest
import subprocess
import sys
from unittest.mock import patch

import pandas as pd

import main


class MainGuardTests(unittest.TestCase):
    def test_missing_sector_mapping_raises_by_default(self):
        with patch.object(main, "REQUIRE_SECTOR_MAPPING", True):
            with self.assertRaisesRegex(ValueError, "SECTORS mapping is missing"):
                main._handle_missing_sector_mapping(["AAA", "BBB"])

    def test_missing_sector_mapping_can_warn_when_disabled(self):
        with patch.object(main, "REQUIRE_SECTOR_MAPPING", False):
            with self.assertLogs("evo_finance.main", level="WARNING") as logs:
                main._handle_missing_sector_mapping(["AAA"])

        self.assertIn("Unknown", "\n".join(logs.output))

    def test_small_sector_universe_raises_by_default(self):
        with patch.object(main, "REQUIRE_SECTOR_MAPPING", True):
            with self.assertRaisesRegex(ValueError, "too few loaded tickers"):
                main._handle_small_sector_universe({"Banking": 1})

    def test_small_sector_universe_can_warn_when_disabled(self):
        with patch.object(main, "REQUIRE_SECTOR_MAPPING", False):
            with self.assertLogs("evo_finance.main", level="WARNING") as logs:
                main._handle_small_sector_universe({"Banking": 1})

        self.assertIn("noisy", "\n".join(logs.output))

    def test_run_requires_market_columns_for_direct_data(self):
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=5), ["ACB", "BID"]],
            names=["date", "ticker"],
        )
        df = pd.DataFrame(
            {
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 1000.0,
            },
            index=idx,
        )

        with self.assertRaisesRegex(ValueError, "missing market columns"):
            main.run(df, time_budget=0.0)

    def test_run_rejects_inconsistent_market_columns_for_direct_data(self):
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=5), ["ACB", "BID"]],
            names=["date", "ticker"],
        )
        df = pd.DataFrame(
            {
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 1000.0,
                "market_open": 1000.0,
                "market_high": 1010.0,
                "market_low": 990.0,
                "market_close": 1005.0,
                "market_volume": 1_000_000.0,
            },
            index=idx,
        )
        df.loc[(pd.Timestamp("2024-01-03"), "BID"), "market_close"] = 2005.0

        with self.assertRaisesRegex(ValueError, "identical within each date"):
            main.run(df, time_budget=0.0)

    def test_main_cli_exposes_walk_forward_overrides(self):
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            cwd=".",
            capture_output=True,
            text=True,
            check=True,
        )

        for flag in (
            "--wf-end",
            "--wf-min-train-months",
            "--wf-val-months",
            "--wf-step-months",
            "--wf-purge-days",
        ):
            self.assertIn(flag, result.stdout)
        self.assertNotIn("--resume-recheck", result.stdout)


if __name__ == "__main__":
    unittest.main()
