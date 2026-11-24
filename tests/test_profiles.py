import unittest

from config.profiles import normalize_profile_name
from config.profiles import sp500_15m, vnindex
from craw_sp500 import MARKET_SYMBOL, SP500_UNIVERSE


class DatasetProfileTests(unittest.TestCase):
    def test_aliases_resolve_to_canonical_profiles(self):
        self.assertEqual(normalize_profile_name("vn"), "vnindex")
        self.assertEqual(normalize_profile_name("sp500"), "sp500_15m")

    def test_vnindex_profile_preserves_legacy_defaults(self):
        self.assertEqual(vnindex.MARKET_INDEX_TICKER, "VNINDEX")
        self.assertEqual(vnindex.DATA_FILE_SUFFIX, "")
        self.assertEqual(vnindex.HOLDING_HORIZON, 10)
        self.assertEqual(vnindex.DEFAULT_DATA_DIR, "data/raw")
        self.assertEqual(vnindex.WF_TRAIN_BAR_STRIDE, 1)

    def test_sp500_profile_has_unique_complete_sector_universe(self):
        tickers = [ticker for values in sp500_15m.SECTORS.values() for ticker in values]
        self.assertEqual(len(tickers), 50)
        self.assertEqual(len(set(tickers)), 50)
        self.assertNotIn(sp500_15m.MARKET_INDEX_TICKER, tickers)
        self.assertEqual(sp500_15m.DATA_FILE_SUFFIX, "_15Min")
        self.assertEqual(sp500_15m.BAR_INTERVAL_MINUTES, 15)
        self.assertIsInstance(sp500_15m.WF_TRAIN_BAR_STRIDE, int)
        self.assertGreaterEqual(sp500_15m.WF_TRAIN_BAR_STRIDE, 1)

    def test_sp500_profile_matches_crawler_universe(self):
        profile_tickers = {
            ticker for values in sp500_15m.SECTORS.values() for ticker in values
        }
        self.assertEqual(profile_tickers, set(SP500_UNIVERSE))
        self.assertEqual(sp500_15m.MARKET_INDEX_TICKER, MARKET_SYMBOL)


if __name__ == "__main__":
    unittest.main()
