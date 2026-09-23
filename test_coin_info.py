import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import coin_info


class TestDescriptions(unittest.TestCase):
    def test_known_coin(self):
        cat, text = coin_info.describe("BTC")
        self.assertEqual(cat, "Layer 1")
        self.assertIn("digital money", text)

    def test_case_insensitive(self):
        self.assertEqual(coin_info.describe("btc"), coin_info.describe("BTC"))

    def test_unknown_coin_returns_none(self):
        self.assertIsNone(coin_info.describe("ZZZZ"))
        self.assertIsNone(coin_info.describe(""))

    def test_every_category_used_is_defined(self):
        for symbol, (cat, _t) in coin_info.COINS.items():
            self.assertIn(cat, coin_info.CATEGORIES, f"{symbol} uses undefined category {cat}")

    def test_descriptions_are_one_sentence_ish(self):
        for symbol, (_c, text) in coin_info.COINS.items():
            self.assertTrue(10 < len(text) < 260, f"{symbol}: {len(text)} chars")
            self.assertTrue(text[0].isupper(), symbol)
            self.assertTrue(text.endswith("."), symbol)

    def test_no_investment_advice_language(self):
        # Phrases that recommend, not verbs that describe — "lets traders buy
        # and sell future yield" is a description of the product.
        banned = ("you should buy", "worth buying", "good investment", "will moon",
                  "guaranteed", "undervalued", "should own", "great buy", "must own")
        for symbol, (_c, text) in coin_info.COINS.items():
            low = text.lower()
            for word in banned:
                self.assertNotIn(word, low, f"{symbol} contains '{word}'")

    def test_category_lookup(self):
        self.assertEqual(coin_info.category_of("PEPE"), "Meme")
        self.assertIsNone(coin_info.category_of("NOPE"))

    def test_coverage_counts(self):
        covered, total = coin_info.coverage(["BTC", "ETH", "ZZZZ"])
        self.assertEqual((covered, total), (2, 3))

    def test_covers_the_traders_watchlist(self):
        for s in ("BTC", "ETH", "SOL", "ZEC", "HYPE", "TAO", "NEAR", "SUI", "UNI",
                  "RENDER", "COMP", "STX", "IMX", "LINK", "XRP", "AVAX", "INJ", "ONDO",
                  "RSR", "PUMP", "DRV", "LIT"):
            self.assertIsNotNone(coin_info.describe(s), s)
