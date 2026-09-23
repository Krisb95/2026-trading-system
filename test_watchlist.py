import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watchlist import resolve, parse_list, suggest, DEFAULT_WATCHLIST

MARKETS = ["BTC", "ETH", "SOL", "ZEC", "HYPE", "TAO", "NEAR", "SUI", "UNI", "RENDER",
           "COMP", "STX", "IMX", "LINK", "XRP", "AVAX", "INJ", "ONDO", "RSR", "PUMP",
           "kPEPE", "WIF"]


class TestParsing(unittest.TestCase):
    def test_splits_on_commas_and_newlines(self):
        self.assertEqual(parse_list("BTC, ETH\nSOL;AVAX"), ["BTC", "ETH", "SOL", "AVAX"])

    def test_ignores_blanks(self):
        self.assertEqual(parse_list("BTC,,  ,ETH"), ["BTC", "ETH"])


class TestResolving(unittest.TestCase):
    def test_plain_symbols(self):
        found, missing = resolve("BTC, ETH, SOL", MARKETS)
        self.assertEqual([r.market for r in found], ["BTC", "ETH", "SOL"])
        self.assertEqual(missing, [])

    def test_case_and_punctuation_insensitive(self):
        found, _ = resolve("btc , eth.", MARKETS)
        self.assertEqual([r.market for r in found], ["BTC", "ETH"])

    def test_old_ticker_resolves_via_alias(self):
        found, _ = resolve("RNDR", MARKETS)
        self.assertEqual(found[0].market, "RENDER")
        self.assertTrue(found[0].via_alias)

    def test_full_names_resolve(self):
        found, _ = resolve("stacks, immutable, chainlink", MARKETS)
        self.assertEqual([r.market for r in found], ["STX", "IMX", "LINK"])

    def test_two_coins_typed_as_one_entry(self):
        found, missing = resolve("immutable link", MARKETS)
        self.assertIn("IMX", [r.market for r in found])
        self.assertIn("LINK", [r.market for r in found])

    def test_partly_matched_entry_still_reports_the_unmatched_word(self):
        """'immutable xlink' resolves IMX but 'xlink' is ambiguous — it must be
        reported, not silently dropped."""
        found, missing = resolve("immutable xlink", MARKETS)
        self.assertIn("IMX", [r.market for r in found])
        self.assertIn("xlink", missing)

    def test_bundled_k_coin_matches_underlying_symbol(self):
        found, _ = resolve("PEPE", MARKETS)
        self.assertEqual(found[0].market, "kPEPE")

    def test_unknown_entries_are_reported_not_guessed(self):
        found, missing = resolve("BTC, CC, CARDS, DRV, LIGHTER", MARKETS)
        self.assertEqual([r.market for r in found], ["BTC"])
        self.assertEqual(missing, ["CC", "CARDS", "DRV", "LIGHTER"])

    def test_no_duplicates(self):
        found, _ = resolve("BTC, bitcoin, BTC", MARKETS)
        self.assertEqual(len(found), 1)

    def test_entry_not_on_the_venue_is_unmatched(self):
        found, missing = resolve("DOGE", MARKETS)
        self.assertEqual(found, [])
        self.assertEqual(missing, ["DOGE"])

    def test_default_watchlist_mostly_resolves(self):
        found, missing = resolve(DEFAULT_WATCHLIST, MARKETS)
        self.assertGreaterEqual(len(found), 18)
        for ambiguous in ("CC", "CARDS", "DRV", "LIGHTER"):
            self.assertIn(ambiguous, missing)

    def test_empty_input(self):
        self.assertEqual(resolve("", MARKETS), ([], []))


class TestSuggestions(unittest.TestCase):
    def test_suggests_near_matches(self):
        self.assertIn("LINK", suggest("LNK", MARKETS))

    def test_no_suggestion_for_nonsense(self):
        self.assertEqual(suggest("", MARKETS), [])


class TestCustomAliases(unittest.TestCase):
    def test_parses_user_mappings(self):
        from watchlist import parse_aliases
        aliases, bad = parse_aliases("lighter = LIT\nderive: DRV\n")
        self.assertEqual(aliases, {"LIGHTER": "LIT", "DERIVE": "DRV"})
        self.assertEqual(bad, [])

    def test_reports_bad_lines_rather_than_ignoring_them(self):
        from watchlist import parse_aliases
        aliases, bad = parse_aliases("lighter = LIT\nnonsense line\n")
        self.assertEqual(list(aliases), ["LIGHTER"])
        self.assertEqual(bad, ["nonsense line"])

    def test_comments_and_blanks_skipped(self):
        from watchlist import parse_aliases
        aliases, bad = parse_aliases("# my notes\n\nlighter=LIT")
        self.assertEqual(aliases, {"LIGHTER": "LIT"})
        self.assertEqual(bad, [])

    def test_custom_alias_resolves_a_name(self):
        aliases, _ = __import__("watchlist").parse_aliases("lighter = LIT")
        found, missing = resolve("LIGHTER", MARKETS + ["LIT"], aliases=aliases)
        self.assertEqual(found[0].market, "LIT")
        self.assertEqual(missing, [])

    def test_custom_alias_to_a_market_that_does_not_exist_still_unmatched(self):
        aliases, _ = __import__("watchlist").parse_aliases("cards = CARDS")
        found, missing = resolve("CARDS", MARKETS, aliases=aliases)
        self.assertEqual(found, [])
        self.assertEqual(missing, ["CARDS"])


class TestMarketSearch(unittest.TestCase):
    def test_finds_by_fragment(self):
        from watchlist import search_markets
        self.assertIn("LINK", search_markets("lin", MARKETS))

    def test_empty_query_lists_markets(self):
        from watchlist import search_markets
        self.assertEqual(len(search_markets("", MARKETS, limit=5)), 5)


class TestTraderConfirmedAliases(unittest.TestCase):
    """Mappings the trader confirmed. An alias only helps if the venue lists
    that market — it cannot conjure one into existence."""

    def test_lighter_resolves_to_lit_when_listed(self):
        found, missing = resolve("LIGHTER", MARKETS + ["LIT"])
        self.assertEqual(found[0].market, "LIT")
        self.assertTrue(found[0].via_alias)

    def test_derivative_and_derive_both_resolve_to_drv(self):
        for name in ("Derivative", "Derive", "DRV"):
            found, _ = resolve(name, MARKETS + ["DRV"])
            self.assertEqual(found[0].market, "DRV", name)

    def test_alias_does_not_help_if_the_market_is_not_listed(self):
        found, missing = resolve("LIGHTER, Derivative", MARKETS)   # no LIT, no DRV
        self.assertEqual(found, [])
        self.assertEqual(missing, ["LIGHTER", "Derivative"])

    def test_cards_matches_only_when_the_venue_lists_it(self):
        self.assertEqual(resolve("CARDS", MARKETS)[1], ["CARDS"])
        found, missing = resolve("CARDS", MARKETS + ["CARDS"])
        self.assertEqual(found[0].market, "CARDS")
        self.assertEqual(missing, [])
