import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import venues
from venues import (VenueListings, fetch_bybit_perps, fetch_hyperliquid_perps,
                     filter_universe, base_asset_from_label)


class FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


UNIVERSE = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Hyperliquid (HYPE)": "HYPE-USD",
    "WhiteBIT Coin (WBT)": "WBT-USD",
    "OKB (OKB)": "OKB-USD",
    "Pepe (PEPE)": "PEPE-USD",
}


class TestBybit(unittest.TestCase):
    def setUp(self):
        self.original = venues.requests.get

    def tearDown(self):
        venues.requests.get = self.original

    def test_parses_usdt_perps(self):
        venues.requests.get = lambda *a, **k: FakeResp({
            "retCode": 0, "result": {"list": [
                {"baseCoin": "BTC", "quoteCoin": "USDT", "status": "Trading"},
                {"baseCoin": "ETH", "quoteCoin": "USDT", "status": "Trading"},
            ]}})
        assets, err = fetch_bybit_perps()
        self.assertIsNone(err)
        self.assertEqual(assets, {"BTC", "ETH"})

    def test_ignores_non_usdt_quote(self):
        venues.requests.get = lambda *a, **k: FakeResp({
            "retCode": 0, "result": {"list": [
                {"baseCoin": "BTC", "quoteCoin": "USDT", "status": "Trading"},
                {"baseCoin": "SOL", "quoteCoin": "USDC", "status": "Trading"},
            ]}})
        assets, err = fetch_bybit_perps()
        self.assertEqual(assets, {"BTC"})

    def test_ignores_non_trading_status(self):
        venues.requests.get = lambda *a, **k: FakeResp({
            "retCode": 0, "result": {"list": [
                {"baseCoin": "BTC", "quoteCoin": "USDT", "status": "Trading"},
                {"baseCoin": "DEAD", "quoteCoin": "USDT", "status": "Delivering"},
            ]}})
        assets, err = fetch_bybit_perps()
        self.assertEqual(assets, {"BTC"})

    def test_network_failure_reported(self):
        venues.requests.get = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down"))
        assets, err = fetch_bybit_perps()
        self.assertEqual(assets, set())
        self.assertIn("unreachable", err)


class TestHyperliquid(unittest.TestCase):
    def setUp(self):
        self.original = venues.requests.post

    def tearDown(self):
        venues.requests.post = self.original

    def test_parses_universe(self):
        venues.requests.post = lambda *a, **k: FakeResp({
            "universe": [{"name": "BTC"}, {"name": "HYPE"}, {"name": "ETH"}]})
        assets, err = fetch_hyperliquid_perps()
        self.assertEqual(assets, {"BTC", "HYPE", "ETH"})

    def test_skips_delisted(self):
        venues.requests.post = lambda *a, **k: FakeResp({
            "universe": [{"name": "BTC"}, {"name": "OLD", "isDelisted": True}]})
        assets, err = fetch_hyperliquid_perps()
        self.assertEqual(assets, {"BTC"})

    def test_failure_reported(self):
        venues.requests.post = lambda *a, **k: (_ for _ in ()).throw(TimeoutError("slow"))
        assets, err = fetch_hyperliquid_perps()
        self.assertEqual(assets, set())
        self.assertIn("unreachable", err)


class TestFiltering(unittest.TestCase):
    def test_drops_exchange_tokens(self):
        listings = VenueListings(bybit={"BTC", "ETH", "PEPE"}, hyperliquid={"BTC", "HYPE"})
        filtered, tags, notes = filter_universe(UNIVERSE, listings)
        self.assertNotIn("WhiteBIT Coin (WBT)", filtered)
        self.assertNotIn("OKB (OKB)", filtered)

    def test_keeps_coins_on_either_venue(self):
        listings = VenueListings(bybit={"PEPE"}, hyperliquid={"HYPE"})
        filtered, tags, notes = filter_universe(UNIVERSE, listings)
        self.assertIn("Pepe (PEPE)", filtered)
        self.assertIn("Hyperliquid (HYPE)", filtered)

    def test_require_both_narrows_further(self):
        listings = VenueListings(bybit={"BTC", "PEPE"}, hyperliquid={"BTC", "HYPE"})
        either, _, _ = filter_universe(UNIVERSE, listings, require_both=False)
        both, _, _ = filter_universe(UNIVERSE, listings, require_both=True)
        self.assertIn("Pepe (PEPE)", either)
        self.assertNotIn("Pepe (PEPE)", both)
        self.assertIn("Bitcoin (BTC)", both)

    def test_venue_tags_identify_where_listed(self):
        listings = VenueListings(bybit={"BTC"}, hyperliquid={"BTC", "HYPE"})
        filtered, tags, notes = filter_universe(UNIVERSE, listings)
        self.assertEqual(sorted(tags["Bitcoin (BTC)"]), ["Bybit", "HL"])
        self.assertEqual(tags["Hyperliquid (HYPE)"], ["HL"])

    def test_unreachable_venues_do_not_silently_pass_everything(self):
        listings = VenueListings(bybit=set(), hyperliquid=set(),
                                  problems=["Bybit unreachable", "HL unreachable"])
        filtered, tags, notes = filter_universe(UNIVERSE, listings)
        self.assertEqual(len(filtered), len(UNIVERSE))       # nothing dropped
        self.assertTrue(any("NOT applied" in n for n in notes))   # but clearly flagged

    def test_partial_outage_is_flagged(self):
        listings = VenueListings(bybit={"BTC"}, hyperliquid=set(),
                                  problems=["Hyperliquid unreachable: TimeoutError"])
        filtered, tags, notes = filter_universe(UNIVERSE, listings)
        self.assertTrue(any("narrower" in n for n in notes))

    def test_note_reports_counts(self):
        listings = VenueListings(bybit={"BTC", "ETH"}, hyperliquid={"HYPE"})
        filtered, tags, notes = filter_universe(UNIVERSE, listings)
        self.assertTrue(any("Filtered to 3 coins" in n for n in notes))

    def test_base_asset_extraction(self):
        self.assertEqual(base_asset_from_label("PEPE-USD"), "PEPE")


if __name__ == "__main__":
    unittest.main()
