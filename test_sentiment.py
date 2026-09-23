import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sentiment


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._p


ALT = {"data": [{"value": "72", "value_classification": "Greed", "timestamp": "1758000000"},
                {"value": "65", "value_classification": "Greed", "timestamp": "1757913600"}]}
CMC = {"data": {"value": 68, "value_classification": "Greed",
                "update_time": "2026-09-21T00:00:00.000Z"}}


class Base(unittest.TestCase):
    def setUp(self):
        self.orig = sentiment.requests.get
        sentiment.set_cmc_api_key("")

    def tearDown(self):
        sentiment.requests.get = self.orig
        sentiment.set_cmc_api_key("")


class TestBands(unittest.TestCase):
    def test_classification_bands(self):
        self.assertEqual(sentiment.classify(10), "Extreme Fear")
        self.assertEqual(sentiment.classify(35), "Fear")
        self.assertEqual(sentiment.classify(50), "Neutral")
        self.assertEqual(sentiment.classify(65), "Greed")
        self.assertEqual(sentiment.classify(90), "Extreme Greed")

    def test_bucket_matches_classification(self):
        for v in (5, 30, 50, 70, 95):
            self.assertEqual(sentiment.bucket(v), sentiment.classify(v))

    def test_bucket_of_none(self):
        self.assertIsNone(sentiment.bucket(None))


class TestFreeFeed(Base):
    def test_reads_value_and_direction(self):
        sentiment.requests.get = lambda *a, **k: FakeResp(ALT)
        fg, err = sentiment.fetch_fear_greed()
        self.assertIsNone(err)
        self.assertEqual(fg.value, 72)
        self.assertEqual(fg.classification, "Greed")
        self.assertEqual(fg.source, "Alternative.me")
        self.assertEqual(fg.previous, 65)
        self.assertEqual(fg.direction, "rising")

    def test_failure_is_reported(self):
        sentiment.requests.get = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("x"))
        fg, err = sentiment.fetch_fear_greed()
        self.assertIsNone(fg)
        self.assertIn("ConnectionError", err)

    def test_empty_payload(self):
        sentiment.requests.get = lambda *a, **k: FakeResp({"data": []})
        self.assertIsNone(sentiment.fetch_fear_greed()[0])


class TestCoinMarketCap(Base):
    def test_used_when_a_key_is_set(self):
        sentiment.set_cmc_api_key("abc")
        seen = {}

        def get(url, **k):
            seen["url"] = url
            seen["headers"] = k.get("headers", {})
            return FakeResp(CMC)

        sentiment.requests.get = get
        fg, err = sentiment.fetch_fear_greed()
        self.assertEqual(fg.source, "CoinMarketCap")
        self.assertEqual(fg.value, 68)
        self.assertIn("coinmarketcap", seen["url"])
        self.assertEqual(seen["headers"].get("X-CMC_PRO_API_KEY"), "abc")

    def test_bad_key_falls_back_to_free_feed(self):
        sentiment.set_cmc_api_key("bad")

        def get(url, **k):
            return FakeResp({}, status=401) if "coinmarketcap" in url else FakeResp(ALT)

        sentiment.requests.get = get
        fg, err = sentiment.fetch_fear_greed()
        self.assertEqual(fg.source, "Alternative.me")
        self.assertIn("CoinMarketCap failed", err)

    def test_free_feed_used_without_a_key(self):
        sentiment.requests.get = lambda *a, **k: FakeResp(ALT)
        self.assertEqual(sentiment.fetch_fear_greed()[0].source, "Alternative.me")

    def test_both_failing_reports_both(self):
        sentiment.set_cmc_api_key("bad")
        sentiment.requests.get = lambda *a, **k: FakeResp({}, status=500)
        fg, err = sentiment.fetch_fear_greed()
        self.assertIsNone(fg)
        self.assertIn("CoinMarketCap", err)
        self.assertIn("Alternative.me", err)


if __name__ == "__main__":
    unittest.main()
