import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import exchanges
from exchanges import (to_binance_symbol, to_kraken_pair, base_asset,
                        fetch_binance_klines, fetch_kraken_ohlc,
                        fetch_binance_price, fetch_kraken_price,
                        fetch_spot, build_frames)

exchanges.configure(max_retries=0, backoff=0.0, spacing=0.0)


class FakeResp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def binance_rows(n=50):
    base = 1758000000000
    return [[base + i * 14400000, "100.0", "110.0", "95.0", "105.0", "12.5",
             base + (i + 1) * 14400000, "0", 10, "0", "0", "0"] for i in range(n)]


def kraken_rows(n=50):
    base = 1758000000
    return [[base + i * 14400, "100.0", "110.0", "95.0", "105.0", "104.0", "12.5", 10]
            for i in range(n)]


class TestSymbolMapping(unittest.TestCase):
    def test_binance_uses_usdt_pairs(self):
        self.assertEqual(to_binance_symbol("BTC-USD"), "BTCUSDT")
        self.assertEqual(to_binance_symbol("SOL-USD"), "SOLUSDT")

    def test_kraken_uses_xbt_for_bitcoin(self):
        self.assertEqual(to_kraken_pair("BTC-USD"), "XBTUSD")

    def test_kraken_uses_xdg_for_doge(self):
        self.assertEqual(to_kraken_pair("DOGE-USD"), "XDGUSD")

    def test_other_assets_pass_through(self):
        self.assertEqual(to_kraken_pair("SOL-USD"), "SOLUSD")

    def test_base_asset_strips_suffix(self):
        self.assertEqual(base_asset("HYPE-USD"), "HYPE")


class TestBinance(unittest.TestCase):
    def setUp(self):
        self.original = exchanges.requests.get

    def tearDown(self):
        exchanges.requests.get = self.original

    def test_klines_parsed_to_ohlcv(self):
        exchanges.requests.get = lambda *a, **k: FakeResp(binance_rows())
        df, err = fetch_binance_klines("BTC-USD", "4h")
        self.assertIsNone(err)
        self.assertEqual(list(df.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(df["High"].iloc[0], 110.0)
        self.assertEqual(df["Volume"].iloc[0], 12.5)

    def test_true_daily_ohlc_unlike_coingecko(self):
        """The whole point of moving off CoinGecko: daily High != Low."""
        exchanges.requests.get = lambda *a, **k: FakeResp(binance_rows())
        df, err = fetch_binance_klines("BTC-USD", "1d")
        self.assertTrue((df["High"] > df["Low"]).all())

    def test_geo_block_451_is_reported_clearly(self):
        exchanges.requests.get = lambda *a, **k: FakeResp({}, status_code=451)
        df, err = fetch_binance_klines("BTC-USD", "4h")
        self.assertIsNone(df)
        self.assertIn("geo-blocked", err)

    def test_unlisted_symbol_reports_empty(self):
        exchanges.requests.get = lambda *a, **k: FakeResp([])
        df, err = fetch_binance_klines("NOTREAL-USD", "4h")
        self.assertIsNone(df)
        self.assertIn("no candles", err)

    def test_unsupported_interval_rejected(self):
        df, err = fetch_binance_klines("BTC-USD", "7m")
        self.assertIsNone(df)
        self.assertIn("Unsupported", err)

    def test_price_endpoint(self):
        exchanges.requests.get = lambda *a, **k: FakeResp({"price": "81544.12"})
        price, err = fetch_binance_price("BTC-USD")
        self.assertAlmostEqual(price, 81544.12)


class TestKraken(unittest.TestCase):
    def setUp(self):
        self.original = exchanges.requests.get

    def tearDown(self):
        exchanges.requests.get = self.original

    def test_ohlc_parsed(self):
        exchanges.requests.get = lambda *a, **k: FakeResp(
            {"error": [], "result": {"XXBTZUSD": kraken_rows(), "last": 1}})
        df, err = fetch_kraken_ohlc("BTC-USD", "4h")
        self.assertIsNone(err)
        self.assertEqual(df["High"].iloc[0], 110.0)

    def test_json_level_error_is_surfaced(self):
        """Kraken returns HTTP 200 with errors in the body."""
        exchanges.requests.get = lambda *a, **k: FakeResp(
            {"error": ["EQuery:Unknown asset pair"], "result": {}})
        df, err = fetch_kraken_ohlc("NOTREAL-USD", "4h")
        self.assertIsNone(df)
        self.assertIn("Unknown asset pair", err)

    def test_price_endpoint(self):
        exchanges.requests.get = lambda *a, **k: FakeResp(
            {"error": [], "result": {"XXBTZUSD": {"c": ["81544.12", "0.01"]}}})
        price, err = fetch_kraken_price("BTC-USD")
        self.assertAlmostEqual(price, 81544.12)


class TestFallbackBehaviour(unittest.TestCase):
    def setUp(self):
        self.original = exchanges.requests.get

    def tearDown(self):
        exchanges.requests.get = self.original

    def test_spot_falls_back_to_kraken_when_binance_blocked(self):
        def _get(url, params=None, **k):
            if "binance" in url:
                return FakeResp({}, status_code=451)
            return FakeResp({"error": [], "result": {"XXBTZUSD": {"c": ["100.0", "1"]}}})
        exchanges.requests.get = _get
        price, source, err = fetch_spot("BTC-USD")
        self.assertEqual(source, "Kraken")
        self.assertAlmostEqual(price, 100.0)

    def test_spot_prefers_binance_when_available(self):
        exchanges.requests.get = lambda *a, **k: FakeResp({"price": "55.5"})
        price, source, err = fetch_spot("BTC-USD")
        self.assertEqual(source, "Binance")

    def test_build_frames_uses_binance_when_available(self):
        exchanges.requests.get = lambda *a, **k: FakeResp(binance_rows())
        result = build_frames("BTC-USD")
        self.assertEqual(result.source, "Binance")
        for tf in ("1d", "4h", "1h", "5m"):
            self.assertFalse(result.frames[tf].empty)

    def test_build_frames_falls_back_to_kraken(self):
        def _get(url, params=None, **k):
            if "binance" in url:
                return FakeResp({}, status_code=451)
            return FakeResp({"error": [], "result": {"XXBTZUSD": kraken_rows(), "last": 1}})
        exchanges.requests.get = _get
        result = build_frames("BTC-USD")
        self.assertEqual(result.source, "Kraken")
        self.assertFalse(result.frames["4h"].empty)
        self.assertTrue(any("Binance unavailable" in p for p in result.problems))

    def test_build_frames_reports_total_failure(self):
        exchanges.requests.get = lambda *a, **k: FakeResp({}, status_code=451)
        result = build_frames("NOTREAL-USD")
        self.assertIsNone(result.source)
        self.assertTrue(all(f.empty for f in result.frames.values()))

    def test_source_is_not_mixed_across_timeframes(self):
        """Once a venue is chosen on 4H, the rest come from the same venue —
        mixing two exchanges' prices in one analysis would be misleading."""
        seen = {"binance": 0, "kraken": 0}

        def _get(url, params=None, **k):
            if "binance" in url:
                seen["binance"] += 1
                return FakeResp(binance_rows())
            seen["kraken"] += 1
            return FakeResp({"error": [], "result": {"X": kraken_rows(), "last": 1}})

        exchanges.requests.get = _get
        build_frames("BTC-USD")
        self.assertEqual(seen["kraken"], 0)


if __name__ == "__main__":
    unittest.main()


class TestBinanceHistoryPaging(unittest.TestCase):
    """Binance caps a request at 1,000 candles; history beyond that is paged."""

    def setUp(self):
        self.original = exchanges.requests.get

    def tearDown(self):
        exchanges.requests.get = self.original

    def _pager(self, available):
        """Serve `available` total candles, newest first, honouring endTime."""
        base = 1_700_000_000_000
        step = 3_600_000
        all_rows = [[base + i * step, "1", "2", "0.5", "1.5", "10",
                     base + (i + 1) * step, "0", 1, "0", "0", "0"]
                    for i in range(available)]
        calls = {"n": 0}

        def _get(url, params=None, **k):
            calls["n"] += 1
            end = params.get("endTime")
            pool = [r for r in all_rows if end is None or r[0] <= end]
            return FakeResp(pool[-params["limit"]:])
        return _get, calls

    def test_pages_beyond_one_thousand(self):
        get, calls = self._pager(2500)
        exchanges.requests.get = get
        df, err = exchanges.fetch_binance_history("BTC-USD", "1h", 2500)
        self.assertIsNone(err)
        self.assertEqual(len(df), 2500)
        self.assertEqual(calls["n"], 3)

    def test_result_is_sorted_and_deduplicated(self):
        get, _ = self._pager(2200)
        exchanges.requests.get = get
        df, _ = exchanges.fetch_binance_history("BTC-USD", "1h", 2200)
        self.assertTrue(df.index.is_monotonic_increasing)
        self.assertFalse(df.index.duplicated().any())

    def test_stops_at_start_of_listing(self):
        get, calls = self._pager(600)          # coin only has 600 candles
        exchanges.requests.get = get
        df, _ = exchanges.fetch_binance_history("NEW-USD", "1h", 4000)
        self.assertEqual(len(df), 600)
        self.assertEqual(calls["n"], 1)

    def test_partial_failure_keeps_pages_already_fetched(self):
        get, _ = self._pager(3000)
        state = {"n": 0}

        def flaky(url, params=None, **k):
            state["n"] += 1
            if state["n"] == 2:
                return FakeResp({}, status_code=451)
            return get(url, params=params)

        exchanges.requests.get = flaky
        df, err = exchanges.fetch_binance_history("BTC-USD", "1h", 3000)
        self.assertEqual(len(df), 1000)       # first page kept, not discarded

    def test_unsupported_interval(self):
        df, err = exchanges.fetch_binance_history("BTC-USD", "7m", 100)
        self.assertIsNone(df)
