import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scanner import derive_levels, atr_value
from universe import fetch_top_cryptos, FALLBACK_CRYPTO, EXCLUDED_SYMBOLS


def ohlc(closes, spread=0.5, freq="4h"):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame({"Open": closes, "High": [c + spread for c in closes],
                          "Low": [c - spread for c in closes], "Close": closes}, index=idx)


def staircase():
    """Uptrend with swing highs near 120 / 140 / 160 and lows near 105 / 112."""
    seq = []
    for a, b in [(100, 120), (120, 105), (105, 140), (140, 112), (112, 160), (160, 98)]:
        seq += list(np.linspace(a, b, 9))[1:]
    return ohlc([100] + seq)


class TestDeriveLevels(unittest.TestCase):
    """Regression tests for the bug where stop=nearest-low and
    target=nearest-high forced reward:risk to ~1:1, making the 2:1
    requirement mathematically unreachable and capping every setup at C/B."""

    def test_target_walks_out_to_meet_min_rr(self):
        df = staircase()
        _, target_low, rr_low, _ = derive_levels(df, 115.0, "Long", min_rr=1.0)
        _, target_high, rr_high, _ = derive_levels(df, 115.0, "Long", min_rr=2.0)
        self.assertGreater(target_high, target_low)
        self.assertGreaterEqual(rr_high, 2.0)

    def test_rr_is_not_pinned_at_one(self):
        df = staircase()
        _, _, rr, _ = derive_levels(df, 115.0, "Long", min_rr=2.0)
        self.assertNotAlmostEqual(rr, 1.0, places=1)

    def test_stop_sits_beyond_the_swing_not_on_it(self):
        df = staircase()
        stop, _, _, _ = derive_levels(df, 115.0, "Long", min_rr=2.0, stop_buffer_atr=0.25)
        # The nearest confirmed swing low below 115 is ~111.5; the stop must
        # be below it, not exactly on it.
        self.assertLess(stop, 111.5)

    def test_zero_buffer_places_stop_on_the_swing(self):
        df = staircase()
        stop, _, _, _ = derive_levels(df, 115.0, "Long", min_rr=2.0, stop_buffer_atr=0.0)
        self.assertAlmostEqual(stop, 111.5, delta=0.6)

    def test_reports_honest_shortfall_when_no_level_reaches_min_rr(self):
        df = staircase()
        stop, target, rr, reason = derive_levels(df, 115.0, "Long", min_rr=50.0)
        self.assertIsNotNone(target)
        self.assertLess(rr, 50.0)
        self.assertIn("fails the reward:risk test", reason)

    def test_short_direction_mirrors_long(self):
        df = staircase()
        stop, target, rr, _ = derive_levels(df, 115.0, "Short", min_rr=1.0)
        self.assertGreater(stop, 115.0)   # stop above entry for a short
        self.assertLess(target, 115.0)    # target below entry

    def test_no_swings_returns_none_not_invented_levels(self):
        df = ohlc([100, 101, 102])
        stop, target, rr, reason = derive_levels(df, 101, "Long")
        self.assertIsNone(stop)
        self.assertIsNone(target)
        self.assertIsNone(rr)

    def test_empty_frame_handled(self):
        stop, target, rr, reason = derive_levels(pd.DataFrame(), 100, "Long")
        self.assertIsNone(stop)
        self.assertIn("No usable", reason)

    def test_invalid_direction_rejected(self):
        stop, _, _, reason = derive_levels(staircase(), 115, "Sideways")
        self.assertIsNone(stop)

    def test_reason_is_always_populated(self):
        for entry in (98.0, 115.0, 155.0):
            for d in ("Long", "Short"):
                _, _, _, reason = derive_levels(staircase(), entry, d, min_rr=2.0)
                self.assertTrue(reason and len(reason) > 10)


class TestAtr(unittest.TestCase):
    def test_atr_positive_on_real_range(self):
        self.assertGreater(atr_value(staircase()), 0)

    def test_atr_none_on_short_frame(self):
        self.assertIsNone(atr_value(ohlc([100, 101])))


class TestCryptoUniverse(unittest.TestCase):
    def test_fallback_is_flagged_as_not_live(self):
        import universe
        original = universe.requests.get

        def boom(*a, **k):
            raise ConnectionError("offline")

        universe.requests.get = boom
        try:
            mapping, is_live, note = fetch_top_cryptos()
        finally:
            universe.requests.get = original

        self.assertFalse(is_live)
        self.assertIn("NOT a live top-100", note)
        self.assertEqual(mapping, FALLBACK_CRYPTO)

    def test_live_response_is_mapped_to_yahoo_tickers(self):
        import universe
        original = universe.requests.get

        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return [
                    {"symbol": "btc", "name": "Bitcoin"},
                    {"symbol": "eth", "name": "Ethereum"},
                    {"symbol": "usdt", "name": "Tether"},   # excluded stablecoin
                ]

        universe.requests.get = lambda *a, **k: FakeResp()
        try:
            mapping, is_live, note = fetch_top_cryptos()
        finally:
            universe.requests.get = original

        self.assertTrue(is_live)
        self.assertEqual(mapping["Bitcoin (BTC)"], "BTC-USD")
        self.assertEqual(mapping["Ethereum (ETH)"], "ETH-USD")
        self.assertNotIn("Tether (USDT)", mapping)

    def test_stablecoins_and_wrapped_are_excluded(self):
        for sym in ("USDT", "USDC", "WBTC", "STETH"):
            self.assertIn(sym, EXCLUDED_SYMBOLS)

    def test_fallback_tickers_all_use_usd_suffix(self):
        for label, ticker in FALLBACK_CRYPTO.items():
            self.assertTrue(ticker.endswith("-USD"), f"{label} -> {ticker}")

    def test_malformed_response_falls_back(self):
        import universe
        original = universe.requests.get

        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"unexpected": "shape"}

        universe.requests.get = lambda *a, **k: FakeResp()
        try:
            mapping, is_live, note = fetch_top_cryptos()
        finally:
            universe.requests.get = original

        self.assertFalse(is_live)
        self.assertEqual(mapping, FALLBACK_CRYPTO)


if __name__ == "__main__":
    unittest.main()
