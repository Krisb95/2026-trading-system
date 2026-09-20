import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scanner import (detect_regime, structure_direction, detect_sweep_and_reclaim,
                      fib_confluence, is_extended, opposing_liquidity_ahead,
                      resample_to_4h, analyze_candidate, nearest_level)


def ohlc_from_closes(closes, spread=0.5, start="2026-01-01", freq="1D"):
    idx = pd.date_range(start=start, periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame({
        "Open": closes,
        "High": [c + spread for c in closes],
        "Low": [c - spread for c in closes],
        "Close": closes,
        "Volume": [1000] * len(closes),
    }, index=idx)


class TestRegime(unittest.TestCase):
    def test_steady_uptrend_is_bullish(self):
        closes = list(np.linspace(100, 200, 120))
        df = ohlc_from_closes(closes)
        regime, reason = detect_regime(df)
        self.assertEqual(regime, "bullish")
        self.assertIn("above", reason)

    def test_steady_downtrend_is_bearish(self):
        closes = list(np.linspace(200, 100, 120))
        df = ohlc_from_closes(closes)
        regime, _ = detect_regime(df)
        self.assertEqual(regime, "bearish")

    def test_choppy_sideways_is_ranging(self):
        closes = [100 + (i % 5) for i in range(120)]
        df = ohlc_from_closes(closes)
        regime, _ = detect_regime(df)
        self.assertEqual(regime, "ranging")

    def test_insufficient_history_is_unknown_not_guessed(self):
        df = ohlc_from_closes(list(np.linspace(100, 110, 10)))
        regime, reason = detect_regime(df)
        self.assertEqual(regime, "unknown")
        self.assertIn("Not enough", reason)


def zigzag(n_legs=8, leg_len=7, start=100.0, rise=15.0, retrace=0.5, rising=True):
    """Zigzag with distinct turning points. Avoids repeating the peak/trough
    value across adjacent legs, which would fail the swing detector's
    uniqueness check."""
    closes = [start]
    level = start
    for _ in range(n_legs):
        if rising:
            high = level + rise
            closes += list(np.linspace(level, high, leg_len))[1:]
            level = high - rise * retrace
            closes += list(np.linspace(high, level, leg_len))[1:]
        else:
            low = level - rise
            closes += list(np.linspace(level, low, leg_len))[1:]
            level = low + rise * retrace
            closes += list(np.linspace(low, level, leg_len))[1:]
    return closes


class TestStructureDirection(unittest.TestCase):
    def test_rising_swings_favour_long(self):
        df = ohlc_from_closes(zigzag(rising=True))
        direction, reason = structure_direction(df)
        self.assertEqual(direction, "Long")
        self.assertIn("HH", reason)

    def test_falling_swings_favour_short(self):
        df = ohlc_from_closes(zigzag(rising=False, start=300.0))
        direction, reason = structure_direction(df)
        self.assertEqual(direction, "Short")

    def test_too_short_returns_none_not_a_guess(self):
        df = ohlc_from_closes([100, 101, 102])
        direction, reason = structure_direction(df)
        self.assertIsNone(direction)


class TestSweepReclaim(unittest.TestCase):
    def test_wick_only_pierce_without_reclaim_scores_false(self):
        # Build history with a clear swing low, then pierce it and STAY below.
        closes = [100, 95, 105, 90, 106, 92, 108, 94, 110] * 5
        df = ohlc_from_closes(closes)
        # Force the tail to break below everything and close there.
        tail = ohlc_from_closes([70] * 35, start="2027-01-01")
        df = pd.concat([df, tail])
        result, reason = detect_sweep_and_reclaim(df, "Long", lookback=30)
        self.assertIn(result, (False, None))
        if result is False:
            self.assertTrue("no reclaim" in reason.lower() or "no sweep" in reason.lower())

    def test_insufficient_data_returns_none(self):
        df = ohlc_from_closes([100, 101, 102])
        result, reason = detect_sweep_and_reclaim(df, "Long")
        self.assertIsNone(result)
        self.assertIn("Not enough", reason)


class TestFibConfluence(unittest.TestCase):
    def test_returns_none_when_no_anchors(self):
        df = ohlc_from_closes([100, 101])
        ok, reason, levels = fib_confluence(df, 100, "Long")
        self.assertIsNone(ok)

    def test_produces_levels_from_real_swings(self):
        closes = [100, 120, 105, 130, 110, 140, 115, 150, 120, 145] * 4
        df = ohlc_from_closes(closes)
        ok, reason, levels = fib_confluence(df, 130, "Long")
        if levels:
            self.assertTrue(any(k.startswith("retr_") for k in levels))


class TestExtension(unittest.TestCase):
    def test_far_from_sma_is_extended(self):
        closes = list(np.linspace(100, 105, 60)) + [180]
        df = ohlc_from_closes(closes)
        extended, reason = is_extended(df, 180)
        self.assertTrue(extended)
        self.assertIn("ATR", reason)

    def test_near_sma_is_not_extended(self):
        closes = list(np.linspace(100, 105, 60))
        df = ohlc_from_closes(closes)
        extended, reason = is_extended(df, float(df["Close"].iloc[-1]))
        self.assertFalse(extended)

    def test_insufficient_bars_returns_none(self):
        df = ohlc_from_closes([100, 101, 102])
        extended, _ = is_extended(df, 101)
        self.assertIsNone(extended)


class TestResample(unittest.TestCase):
    def test_1h_resamples_to_4h_with_correct_ohlc(self):
        idx = pd.date_range("2026-01-01 00:00", periods=8, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "Open": [1, 2, 3, 4, 5, 6, 7, 8],
            "High": [10, 11, 12, 13, 14, 15, 16, 17],
            "Low": [0.5, 0.4, 0.3, 0.2, 0.1, 0.6, 0.7, 0.8],
            "Close": [2, 3, 4, 5, 6, 7, 8, 9],
        }, index=idx)
        out = resample_to_4h(df)
        self.assertEqual(len(out), 2)
        self.assertEqual(out["Open"].iloc[0], 1)      # first open of the 4 bars
        self.assertEqual(out["Close"].iloc[0], 5)     # last close of the 4 bars
        self.assertEqual(out["High"].iloc[0], 13)     # max high
        self.assertAlmostEqual(out["Low"].iloc[0], 0.2)  # min low

    def test_empty_input_returns_empty(self):
        self.assertTrue(resample_to_4h(pd.DataFrame()).empty)


class TestAnalyzeCandidate(unittest.TestCase):
    def _frames(self, closes_1d, closes_1h):
        return {
            "1d": ohlc_from_closes(closes_1d),
            "1h": ohlc_from_closes(closes_1h, start="2026-06-01", freq="1h"),
            "4h": resample_to_4h(ohlc_from_closes(closes_1h, start="2026-06-01", freq="1h")),
        }

    def test_uptrend_suggests_long(self):
        frames = self._frames(list(np.linspace(100, 200, 150)),
                               list(np.linspace(180, 200, 400)))
        result = analyze_candidate("TEST", frames)
        self.assertEqual(result.regime_1d, "bullish")
        self.assertEqual(result.direction, "Long")

    def test_missing_data_is_reported_not_invented(self):
        frames = {"1d": pd.DataFrame(), "1h": pd.DataFrame(), "4h": pd.DataFrame()}
        result = analyze_candidate("TEST", frames)
        self.assertIsNone(result.current_price)
        self.assertTrue(len(result.data_problems) > 0)

    def test_unavailable_evidence_is_none_not_false(self):
        # Tiny dataset: most checks cannot be evaluated and must return None
        # (which scores zero) rather than a fabricated False/True.
        frames = self._frames([100, 101, 102], [100, 101, 102])
        result = analyze_candidate("TEST", frames)
        values = [item.value for item in result.score_evidence.values()]
        self.assertIn(None, values)

    def test_every_evidence_item_carries_a_reason(self):
        frames = self._frames(list(np.linspace(100, 200, 150)),
                               list(np.linspace(180, 200, 400)))
        result = analyze_candidate("TEST", frames)
        for key, item in result.score_evidence.items():
            self.assertTrue(item.reason and len(item.reason) > 5,
                             f"{key} has no usable reason string")
        for key, item in result.sequence_evidence.items():
            self.assertTrue(item.reason and len(item.reason) > 5,
                             f"{key} has no usable reason string")

    def test_score_dict_is_consumable_by_scoring_module(self):
        from scoring import score_setup
        frames = self._frames(list(np.linspace(100, 200, 150)),
                               list(np.linspace(180, 200, 400)))
        result = analyze_candidate("TEST", frames)
        scored = score_setup(result.score_dict())
        self.assertGreaterEqual(scored.normalized_score, 0.0)
        self.assertLessEqual(scored.normalized_score, 10.0)

    def test_rr_uses_derived_structural_levels(self):
        frames = self._frames(list(np.linspace(100, 200, 150)),
                               list(np.linspace(180, 200, 400)))
        result = analyze_candidate("TEST", frames)
        if result.stop is not None and result.target is not None:
            risk = abs(result.entry - result.stop)
            reward = abs(result.target - result.entry)
            self.assertAlmostEqual(result.reward_risk, reward / risk, places=4)


if __name__ == "__main__":
    unittest.main()


class TestRegimeFallbackWhenDailyMissing(unittest.TestCase):
    """A rate-limited daily request previously cascaded into: regime unknown
    -> no direction -> every check unevaluated -> 0.0/10. The 4H frame should
    carry the regime read instead, clearly labelled as weaker."""

    def _trending_4h(self):
        # A zigzag, not a straight line: linspace has no swing points at all,
        # so every structural check would be unevaluable for reasons that have
        # nothing to do with the regime fallback being tested here.
        return ohlc_from_closes(zigzag(n_legs=14, rising=True),
                                 start="2026-06-01", freq="4h")

    def test_missing_daily_falls_back_to_4h_regime(self):
        frames = {"1d": pd.DataFrame(), "4h": self._trending_4h(),
                   "1h": self._trending_4h()}
        result = analyze_candidate("TEST", frames)
        self.assertNotEqual(result.regime_1d, "unknown")
        self.assertIsNotNone(result.direction)

    def test_fallback_is_disclosed_not_hidden(self):
        frames = {"1d": pd.DataFrame(), "4h": self._trending_4h(),
                   "1h": self._trending_4h()}
        result = analyze_candidate("TEST", frames)
        joined = " ".join(result.data_problems).lower()
        self.assertIn("inferred from 4h", joined)

    def test_regime_reason_flags_the_weaker_read(self):
        frames = {"1d": pd.DataFrame(), "4h": self._trending_4h(),
                   "1h": self._trending_4h()}
        result = analyze_candidate("TEST", frames)
        item = result.score_evidence.get("regime_alignment_1d_4h")
        self.assertIn("Daily data unavailable", item.reason)

    def test_score_is_not_zero_when_only_daily_is_missing(self):
        from scoring import score_setup
        frames = {"1d": pd.DataFrame(), "4h": self._trending_4h(),
                   "1h": self._trending_4h()}
        result = analyze_candidate("TEST", frames)
        scored = score_setup(result.score_dict())
        self.assertGreater(scored.normalized_score, 0.0)

    def test_no_fallback_when_4h_also_too_short(self):
        frames = {"1d": pd.DataFrame(), "4h": ohlc_from_closes([1, 2, 3]),
                   "1h": pd.DataFrame()}
        result = analyze_candidate("TEST", frames)
        self.assertEqual(result.regime_1d, "unknown")


class TestUniverseScan(unittest.TestCase):
    """Ranked shortlist across many instruments."""

    def _good_frames(self, rising=True):
        df = ohlc_from_closes(zigzag(n_legs=14, rising=rising),
                               start="2026-06-01", freq="4h")
        return {"4h": df, "1d": pd.DataFrame(), "1h": pd.DataFrame()}, []

    def test_returns_one_result_per_instrument(self):
        from scanner import scan_universe
        insts = [("Bitcoin (BTC)", "BTC-USD", "bitcoin"),
                 ("Ethereum (ETH)", "ETH-USD", "ethereum")]
        out = scan_universe(insts, lambda k: self._good_frames())
        self.assertEqual(len(out), 2)

    def test_results_are_sorted_best_first(self):
        from scanner import scan_universe
        insts = [(f"C{i}", f"C{i}-USD", f"c{i}") for i in range(4)]

        def loader(key):
            # Alternate trending vs flat so scores genuinely differ.
            if key in ("c0", "c2"):
                return self._good_frames()
            return {"4h": ohlc_from_closes([100] * 80, start="2026-06-01", freq="4h"),
                     "1d": pd.DataFrame(), "1h": pd.DataFrame()}, []

        out = scan_universe(insts, loader)
        scores = [r.score for r in out if r.error is None]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_failed_instruments_are_reported_not_dropped(self):
        from scanner import scan_universe
        insts = [("Good", "G-USD", "good"), ("Bad", "B-USD", "bad")]

        def loader(key):
            if key == "bad":
                raise ConnectionError("429")
            return self._good_frames()

        out = scan_universe(insts, loader)
        self.assertEqual(len(out), 2)
        errored = [r for r in out if r.error]
        self.assertEqual(len(errored), 1)
        self.assertIn("429", errored[0].error)

    def test_empty_frames_reported_as_no_data(self):
        from scanner import scan_universe
        insts = [("Empty", "E-USD", "empty")]
        out = scan_universe(insts, lambda k: ({"4h": pd.DataFrame()}, []))
        self.assertEqual(out[0].error, "No data returned.")

    def test_failures_sort_last(self):
        from scanner import scan_universe
        insts = [("Bad", "B-USD", "bad"), ("Good", "G-USD", "good")]

        def loader(key):
            if key == "bad":
                raise ValueError("boom")
            return self._good_frames()

        out = scan_universe(insts, loader)
        self.assertIsNone(out[0].error)
        self.assertIsNotNone(out[-1].error)

    def test_progress_callback_is_invoked(self):
        from scanner import scan_universe
        seen = []
        insts = [("A", "A-USD", "a"), ("B", "B-USD", "b")]
        scan_universe(insts, lambda k: self._good_frames(),
                       progress_callback=lambda i, t, l: seen.append((i, t)))
        self.assertEqual(seen[-1], (2, 2))

    def test_direction_override_is_passed_through(self):
        from scanner import scan_universe
        insts = [("A", "A-USD", "a")]
        out = scan_universe(insts, lambda k: self._good_frames(),
                             direction_override="Short")
        self.assertEqual(out[0].direction, "Short")

    def test_empty_instrument_list_is_safe(self):
        from scanner import scan_universe
        self.assertEqual(scan_universe([], lambda k: self._good_frames()), [])
