import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from stop_manager import (suggest, structural_stop, trailing_stop, is_tighter,
                           trailing_plan, TRAIL, BREAKEVEN, STRUCTURE, HOLD)


def bars(closes, start="2026-06-01", freq="5min", wick=0.4):
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    o = [closes[0]] + list(closes[:-1])
    nudge = [1e-6 * i for i in range(len(closes))]
    return pd.DataFrame({
        "Open": o,
        "High": [max(a, b) + wick + d for a, b, d in zip(o, closes, nudge)],
        "Low": [min(a, b) - wick - d for a, b, d in zip(o, closes, nudge)],
        "Close": closes}, index=idx)


def rising_with_pullbacks(start=100.0, legs=5, up=6.0, down=2.0):
    closes, p = [start], start
    for _ in range(legs):
        closes += list(np.linspace(p, p + up, 7))[1:]
        p += up
        closes += list(np.linspace(p, p - down, 5))[1:]
        p -= down
    return bars(closes)


class TestNeverWiden(unittest.TestCase):
    """The one rule that cannot bend."""

    def test_is_tighter_for_a_long(self):
        self.assertTrue(is_tighter("Long", 98.0, 96.0))
        self.assertFalse(is_tighter("Long", 94.0, 96.0))

    def test_is_tighter_for_a_short(self):
        self.assertTrue(is_tighter("Short", 102.0, 104.0))
        self.assertFalse(is_tighter("Short", 106.0, 104.0))

    def test_suggestion_never_widens_a_long(self):
        df = rising_with_pullbacks()
        price = float(df["Close"].iloc[-1])
        for stop in (price - 30, price - 10, price - 3):
            s = suggest("Long", entry=price - 8, price=price, candles=df, atr=0.8,
                        current_stop=stop)
            if s.price is not None:
                self.assertGreaterEqual(s.price, stop - 1e-9,
                                        "a long's stop must never move down")

    def test_suggestion_never_widens_a_short(self):
        closes = list(np.linspace(140, 100, 60))
        df = bars(closes)
        price = float(df["Close"].iloc[-1])
        for stop in (price + 30, price + 10, price + 3):
            s = suggest("Short", entry=price + 8, price=price, candles=df, atr=0.8,
                        current_stop=stop)
            if s.price is not None:
                self.assertLessEqual(s.price, stop + 1e-9,
                                     "a short's stop must never move up")


class TestStructuralStop(unittest.TestCase):
    def test_sits_below_the_last_swing_low_for_a_long(self):
        df = rising_with_pullbacks()
        stop = structural_stop("Long", entry=float(df["Close"].iloc[-1]), candles=df,
                               buffer=0.4)
        self.assertIsNotNone(stop)
        self.assertLess(stop, float(df["Close"].iloc[-1]))

    def test_none_when_there_is_no_swing(self):
        self.assertIsNone(structural_stop("Long", 100, bars([100] * 10), 0.4))


class TestTrailing(unittest.TestCase):
    def test_trails_to_a_newer_swing(self):
        df = rising_with_pullbacks()
        price = float(df["Close"].iloc[-1])
        trail = trailing_stop("Long", price, current_stop=price - 25, candles=df, buffer=0.4)
        self.assertIsNotNone(trail)
        self.assertGreater(trail, price - 25)
        self.assertLess(trail, price)

    def test_nothing_to_trail_to_yet(self):
        df = bars(list(np.linspace(100, 101, 20)))
        self.assertIsNone(trailing_stop("Long", 101, current_stop=90, candles=df, buffer=0.4))

    def test_never_returns_a_stop_above_price(self):
        df = rising_with_pullbacks()
        price = float(df["Close"].iloc[-1])
        trail = trailing_stop("Long", price, None, df, buffer=0.4)
        if trail is not None:
            self.assertLess(trail, price)


class TestSuggestions(unittest.TestCase):
    def test_first_stop_comes_from_structure(self):
        df = rising_with_pullbacks()
        entry = float(df["Close"].iloc[-1])
        s = suggest("Long", entry=entry, price=entry, candles=df, atr=0.8)
        self.assertEqual(s.basis, STRUCTURE)
        self.assertLess(s.price, entry)
        self.assertIn("wrong", s.reason)

    def test_breakeven_offered_once_up_one_r(self):
        df = rising_with_pullbacks()
        price = float(df["Close"].iloc[-1])
        entry, stop = price - 6, price - 9      # 3 of risk, now 6 in profit = 2R
        s = suggest("Long", entry=entry, price=price, candles=df, atr=0.3,
                    current_stop=stop)
        self.assertIn(s.basis, (BREAKEVEN, TRAIL))
        self.assertGreaterEqual(s.price, entry - 1e-9)

    def test_breakeven_not_offered_too_early(self):
        df = rising_with_pullbacks()
        price = float(df["Close"].iloc[-1])
        entry, stop = price - 0.2, price - 5    # barely in profit
        s = suggest("Long", entry=entry, price=price, candles=df, atr=0.8,
                    current_stop=stop, allow_breakeven=True)
        self.assertNotEqual(s.basis, BREAKEVEN)

    def test_reports_what_a_stop_locks_in(self):
        df = rising_with_pullbacks()
        price = float(df["Close"].iloc[-1])
        s = suggest("Long", entry=price - 6, price=price, candles=df, atr=0.3,
                    current_stop=price - 9)
        if s.price is not None:
            self.assertAlmostEqual(s.locked_in, s.price - (price - 6))

    def test_current_r_is_reported(self):
        df = rising_with_pullbacks()
        price = float(df["Close"].iloc[-1])
        s = suggest("Long", entry=price - 4, price=price, candles=df, atr=0.5,
                    current_stop=price - 8)
        self.assertAlmostEqual(s.current_r, 1.0, delta=0.01)

    def test_hold_when_nothing_can_be_tightened(self):
        df = bars(list(np.linspace(100, 100.5, 30)))
        s = suggest("Long", entry=100.4, price=100.5, candles=df, atr=0.5,
                    current_stop=99.0)
        self.assertEqual(s.basis, HOLD)
        self.assertFalse(s.is_tighter)

    def test_every_suggestion_explains_itself(self):
        df = rising_with_pullbacks()
        price = float(df["Close"].iloc[-1])
        for stop in (None, price - 20, price - 3):
            s = suggest("Long", entry=price - 5, price=price, candles=df, atr=0.5,
                        current_stop=stop)
            self.assertGreater(len(s.reason), 40)

    def test_bad_inputs_rejected(self):
        df = rising_with_pullbacks()
        with self.assertRaises(ValueError):
            suggest("Sideways", 100, 100, df, 1.0)
        with self.assertRaises(ValueError):
            suggest("Long", 0, 100, df, 1.0)


class TestFirstStopVersusTrailing(unittest.TestCase):
    """Trailing means moving an EXISTING stop. With none set, the first stop
    comes from structure — the level that would say the trade was wrong."""

    def test_no_existing_stop_gives_a_structural_one(self):
        df = rising_with_pullbacks()
        entry = float(df["Close"].iloc[-1])
        self.assertEqual(suggest("Long", entry, entry, df, 0.8).basis, STRUCTURE)

    def test_existing_stop_can_be_trailed(self):
        df = rising_with_pullbacks()
        price = float(df["Close"].iloc[-1])
        s = suggest("Long", entry=price - 8, price=price, candles=df, atr=0.5,
                    current_stop=price - 25)
        self.assertIn(s.basis, (TRAIL, BREAKEVEN))
        self.assertTrue(s.is_tighter)


class TestTrailingPlan(unittest.TestCase):
    """The two numbers an exchange needs: activation price and trail distance."""

    def test_activation_is_one_r_above_entry_for_a_long(self):
        p = trailing_plan("Long", entry=100, stop=97, atr=1.0, activate_at_r=1.0)
        self.assertAlmostEqual(p.activate_at, 103.0)
        self.assertAlmostEqual(p.activate_at_r, 1.0)

    def test_activation_is_below_entry_for_a_short(self):
        p = trailing_plan("Short", entry=100, stop=103, atr=1.0, activate_at_r=1.0)
        self.assertAlmostEqual(p.activate_at, 97.0)

    def test_distance_comes_from_atr(self):
        p = trailing_plan("Long", 100, 97, atr=1.5, trail_atr_mult=2.0)
        self.assertAlmostEqual(p.distance, 3.0)
        self.assertAlmostEqual(p.distance_pct, 3.0 / p.activate_at * 100)

    def test_reports_what_it_locks_in(self):
        p = trailing_plan("Long", 100, 97, atr=1.0, activate_at_r=2.0, trail_atr_mult=2.0)
        # activates at 106, trails 2 behind -> stop 104 = +1.33R
        self.assertAlmostEqual(p.locked_in_at_activation, (104 - 100) / 3, places=6)
        self.assertIn("locking in", p.description)

    def test_warns_when_the_trail_is_wider_than_the_gain(self):
        p = trailing_plan("Long", 100, 99, atr=3.0, activate_at_r=1.0, trail_atr_mult=2.0)
        self.assertLess(p.locked_in_at_activation, 0)
        self.assertIn("wouldn't protect anything yet", p.description)

    def test_scales_with_the_trade_s_own_risk(self):
        tight = trailing_plan("Long", 100, 99, atr=0.5, activate_at_r=1.0)
        wide = trailing_plan("Long", 100, 90, atr=0.5, activate_at_r=1.0)
        self.assertAlmostEqual(tight.activate_at, 101.0)
        self.assertAlmostEqual(wide.activate_at, 110.0)

    def test_summary_has_both_numbers(self):
        s = trailing_plan("Long", 100, 97, atr=1.0).summary
        self.assertIn("activate at", s)
        self.assertIn("trail by", s)

    def test_none_without_atr_or_risk(self):
        self.assertIsNone(trailing_plan("Long", 100, 97, atr=None))
        self.assertIsNone(trailing_plan("Long", 100, 100, atr=1.0))
