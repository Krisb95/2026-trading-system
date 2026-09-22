import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from trade_review import (review, excursions, tighter_stop, HOLD, TIGHTEN, CLOSE,
                           CLOSE_THESIS)

T0 = pd.Timestamp("2026-06-01 00:00", tz="UTC")


def candles(rows, freq, start=T0):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)


UP_4H = candles([(100, 105, 98, 104), (104, 108, 101, 107), (107, 112, 104, 111)], "4h")
DOWN_4H = candles([(111, 112, 104, 105), (105, 108, 101, 102), (102, 105, 98, 99)], "4h")
FLAT_4H = candles([(100, 105, 98, 104), (104, 104, 99, 101), (101, 106, 97, 102)], "4h")
BULL_1H = candles([(100, 101, 99, 100.8)], "1h")
BEAR_1H = candles([(100.8, 101, 99, 99.2)], "1h")


def sideways(n=100, level=100.0, wiggle=0.3):
    rows = []
    for i in range(n):
        c = level + wiggle * np.sin(i / 3)
        rows.append((c, c + 0.1, c - 0.1, c))
    return candles(rows, "5min", start=T0 + pd.Timedelta(minutes=5))


def run(**kw):
    args = dict(direction="Long", entry=100.0, stop=97.0, target=109.0,
                entry_time=T0, price=100.1, now=T0 + pd.Timedelta(hours=2),
                df_4h=UP_4H, df_1h=BULL_1H, candles_since_entry=sideways(20),
                structure_candles=sideways(20), atr_value=0.2)
    args.update(kw)
    return review(**args)


class TestThesis(unittest.TestCase):
    def test_reversed_4h_trend_says_close(self):
        r = run(df_4h=DOWN_4H)
        self.assertEqual(r.verdict, CLOSE_THESIS)
        self.assertEqual(r.thesis, "broken")

    def test_intact_trend_early_in_trade_holds(self):
        r = run()
        self.assertEqual(r.verdict, HOLD)
        self.assertEqual(r.thesis, "intact")

    def test_paused_trend_is_weakening_not_broken(self):
        r = run(df_4h=FLAT_4H)
        self.assertEqual(r.thesis, "weakening")
        self.assertNotEqual(r.verdict, CLOSE_THESIS)


class TestStaleTrades(unittest.TestCase):
    def test_long_sideways_trade_is_stale(self):
        r = run(now=T0 + pd.Timedelta(hours=20), candles_since_entry=sideways(200))
        self.assertTrue(r.stale)

    def test_short_time_is_not_stale(self):
        self.assertFalse(run(now=T0 + pd.Timedelta(hours=3)).stale)

    def test_stale_with_1h_against_suggests_closing(self):
        r = run(now=T0 + pd.Timedelta(hours=20), df_1h=BEAR_1H,
                candles_since_entry=sideways(200))
        self.assertEqual(r.verdict, CLOSE)
        self.assertTrue(any("Caution" in a for a in r.actions))

    def test_stale_trade_that_ran_1r_is_not_stale(self):
        moved = sideways(200)
        moved.iloc[50, moved.columns.get_loc("High")] = 104.0    # touched +1.33R
        r = run(now=T0 + pd.Timedelta(hours=20), candles_since_entry=moved)
        self.assertFalse(r.stale)


class TestTightening(unittest.TestCase):
    def test_gave_back_gains_moves_stop_to_at_least_breakeven(self):
        moved = sideways(60)
        moved.iloc[20, moved.columns.get_loc("High")] = 104.5    # +1.5R then back
        r = run(price=100.5, candles_since_entry=moved)
        self.assertEqual(r.verdict, TIGHTEN)
        self.assertGreaterEqual(r.suggested_stop, 100.0)

    def test_suggested_stop_never_widens_for_long(self):
        for price in (99.0, 100.1, 102.0, 105.0):
            for hours in (2, 20):
                r = run(price=price, now=T0 + pd.Timedelta(hours=hours),
                        candles_since_entry=sideways(200))
                if r.suggested_stop is not None:
                    self.assertGreater(r.suggested_stop, 97.0)
                    self.assertLess(r.suggested_stop, price)

    def test_suggested_stop_never_widens_for_short(self):
        for price in (98.0, 99.9, 101.0):
            r = run(direction="Short", entry=100.0, stop=103.0, target=91.0, price=price,
                    df_4h=DOWN_4H, df_1h=BEAR_1H, now=T0 + pd.Timedelta(hours=20),
                    candles_since_entry=sideways(200), structure_candles=sideways(200))
            if r.suggested_stop is not None:
                self.assertLess(r.suggested_stop, 103.0)
                self.assertGreater(r.suggested_stop, price)

    def test_tighter_stop_uses_a_swing_between_stop_and_price(self):
        rows = [(100, 100.2, 99.8, 100)] * 5 + [(99.5, 99.6, 98.5, 99)] + \
               [(100, 100.3, 99.9, 100.2)] * 6
        c = candles(rows, "5min")
        s = tighter_stop("Long", 97.0, 101.0, c, buffer=0.1)
        self.assertIsNotNone(s)
        self.assertTrue(97.0 < s < 101.0)

    def test_no_swing_between_means_no_tighter_stop(self):
        c = candles([(100, 100.1, 99.9, 100)] * 12, "5min")
        self.assertIsNone(tighter_stop("Long", 97.0, 100.0, c, buffer=0.1))


class TestCostComparison(unittest.TestCase):
    def test_shows_close_now_stop_and_target_in_r(self):
        r = run(price=99.0)
        self.assertAlmostEqual(r.close_now_r, -1 / 3)
        self.assertAlmostEqual(r.stop_r, -1.0)
        self.assertAlmostEqual(r.target_r, 3.0)

    def test_excursions(self):
        c = candles([(100, 103, 98.5, 101)], "5min")
        mfe, mae = excursions("Long", 100, 97, c)
        self.assertAlmostEqual(mfe, 1.0)
        self.assertAlmostEqual(mae, -0.5)

    def test_every_review_explains_itself(self):
        for kw in ({}, {"df_4h": DOWN_4H}, {"now": T0 + pd.Timedelta(hours=20)}):
            r = run(**kw)
            self.assertTrue(r.reasons)
            self.assertTrue(r.actions)


if __name__ == "__main__":
    unittest.main()
