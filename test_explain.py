import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from explain import (short_plan, full_plan, AT_ENTRY, WAIT_RETRACE, WAIT_1H,
                      NO_SUPPORT, NO_TREND)


class TestShortPlan(unittest.TestCase):
    def test_long_waiting_for_retrace_is_a_limit_buy(self):
        s = short_plan("Long", WAIT_RETRACE, 81200, 80650, 82850, 3.0, 81850)
        self.assertIn("limit buy 81,200", s)
        self.assertIn("stop 80,650", s)
        self.assertIn("TP 82,850", s)
        self.assertIn("3R", s)
        self.assertIn("-0.79%", s)

    def test_short_is_a_limit_sell(self):
        s = short_plan("Short", WAIT_RETRACE, 100, 103, 91, 3.0, 98)
        self.assertIn("limit sell", s)

    def test_at_entry_says_now(self):
        s = short_plan("Long", AT_ENTRY, 100, 97, 109, 3.0, 100.01)
        self.assertIn("buy now", s)

    def test_waiting_for_1h_says_wait(self):
        s = short_plan("Long", WAIT_1H, 100, 97, 109, 3.0, 102)
        self.assertIn("wait for a bullish 1H close", s)

    def test_no_trend(self):
        self.assertIn("No trade", short_plan(None, NO_TREND, None, None, None, None, 100))

    def test_no_support(self):
        s = short_plan("Long", NO_SUPPORT, None, None, None, None, 100)
        self.assertIn("no entry", s)

    def test_tiny_prices_are_readable(self):
        s = short_plan("Long", WAIT_RETRACE, 0.000003991, 0.00000393, 0.0000042, 3.0,
                       0.0000041)
        self.assertNotIn("e-06", s)


class TestFullPlan(unittest.TestCase):
    def _plan(self, **kw):
        base = dict(ticker="BTC-USD", direction="Long", stage=WAIT_RETRACE, entry=81200.0,
                    stop=80650.0, target=82850.0, rr=3.0, price=81850.0,
                    reasons=["4H made two higher highs and lows", "1H closed bullish"],
                    stop_method="3x 5m ATR below the support", target_method="3x the risk")
        base.update(kw)
        return "\n".join(full_plan(**base))

    def test_contains_entry_stop_target(self):
        p = self._plan()
        self.assertIn("81,200", p)
        self.assertIn("80,650", p)
        self.assertIn("82,850", p)

    def test_explains_why(self):
        self.assertIn("higher highs", self._plan())

    def test_states_risk_and_reward_per_coin(self):
        p = self._plan()
        self.assertIn("Risk: 550", p)
        self.assertIn("Reward: 1,650", p)

    def test_includes_cancel_rule(self):
        self.assertIn("Cancel the order", self._plan())

    def test_includes_reentry_when_given(self):
        p = self._plan(reentry_steps=["Wait for a **bullish** 4H candle", "Re-enter **50%**"])
        self.assertIn("If you're stopped out", p)
        self.assertNotIn("**bullish**", p)

    def test_short_uses_sell_and_resistance(self):
        p = self._plan(direction="Short", entry=100.0, stop=103.0, target=91.0, price=98.0)
        self.assertIn("SELL", p)
        self.assertIn("resistance", p)

    def test_no_trend_plan_says_no_trade(self):
        p = "\n".join(full_plan("X", None, NO_TREND, None, None, None, None, 100))
        self.assertIn("no trade", p)

    def test_incomplete_plan_does_not_invent_levels(self):
        p = self._plan(entry=None, stop=None, target=None, stage=NO_SUPPORT)
        self.assertIn("No complete plan", p)

    def test_never_promises_an_outcome(self):
        p = self._plan().lower()
        for word in ("guarantee", "will profit", "will win", "sure thing"):
            self.assertNotIn(word, p)


if __name__ == "__main__":
    unittest.main()


class TestTargetShortOfLivePrice(unittest.TestCase):
    """A long's target can sit below today's price when the entry is a deep
    retrace. That previously made a long read like a short — so it's explained."""

    def test_long_target_below_live_is_explained(self):
        p = "\n".join(full_plan("X", "Long", WAIT_RETRACE, 99.7, 99.26, 101.03, 3.0, 103.0))
        self.assertIn("take profit is below today's price", p)
        self.assertIn("still a long", p)

    def test_short_target_above_live_is_explained(self):
        p = "\n".join(full_plan("X", "Short", WAIT_RETRACE, 100.3, 100.7, 99.1, 3.0, 97.0))
        self.assertIn("take profit is above today's price", p)

    def test_no_note_when_target_beyond_live(self):
        p = "\n".join(full_plan("X", "Long", WAIT_RETRACE, 100.0, 97.0, 109.0, 3.0, 102.0))
        self.assertNotIn("Note:", p)
