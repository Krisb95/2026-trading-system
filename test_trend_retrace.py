import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import trend_retrace as tr
from trend_retrace import (four_hour_trend, one_hour_confirms, five_minute_support,
                            analyze, run_backtest, backtest_stats, StrategyParams,
                            NO_TREND, WAIT_1H, WAIT_RETRACE, AT_ENTRY, NO_SUPPORT,
                            KIND_INITIAL, KIND_RE1, KIND_RE2, STOP_BELOW_4H_CANDLE)


def candles(rows, freq="4h", start="2026-01-01"):
    """rows: list of (open, high, low, close)."""
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)


def uptrend_4h():
    return candles([(100, 105, 98, 104), (104, 108, 101, 107), (107, 112, 104, 111)])


def downtrend_4h():
    return candles([(111, 112, 104, 105), (105, 108, 101, 102), (102, 105, 98, 99)])


def bullish_1h():
    return candles([(100, 101, 99, 100.5), (100.5, 102, 100, 101.8)], freq="1h")


def bearish_1h():
    return candles([(101, 101.5, 99, 100), (100, 100.5, 98, 98.5)], freq="1h")


def five_min_with_support(support=100.0, now=103.0, n=80):
    """Clear dip to `support` a while ago, then a rise to `now`."""
    closes = list(np.linspace(104, support + 0.2, 20)) + [support] + \
             list(np.linspace(support + 0.2, now, n - 21))
    rows = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i else c
        rows.append((o, max(o, c) + 0.05, min(o, c) - 0.05, c))
    # The swing low must be a UNIQUE low — the neighbouring candle opens at
    # `support`, so its low would otherwise tie with this one and the swing
    # detector would (correctly) refuse to call either a swing.
    rows[20] = (support + 0.1, support + 0.2, support - 0.3, support)
    return candles(rows, freq="5min")


class TestRule1FourHourTrend(unittest.TestCase):
    def test_two_higher_highs_and_lows_is_long(self):
        d, _ = four_hour_trend(uptrend_4h(), 2)
        self.assertEqual(d, "Long")

    def test_two_lower_highs_and_lows_is_short(self):
        d, _ = four_hour_trend(downtrend_4h(), 2)
        self.assertEqual(d, "Short")

    def test_higher_high_but_lower_low_is_not_a_trend(self):
        df = candles([(100, 105, 98, 104), (104, 108, 101, 107), (107, 112, 99, 111)])
        d, _ = four_hour_trend(df, 2)
        self.assertIsNone(d)

    def test_only_one_of_two_candles_qualifying_is_not_enough(self):
        df = candles([(100, 105, 98, 104), (104, 104, 97, 100), (100, 110, 99, 109)])
        d, _ = four_hour_trend(df, 2)
        self.assertIsNone(d)

    def test_needs_three_candles_for_two_comparisons(self):
        d, reason = four_hour_trend(uptrend_4h().iloc[:2], 2)
        self.assertIsNone(d)
        self.assertIn("at least 3", reason)

    def test_uses_only_the_most_recent_candles(self):
        df = pd.concat([downtrend_4h(), candles(
            [(99, 100, 97, 99.5), (99.5, 103, 98, 102), (102, 106, 99, 105)],
            start="2026-01-01 12:00")])
        d, _ = four_hour_trend(df, 2)
        self.assertEqual(d, "Long")


class TestRule2OneHourConfirm(unittest.TestCase):
    def test_bullish_candle_confirms_long(self):
        self.assertTrue(one_hour_confirms(bullish_1h(), "Long").passed)

    def test_bearish_candle_does_not_confirm_long(self):
        self.assertFalse(one_hour_confirms(bearish_1h(), "Long").passed)

    def test_bearish_candle_confirms_short(self):
        self.assertTrue(one_hour_confirms(bearish_1h(), "Short").passed)

    def test_doji_confirms_nothing(self):
        doji = candles([(100, 101, 99, 100)], freq="1h")
        self.assertFalse(one_hour_confirms(doji, "Long").passed)
        self.assertFalse(one_hour_confirms(doji, "Short").passed)

    def test_empty_fails(self):
        self.assertFalse(one_hour_confirms(pd.DataFrame(), "Long").passed)


class TestRule3FiveMinuteSupport(unittest.TestCase):
    def test_finds_previous_support_below_price(self):
        lvl, _ = five_minute_support(five_min_with_support(100.0, 103.0), "Long",
                                      103.0, StrategyParams())
        self.assertIsNotNone(lvl)
        self.assertLess(lvl, 103.0)
        self.assertAlmostEqual(lvl, 99.7, delta=0.5)

    def test_no_support_below_price_returns_none(self):
        df = five_min_with_support(100.0, 103.0)
        lvl, reason = five_minute_support(df, "Long", 90.0, StrategyParams())
        self.assertIsNone(lvl)

    def test_short_looks_for_resistance_above(self):
        df = five_min_with_support(100.0, 103.0)
        lvl, _ = five_minute_support(df, "Short", 101.0, StrategyParams())
        if lvl is not None:
            self.assertGreater(lvl, 101.0)


class TestLiveAnalysis(unittest.TestCase):
    def test_complete_setup_scores_ten_and_a_plus(self):
        p = analyze("X", uptrend_4h(), bullish_1h(), five_min_with_support(), live_price=103.0)
        self.assertEqual(p.direction, "Long")
        self.assertEqual(p.score, 10)
        self.assertEqual(p.grade, "A+")
        self.assertIn(p.stage, (WAIT_RETRACE, AT_ENTRY))

    def test_no_trend_scores_zero(self):
        flat = candles([(100, 101, 99, 100)] * 3)
        p = analyze("X", flat, bullish_1h(), five_min_with_support(), live_price=103.0)
        self.assertEqual(p.stage, NO_TREND)
        self.assertEqual(p.score, 0)

    def test_trend_without_1h_confirmation_waits(self):
        p = analyze("X", uptrend_4h(), bearish_1h(), five_min_with_support(), live_price=103.0)
        self.assertEqual(p.stage, WAIT_1H)
        self.assertEqual(p.score, 7)          # 4H + 5m, missing 1H
        self.assertNotEqual(p.grade, "A+")

    def test_no_support_stage(self):
        p = analyze("X", uptrend_4h(), bullish_1h(), five_min_with_support(), live_price=80.0)
        self.assertEqual(p.stage, NO_SUPPORT)
        self.assertEqual(p.score, 7)          # 4H + 1H, missing 5m

    def test_entry_is_the_support_not_the_live_price(self):
        p = analyze("X", uptrend_4h(), bullish_1h(), five_min_with_support(), live_price=103.0)
        self.assertLess(p.entry, 103.0)

    def test_long_levels_are_ordered(self):
        p = analyze("X", uptrend_4h(), bullish_1h(), five_min_with_support(), live_price=103.0)
        self.assertTrue(p.stop < p.entry < p.target)

    def test_target_is_the_chosen_r_multiple(self):
        p = analyze("X", uptrend_4h(), bullish_1h(), five_min_with_support(),
                    live_price=103.0, params=StrategyParams(target_r=3.0))
        risk = p.entry - p.stop
        self.assertAlmostEqual((p.target - p.entry) / risk, 3.0, places=6)

    def test_at_entry_when_price_sits_on_support(self):
        df = five_min_with_support(100.0, 103.0)
        lvl, _ = five_minute_support(df, "Long", 103.0, StrategyParams())
        p = analyze("X", uptrend_4h(), bullish_1h(), df, live_price=lvl + 0.001)
        self.assertEqual(p.stage, AT_ENTRY)

    def test_4h_candle_stop_mode(self):
        p = analyze("X", uptrend_4h(), bullish_1h(), five_min_with_support(), live_price=103.0,
                    params=StrategyParams(stop_mode=STOP_BELOW_4H_CANDLE))
        if p.stop is not None:
            self.assertAlmostEqual(p.stop, 104.0)   # low of the last 4H candle

    def test_every_rule_has_a_reason(self):
        p = analyze("X", uptrend_4h(), bullish_1h(), five_min_with_support(), live_price=103.0)
        for key in ("4h", "1h", "5m"):
            self.assertTrue(p.rules[key].reason)

    def test_reentry_checklist_has_three_steps(self):
        steps = tr.reentry_plan_text("Long")
        self.assertEqual(len(steps), 3)
        self.assertIn("50%", steps[1])
        self.assertIn("50%", steps[2])


# ---------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------

def market(seed=1, days=25, drift=0.0, vol=0.0022):
    rng = np.random.default_rng(seed)
    n = days * 288
    c = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    idx = pd.date_range("2026-05-01", periods=n, freq="5min", tz="UTC")
    m5 = pd.DataFrame({"Open": np.r_[c[0], c[:-1]], "Close": c}, index=idx)
    m5["High"] = np.maximum(m5["Open"], m5["Close"]) * (1 + np.abs(rng.normal(0, .0008, n)))
    m5["Low"] = np.minimum(m5["Open"], m5["Close"]) * (1 - np.abs(rng.normal(0, .0008, n)))
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    return m5, m5.resample("1h").agg(agg).dropna(), m5.resample("4h").agg(agg).dropna()


class TestBacktestNoLookAhead(unittest.TestCase):
    def test_orders_before_cutoff_are_unaffected_by_later_prices(self):
        m5, h1, h4 = market()
        base = run_backtest(m5, h1, h4, "X")
        cutoff = m5.index[4000]
        agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        t5 = m5.copy()
        t5.loc[t5.index > cutoff, ["Open", "High", "Low", "Close"]] *= 2.5
        t1 = t5.resample("1h").agg(agg).dropna()
        t4 = t5.resample("4h").agg(agg).dropna()
        altered = run_backtest(t5, t1, t4, "X")

        def plans(trades):
            # Only orders placed AND resolved before the cutoff are fully
            # determined by past data.
            return [(t.kind, t.signal_time, round(t.entry, 8), round(t.stop, 8),
                     round(t.target, 8), t.status)
                    for t in trades if t.exit_time is not None and t.exit_time < cutoff]

        self.assertTrue(plans(base), "need resolved trades before the cutoff")
        self.assertEqual(plans(base), plans(altered))


class TestBacktestBehaviour(unittest.TestCase):
    def setUp(self):
        self.m5, self.h1, self.h4 = market()
        self.trades = run_backtest(self.m5, self.h1, self.h4, "X")

    def test_produces_trades(self):
        self.assertTrue(self.trades)

    def test_levels_ordered_for_every_trade(self):
        for t in self.trades:
            if t.kind == KIND_RE2:
                continue          # second half enters at market, same stop/target
            if t.direction == "Long":
                self.assertTrue(t.stop < t.entry < t.target)
            else:
                self.assertTrue(t.stop > t.entry > t.target)

    def test_reentries_are_half_size(self):
        for t in self.trades:
            expected = 1.0 if t.kind == KIND_INITIAL else 0.5
            self.assertEqual(t.weight, expected)

    def test_reentry_only_follows_a_loss(self):
        ordered = sorted([t for t in self.trades if t.exit_time is not None
                          or t.kind == KIND_RE1], key=lambda t: t.signal_time)
        for i, t in enumerate(ordered):
            if t.kind == KIND_RE1:
                earlier = [x for x in ordered[:i] if x.exit_time is not None]
                self.assertTrue(earlier, "a re-entry needs a prior closed trade")
                self.assertEqual(earlier[-1].status, "LOSS")

    def test_second_half_only_after_first_half(self):
        firsts = {t.fill_time for t in self.trades if t.kind == KIND_RE1}
        for t in self.trades:
            if t.kind == KIND_RE2:
                self.assertTrue(any(f is not None and f <= t.fill_time for f in firsts))

    def test_second_halves_occur_with_default_stop(self):
        """Regression: a too-tight stop made trades finish before any 1H
        candle closed, so the second 50% could never be added."""
        m5, h1, h4 = market(seed=1, days=30)
        trades = run_backtest(m5, h1, h4, "X")
        self.assertTrue(any(t.kind == KIND_RE2 for t in trades))

    def test_loss_is_minus_one_r_per_leg(self):
        for t in self.trades:
            if t.status == "LOSS":
                self.assertAlmostEqual(t.r_result, -1.0)
                self.assertAlmostEqual(t.weighted_r, -1.0 * t.weight)

    def test_disabling_reentry_removes_reentries(self):
        trades = run_backtest(self.m5, self.h1, self.h4, "X", enable_reentry=False)
        self.assertFalse(any(t.kind != KIND_INITIAL for t in trades))

    def test_empty_inputs(self):
        self.assertEqual(run_backtest(pd.DataFrame(), self.h1, self.h4, "X"), [])

    def test_stats_groups(self):
        groups = {s.group for s in backtest_stats(self.trades)}
        self.assertEqual(groups, {KIND_INITIAL, KIND_RE1, KIND_RE2, "All"})

    def test_stats_total_is_weighted(self):
        stats = {s.group: s for s in backtest_stats(self.trades)}
        manual = sum(t.weighted_r for t in self.trades
                     if t.status in ("WIN", "LOSS") and t.weighted_r is not None)
        self.assertAlmostEqual(stats["All"].total_r, manual)


class TestNoFakeEdgeOnRandomData(unittest.TestCase):
    """A random walk has no edge. Large profits here would mean the backtest
    is leaking future information."""

    def test_random_walks_do_not_show_a_solid_edge(self):
        trades = []
        for seed in range(6):
            m5, h1, h4 = market(seed=seed + 20, days=25)
            trades += run_backtest(m5, h1, h4, f"RW{seed}")
        allrow = {s.group: s for s in backtest_stats(trades)}["All"]
        self.assertGreater(allrow.entries, 60)
        self.assertLess(allrow.avg_r_per_entry, 0.2,
                        f"edge found in pure noise ({allrow.avg_r_per_entry:+.2f}R)")


if __name__ == "__main__":
    unittest.main()


class TestLiveScanHelpers(unittest.TestCase):
    def test_forming_candle_is_dropped(self):
        idx = pd.date_range("2026-06-01 10:00", periods=3, freq="1h", tz="UTC")
        df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0}, index=idx)
        now = pd.Timestamp("2026-06-01 12:20", tz="UTC")   # 12:00 candle still forming
        out = tr.drop_forming(df, "1h", now)
        self.assertEqual(out.index[-1], pd.Timestamp("2026-06-01 11:00", tz="UTC"))

    def test_candle_kept_once_closed(self):
        idx = pd.date_range("2026-06-01 10:00", periods=3, freq="1h", tz="UTC")
        df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0}, index=idx)
        out = tr.drop_forming(df, "1h", pd.Timestamp("2026-06-01 13:00", tz="UTC"))
        self.assertEqual(len(out), 3)

    def test_green_forming_candle_cannot_confirm(self):
        """A 1H candle green mid-hour may close red — it must not count."""
        closed = candles([(100, 101, 99, 99.5)], freq="1h", start="2026-06-01 10:00")
        forming = candles([(99.5, 102, 99.4, 101.9)], freq="1h", start="2026-06-01 11:00")
        both = pd.concat([closed, forming])
        now = pd.Timestamp("2026-06-01 11:30", tz="UTC")
        self.assertFalse(one_hour_confirms(tr.drop_forming(both, "1h", now), "Long").passed)

    def _loader(self, _key):
        return {"4h": uptrend_4h(), "1h": bullish_1h(), "5m": five_min_with_support()}

    def test_universe_scan_ranks_complete_setups_first(self):
        far_future = pd.Timestamp("2030-01-01", tz="UTC")

        def loader(key):
            if key == "full":
                return self._loader(key)
            return {"4h": candles([(100, 101, 99, 100)] * 3), "1h": bullish_1h(),
                    "5m": five_min_with_support()}

        out = tr.scan_universe_tr([("Flat", "FLAT-USD", "flat"), ("Full", "FULL-USD", "full")],
                                   loader, spot_loader=lambda k: 103.0, now=far_future)
        self.assertEqual(out[0].label, "Full")
        self.assertEqual(out[0].score, 10)

    def test_universe_rows_carry_entry_and_stage(self):
        out = tr.scan_universe_tr([("Full", "FULL-USD", "full")], self._loader,
                                   spot_loader=lambda k: 103.0,
                                   now=pd.Timestamp("2030-01-01", tz="UTC"))
        row = out[0]
        self.assertIsNotNone(row.entry)
        self.assertIn(row.entry_status, (WAIT_RETRACE, AT_ENTRY))
        self.assertTrue(row.price_is_live)

    def test_failed_loader_reported_not_dropped(self):
        def boom(k):
            raise ConnectionError("down")
        out = tr.scan_universe_tr([("Bad", "BAD-USD", "bad")], boom)
        self.assertIsNotNone(out[0].error)


class TestLuckRange(unittest.TestCase):
    """Real results must be distinguishable from noise before trusting them."""

    def _trades(self, rs):
        ts = pd.Timestamp("2026-01-01", tz="UTC")
        return [tr.TRTrade("X", KIND_INITIAL, 1.0, "Long", ts, 100, 95, 110,
                           "WIN" if r > 0 else "LOSS", r) for r in rs]

    def test_random_data_shows_no_edge(self):
        """Asserting the verdict itself would be flaky by design: a ~95% luck
        band means pure noise lands outside it about 1 time in 20 (an 8-series
        sample did exactly that). Test the real property instead — on noise,
        the average result per entry sits near zero, well inside a wide band."""
        trades = []
        for seed in range(20):
            m5, h1, h4 = market(seed=seed + 1000, days=25)
            trades += run_backtest(m5, h1, h4, "RW")
        allrow = {s.group: s for s in backtest_stats(trades)}["All"]
        self.assertGreater(allrow.entries, 300)
        self.assertLess(abs(allrow.avg_r_per_entry), 0.1)
        # ~3.5 standard errors: chance exceeds this far less than 1 time in 1,000
        self.assertLess(abs(allrow.total_r), 1.75 * allrow.luck_range)

    def test_consistent_winner_reads_clearly_positive(self):
        rs = [2.0] * 60 + [-1.0] * 40      # 60% wins at 2R: strongly positive
        allrow = {s.group: s for s in backtest_stats(self._trades(rs))}["All"]
        self.assertEqual(allrow.verdict, "Clearly positive")

    def test_consistent_loser_reads_clearly_negative(self):
        rs = [2.0] * 10 + [-1.0] * 90
        allrow = {s.group: s for s in backtest_stats(self._trades(rs))}["All"]
        self.assertEqual(allrow.verdict, "Clearly negative")

    def test_small_sample_refuses_to_judge(self):
        allrow = {s.group: s for s in backtest_stats(self._trades([2.0] * 10))}["All"]
        self.assertEqual(allrow.verdict, "Too few trades to judge")

    def test_luck_range_grows_with_more_trades(self):
        small = {s.group: s for s in backtest_stats(self._trades([2.0, -1.0] * 20))}["All"]
        large = {s.group: s for s in backtest_stats(self._trades([2.0, -1.0] * 200))}["All"]
        self.assertGreater(large.luck_range, small.luck_range)


class TestThreeToOneDefault(unittest.TestCase):
    def test_default_target_is_three_r(self):
        self.assertEqual(StrategyParams().target_r, 3.0)

    def test_default_plan_is_three_to_one(self):
        p = analyze("X", uptrend_4h(), bullish_1h(), five_min_with_support(), live_price=103.0)
        self.assertAlmostEqual(p.reward_risk, 3.0)
        self.assertAlmostEqual((p.target - p.entry) / (p.entry - p.stop), 3.0, places=6)

    def test_backtest_wins_pay_three_r_by_default(self):
        m5, h1, h4 = market()
        for t in run_backtest(m5, h1, h4, "X", enable_reentry=False):
            if t.status == "WIN":
                self.assertAlmostEqual(t.r_result, 3.0, places=6)
