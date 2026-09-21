import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from strategy_backtest import (run_strategy_backtest, closed_daily_slice,
                                stats_by_grade, verdict, BacktestTrade)
from trade_sim import WIN, LOSS, EXPIRED


def market(n=420, seed=3, drift=0.0012, vol=0.016, freq="4h"):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    hi = close * (1 + np.abs(rng.normal(0, 0.007, n)))
    lo = close * (1 - np.abs(rng.normal(0, 0.007, n)))
    return pd.DataFrame({"Open": close, "High": hi, "Low": lo, "Close": close}, index=idx)


def daily_from(df4):
    return df4.resample("1D").agg({"Open": "first", "High": "max",
                                    "Low": "min", "Close": "last"}).dropna()


class TestNoLookAhead(unittest.TestCase):
    """The single most important property of a backtest."""

    def test_plans_before_a_cutoff_ignore_all_later_candles(self):
        df = market()
        base = run_strategy_backtest(df, daily_from(df), "X", step=3)
        self.assertTrue(base, "expected at least one signal to compare")

        cutoff = df.index[300]
        tampered = df.copy()
        # Wreck everything after the cutoff.
        tampered.loc[tampered.index > cutoff, ["Open", "High", "Low", "Close"]] *= 3.0
        altered = run_strategy_backtest(tampered, daily_from(tampered), "X", step=3)

        def plans(trades):
            return [(t.signal_time, t.direction, round(t.entry, 8), round(t.stop, 8),
                     round(t.target, 8), t.grade)
                    for t in trades if t.signal_time <= cutoff]

        # Decisions made before the cutoff must be identical: the strategy
        # cannot have known what happened afterwards. (Their OUTCOMES may
        # differ, since outcomes legitimately depend on later prices.)
        self.assertEqual(plans(base), plans(altered))

    def test_daily_candle_excluded_until_closed(self):
        idx = pd.date_range("2026-03-01", periods=5, freq="1D", tz="UTC")
        d = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0}, index=idx)
        # At 08:00 on 03-05 the 03-05 candle is still forming.
        decision = pd.Timestamp("2026-03-05 08:00", tz="UTC")
        sliced = closed_daily_slice(d, decision)
        self.assertEqual(sliced.index[-1], pd.Timestamp("2026-03-04", tz="UTC"))

    def test_daily_candle_included_once_closed(self):
        idx = pd.date_range("2026-03-01", periods=5, freq="1D", tz="UTC")
        d = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0}, index=idx)
        decision = pd.Timestamp("2026-03-05 00:00", tz="UTC")   # 03-04 just closed
        self.assertEqual(closed_daily_slice(d, decision).index[-1],
                         pd.Timestamp("2026-03-04", tz="UTC"))

    def test_signal_time_is_after_the_bar_closes(self):
        df = market()
        for t in run_strategy_backtest(df, daily_from(df), "X", step=3):
            # decision time is a 4H bar close, i.e. a bar open + 4h
            self.assertIn(t.signal_time - pd.Timedelta(hours=4), df.index)


class TestBacktestBehaviour(unittest.TestCase):
    def test_no_overlapping_trades(self):
        df = market()
        trades = run_strategy_backtest(df, daily_from(df), "X", step=1)
        resolved = [t for t in trades if t.exit_time is not None]
        for a, b in zip(resolved, resolved[1:]):
            self.assertGreaterEqual(b.signal_time, a.exit_time)

    def test_every_trade_meets_min_rr(self):
        df = market()
        for t in run_strategy_backtest(df, daily_from(df), "X", min_rr=2.0, step=3):
            self.assertGreaterEqual(t.planned_rr, 2.0)

    def test_levels_ordered_correctly(self):
        df = market()
        for t in run_strategy_backtest(df, daily_from(df), "X", step=3):
            if t.direction == "Long":
                self.assertLess(t.stop, t.entry)
                self.assertGreater(t.target, t.entry)
            else:
                self.assertGreater(t.stop, t.entry)
                self.assertLess(t.target, t.entry)

    def test_short_history_returns_nothing(self):
        df = market(n=50)
        self.assertEqual(run_strategy_backtest(df, None, "X"), [])

    def test_works_without_daily_data(self):
        df = market()
        trades = run_strategy_backtest(df, None, "X", step=3)
        self.assertIsInstance(trades, list)

    def test_progress_reported(self):
        df = market()
        seen = []
        run_strategy_backtest(df, None, "X", step=6, progress=lambda i, n: seen.append((i, n)))
        self.assertEqual(seen[-1][0], seen[-1][1])


def trade(grade, status, r):
    ts = pd.Timestamp("2026-01-01", tz="UTC")
    return BacktestTrade("X", ts, "Long", 8.0, grade, 100, 95, 110, 2.0, status, r)


class TestStats(unittest.TestCase):
    def test_win_rate_and_expectancy(self):
        trades = [trade("A+", WIN, 2.0), trade("A+", LOSS, -1.0), trade("A+", LOSS, -1.0)]
        a = {s.grade: s for s in stats_by_grade(trades)}["A+"]
        self.assertAlmostEqual(a.win_rate, 1 / 3)
        self.assertAlmostEqual(a.avg_r, 0.0)

    def test_expired_excluded_from_win_rate(self):
        trades = [trade("B", WIN, 2.0), trade("B", EXPIRED, None)]
        b = {s.grade: s for s in stats_by_grade(trades)}["B"]
        self.assertEqual(b.win_rate, 1.0)
        self.assertEqual(b.expired, 1)
        self.assertAlmostEqual(b.fill_rate, 0.5)

    def test_small_samples_flagged(self):
        a = {s.grade: s for s in stats_by_grade([trade("A+", WIN, 2.0)])}["A+"]
        self.assertFalse(a.enough_data)

    def test_all_row_aggregates(self):
        trades = [trade("A+", WIN, 2.0), trade("C", LOSS, -1.0)]
        allrow = {s.grade: s for s in stats_by_grade(trades)}["All"]
        self.assertEqual(allrow.signals, 2)
        self.assertAlmostEqual(allrow.total_r, 1.0)


class TestVerdict(unittest.TestCase):
    def test_reports_when_grades_do_not_separate(self):
        trades = ([trade("A+", LOSS, -1.0)] * 5 + [trade("C", WIN, 2.0)] * 5)
        self.assertIn("did NOT beat", verdict(stats_by_grade(trades)))

    def test_reports_when_grades_do_separate(self):
        trades = ([trade("A+", WIN, 2.0)] * 5 + [trade("C", LOSS, -1.0)] * 5)
        self.assertIn("separated better setups", verdict(stats_by_grade(trades)))

    def test_flags_thin_samples(self):
        trades = ([trade("A+", WIN, 2.0)] * 3 + [trade("C", LOSS, -1.0)] * 3)
        self.assertIn("could easily be luck", verdict(stats_by_grade(trades)))

    def test_insufficient_data_message(self):
        self.assertIn("Not enough", verdict(stats_by_grade([])))


if __name__ == "__main__":
    unittest.main()


class TestNoFakeEdgeOnRandomData(unittest.TestCase):
    """A random walk has no edge by construction. If the backtester ever
    reports solid profits on one, it is leaking future information. This
    guards every future change to the scanner against that failure."""

    def test_random_walk_shows_no_positive_edge(self):
        trades = []
        for seed in range(8):
            rng = np.random.default_rng(seed)
            n = 600
            close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.015, n)))
            idx = pd.date_range("2025-06-01", periods=n, freq="4h", tz="UTC")
            hi = close * (1 + np.abs(rng.normal(0, 0.006, n)))
            lo = close * (1 - np.abs(rng.normal(0, 0.006, n)))
            df = pd.DataFrame({"Open": close, "High": hi, "Low": lo, "Close": close}, index=idx)
            trades += run_strategy_backtest(df, daily_from(df), f"RW{seed}", step=4)
        overall = {s.grade: s for s in stats_by_grade(trades)}["All"]
        self.assertGreater(overall.wins + overall.losses, 50, "need a real sample")
        self.assertLess(overall.avg_r, 0.15,
                        f"backtester found an edge in pure noise ({overall.avg_r:+.2f}R) — "
                        f"look for look-ahead bias")
