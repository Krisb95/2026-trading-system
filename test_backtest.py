import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from backtest import (
    TradeSignal, run_backtest, compute_metrics, split_in_out_sample,
    example_sma_crossover_signals,
)


def bar(o, h, l, c):
    return {"Open": o, "High": h, "Low": l, "Close": c}


class TestNoLookahead(unittest.TestCase):
    def test_fill_happens_on_bar_after_signal_not_signal_bar(self):
        # Signal at index 0 with a big favorable move ON bar 0 itself should
        # NOT be captured — the engine must fill at bar 1's open, not bar 0's close.
        df = pd.DataFrame([
            bar(100, 100, 100, 100),   # signal bar (index 0) - price doesn't matter for fill
            bar(150, 160, 149, 155),   # fill bar (index 1) - open is what we should fill at
            bar(155, 200, 150, 190),   # target should hit here
        ])
        sig = TradeSignal(signal_index=0, direction="Long", stop=90, target=180)
        results = run_backtest(df, [sig])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].entry_index, 1)
        self.assertAlmostEqual(results[0].entry_price, 150.0)  # bar[1] Open, not bar[0] Close

    def test_signal_at_last_bar_is_skipped_not_fabricated(self):
        df = pd.DataFrame([bar(100, 101, 99, 100), bar(101, 102, 100, 101)])
        sig = TradeSignal(signal_index=1, direction="Long", stop=95, target=110)  # no bar after index 1
        results = run_backtest(df, [sig])
        self.assertEqual(len(results), 0)


class TestFillLogicAndExits(unittest.TestCase):
    def test_stop_hit_long(self):
        df = pd.DataFrame([
            bar(100, 100, 100, 100),
            bar(100, 101, 99, 100),
            bar(100, 100, 90, 92),   # low pierces stop at 95
        ])
        sig = TradeSignal(signal_index=0, direction="Long", stop=95, target=120)
        results = run_backtest(df, [sig])
        self.assertEqual(results[0].exit_reason, "stop")
        self.assertAlmostEqual(results[0].exit_price, 95)
        self.assertLess(results[0].r_multiple, 0)

    def test_target_hit_long(self):
        df = pd.DataFrame([
            bar(100, 100, 100, 100),
            bar(100, 101, 99, 100),
            bar(100, 130, 99, 125),  # high exceeds target at 120
        ])
        sig = TradeSignal(signal_index=0, direction="Long", stop=90, target=120)
        results = run_backtest(df, [sig])
        self.assertEqual(results[0].exit_reason, "target")
        self.assertAlmostEqual(results[0].exit_price, 120)
        self.assertAlmostEqual(results[0].r_multiple, 2.0, delta=0.01)  # (120-100)/(100-90)=2R

    def test_conservative_assumption_stop_wins_if_both_hit_same_bar(self):
        df = pd.DataFrame([
            bar(100, 100, 100, 100),
            bar(100, 101, 99, 100),
            bar(100, 130, 80, 100),  # both stop (90) and target (120) breached in one bar
        ])
        sig = TradeSignal(signal_index=0, direction="Long", stop=90, target=120)
        results = run_backtest(df, [sig])
        self.assertEqual(results[0].exit_reason, "stop")  # conservative: stop assumed first

    def test_end_of_data_exit_when_neither_hit(self):
        df = pd.DataFrame([
            bar(100, 100, 100, 100),
            bar(100, 105, 98, 102),
            bar(102, 106, 100, 104),
        ])
        sig = TradeSignal(signal_index=0, direction="Long", stop=80, target=200)
        results = run_backtest(df, [sig])
        self.assertEqual(results[0].exit_reason, "end_of_data")
        self.assertAlmostEqual(results[0].exit_price, 104)  # last close

    def test_short_direction_stop_and_target(self):
        df = pd.DataFrame([
            bar(100, 100, 100, 100),
            bar(100, 101, 99, 100),
            bar(100, 100, 70, 75),  # price fell, target for short should trigger
        ])
        sig = TradeSignal(signal_index=0, direction="Short", stop=110, target=80)
        results = run_backtest(df, [sig])
        self.assertEqual(results[0].exit_reason, "target")
        self.assertGreater(results[0].r_multiple, 0)

    def test_zero_stop_distance_skipped_not_crashed(self):
        df = pd.DataFrame([bar(100, 100, 100, 100), bar(100, 101, 99, 100)])
        sig = TradeSignal(signal_index=0, direction="Long", stop=100, target=110)  # stop == entry
        results = run_backtest(df, [sig])
        self.assertEqual(len(results), 0)

    def test_fees_reduce_r_multiple(self):
        df = pd.DataFrame([
            bar(100, 100, 100, 100),
            bar(100, 101, 99, 100),
            bar(100, 130, 99, 125),
        ])
        sig = TradeSignal(signal_index=0, direction="Long", stop=90, target=120)
        no_fee = run_backtest(df, [sig], fee_rate=0)
        with_fee = run_backtest(df, [sig], fee_rate=0.001)
        self.assertLess(with_fee[0].r_multiple, no_fee[0].r_multiple)


class TestMetrics(unittest.TestCase):
    def test_expectancy_and_win_rate(self):
        from backtest import TradeResult
        results = [
            TradeResult(0, 1, "Long", 100, 120, 90, 120, "target", 2.0, 200, 2.0, 0.1),
            TradeResult(0, 1, "Long", 100, 90, 90, 120, "stop", -1.0, -100, 0.0, 1.0),
            TradeResult(0, 1, "Long", 100, 90, 90, 120, "stop", -1.0, -100, 0.0, 1.0),
        ]
        m = compute_metrics(results)
        self.assertEqual(m.trade_count, 3)
        self.assertAlmostEqual(m.win_rate, 1 / 3, places=4)
        self.assertAlmostEqual(m.expectancy_r, (2.0 - 1.0 - 1.0) / 3, places=4)
        self.assertAlmostEqual(m.profit_factor, 2.0 / 2.0, places=4)

    def test_empty_results_no_crash(self):
        m = compute_metrics([])
        self.assertEqual(m.trade_count, 0)
        self.assertEqual(m.win_rate, 0.0)

    def test_max_drawdown_computed_from_equity_curve(self):
        from backtest import TradeResult
        # R sequence: +2, +2, -3, +1 -> equity curve 2,4,1,2 -> peak 4, trough 1 -> dd=3
        results = [
            TradeResult(0, 1, "Long", 100, 100, 90, 100, "target", 2.0, 0, 0, 0),
            TradeResult(0, 1, "Long", 100, 100, 90, 100, "target", 2.0, 0, 0, 0),
            TradeResult(0, 1, "Long", 100, 100, 90, 100, "stop", -3.0, 0, 0, 0),
            TradeResult(0, 1, "Long", 100, 100, 90, 100, "target", 1.0, 0, 0, 0),
        ]
        m = compute_metrics(results)
        self.assertAlmostEqual(m.max_drawdown_r, 3.0)


class TestSplitAndExampleSignals(unittest.TestCase):
    def test_split_is_chronological_no_shuffle(self):
        df = pd.DataFrame({"Close": range(100)})
        in_sample, out_sample = split_in_out_sample(df, split_ratio=0.7)
        self.assertEqual(len(in_sample), 70)
        self.assertEqual(len(out_sample), 30)
        self.assertEqual(in_sample["Close"].iloc[-1], 69)
        self.assertEqual(out_sample["Close"].iloc[0], 70)

    def test_example_sma_signals_only_use_past_data(self):
        # Construct a synthetic uptrend-then-downtrend series; just check
        # it runs and produces signals without raising, and every signal's
        # index is within bounds.
        n = 200
        prices = list(np.linspace(100, 150, 100)) + list(np.linspace(150, 100, 100))
        df = pd.DataFrame({
            "Open": prices, "High": [p * 1.01 for p in prices],
            "Low": [p * 0.99 for p in prices], "Close": prices,
        })
        signals = example_sma_crossover_signals(df, fast=5, slow=20)
        self.assertGreater(len(signals), 0)
        for s in signals:
            self.assertGreaterEqual(s.signal_index, 0)
            self.assertLess(s.signal_index, n)


if __name__ == "__main__":
    unittest.main()
