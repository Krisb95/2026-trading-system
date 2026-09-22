import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Optional
from expectancy import (summarise, segments, EvidenceBook, expected_r_per_trade,
                         break_even_win_rate, simulate_expectations,
                         PROVEN_POSITIVE, PROVEN_NEGATIVE, UNPROVEN, TOO_FEW)


@dataclass
class T:
    ticker: str
    direction: str
    status: str
    r_result: Optional[float]
    weighted_r: Optional[float] = None


def book(rows):
    trades = []
    for ticker, direction, rs in rows:
        for r in rs:
            trades.append(T(ticker, direction, "WIN" if r > 0 else "LOSS", r))
    return trades


class TestExpectedValue(unittest.TestCase):
    def test_three_to_one_break_even_is_25_percent(self):
        self.assertAlmostEqual(break_even_win_rate(3.0), 0.25)

    def test_three_wins_in_ten_at_three_to_one_is_profitable(self):
        self.assertAlmostEqual(expected_r_per_trade(0.30, 3.0), 0.20)

    def test_two_wins_in_ten_at_three_to_one_loses(self):
        self.assertAlmostEqual(expected_r_per_trade(0.20, 3.0), -0.20)

    def test_break_even_gives_zero(self):
        self.assertAlmostEqual(expected_r_per_trade(0.25, 3.0), 0.0)


class TestSummarise(unittest.TestCase):
    def test_clearly_profitable_segment_is_proven_positive(self):
        n, wins, wr, avg, total, luck, v = summarise([3.0] * 40 + [-1.0] * 60)
        self.assertEqual(v, PROVEN_POSITIVE)
        self.assertAlmostEqual(avg, 0.6)

    def test_clearly_losing_segment_is_proven_negative(self):
        *_, v = summarise([3.0] * 5 + [-1.0] * 95)
        self.assertEqual(v, PROVEN_NEGATIVE)

    def test_break_even_segment_is_unproven(self):
        *_, v = summarise([3.0, -1.0, -1.0, -1.0] * 15)
        self.assertEqual(v, UNPROVEN)

    def test_small_sample_is_too_few(self):
        *_, v = summarise([3.0] * 10)
        self.assertEqual(v, TOO_FEW)

    def test_empty(self):
        n, *_, v = summarise([])
        self.assertEqual(n, 0)
        self.assertEqual(v, TOO_FEW)


class TestSegments(unittest.TestCase):
    def test_groups_by_key(self):
        trades = book([("BTC", "Long", [3.0, -1.0]), ("ETH", "Long", [-1.0])])
        segs = segments(trades, lambda t: (t.direction,))
        self.assertEqual(segs[("Long",)].n, 3)

    def test_ignores_unresolved(self):
        trades = book([("BTC", "Long", [3.0])]) + [T("BTC", "Long", "EXPIRED", None)]
        self.assertEqual(segments(trades, lambda t: ("All",))[("All",)].n, 1)

    def test_uses_weighted_r_when_present(self):
        trades = [T("BTC", "Long", "WIN", 3.0, weighted_r=1.5)]
        self.assertAlmostEqual(segments(trades, lambda t: ("All",))[("All",)].total_r, 1.5)


class TestEvidenceBook(unittest.TestCase):
    def test_prefers_most_specific_segment_with_enough_trades(self):
        trades = book([("BTC-USD", "Long", [3.0] * 20 + [-1.0] * 20),
                       ("ETH-USD", "Long", [-1.0] * 40)])
        verdict, why = EvidenceBook.from_trades(trades).evidence_for("BTC-USD", "Long")
        self.assertIn("BTC-USD Longs", why)
        self.assertEqual(verdict, PROVEN_POSITIVE)

    def test_falls_back_to_direction_when_coin_sample_small(self):
        trades = book([("BTC-USD", "Long", [3.0] * 3),
                       ("ETH-USD", "Long", [3.0] * 20 + [-1.0] * 30)])
        verdict, why = EvidenceBook.from_trades(trades).evidence_for("BTC-USD", "Long")
        self.assertIn("all Longs", why)

    def test_losing_segment_labelled_negative(self):
        trades = book([("X", "Short", [-1.0] * 45 + [3.0] * 2)])
        verdict, _ = EvidenceBook.from_trades(trades).evidence_for("X", "Short")
        self.assertEqual(verdict, PROVEN_NEGATIVE)

    def test_no_direction_has_no_evidence(self):
        verdict, _ = EvidenceBook.from_trades([]).evidence_for("X", None)
        self.assertEqual(verdict, TOO_FEW)

    def test_empty_book_says_too_few(self):
        verdict, _ = EvidenceBook.from_trades([]).evidence_for("X", "Long")
        self.assertEqual(verdict, TOO_FEW)

    def test_round_trip_through_dict(self):
        trades = book([("BTC-USD", "Long", [3.0] * 20 + [-1.0] * 20)])
        b = EvidenceBook.from_trades(trades)
        b2 = EvidenceBook.from_dict(b.to_dict())
        self.assertEqual(b.evidence_for("BTC-USD", "Long"), b2.evidence_for("BTC-USD", "Long"))


class TestSimulatedExpectations(unittest.TestCase):
    def test_losing_streaks_are_long_at_low_win_rates(self):
        e = simulate_expectations(0.30, 3.0, n_trades=100)
        self.assertGreaterEqual(e.streak_typical, 7)   # long runs are normal at 30%
        self.assertGreater(e.streak_bad, e.streak_typical)

    def test_profitable_system_still_has_drawdowns(self):
        e = simulate_expectations(0.35, 3.0, risk_pct=1.0, n_trades=100)
        self.assertGreater(e.r_per_trade, 0)
        self.assertGreater(e.drawdown_typical_pct, 3.0)

    def test_higher_risk_means_deeper_drawdown(self):
        lo = simulate_expectations(0.30, 3.0, risk_pct=1.0)
        hi = simulate_expectations(0.30, 3.0, risk_pct=3.0)
        self.assertAlmostEqual(hi.drawdown_typical_pct, lo.drawdown_typical_pct * 3, delta=0.5)

    def test_positive_edge_rarely_ends_down_over_many_trades(self):
        e = simulate_expectations(0.35, 3.0, n_trades=300)
        self.assertLess(e.chance_of_loss_pct, 5)

    def test_break_even_edge_ends_down_about_half_the_time(self):
        e = simulate_expectations(0.25, 3.0, n_trades=300)
        self.assertTrue(35 < e.chance_of_loss_pct < 65)

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            simulate_expectations(1.5, 3.0)
        with self.assertRaises(ValueError):
            simulate_expectations(0.3, 0)


if __name__ == "__main__":
    unittest.main()
