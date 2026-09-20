import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from position_math import PositionSnapshot, analyse_scale_in, ScaleVerdict


def long_pos(price=110.0, entry=100.0, stop=95.0, target=120.0, qty=10.0):
    return PositionSnapshot(direction="Long", entry=entry, stop=stop, target=target,
                             quantity=qty, current_price=price)


def short_pos(price=90.0, entry=100.0, stop=105.0, target=80.0, qty=10.0):
    return PositionSnapshot(direction="Short", entry=entry, stop=stop, target=target,
                             quantity=qty, current_price=price)


class TestUnrealisedPL(unittest.TestCase):
    def test_long_in_profit(self):
        self.assertAlmostEqual(long_pos(price=110).unrealised_pl, 100.0)

    def test_long_in_loss(self):
        self.assertAlmostEqual(long_pos(price=98).unrealised_pl, -20.0)

    def test_short_in_profit(self):
        self.assertAlmostEqual(short_pos(price=90).unrealised_pl, 100.0)

    def test_short_in_loss(self):
        self.assertAlmostEqual(short_pos(price=104).unrealised_pl, -40.0)

    def test_percentage_uses_notional(self):
        p = long_pos(price=110)
        self.assertAlmostEqual(p.unrealised_pct, 10.0)

    def test_contract_multiplier_scales_pl(self):
        p = PositionSnapshot("Long", 2000, 1980, 2050, 1, 2010, contract_multiplier=100)
        self.assertAlmostEqual(p.unrealised_pl, 1000.0)


class TestOutcomes(unittest.TestCase):
    def test_loss_at_stop_is_negative_for_normal_stop(self):
        self.assertLess(long_pos().loss_at_stop, 0)

    def test_loss_at_stop_becomes_profit_when_stop_beyond_entry(self):
        p = long_pos(stop=105)
        self.assertGreater(p.loss_at_stop, 0)
        self.assertTrue(p.stop_is_protecting_profit)

    def test_profit_at_target(self):
        self.assertAlmostEqual(long_pos().profit_at_target, 200.0)

    def test_short_profit_at_target(self):
        self.assertAlmostEqual(short_pos().profit_at_target, 200.0)

    def test_no_target_returns_none(self):
        p = long_pos(); p.target = None
        self.assertIsNone(p.profit_at_target)
        self.assertIsNone(p.reward_risk)

    def test_reward_risk(self):
        self.assertAlmostEqual(long_pos().reward_risk, 4.0)   # 20 reward / 5 risk

    def test_r_multiple(self):
        self.assertAlmostEqual(long_pos(price=110).r_multiple, 2.0)

    def test_realised_pl_at_exit(self):
        self.assertAlmostEqual(long_pos().realised_pl(115), 150.0)

    def test_zero_risk_does_not_divide_by_zero(self):
        p = long_pos(stop=100)
        self.assertEqual(p.r_multiple, 0.0)


class TestScaleInGuards(unittest.TestCase):
    """The rulebook: never add to a losing position unless preplanned, and
    never exceed the risk budget."""

    def test_adding_to_loser_unplanned_is_blocked(self):
        p = long_pos(price=97)          # underwater
        r = analyse_scale_in(p, 5, 97, account_equity=10000, max_risk_pct=5.0)
        self.assertEqual(r.verdict, ScaleVerdict.BLOCKED)
        self.assertTrue(any("not planned in advance" in x for x in r.reasons))

    def test_adding_to_loser_preplanned_is_caution_not_blocked(self):
        p = long_pos(price=97)
        r = analyse_scale_in(p, 5, 97, account_equity=100000, max_risk_pct=5.0,
                              preplanned=True)
        self.assertEqual(r.verdict, ScaleVerdict.CAUTION)
        self.assertTrue(any("planned in advance" in x for x in r.reasons))

    def test_exceeding_risk_budget_is_blocked(self):
        p = long_pos(price=110)
        r = analyse_scale_in(p, 1000, 110, account_equity=10000, max_risk_pct=1.0)
        self.assertEqual(r.verdict, ScaleVerdict.BLOCKED)
        self.assertTrue(any("over your" in x for x in r.reasons))

    def test_adding_in_profit_within_budget_is_allowed(self):
        p = long_pos(price=110)
        r = analyse_scale_in(p, 2, 110, account_equity=100000, max_risk_pct=5.0)
        self.assertEqual(r.verdict, ScaleVerdict.ALLOWED)

    def test_add_price_past_stop_is_blocked(self):
        """Averaging down so far that the new average sits beyond the stop."""
        p = long_pos(price=90, entry=100, stop=95, qty=10)
        r = analyse_scale_in(p, 1000, 90, account_equity=10_000_000,
                              max_risk_pct=99.0, preplanned=True)
        self.assertEqual(r.verdict, ScaleVerdict.BLOCKED)
        self.assertTrue(any("wrong side of the stop" in x for x in r.reasons))

    def test_poor_resulting_rr_downgrades_to_caution(self):
        p = long_pos(price=118, entry=100, stop=95, target=120, qty=10)
        r = analyse_scale_in(p, 10, 118, account_equity=1_000_000, max_risk_pct=90.0)
        self.assertEqual(r.verdict, ScaleVerdict.CAUTION)
        self.assertTrue(any("reward:risk" in x for x in r.reasons))


class TestScaleInMaths(unittest.TestCase):
    def test_new_average_entry_is_weighted(self):
        p = long_pos(price=110, entry=100, qty=10)
        r = analyse_scale_in(p, 10, 120, account_equity=1_000_000, max_risk_pct=90.0)
        self.assertAlmostEqual(r.new_average_entry, 110.0)

    def test_new_quantity_adds_up(self):
        p = long_pos(qty=10)
        r = analyse_scale_in(p, 7, 110, account_equity=1_000_000, max_risk_pct=90.0)
        self.assertAlmostEqual(r.new_quantity, 17.0)

    def test_risk_measured_from_new_average(self):
        p = long_pos(price=110, entry=100, stop=95, qty=10)
        r = analyse_scale_in(p, 10, 120, account_equity=1_000_000, max_risk_pct=90.0)
        # new avg 110, stop 95 -> 15 per unit x 20 units
        self.assertAlmostEqual(r.new_total_risk, 300.0)

    def test_added_notional(self):
        p = long_pos()
        r = analyse_scale_in(p, 5, 110, account_equity=1_000_000, max_risk_pct=90.0)
        self.assertAlmostEqual(r.added_notional, 550.0)

    def test_negative_add_quantity_rejected(self):
        r = analyse_scale_in(long_pos(), -5, 110, account_equity=10000)
        self.assertEqual(r.verdict, ScaleVerdict.BLOCKED)

    def test_zero_equity_rejected(self):
        r = analyse_scale_in(long_pos(), 5, 110, account_equity=0)
        self.assertEqual(r.verdict, ScaleVerdict.BLOCKED)

    def test_short_scale_in_maths(self):
        p = short_pos(price=90, entry=100, stop=105, qty=10)
        r = analyse_scale_in(p, 10, 90, account_equity=1_000_000, max_risk_pct=90.0)
        self.assertAlmostEqual(r.new_average_entry, 95.0)
        self.assertEqual(r.verdict, ScaleVerdict.ALLOWED)

    def test_reasons_always_populated(self):
        r = analyse_scale_in(long_pos(price=110), 2, 110,
                              account_equity=1_000_000, max_risk_pct=90.0)
        self.assertTrue(r.reasons)


if __name__ == "__main__":
    unittest.main()
