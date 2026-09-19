import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from portfolio import OpenPosition, summarize_portfolio_risk, correlation_matrix, risk_to_stop
from stops import StopManager, StopManagementError, ManagementLabel, suggest_management_label


class TestPortfolio(unittest.TestCase):
    def test_total_open_risk_sums_correctly(self):
        positions = [
            OpenPosition("BTC-USD", "crypto", "Long", entry=50000, stop=48000, quantity=0.1),
            OpenPosition("AAPL", "stock", "Long", entry=200, stop=190, quantity=10),
        ]
        summary = summarize_portfolio_risk(positions)
        expected = (50000 - 48000) * 0.1 + (200 - 190) * 10
        self.assertAlmostEqual(summary.total_open_risk, expected)

    def test_btc_short_vs_crypto_longs_flag(self):
        positions = [
            OpenPosition("BTC-USD", "crypto", "Short", entry=50000, stop=52000, quantity=0.1),
            OpenPosition("ETH-USD", "crypto", "Long", entry=3000, stop=2800, quantity=1),
            OpenPosition("SOL-USD", "crypto", "Long", entry=150, stop=140, quantity=5),
        ]
        summary = summarize_portfolio_risk(positions)
        self.assertTrue(any("BTC" in w for w in summary.concentration_warnings))

    def test_no_hedge_auto_classified_as_risk_free(self):
        # Even with a plausible hedge, we never claim it's "risk-free" — just flag it.
        positions = [
            OpenPosition("BTC-USD", "crypto", "Short", entry=50000, stop=52000, quantity=0.1),
            OpenPosition("ETH-USD", "crypto", "Long", entry=3000, stop=2800, quantity=1),
            OpenPosition("SOL-USD", "crypto", "Long", entry=150, stop=140, quantity=5),
        ]
        summary = summarize_portfolio_risk(positions)
        for w in summary.concentration_warnings:
            self.assertNotIn("risk-free", w.lower())

    def test_directional_exposure_sign(self):
        positions = [
            OpenPosition("AAPL", "stock", "Long", entry=100, stop=95, quantity=10),
            OpenPosition("MSFT", "stock", "Short", entry=100, stop=105, quantity=10),
        ]
        summary = summarize_portfolio_risk(positions)
        # net exposure should be long_notional - short_notional = 1000 - 1000 = 0
        self.assertAlmostEqual(summary.directional_exposure_by_class["stock"], 0.0)

    def test_correlation_matrix_from_real_returns(self):
        s1 = pd.Series([100, 102, 101, 105, 107])
        s2 = pd.Series([50, 51, 50.5, 52.5, 53.5])  # moves with s1
        corr = correlation_matrix({"A": s1, "B": s2})
        self.assertGreater(corr.loc["A", "B"], 0.9)

    def test_empty_portfolio_no_crash(self):
        summary = summarize_portfolio_risk([])
        self.assertEqual(summary.total_open_risk, 0.0)
        self.assertEqual(summary.concentration_warnings, [])


class TestStopManager(unittest.TestCase):
    def test_tightening_stop_allowed_for_long(self):
        sm = StopManager(direction="Long", entry=100, current_stop=90, current_target=130)
        sm.update_stop(95, note="moved to breakeven+")
        self.assertEqual(sm.current_stop, 95)
        self.assertEqual(len(sm.log), 1)

    def test_widening_stop_blocked_for_long(self):
        sm = StopManager(direction="Long", entry=100, current_stop=95, current_target=130)
        with self.assertRaises(StopManagementError):
            sm.update_stop(90)  # moving stop further from price = more risk

    def test_widening_stop_blocked_for_short(self):
        sm = StopManager(direction="Short", entry=100, current_stop=105, current_target=80)
        with self.assertRaises(StopManagementError):
            sm.update_stop(110)

    def test_allow_initial_widen_flag_bypasses_check(self):
        sm = StopManager(direction="Long", entry=100, current_stop=95, current_target=130)
        sm.update_stop(90, note="re-basing initial structural stop", allow_initial_widen=True)
        self.assertEqual(sm.current_stop, 90)

    def test_r_multiple_calculation(self):
        sm = StopManager(direction="Long", entry=100, current_stop=90, current_target=130)
        self.assertAlmostEqual(sm.current_r_multiple(120), 2.0)  # (120-100)/(100-90)=2R

    def test_transition_suggestion_activation(self):
        sm = StopManager(direction="Long", entry=100, current_stop=90, current_target=130)
        self.assertFalse(sm.suggest_transition_to_trailing(105))  # only 0.5R
        self.assertTrue(sm.suggest_transition_to_trailing(115))   # 1.5R

    def test_profit_protecting_validation_long(self):
        sm = StopManager(direction="Long", entry=100, current_stop=105, current_target=130)
        # stop above current price is invalid for a long
        err = sm.validate_profit_protecting_stop(current_price=103)
        self.assertIsNotNone(err)
        err2 = sm.validate_profit_protecting_stop(current_price=110)
        self.assertIsNone(err2)

    def test_profit_protecting_validation_accounts_for_costs(self):
        # Stop at 100.05 clears raw entry (100) but not after a 0.1% round-trip cost buffer.
        sm = StopManager(direction="Long", entry=100, current_stop=100.05, current_target=130)
        err_with_costs = sm.validate_profit_protecting_stop(current_price=105, round_trip_cost_pct=0.001)
        self.assertIsNotNone(err_with_costs)
        err_no_costs = sm.validate_profit_protecting_stop(current_price=105, round_trip_cost_pct=0.0)
        self.assertIsNone(err_no_costs)


class TestManagementLabels(unittest.TestCase):
    def test_invalidation_gives_exit(self):
        label = suggest_management_label("Long", invalidation_hit=True, risk_rule_breached=False,
                                          structure_supports_tightening=True, r_multiple=1.0)
        self.assertEqual(label, ManagementLabel.EXIT)

    def test_risk_rule_breach_gives_exit_even_if_structure_ok(self):
        label = suggest_management_label("Long", invalidation_hit=False, risk_rule_breached=True,
                                          structure_supports_tightening=True, r_multiple=1.0)
        self.assertEqual(label, ManagementLabel.EXIT)

    def test_structure_supports_tightening_gives_protect(self):
        label = suggest_management_label("Long", invalidation_hit=False, risk_rule_breached=False,
                                          structure_supports_tightening=True, r_multiple=1.5)
        self.assertEqual(label, ManagementLabel.PROTECT)

    def test_default_is_hold(self):
        label = suggest_management_label("Long", invalidation_hit=False, risk_rule_breached=False,
                                          structure_supports_tightening=False, r_multiple=0.5)
        self.assertEqual(label, ManagementLabel.HOLD)

    def test_reduce_threshold(self):
        label = suggest_management_label("Long", invalidation_hit=False, risk_rule_breached=False,
                                          structure_supports_tightening=False, r_multiple=-1.5,
                                          reduce_threshold_r=1.0)
        self.assertEqual(label, ManagementLabel.REDUCE)


if __name__ == "__main__":
    unittest.main()
