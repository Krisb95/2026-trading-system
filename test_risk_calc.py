import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_calc import calculate_risk, InvalidRiskInputError, get_instrument_spec


class TestRiskCalcLinear(unittest.TestCase):
    def test_basic_long_equity(self):
        r = calculate_risk(account_equity=10000, risk_pct=2, entry=100, stop=95,
                            direction="Long", target=115)
        self.assertAlmostEqual(r.max_permitted_loss, 200.0)
        self.assertAlmostEqual(r.stop_distance_price, 5.0)
        self.assertAlmostEqual(r.stop_distance_pct, 0.05)
        # position_notional = risk_amount / stop_pct = 200/0.05 = 4000
        self.assertAlmostEqual(r.position_notional, 4000.0, delta=100)
        # quantity ~ 4000/100 = 40 shares
        self.assertAlmostEqual(r.quantity, 40.0, delta=1)
        # gross loss at stop should be close to the intended risk amount
        self.assertAlmostEqual(r.gross_loss_at_stop, 200.0, delta=10)
        self.assertAlmostEqual(r.gross_profit_at_target, 40.0 * (115 - 100), delta=50)
        self.assertFalse(r.exceeds_account_equity)
        self.assertIsNone(r.liquidation_warning)

    def test_short_direction_math(self):
        r = calculate_risk(account_equity=10000, risk_pct=1, entry=100, stop=105,
                            direction="Short", target=90)
        self.assertGreater(r.quantity, 0)
        self.assertGreater(r.gross_profit_at_target, 0)  # short profits as price falls

    def test_invalid_long_stop_above_entry(self):
        with self.assertRaises(InvalidRiskInputError):
            calculate_risk(account_equity=10000, risk_pct=1, entry=100, stop=105, direction="Long")

    def test_invalid_short_stop_below_entry(self):
        with self.assertRaises(InvalidRiskInputError):
            calculate_risk(account_equity=10000, risk_pct=1, entry=100, stop=95, direction="Short")

    def test_rejects_bad_risk_pct(self):
        with self.assertRaises(InvalidRiskInputError):
            calculate_risk(account_equity=10000, risk_pct=0, entry=100, stop=95, direction="Long")
        with self.assertRaises(InvalidRiskInputError):
            calculate_risk(account_equity=10000, risk_pct=150, entry=100, stop=95, direction="Long")

    def test_margin_exceeds_equity_flag(self):
        # Tiny account, tight stop => huge notional relative to equity, no leverage
        r = calculate_risk(account_equity=100, risk_pct=2, entry=100, stop=99.5, direction="Long")
        self.assertTrue(r.exceeds_account_equity)

    def test_fees_and_slippage_reduce_net_profit(self):
        no_fee = calculate_risk(account_equity=10000, risk_pct=2, entry=100, stop=95,
                                 direction="Long", target=115, fee_rate=0, slippage_pct=0)
        with_fee = calculate_risk(account_equity=10000, risk_pct=2, entry=100, stop=95,
                                   direction="Long", target=115, fee_rate=0.001, slippage_pct=0.001)
        self.assertLess(with_fee.net_profit_at_target, no_fee.net_profit_at_target)
        self.assertGreater(with_fee.net_loss_at_stop, no_fee.net_loss_at_stop)


class TestRiskCalcFuturesContracts(unittest.TestCase):
    def test_gold_futures_multiplier_applied(self):
        spec = get_instrument_spec("GC=F")
        self.assertEqual(spec.contract_multiplier, 100)
        r = calculate_risk(account_equity=50000, risk_pct=1, entry=2000, stop=1980,
                            direction="Long", target=2050, ticker="GC=F")
        # 1 contract minimum granularity — quantity should be a whole number of contracts
        self.assertEqual(r.quantity, round(r.quantity))

    def test_leverage_reduces_margin(self):
        no_lev = calculate_risk(account_equity=10000, risk_pct=2, entry=100, stop=95,
                                 direction="Long", leverage=1)
        with_lev = calculate_risk(account_equity=10000, risk_pct=2, entry=100, stop=95,
                                   direction="Long", leverage=5)
        self.assertAlmostEqual(with_lev.margin_required, no_lev.margin_required / 5, delta=1)

    def test_liquidation_warning_fires_when_stop_beyond_buffer(self):
        # 10x leverage => naive liquidation buffer ~10%. Stop at 15% away should warn danger.
        r = calculate_risk(account_equity=10000, risk_pct=2, entry=100, stop=85,
                            direction="Long", leverage=10)
        self.assertIn("liquidated", r.liquidation_warning.lower())

    def test_liquidation_warning_safe_when_stop_inside_buffer(self):
        r = calculate_risk(account_equity=10000, risk_pct=2, entry=100, stop=98,
                            direction="Long", leverage=5)
        self.assertIsNotNone(r.liquidation_warning)
        self.assertNotIn("could be liquidated", r.liquidation_warning.lower())


if __name__ == "__main__":
    unittest.main()
