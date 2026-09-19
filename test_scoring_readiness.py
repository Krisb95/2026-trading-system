import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring import score_setup, weakest_components, trade_policy_for_grade
from readiness import (determine_readiness, ReadinessInputs, Readiness,
                        display_label, missing_entry_sequence_stages)


class TestScoring(unittest.TestCase):
    def test_perfect_score_caps_at_ten(self):
        evidence = {
            "regime_alignment_1d_4h": True, "support_resistance": True,
            "liquidity_sweep_reclaim": True, "fib_confluence": True,
            "structure_4h_supports": True, "entry_confirmation_1h": True,
            "invalidation_defined": True, "rr_at_least_2": True,
        }
        result = score_setup(evidence)
        self.assertEqual(result.normalized_score, 10.0)
        self.assertEqual(result.label, "A+")

    def test_missing_evidence_scores_zero_not_assumed(self):
        # Only two components have real evidence; everything else is
        # simply absent from the dict (i.e. unavailable).
        evidence = {"regime_alignment_1d_4h": True, "rr_at_least_2": True}
        result = score_setup(evidence)
        self.assertEqual(result.normalized_score, 3.0)  # 2 + 1
        self.assertEqual(result.label, "C")

    def test_false_scores_zero_same_as_missing(self):
        evidence = {"regime_alignment_1d_4h": False}
        result = score_setup(evidence)
        self.assertEqual(result.normalized_score, 0.0)

    def test_negative_components_reduce_score_but_floor_at_zero(self):
        evidence = {
            "rr_at_least_2": True,  # +1
            "opposing_liquidity_ahead": True,  # -1
            "chasing_extended_move": True,  # -1
        }
        result = score_setup(evidence)
        self.assertEqual(result.normalized_score, 0.0)  # 1-1-1 = -1, floored to 0

    def test_labels_match_spec_thresholds(self):
        # 6 points -> B ; 8 points -> A+
        six_pt = score_setup({"regime_alignment_1d_4h": True, "liquidity_sweep_reclaim": True,
                               "support_resistance": True, "fib_confluence": True})
        self.assertEqual(six_pt.normalized_score, 6.0)
        self.assertEqual(six_pt.label, "B")

        eight_pt = score_setup({"regime_alignment_1d_4h": True, "liquidity_sweep_reclaim": True,
                                 "structure_4h_supports": True, "entry_confirmation_1h": True,
                                 "invalidation_defined": True, "rr_at_least_2": True})
        self.assertEqual(eight_pt.normalized_score, 8.0)
        self.assertEqual(eight_pt.label, "A+")

    def test_no_duplicate_counting_same_key_read_once(self):
        # Passing the same key doesn't let it contribute twice — a plain
        # dict can only have one value per key by construction, but this
        # test documents/locks that invariant explicitly.
        evidence = {"rr_at_least_2": True}
        result = score_setup(evidence)
        rr_items = [li for li in result.breakdown if li.key == "rr_at_least_2"]
        self.assertEqual(len(rr_items), 1)


class TestReadiness(unittest.TestCase):
    def test_data_error_takes_precedence_over_everything(self):
        inputs = ReadinessInputs(data_is_valid=False, user_marked_active=True,
                                  near_actionable_location=True, confirmation_triggered=True,
                                  invalidation_defined=True, risk_checks_pass=True)
        self.assertEqual(determine_readiness(inputs), Readiness.DATA_ERROR)

    def test_invalidated_beats_active(self):
        inputs = ReadinessInputs(data_is_valid=True, invalidation_hit=True, user_marked_active=True)
        self.assertEqual(determine_readiness(inputs), Readiness.INVALIDATED)

    def test_high_score_does_not_imply_ready(self):
        # Even with data valid and near location, missing confirmation => CONDITIONAL not READY.
        inputs = ReadinessInputs(data_is_valid=True, near_actionable_location=True,
                                  confirmation_triggered=False, invalidation_defined=True,
                                  risk_checks_pass=True)
        self.assertEqual(determine_readiness(inputs), Readiness.CONDITIONAL)

    def test_all_conditions_met_gives_ready(self):
        inputs = ReadinessInputs(data_is_valid=True, near_actionable_location=True,
                                  confirmation_triggered=True, invalidation_defined=True,
                                  risk_checks_pass=True)
        self.assertEqual(determine_readiness(inputs), Readiness.READY)

    def test_default_not_ready(self):
        inputs = ReadinessInputs(data_is_valid=True)
        self.assertEqual(determine_readiness(inputs), Readiness.NOT_READY)


class TestTradePolicy(unittest.TestCase):
    def test_c_grade_is_no_trade(self):
        result = score_setup({"rr_at_least_2": True})  # score = 1 -> C
        self.assertEqual(result.label, "C")
        self.assertIn("NO TRADE", trade_policy_for_grade(result.label))

    def test_b_grade_is_cautious(self):
        result = score_setup({"regime_alignment_1d_4h": True, "liquidity_sweep_reclaim": True,
                               "support_resistance": True, "fib_confluence": True})  # 6 -> B
        self.assertIn("CAUTIOUSLY", trade_policy_for_grade(result.label))

    def test_weakest_components_lists_missing_positive_items(self):
        result = score_setup({"regime_alignment_1d_4h": True})  # only 2 points, rest missing
        missing = weakest_components(result, top_n=3)
        self.assertTrue(all(li.points_awarded == 0 for li in missing))
        self.assertTrue(all(li.points_possible > 0 for li in missing))
        # liquidity_sweep_reclaim (2 pts) should outrank 1-pt items
        self.assertEqual(missing[0].key, "liquidity_sweep_reclaim")


class TestReadinessDisplayAndSequence(unittest.TestCase):
    def test_display_label_maps_conditional_to_waiting(self):
        self.assertEqual(display_label(Readiness.CONDITIONAL), "WAITING")

    def test_display_label_maps_not_ready_and_invalidated_to_no_trade(self):
        self.assertEqual(display_label(Readiness.NOT_READY), "NO TRADE")
        self.assertEqual(display_label(Readiness.INVALIDATED), "NO TRADE")
        self.assertEqual(display_label(Readiness.DATA_ERROR), "NO TRADE")

    def test_display_label_ready_and_active_both_show_ready(self):
        self.assertEqual(display_label(Readiness.READY), "READY")
        self.assertEqual(display_label(Readiness.ACTIVE), "READY")

    def test_missing_entry_sequence_stages_lists_only_unmet(self):
        evidence = {"location": True, "liquidity_event": True, "reclaim_or_rejection": False}
        missing = missing_entry_sequence_stages(evidence)
        self.assertTrue(any("reject" in m.lower() for m in missing))
        self.assertFalse(any(m.lower().startswith("price reaches") for m in missing))


if __name__ == "__main__":
    unittest.main()
