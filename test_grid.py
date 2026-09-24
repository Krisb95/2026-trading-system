import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grid import (build_grid, usable_levels, weights_for, EQUAL, HEAVIER_LOWER,
                   MIN_LEVELS)


class TestLevelSelection(unittest.TestCase):
    def test_keeps_distinct_levels_below_price_for_a_long(self):
        got = usable_levels([98.0, 96.0, 94.0], "Long", price=100.0, atr=1.0)
        self.assertEqual(got, [98.0, 96.0, 94.0])

    def test_drops_levels_too_close_together(self):
        got = usable_levels([98.0, 97.9, 96.0], "Long", price=100.0, atr=1.0,
                            min_gap_atr=0.6)
        self.assertEqual(got, [98.0, 96.0])

    def test_drops_levels_too_far_away(self):
        got = usable_levels([98.0, 50.0], "Long", price=100.0, atr=1.0, max_distance_atr=8)
        self.assertEqual(got, [98.0])

    def test_ignores_levels_on_the_wrong_side(self):
        got = usable_levels([102.0, 98.0], "Long", price=100.0, atr=1.0)
        self.assertEqual(got, [98.0])

    def test_short_looks_above_price(self):
        got = usable_levels([102.0, 104.0, 98.0], "Short", price=100.0, atr=1.0)
        self.assertEqual(got, [102.0, 104.0])

    def test_respects_the_maximum(self):
        got = usable_levels([98, 96, 94, 92, 90], "Long", price=100.0, atr=1.0, max_levels=3)
        self.assertEqual(len(got), 3)

    def test_no_levels(self):
        self.assertEqual(usable_levels([], "Long", 100.0, 1.0), [])


class TestWeights(unittest.TestCase):
    def test_equal_weights_sum_to_one(self):
        w = weights_for(3, EQUAL)
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertAlmostEqual(w[0], w[2])

    def test_heavier_lower_puts_most_at_the_best_price(self):
        w = weights_for(3, HEAVIER_LOWER)
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertGreater(w[2], w[0])


class TestGridRisk(unittest.TestCase):
    """The rule that cannot bend: a full fill risks exactly the budget."""

    def _grid(self, **kw):
        args = dict(direction="Long", levels=[98.0, 96.0, 94.0], price=100.0, stop=93.0,
                    risk_amount=100.0, target_r=3.0, weighting=EQUAL)
        args.update(kw)
        return build_grid(**args)

    def test_full_fill_risks_exactly_the_budget(self):
        g = self._grid()
        self.assertAlmostEqual(g.total_risk, 100.0, places=6)

    def test_holds_for_every_weighting(self):
        for style in (EQUAL, HEAVIER_LOWER):
            self.assertAlmostEqual(self._grid(weighting=style).total_risk, 100.0, places=6)

    def test_holds_for_shorts(self):
        g = self._grid(direction="Short", levels=[102.0, 104.0, 106.0], stop=107.0)
        self.assertAlmostEqual(g.total_risk, 100.0, places=6)

    def test_partial_fill_risks_less_not_more(self):
        g = self._grid()
        for level in g.levels[:-1]:
            self.assertLess(level.cumulative_risk, g.total_risk)

    def test_cumulative_risk_reaches_the_budget_at_the_last_level(self):
        g = self._grid()
        self.assertAlmostEqual(g.levels[-1].cumulative_risk, 100.0, places=6)

    def test_more_levels_does_not_multiply_risk(self):
        two = build_grid("Long", [98.0, 96.0], 100.0, 95.0, 100.0)
        five = build_grid("Long", [98.0, 96.0, 94.0, 92.0, 90.0], 100.0, 89.0, 100.0)
        self.assertAlmostEqual(two.total_risk, 100.0, places=6)
        self.assertAlmostEqual(five.total_risk, 100.0, places=6)


class TestGridLevels(unittest.TestCase):
    def test_target_gives_the_requested_multiple(self):
        g = build_grid("Long", [98.0, 96.0, 94.0], 100.0, 93.0, 100.0, target_r=3.0)
        self.assertAlmostEqual(g.reward_at_target / g.total_risk, 3.0, places=6)

    def test_average_entry_sits_between_the_levels(self):
        g = build_grid("Long", [98.0, 96.0, 94.0], 100.0, 93.0, 100.0)
        self.assertTrue(94.0 < g.average_entry < 98.0)

    def test_average_entry_beats_a_single_entry_for_a_long(self):
        g = build_grid("Long", [98.0, 96.0, 94.0], 100.0, 93.0, 100.0)
        self.assertLess(g.average_entry, g.single_entry_price)

    def test_heavier_lower_improves_the_average_entry(self):
        eq = build_grid("Long", [98.0, 96.0, 94.0], 100.0, 93.0, 100.0, weighting=EQUAL)
        hv = build_grid("Long", [98.0, 96.0, 94.0], 100.0, 93.0, 100.0,
                        weighting=HEAVIER_LOWER)
        self.assertLess(hv.average_entry, eq.average_entry)

    def test_levels_ordered_from_nearest_to_furthest(self):
        g = build_grid("Long", [94.0, 98.0, 96.0], 100.0, 93.0, 100.0)
        self.assertEqual([l.price for l in g.levels], [98.0, 96.0, 94.0])

    def test_distances_are_negative_for_a_long(self):
        g = build_grid("Long", [98.0, 96.0], 100.0, 95.0, 100.0)
        self.assertTrue(all(l.distance_pct < 0 for l in g.levels))

    def test_warns_that_the_stop_is_further_away(self):
        g = build_grid("Long", [98.0, 96.0, 94.0], 100.0, 93.0, 100.0)
        self.assertTrue(any("further than a single entry" in w for w in g.warnings))

    def test_partial_fill_note_mentions_the_smaller_win(self):
        g = build_grid("Long", [98.0, 96.0], 100.0, 95.0, 100.0)
        self.assertIn("smaller win", g.partial_fill_note)


class TestGridRefusals(unittest.TestCase):
    def test_one_level_is_not_a_grid(self):
        self.assertIsNone(build_grid("Long", [98.0], 100.0, 95.0, 100.0))

    def test_stop_must_sit_beyond_every_level(self):
        self.assertIsNone(build_grid("Long", [98.0, 96.0], 100.0, 97.0, 100.0))

    def test_short_stop_must_sit_above_every_level(self):
        self.assertIsNone(build_grid("Short", [102.0, 104.0], 100.0, 103.0, 100.0))

    def test_zero_risk_budget(self):
        self.assertIsNone(build_grid("Long", [98.0, 96.0], 100.0, 95.0, 0.0))

    def test_bad_direction(self):
        with self.assertRaises(ValueError):
            build_grid("Sideways", [98.0, 96.0], 100.0, 95.0, 100.0)
