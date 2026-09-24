import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uihelpers import scan_count


class TestScanCount(unittest.TestCase):
    def test_short_list_needs_no_slider(self):
        """The commodities list is three entries — a 3-to-3 slider crashes."""
        lo, hi, value = scan_count(3)
        self.assertIsNone(lo)
        self.assertIsNone(hi)
        self.assertEqual(value, 3)

    def test_empty_list(self):
        self.assertEqual(scan_count(0), (None, None, 0))

    def test_long_list_gets_a_slider(self):
        lo, hi, value = scan_count(94)
        self.assertEqual((lo, hi), (3, 94))
        self.assertEqual(value, 10)

    def test_default_is_capped_by_what_exists(self):
        lo, hi, value = scan_count(6)
        self.assertEqual((lo, hi), (3, 6))
        self.assertEqual(value, 6)

    def test_minimum_is_respected(self):
        lo, hi, value = scan_count(20, default=1, minimum=3)
        self.assertEqual(value, 3)

    def test_bounds_are_always_valid_for_a_slider(self):
        for n in range(0, 120):
            lo, hi, value = scan_count(n)
            if lo is not None:
                self.assertLess(lo, hi, f"{n}: slider bounds must differ")
                self.assertTrue(lo <= value <= hi, f"{n}: value outside bounds")
            else:
                self.assertEqual(value, n)
