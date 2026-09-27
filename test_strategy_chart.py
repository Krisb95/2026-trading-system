import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Optional
import pandas as pd
from strategy_chart import (chart_spec, levels_for, entry_zone_of, ChartLevel,
                             GREEN, RED, AMBER, BLUE)


def candles(n=50, start=100.0):
    idx = pd.date_range("2026-06-01", periods=n, freq="4h", tz="UTC")
    closes = [start + i * 0.5 for i in range(n)]
    return pd.DataFrame({"Open": closes, "High": [c + 1 for c in closes],
                          "Low": [c - 1 for c in closes], "Close": closes}, index=idx)


@dataclass
class Plan:
    entry: Optional[float] = 100.0
    stop: Optional[float] = 97.0
    target: Optional[float] = 109.0
    current_price: Optional[float] = 101.0
    entry_plan: object = None
    level: object = None


@dataclass
class Zone:
    zone_low: float
    zone_high: float


@dataclass
class Lvl:
    price: float
    touches: int
    kind: str


class TestLevels(unittest.TestCase):
    def test_entry_stop_target_and_price_are_drawn(self):
        labels = [l.label for l in levels_for(Plan())]
        self.assertEqual(labels, ["Now", "Entry", "Stop", "Target"])

    def test_colours_carry_the_usual_meaning(self):
        by = {l.label: l.colour for l in levels_for(Plan())}
        self.assertEqual(by["Target"], GREEN)
        self.assertEqual(by["Stop"], RED)
        self.assertEqual(by["Entry"], AMBER)
        self.assertEqual(by["Now"], BLUE)

    def test_labels_include_the_price_in_the_chart(self):
        spec = chart_spec(candles(), levels_for(Plan()))
        texts = [l["data"]["values"][0]["label"] for l in spec["layer"]
                 if l.get("mark", {}).get("type") == "text"]
        self.assertTrue(any("Entry 100" in t for t in texts))

    def test_incomplete_plan_draws_what_it_has(self):
        labels = [l.label for l in levels_for(Plan(target=None, stop=None))]
        self.assertEqual(labels, ["Now", "Entry"])

    def test_swing_level_is_drawn_with_its_touch_count(self):
        labels = [l.label for l in levels_for(Plan(level=Lvl(95.0, 3, "support")))]
        self.assertIn("Support ×3", labels)

    def test_no_plan_draws_nothing(self):
        self.assertEqual(levels_for(None), [])


class TestEntryZone(unittest.TestCase):
    def test_zone_extracted_and_ordered(self):
        z = entry_zone_of(Plan(entry_plan=Zone(101.0, 99.0)))
        self.assertEqual(z, {"low": 99.0, "high": 101.0})

    def test_no_zone(self):
        self.assertIsNone(entry_zone_of(Plan()))
        self.assertIsNone(entry_zone_of(None))


class TestChartSpec(unittest.TestCase):
    def test_has_candles(self):
        spec = chart_spec(candles(30))
        self.assertEqual(len(spec["data"]["values"]), 30)
        self.assertIn("open", spec["data"]["values"][0])

    def test_limits_how_many_bars_are_drawn(self):
        spec = chart_spec(candles(500), max_bars=120)
        self.assertEqual(len(spec["data"]["values"]), 120)

    def test_up_and_down_candles_are_marked(self):
        idx = pd.date_range("2026-06-01", periods=2, freq="4h", tz="UTC")
        df = pd.DataFrame({"Open": [100, 105], "High": [106, 106], "Low": [99, 99],
                            "Close": [105, 100]}, index=idx)
        vals = chart_spec(df)["data"]["values"]
        self.assertTrue(vals[0]["up"])
        self.assertFalse(vals[1]["up"])

    def test_each_level_adds_a_line_and_a_label(self):
        base = len(chart_spec(candles())["layer"])
        with_levels = len(chart_spec(candles(), [ChartLevel("Entry", 100.0, AMBER)])["layer"])
        self.assertEqual(with_levels, base + 2)

    def test_zone_is_shaded_behind_the_candles(self):
        spec = chart_spec(candles(), zone={"low": 99.0, "high": 101.0})
        self.assertEqual(spec["layer"][0]["mark"]["type"], "rect")

    def test_price_axis_does_not_force_zero(self):
        spec = chart_spec(candles())
        wick = next(l for l in spec["layer"] if l.get("mark", {}).get("type") == "rule"
                    and "y2" in l.get("encoding", {}))
        self.assertFalse(wick["encoding"]["y"]["scale"]["zero"])

    def test_empty_data_is_safe(self):
        spec = chart_spec(pd.DataFrame())
        self.assertEqual(spec["data"]["values"], [])

    def test_dashed_levels_are_dashed(self):
        spec = chart_spec(candles(), [ChartLevel("Now", 100.0, BLUE, dash=True)])
        rule = next(l for l in spec["layer"]
                    if l.get("mark", {}).get("color") == BLUE
                    and l["mark"]["type"] == "rule")
        self.assertEqual(rule["mark"]["strokeDash"], [6, 4])


def zigzag_candles(n_cycles=5, up=8.0, down=3.0, start=100.0):
    closes, p = [start], start
    for _ in range(n_cycles):
        closes += list(pd.Series(range(8)).apply(lambda i: p + up * (i + 1) / 8))[:]
        p += up
        closes += list(pd.Series(range(5)).apply(lambda i: p - down * (i + 1) / 5))[:]
        p -= down
    idx = pd.date_range("2026-05-01", periods=len(closes), freq="4h", tz="UTC")
    o = [closes[0]] + list(closes[:-1])
    nudge = [1e-6 * i for i in range(len(closes))]
    return pd.DataFrame({"Open": o,
                          "High": [max(a, b) + 0.4 + d for a, b, d in zip(o, closes, nudge)],
                          "Low": [min(a, b) - 0.4 - d for a, b, d in zip(o, closes, nudge)],
                          "Close": closes}, index=idx)


class TestTrendLineAndMarkers(unittest.TestCase):
    def test_trend_line_connects_two_swing_lows_for_a_long(self):
        from strategy_chart import trend_line
        pts = trend_line(zigzag_candles(), "Long")
        self.assertEqual(len(pts), 2)
        self.assertLess(pts[0]["line"], pts[1]["line"], "an uptrend line should rise")

    def test_trend_line_uses_highs_for_a_short(self):
        from strategy_chart import trend_line
        pts = trend_line(zigzag_candles(), "Short")
        self.assertEqual(len(pts), 2)

    def test_no_trend_line_without_two_swings(self):
        from strategy_chart import trend_line
        flat = candles(12)
        self.assertEqual(trend_line(flat, "Long"), [])

    def test_no_trend_line_without_a_direction(self):
        from strategy_chart import trend_line
        self.assertEqual(trend_line(zigzag_candles(), None), [])

    def test_swing_markers_found(self):
        from strategy_chart import swing_markers
        marks = swing_markers(zigzag_candles())
        self.assertTrue(marks)
        self.assertTrue({"low", "high"} & {m["kind"] for m in marks})

    def test_moving_average_points(self):
        from strategy_chart import moving_average
        ma = moving_average(candles(60), 20)
        self.assertEqual(len(ma), 41)
        self.assertIn("ma", ma[0])

    def test_moving_average_needs_enough_bars(self):
        from strategy_chart import moving_average
        self.assertEqual(moving_average(candles(5), 20), [])


class TestSpecIncludesStrategyLayers(unittest.TestCase):
    def test_trend_line_layer_present(self):
        spec = chart_spec(zigzag_candles(), direction="Long")
        dashes = [l for l in spec["layer"]
                  if l.get("mark", {}).get("type") == "line"
                  and l["mark"].get("strokeDash")]
        self.assertTrue(dashes, "expected a dashed trend line")

    def test_moving_average_layer_present(self):
        spec = chart_spec(zigzag_candles(), sma_period=20)
        lines = [l for l in spec["layer"] if l.get("mark", {}).get("type") == "line"]
        self.assertTrue(lines)

    def test_swing_markers_layer_present(self):
        spec = chart_spec(zigzag_candles(), direction="Long")
        points = [l for l in spec["layer"] if l.get("mark", {}).get("type") == "point"]
        self.assertTrue(points)

    def test_layers_can_be_switched_off(self):
        spec = chart_spec(zigzag_candles(), direction="Long", show_swings=False,
                          show_trend_line=False)
        self.assertFalse([l for l in spec["layer"]
                          if l.get("mark", {}).get("type") == "point"])

    def test_still_safe_with_no_data(self):
        spec = chart_spec(pd.DataFrame(), direction="Long", sma_period=20)
        self.assertEqual(spec["data"]["values"], [])
