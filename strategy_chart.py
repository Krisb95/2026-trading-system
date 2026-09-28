"""
The chart the strategy sees, with its own levels drawn on it.

Built as a Vega-Lite spec, which ships inside Streamlit — no charting library to
install and nothing that can fail on deploy. The spec is a plain dictionary, so
the levels can be tested without rendering anything.

It draws the candles the strategy actually read, not a separate feed, so what
you see is what the rules were applied to. Where there is no setup, the trend
line, swing points and moving average are still drawn: the chart has a
direction whether or not it has a trade.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

GREEN = "#00E5A0"
RED = "#FF5C7A"
AMBER = "#FFB65C"
BLUE = "#5CC8FF"
GREY = "#8A93AB"
PURPLE = "#B58CFF"


@dataclass
class ChartLevel:
    label: str
    price: float
    colour: str
    dash: bool = False


def levels_for(plan) -> List[ChartLevel]:
    """The lines to draw for a plan from any of the strategies."""
    out: List[ChartLevel] = []
    if plan is None:
        return out
    price = getattr(plan, "current_price", None) or getattr(plan, "price", None)
    if price:
        out.append(ChartLevel("Now", float(price), BLUE, dash=True))
    for label, attr, colour in (("Entry", "entry", AMBER), ("Stop", "stop", RED),
                                 ("Target", "target", GREEN)):
        value = getattr(plan, attr, None)
        if value:
            out.append(ChartLevel(label, float(value), colour))
    level = getattr(plan, "level", None)
    if level is not None and getattr(level, "price", None):
        out.append(ChartLevel(f"{level.kind.title()} ×{level.touches}",
                               float(level.price), GREY, dash=True))
    return out


def entry_zone_of(plan) -> Optional[Dict[str, float]]:
    zone = getattr(plan, "entry_plan", None)
    if zone is None:
        return None
    low, high = getattr(zone, "zone_low", None), getattr(zone, "zone_high", None)
    if low is None or high is None:
        return None
    return {"low": float(min(low, high)), "high": float(max(low, high))}


def moving_average(df: pd.DataFrame, period: int) -> List[Dict]:
    if df is None or len(df) < period:
        return []
    ma = df["Close"].rolling(period).mean()
    return [{"t": ts.isoformat(), "ma": float(v)} for ts, v in ma.items() if pd.notna(v)]


def infer_direction(df: pd.DataFrame, left: int = 3, right: int = 3) -> Optional[str]:
    """Which way the chart is trending, from its own swing points.

    Needed because most of the time there is no setup, and a trend line that
    only appears alongside a trade would almost never appear.
    """
    from technical import find_swing_points
    if df is None or len(df) < left + right + 4:
        return None
    swings = [s for s in find_swing_points(df, left, right) if s.confirmed]
    lows = [s.price for s in swings if s.kind == "low"][-2:]
    highs = [s.price for s in swings if s.kind == "high"][-2:]
    if len(lows) == 2 and lows[1] > lows[0]:
        return "Long"
    if len(highs) == 2 and highs[1] < highs[0]:
        return "Short"
    if len(lows) == 2 and lows[1] < lows[0]:
        return "Short"
    if len(highs) == 2 and highs[1] > highs[0]:
        return "Long"
    return None


def trend_line(df: pd.DataFrame, direction: Optional[str], left: int = 3,
               right: int = 3) -> List[Dict]:
    """A line through the last two confirmed swings, out to the right edge.

    Through swing LOWS in an uptrend and HIGHS in a downtrend — the points you
    would join by hand. With fewer than two, nothing is drawn rather than a
    guess, because an unconfirmed swing moves with every candle.
    """
    from technical import find_swing_points
    if df is None or len(df) < left + right + 2 or direction not in ("Long", "Short"):
        return []
    swings = [s for s in find_swing_points(df, left, right) if s.confirmed]
    kind = "low" if direction == "Long" else "high"
    picked = [s for s in swings if s.kind == kind][-2:]
    if len(picked) < 2 or picked[1].index == picked[0].index:
        return []
    a, b = picked
    slope = (b.price - a.price) / (b.index - a.index)
    last = len(df) - 1
    return [{"t": df.index[a.index].isoformat(), "line": float(a.price)},
            {"t": df.index[last].isoformat(),
             "line": float(b.price + slope * (last - b.index))}]


def swing_markers(df: pd.DataFrame, left: int = 3, right: int = 3,
                  limit: int = 12) -> List[Dict]:
    from technical import find_swing_points
    if df is None or len(df) < left + right + 1:
        return []
    swings = [s for s in find_swing_points(df, left, right) if s.confirmed][-limit:]
    return [{"t": df.index[s.index].isoformat(), "swing": float(s.price), "kind": s.kind}
            for s in swings]


def _rows(df: pd.DataFrame, max_bars: int) -> List[Dict]:
    return [{"t": ts.isoformat(), "open": float(r["Open"]), "high": float(r["High"]),
             "low": float(r["Low"]), "close": float(r["Close"]),
             "up": bool(r["Close"] >= r["Open"])}
            for ts, r in df.tail(max_bars).iterrows()]


def chart_spec(df: pd.DataFrame, levels: Optional[List[ChartLevel]] = None,
               zone: Optional[Dict[str, float]] = None, max_bars: int = 140,
               height: int = 360, direction: Optional[str] = None,
               sma_period: Optional[int] = 20, show_swings: bool = True,
               show_trend_line: bool = True) -> Dict:
    """Candlesticks with the strategy's levels, trend line and swing points."""
    levels = levels or []
    view = df.tail(max_bars) if df is not None and not df.empty else df
    candles = _rows(df, max_bars) if df is not None and not df.empty else []
    layers: List[Dict] = []

    if zone and candles:
        layers.append({
            "data": {"values": [{"low": zone["low"], "high": zone["high"]}]},
            "mark": {"type": "rect", "opacity": 0.12, "color": AMBER},
            "encoding": {"y": {"field": "low", "type": "quantitative"},
                          "y2": {"field": "high"}}})

    colour = {"field": "up", "type": "nominal", "legend": None,
              "scale": {"domain": [True, False], "range": [GREEN, RED]}}
    layers.append({
        "mark": {"type": "rule"},
        "encoding": {"x": {"field": "t", "type": "temporal", "title": None},
                      "y": {"field": "low", "type": "quantitative",
                            "scale": {"zero": False}, "title": None},
                      "y2": {"field": "high"}, "color": colour}})
    layers.append({
        "mark": {"type": "bar"},
        "encoding": {"x": {"field": "t", "type": "temporal"},
                      "y": {"field": "open", "type": "quantitative"},
                      "y2": {"field": "close"}, "color": colour,
                      "tooltip": [{"field": "t", "type": "temporal", "title": "Time"},
                                   {"field": "open", "type": "quantitative"},
                                   {"field": "high", "type": "quantitative"},
                                   {"field": "low", "type": "quantitative"},
                                   {"field": "close", "type": "quantitative"}]}})

    if sma_period and view is not None and not view.empty:
        ma = moving_average(view, sma_period)
        if ma:
            layers.append({
                "data": {"values": ma},
                "mark": {"type": "line", "color": BLUE, "opacity": 0.75, "size": 1.5},
                "encoding": {"x": {"field": "t", "type": "temporal"},
                              "y": {"field": "ma", "type": "quantitative"}}})

    if show_trend_line and view is not None and not view.empty:
        tl = trend_line(view, direction or infer_direction(view))
        if tl:
            layers.append({
                "data": {"values": tl},
                "mark": {"type": "line", "color": PURPLE, "size": 1.8,
                          "strokeDash": [5, 3]},
                "encoding": {"x": {"field": "t", "type": "temporal"},
                              "y": {"field": "line", "type": "quantitative"}}})

    if show_swings and view is not None and not view.empty:
        marks = swing_markers(view)
        if marks:
            layers.append({
                "data": {"values": marks},
                "mark": {"type": "point", "filled": True, "size": 30, "opacity": 0.85},
                "encoding": {"x": {"field": "t", "type": "temporal"},
                              "y": {"field": "swing", "type": "quantitative"},
                              "color": {"field": "kind", "type": "nominal", "legend": None,
                                         "scale": {"domain": ["low", "high"],
                                                    "range": [GREEN, RED]}}}})

    for lv in levels:
        point = [{"price": lv.price, "label": f"{lv.label} {lv.price:,.6g}"}]
        layers.append({
            "data": {"values": point},
            "mark": {"type": "rule", "color": lv.colour, "size": 1.6,
                      "strokeDash": [6, 4] if lv.dash else [0]},
            "encoding": {"y": {"field": "price", "type": "quantitative"}}})
        layers.append({
            "data": {"values": point},
            "mark": {"type": "text", "align": "left", "dx": 4, "dy": -6,
                      "color": lv.colour, "fontSize": 11},
            "encoding": {"y": {"field": "price", "type": "quantitative"},
                          "text": {"field": "label", "type": "nominal"}}})

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": candles},
        "params": [{"name": "grid",
                     "select": {"type": "interval", "encodings": ["x", "y"]},
                     "bind": "scales"}],
        "layer": layers,
        "height": height,
        "autosize": {"type": "fit", "contains": "padding"},
        "config": {"background": "transparent",
                    "view": {"stroke": "transparent"},
                    "axis": {"labelColor": "#9FB0D0",
                             "gridColor": "rgba(255,255,255,.06)",
                             "domainColor": "rgba(255,255,255,.15)"}},
    }
