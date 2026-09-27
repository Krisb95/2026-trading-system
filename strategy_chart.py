"""
A chart that draws the strategy's own levels, automatically.

WHY NOT DRAW ON THE TRADINGVIEW CHART: the free embedded widget has no API for
placing lines on it — programmatic drawing needs their paid Charting Library
with a self-hosted data feed. So TradingView stays for manual drawing, and this
draws what the strategy actually calculated.

WHY VEGA-LITE: it ships inside Streamlit, so there is no extra package to
install and nothing that can break on deploy. The chart is a plain dictionary,
which also means the levels can be tested without rendering anything.

WHAT IT SHOWS: the candles the strategy read, with its entry, stop and target,
the level the plan is built on, and the entry zone where one exists. Every line
is labelled with what it is and its price, so nothing depends on remembering
which colour means what.
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

UP = GREEN
DOWN = RED


@dataclass
class ChartLevel:
    label: str
    price: float
    colour: str
    dash: bool = False


def levels_for(plan) -> List[ChartLevel]:
    """The lines to draw for a plan from any of the strategies.

    Reads whatever the plan happens to expose, so a Trend Retrace plan, a Swing
    plan or a scanner row all work without special-casing each one.
    """
    out: List[ChartLevel] = []
    if plan is None:
        return out
    entry = getattr(plan, "entry", None)
    stop = getattr(plan, "stop", None)
    target = getattr(plan, "target", None)
    price = getattr(plan, "current_price", None) or getattr(plan, "price", None)

    if price:
        out.append(ChartLevel("Now", float(price), BLUE, dash=True))
    if entry:
        out.append(ChartLevel("Entry", float(entry), AMBER))
    if stop:
        out.append(ChartLevel("Stop", float(stop), RED))
    if target:
        out.append(ChartLevel("Target", float(target), GREEN))

    level = getattr(plan, "level", None)          # swing plans carry the level
    if level is not None and getattr(level, "price", None):
        out.append(ChartLevel(f"{level.kind.title()} ×{level.touches}",
                               float(level.price), GREY, dash=True))
    return out


def entry_zone_of(plan) -> Optional[Dict[str, float]]:
    """The shaded entry band, when the plan has one."""
    zone = getattr(plan, "entry_plan", None)
    if zone is None:
        return None
    low, high = getattr(zone, "zone_low", None), getattr(zone, "zone_high", None)
    if low is None or high is None:
        return None
    return {"low": float(min(low, high)), "high": float(max(low, high))}


def moving_average(df: pd.DataFrame, period: int) -> List[Dict]:
    """Points for a simple moving average line."""
    if df is None or len(df) < period:
        return []
    ma = df["Close"].rolling(period).mean()
    return [{"t": ts.isoformat(), "ma": float(v)}
            for ts, v in ma.items() if pd.notna(v)]


def trend_line(df: pd.DataFrame, direction: Optional[str], left: int = 3,
               right: int = 3) -> List[Dict]:
    """A line through the last two confirmed swings, extended to the right edge.

    Drawn through swing LOWS in an uptrend and swing HIGHS in a downtrend —
    the same points you would connect by hand. Needs two confirmed swings of
    the right kind; with fewer, no line is drawn rather than a guessed one.
    """
    from technical import find_swing_points
    if df is None or len(df) < left + right + 2 or direction not in ("Long", "Short"):
        return []
    swings = [s for s in find_swing_points(df, left, right) if s.confirmed]
    kind = "low" if direction == "Long" else "high"
    picked = [s for s in swings if s.kind == kind][-2:]
    if len(picked) < 2:
        return []
    a, b = picked
    if b.index == a.index:
        return []
    slope = (b.price - a.price) / (b.index - a.index)
    last = len(df) - 1
    end_price = b.price + slope * (last - b.index)
    return [{"t": df.index[a.index].isoformat(), "line": float(a.price)},
            {"t": df.index[last].isoformat(), "line": float(end_price)}]


def swing_markers(df: pd.DataFrame, left: int = 3, right: int = 3,
                  limit: int = 12) -> List[Dict]:
    """The confirmed swing highs and lows the strategy reads."""
    from technical import find_swing_points
    if df is None or len(df) < left + right + 1:
        return []
    swings = [s for s in find_swing_points(df, left, right) if s.confirmed][-limit:]
    return [{"t": df.index[s.index].isoformat(), "swing": float(s.price),
             "kind": s.kind} for s in swings]


def _rows(df: pd.DataFrame, max_bars: int) -> List[Dict]:
    tail = df.tail(max_bars)
    return [{"t": ts.isoformat(),
             "open": float(r["Open"]), "high": float(r["High"]),
             "low": float(r["Low"]), "close": float(r["Close"]),
             "up": bool(r["Close"] >= r["Open"])}
            for ts, r in tail.iterrows()]


def chart_spec(df: pd.DataFrame, levels: Optional[List[ChartLevel]] = None,
               zone: Optional[Dict[str, float]] = None, max_bars: int = 180,
               height: int = 420, direction: Optional[str] = None,
               sma_period: Optional[int] = None, show_swings: bool = True,
               show_trend_line: bool = True) -> Dict:
    """A candlestick chart with everything the strategy looked at drawn on it:
    its levels, the swing points it read, a trend line through them, and the
    moving average it used for the trend."""
    levels = levels or []
    view = df.tail(max_bars) if df is not None and not df.empty else df
    candles = _rows(df, max_bars) if df is not None and not df.empty else []

    layers: List[Dict] = []

    if zone and candles:
        layers.append({
            "data": {"values": [{"low": zone["low"], "high": zone["high"]}]},
            "mark": {"type": "rect", "opacity": 0.12, "color": AMBER},
            "encoding": {"y": {"field": "low", "type": "quantitative"},
                          "y2": {"field": "high"}},
        })

    # Wicks, then bodies — two marks over the same data.
    layers.append({
        "mark": {"type": "rule"},
        "encoding": {
            "x": {"field": "t", "type": "temporal", "title": None},
            "y": {"field": "low", "type": "quantitative", "scale": {"zero": False},
                   "title": None},
            "y2": {"field": "high"},
            "color": {"field": "up", "type": "nominal", "legend": None,
                       "scale": {"domain": [True, False], "range": [UP, DOWN]}},
        },
    })
    layers.append({
        "mark": {"type": "bar"},
        "encoding": {
            "x": {"field": "t", "type": "temporal"},
            "y": {"field": "open", "type": "quantitative"},
            "y2": {"field": "close"},
            "color": {"field": "up", "type": "nominal", "legend": None,
                       "scale": {"domain": [True, False], "range": [UP, DOWN]}},
            "tooltip": [{"field": "t", "type": "temporal", "title": "Time"},
                         {"field": "open", "type": "quantitative"},
                         {"field": "high", "type": "quantitative"},
                         {"field": "low", "type": "quantitative"},
                         {"field": "close", "type": "quantitative"}],
        },
    })

    if sma_period and view is not None and not view.empty:
        ma = moving_average(view, sma_period)
        if ma:
            layers.append({
                "data": {"values": ma},
                "mark": {"type": "line", "color": BLUE, "opacity": 0.75, "size": 1.5},
                "encoding": {"x": {"field": "t", "type": "temporal"},
                              "y": {"field": "ma", "type": "quantitative"}},
            })

    if show_trend_line and view is not None and not view.empty:
        tl = trend_line(view, direction)
        if tl:
            layers.append({
                "data": {"values": tl},
                "mark": {"type": "line", "color": PURPLE, "size": 1.8,
                          "strokeDash": [5, 3]},
                "encoding": {"x": {"field": "t", "type": "temporal"},
                              "y": {"field": "line", "type": "quantitative"}},
            })

    if show_swings and view is not None and not view.empty:
        marks = swing_markers(view)
        if marks:
            layers.append({
                "data": {"values": marks},
                "mark": {"type": "point", "filled": True, "size": 34, "opacity": 0.85},
                "encoding": {
                    "x": {"field": "t", "type": "temporal"},
                    "y": {"field": "swing", "type": "quantitative"},
                    "color": {"field": "kind", "type": "nominal", "legend": None,
                               "scale": {"domain": ["low", "high"],
                                          "range": [GREEN, RED]}},
                    "tooltip": [{"field": "kind", "type": "nominal", "title": "Swing"},
                                 {"field": "swing", "type": "quantitative"}],
                },
            })

    for lv in levels:
        layers.append({
            "data": {"values": [{"price": lv.price, "label": f"{lv.label} {lv.price:,.6g}"}]},
            "mark": {"type": "rule", "color": lv.colour, "size": 1.6,
                      "strokeDash": [6, 4] if lv.dash else [0]},
            "encoding": {"y": {"field": "price", "type": "quantitative"}},
        })
        layers.append({
            "data": {"values": [{"price": lv.price, "label": f"{lv.label} {lv.price:,.6g}"}]},
            "mark": {"type": "text", "align": "left", "dx": 4, "dy": -6,
                      "color": lv.colour, "fontSize": 11},
            "encoding": {"y": {"field": "price", "type": "quantitative"},
                          "text": {"field": "label", "type": "nominal"}},
        })

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": candles},
        "layer": layers,
        "height": height,
        "width": "container",
        "config": {"background": "transparent",
                    "view": {"stroke": "transparent"},
                    "axis": {"labelColor": "#9FB0D0", "gridColor": "rgba(255,255,255,.06)",
                             "domainColor": "rgba(255,255,255,.15)"}},
    }
