"""
Automated candidate analysis: derives the scoring evidence and entry-sequence
evidence from ACTUAL multi-timeframe price data, so the user doesn't have to
tick every box by hand.

HONEST SCOPE NOTE — read this before trusting the output:

This is rule-based structural analysis computed from real OHLC candles. It is
NOT discretionary chart reading and it is NOT an oracle. Specifically:

  * "Liquidity" here means equal-highs/equal-lows clustering and prior swing
    levels. There is no order-book, volume-profile or order-flow data behind
    it, so a "sweep" is a close-based rule, not observed liquidation activity.
  * 4H candles are RESAMPLED from 1H data, because the provider does not serve
    a native 4H interval. Resampled bars can differ slightly from an exchange's
    own 4H candles (session boundary alignment).
  * Every derived level is computed from confirmed swing points only — nothing
    is invented — but a rule that says "this is support" is still a heuristic,
    not a fact about where other traders will act.
  * Auto-derived evidence is a STARTING POINT. Every field is overridable in
    the UI, and disagreement between you and the scanner is information, not
    an error.

Each piece of evidence returned carries a plain-English reason so you can audit
why it was set, rather than trusting a bare True/False.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from technical import (find_swing_points, label_structure, find_equal_levels,
                        fib_levels, SwingPoint)


@dataclass
class EvidenceItem:
    value: Optional[bool]   # True / False / None (couldn't evaluate)
    reason: str             # why it was set this way


@dataclass
class CandidateAnalysis:
    ticker: str
    direction: Optional[str]                  # suggested "Long" / "Short" / None
    regime_1d: str                            # "bullish" / "bearish" / "ranging" / "unknown"
    current_price: Optional[float]
    entry: Optional[float]
    stop: Optional[float]
    target: Optional[float]
    reward_risk: Optional[float]
    score_evidence: Dict[str, EvidenceItem] = field(default_factory=dict)
    sequence_evidence: Dict[str, EvidenceItem] = field(default_factory=dict)
    key_levels: Dict[str, float] = field(default_factory=dict)
    data_problems: List[str] = field(default_factory=list)

    def score_dict(self) -> Dict[str, Optional[bool]]:
        return {k: v.value for k, v in self.score_evidence.items()}

    def sequence_dict(self) -> Dict[str, bool]:
        return {k: bool(v.value) for k, v in self.sequence_evidence.items()}


# ---------------------------------------------------------------------
# Data fetching / resampling
# ---------------------------------------------------------------------

def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Build 4H candles from 1H data. The provider has no native 4H interval,
    so this is a genuine resample — noted in the scope warning above."""
    if df_1h is None or df_1h.empty:
        return pd.DataFrame()
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df_1h.columns:
        agg["Volume"] = "sum"
    out = df_1h.resample("4h").agg(agg).dropna(subset=["Close"])
    return out


def fetch_multi_timeframe(ticker: str, yf_module, ticker_builder=None) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Fetch 1D / 1H (and derive 4H). Returns (frames, problems).

    `ticker_builder` lets the caller inject a session-aware Ticker factory
    (and lets tests inject a fake) instead of hitting the network.
    """
    problems: List[str] = []
    frames: Dict[str, pd.DataFrame] = {}

    def _build(sym):
        if ticker_builder is not None:
            return ticker_builder(sym)
        return yf_module.Ticker(sym)

    specs = [("1d", "1y", "1d"), ("1h", "60d", "1h")]
    for label, period, interval in specs:
        try:
            t = _build(ticker)
            try:
                df = t.history(period=period, interval=interval)
            except TypeError:
                df = t.history(period=period)
            if df is None or df.empty:
                problems.append(f"No {label} data returned.")
                frames[label] = pd.DataFrame()
                continue
            if "Close" in df.columns:
                df = df[df["Close"].notna()]
            frames[label] = df
        except Exception as e:
            problems.append(f"{label} fetch failed: {type(e).__name__}: {e}")
            frames[label] = pd.DataFrame()

    frames["4h"] = resample_to_4h(frames.get("1h", pd.DataFrame()))
    if frames["4h"].empty and not frames.get("1h", pd.DataFrame()).empty:
        problems.append("Could not resample 1H data into 4H candles.")

    return frames, problems


# ---------------------------------------------------------------------
# Analysis primitives
# ---------------------------------------------------------------------

def detect_regime(df_1d: pd.DataFrame) -> Tuple[str, str]:
    """1D regime from moving-average alignment plus swing structure.
    Returns (regime, reason)."""
    if df_1d is None or len(df_1d) < 60:
        return "unknown", "Not enough daily history to judge regime (need ~60 bars)."

    close = df_1d["Close"]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    price = float(close.iloc[-1])

    swings = find_swing_points(df_1d, left=3, right=3)
    labeled = label_structure(swings)
    recent = [l for l in labeled if l.label][-4:]
    bullish_structure = sum(1 for l in recent if l.label in ("HH", "HL"))
    bearish_structure = sum(1 for l in recent if l.label in ("LH", "LL"))

    ma_bullish = price > sma20 > sma50
    ma_bearish = price < sma20 < sma50

    if ma_bullish and bullish_structure >= bearish_structure:
        return "bullish", (f"Price {price:.4g} above SMA20 {sma20:.4g} above SMA50 {sma50:.4g}, "
                            f"with {bullish_structure}/{len(recent)} recent swings higher.")
    if ma_bearish and bearish_structure >= bullish_structure:
        return "bearish", (f"Price {price:.4g} below SMA20 {sma20:.4g} below SMA50 {sma50:.4g}, "
                            f"with {bearish_structure}/{len(recent)} recent swings lower.")
    return "ranging", (f"No clean MA stack (price {price:.4g}, SMA20 {sma20:.4g}, SMA50 {sma50:.4g}) "
                        f"or mixed swing structure — treat as ranging/transitioning.")


def structure_direction(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """Judge whether a timeframe's swing structure supports Long or Short."""
    if df is None or len(df) < 20:
        return None, "Not enough bars to read structure."
    swings = find_swing_points(df, left=2, right=2)
    labeled = [l for l in label_structure(swings) if l.label]
    if len(labeled) < 2:
        return None, "Too few labelled swings to read structure."
    recent = labeled[-4:]
    up = sum(1 for l in recent if l.label in ("HH", "HL"))
    down = sum(1 for l in recent if l.label in ("LH", "LL"))
    seq = ", ".join(l.label for l in recent)
    if up > down:
        return "Long", f"Recent swing sequence [{seq}] favours longs."
    if down > up:
        return "Short", f"Recent swing sequence [{seq}] favours shorts."
    return None, f"Recent swing sequence [{seq}] is mixed — no structural bias."


def nearest_level(df: pd.DataFrame, price: float, kind: str,
                   tolerance_pct: float = 0.02) -> Optional[Tuple[float, int]]:
    """Nearest confirmed swing level of `kind` within tolerance of price.
    Returns (level_price, touch_count) or None."""
    if df is None or len(df) < 10 or price <= 0:
        return None
    swings = [s for s in find_swing_points(df, left=2, right=2) if s.confirmed and s.kind == kind]
    if not swings:
        return None
    clusters = find_equal_levels(swings, tolerance_pct=0.003)
    candidates = [(c["price_avg"], c["touches"]) for c in clusters if c["kind"] == kind]
    candidates += [(s.price, 1) for s in swings]
    in_range = [(lvl, touches) for lvl, touches in candidates
                if abs(lvl - price) / price <= tolerance_pct]
    if not in_range:
        return None
    return min(in_range, key=lambda x: abs(x[0] - price))


def detect_sweep_and_reclaim(df: pd.DataFrame, direction: str,
                              lookback: int = 30) -> Tuple[Optional[bool], str]:
    """Close-based sweep + reclaim detection.

    For a Long: a recent bar's LOW pierced a prior confirmed swing low, but the
    bar CLOSED back above it (wick-only rejection is explicitly not enough on
    its own — the close must reclaim). Inverse for a Short.
    """
    if df is None or len(df) < lookback + 5:
        return None, "Not enough bars to evaluate a sweep/reclaim."

    window = df.iloc[-lookback:]
    prior = df.iloc[:-lookback]
    if prior.empty:
        return None, "No prior history to define the swept level."

    swings = [s for s in find_swing_points(prior, left=2, right=2) if s.confirmed]
    if direction == "Long":
        levels = [s.price for s in swings if s.kind == "low"]
        if not levels:
            return None, "No confirmed prior swing low to sweep."
        level = max(levels)  # most recent relevant support region
        pierced = window["Low"].min() < level
        reclaimed = float(window["Close"].iloc[-1]) > level
        if pierced and reclaimed:
            return True, (f"Low {window['Low'].min():.4g} pierced prior swing low {level:.4g}, "
                           f"and price closed back above at {window['Close'].iloc[-1]:.4g} — reclaim confirmed.")
        if pierced and not reclaimed:
            return False, (f"Level {level:.4g} was pierced but price has NOT closed back above it "
                            f"(last close {window['Close'].iloc[-1]:.4g}) — no reclaim, so no points.")
        return False, f"No sweep of prior swing low {level:.4g} in the last {lookback} bars."
    else:
        levels = [s.price for s in swings if s.kind == "high"]
        if not levels:
            return None, "No confirmed prior swing high to sweep."
        level = min(levels)
        pierced = window["High"].max() > level
        rejected = float(window["Close"].iloc[-1]) < level
        if pierced and rejected:
            return True, (f"High {window['High'].max():.4g} pierced prior swing high {level:.4g}, "
                           f"and price closed back below at {window['Close'].iloc[-1]:.4g} — rejection confirmed.")
        if pierced and not rejected:
            return False, (f"Level {level:.4g} was pierced but price has NOT closed back below it "
                            f"(last close {window['Close'].iloc[-1]:.4g}) — no rejection, so no points.")
        return False, f"No sweep of prior swing high {level:.4g} in the last {lookback} bars."


def fib_confluence(df: pd.DataFrame, price: float, direction: str,
                    tolerance_pct: float = 0.015) -> Tuple[Optional[bool], str, Dict[str, float]]:
    """Check whether price sits near a Fibonacci level drawn between the two
    most recent confirmed opposing swings."""
    if df is None or len(df) < 20 or price <= 0:
        return None, "Not enough data for Fibonacci anchors.", {}
    swings = [s for s in find_swing_points(df, left=2, right=2) if s.confirmed]
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if not highs or not lows:
        return None, "No confirmed swing high/low pair to anchor Fibonacci.", {}

    last_high = max(highs, key=lambda s: s.index)
    last_low = max(lows, key=lambda s: s.index)
    if last_high.price <= last_low.price:
        return None, "Swing anchors are degenerate (high not above low).", {}

    try:
        levels = fib_levels(last_low.price, last_high.price, direction)
    except ValueError as e:
        return None, f"Could not compute Fibonacci levels: {e}", {}

    near = [(name, lvl) for name, lvl in levels.items()
            if lvl > 0 and abs(lvl - price) / price <= tolerance_pct]
    if near:
        name, lvl = min(near, key=lambda x: abs(x[1] - price))
        return True, (f"Price {price:.4g} is within {tolerance_pct:.1%} of {name} at {lvl:.4g} "
                       f"(anchored on swing low {last_low.price:.4g} / high {last_high.price:.4g})."), levels
    return False, (f"Price {price:.4g} is not near any Fibonacci level from the latest confirmed "
                    f"swing pair ({last_low.price:.4g} → {last_high.price:.4g})."), levels


def is_extended(df: pd.DataFrame, price: float, atr_mult: float = 2.5,
                 period: int = 14) -> Tuple[Optional[bool], str]:
    """Flag 'chasing an extended move': price stretched far from its 20-SMA in
    ATR terms, i.e. entering after the move rather than at a location."""
    if df is None or len(df) < period + 20:
        return None, "Not enough bars to judge extension."
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
                    axis=1).max(axis=1)
    atr = float(tr.rolling(period).mean().iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    if atr <= 0 or not np.isfinite(atr):
        return None, "ATR unavailable — cannot judge extension."
    distance_atr = abs(price - sma20) / atr
    if distance_atr >= atr_mult:
        return True, (f"Price is {distance_atr:.1f} ATR from its 20-SMA ({sma20:.4g}) — "
                       f"that is an extended move; entering here is chasing.")
    return False, f"Price is {distance_atr:.1f} ATR from its 20-SMA — not extended."


def opposing_liquidity_ahead(df: pd.DataFrame, price: float, target: Optional[float],
                              direction: str) -> Tuple[Optional[bool], str]:
    """Is there a clustered equal-high/low pool between price and the target?"""
    if df is None or target is None or price <= 0:
        return None, "No target defined — cannot check for liquidity in the path."
    swings = [s for s in find_swing_points(df, left=2, right=2) if s.confirmed]
    clusters = find_equal_levels(swings, tolerance_pct=0.003)
    if not clusters:
        return False, "No clustered equal highs/lows detected in the path to target."

    if direction == "Long":
        blocking = [c for c in clusters if c["kind"] == "high" and price < c["price_avg"] < target]
    else:
        blocking = [c for c in clusters if c["kind"] == "low" and target < c["price_avg"] < price]

    if blocking:
        nearest = min(blocking, key=lambda c: abs(c["price_avg"] - price))
        return True, (f"A cluster of {nearest['touches']} equal {nearest['kind']}s sits at "
                       f"{nearest['price_avg']:.4g}, between entry and target — price may stall there.")
    return False, "No clustered opposing liquidity between entry and target."


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def analyze_candidate(ticker: str, frames: Dict[str, pd.DataFrame],
                       direction_override: Optional[str] = None,
                       min_rr: float = 2.0,
                       data_problems: Optional[List[str]] = None) -> CandidateAnalysis:
    """Derive score + sequence evidence from real multi-timeframe data."""
    problems = list(data_problems or [])
    df_1d = frames.get("1d", pd.DataFrame())
    df_4h = frames.get("4h", pd.DataFrame())
    df_1h = frames.get("1h", pd.DataFrame())

    price = None
    for df in (df_1h, df_4h, df_1d):
        if df is not None and not df.empty:
            price = float(df["Close"].iloc[-1])
            break

    regime, regime_reason = detect_regime(df_1d)
    dir_4h, dir_4h_reason = structure_direction(df_4h)
    dir_1h, dir_1h_reason = structure_direction(df_1h)

    # Direction: explicit override wins; otherwise regime, then 4H structure.
    if direction_override:
        direction = direction_override
    elif regime == "bullish":
        direction = "Long"
    elif regime == "bearish":
        direction = "Short"
    else:
        direction = dir_4h

    analysis = CandidateAnalysis(
        ticker=ticker, direction=direction, regime_1d=regime, current_price=price,
        entry=None, stop=None, target=None, reward_risk=None, data_problems=problems,
    )

    if price is None:
        problems.append("No usable price on any timeframe — analysis cannot proceed.")
        return analysis
    if direction is None:
        problems.append("No directional bias could be determined (regime ranging, structure mixed).")

    # --- derive levels from confirmed structure -----------------------
    entry = price
    stop = None
    target = None
    if direction and not df_4h.empty:
        swings = [s for s in find_swing_points(df_4h, left=2, right=2) if s.confirmed]
        lows = [s.price for s in swings if s.kind == "low" and s.price < price]
        highs = [s.price for s in swings if s.kind == "high" and s.price > price]
        if direction == "Long" and lows:
            stop = max(lows)                       # nearest confirmed swing low below
            if highs:
                target = min(highs)                # nearest confirmed swing high above
        elif direction == "Short" and highs:
            stop = min(highs)
            if lows:
                target = max(lows)

    rr = None
    if stop is not None and target is not None and direction:
        risk = abs(entry - stop)
        reward = abs(target - entry)
        if risk > 0:
            rr = reward / risk

    analysis.entry, analysis.stop, analysis.target, analysis.reward_risk = entry, stop, target, rr
    if stop is not None:
        analysis.key_levels["structural_stop"] = stop
    if target is not None:
        analysis.key_levels["structural_target"] = target

    # --- scoring evidence ---------------------------------------------
    ev: Dict[str, EvidenceItem] = {}

    if regime == "unknown" or direction is None:
        ev["regime_alignment_1d_4h"] = EvidenceItem(None, f"{regime_reason} {dir_4h_reason}")
    else:
        aligned = ((regime == "bullish" and direction == "Long" and dir_4h == "Long")
                   or (regime == "bearish" and direction == "Short" and dir_4h == "Short"))
        ev["regime_alignment_1d_4h"] = EvidenceItem(
            aligned, f"1D: {regime_reason} 4H: {dir_4h_reason}"
        )

    sr = nearest_level(df_4h, price, "low" if direction == "Long" else "high") if direction else None
    if direction is None:
        ev["support_resistance"] = EvidenceItem(None, "No direction — cannot pick the relevant level.")
    elif sr:
        lvl, touches = sr
        ev["support_resistance"] = EvidenceItem(
            True, f"Price is within 2% of a confirmed 4H level at {lvl:.4g} ({touches} touch(es))."
        )
    else:
        ev["support_resistance"] = EvidenceItem(
            False, "Price is not near a confirmed 4H swing level (within 2%)."
        )

    if direction:
        swept, sweep_reason = detect_sweep_and_reclaim(df_4h, direction)
        ev["liquidity_sweep_reclaim"] = EvidenceItem(swept, sweep_reason)
    else:
        ev["liquidity_sweep_reclaim"] = EvidenceItem(None, "No direction — cannot evaluate a sweep.")

    if direction:
        fib_ok, fib_reason, fib_lv = fib_confluence(df_4h, price, direction)
        ev["fib_confluence"] = EvidenceItem(fib_ok, fib_reason)
        for k, v in fib_lv.items():
            analysis.key_levels[f"fib_{k}"] = v
    else:
        ev["fib_confluence"] = EvidenceItem(None, "No direction — cannot anchor Fibonacci.")

    if dir_4h is None:
        ev["structure_4h_supports"] = EvidenceItem(None, dir_4h_reason)
    else:
        ev["structure_4h_supports"] = EvidenceItem(dir_4h == direction, dir_4h_reason)

    if dir_1h is None:
        ev["entry_confirmation_1h"] = EvidenceItem(None, dir_1h_reason)
    else:
        ev["entry_confirmation_1h"] = EvidenceItem(dir_1h == direction, dir_1h_reason)

    if stop is None:
        ev["invalidation_defined"] = EvidenceItem(
            False, "No confirmed swing level available to place a structural stop."
        )
    else:
        ev["invalidation_defined"] = EvidenceItem(
            True, f"Structural invalidation at {stop:.4g} (nearest confirmed 4H swing)."
        )

    if rr is None:
        ev["rr_at_least_2"] = EvidenceItem(
            None, "Reward:risk not computable — entry, stop or target missing."
        )
    else:
        ev["rr_at_least_2"] = EvidenceItem(
            rr >= min_rr,
            f"Reward:risk is {rr:.2f}:1 against a {min_rr:.1f}:1 minimum "
            f"(entry {entry:.4g}, stop {stop:.4g}, target {target:.4g})."
        )

    if direction:
        opp, opp_reason = opposing_liquidity_ahead(df_4h, price, target, direction)
        ev["opposing_liquidity_ahead"] = EvidenceItem(opp, opp_reason)
    else:
        ev["opposing_liquidity_ahead"] = EvidenceItem(None, "No direction — cannot check the path.")

    extended, ext_reason = is_extended(df_4h, price)
    ev["chasing_extended_move"] = EvidenceItem(extended, ext_reason)

    analysis.score_evidence = ev

    # --- entry-sequence evidence --------------------------------------
    seq: Dict[str, EvidenceItem] = {}
    seq["location"] = EvidenceItem(
        bool(ev["support_resistance"].value) or bool(ev["fib_confluence"].value),
        "At a confirmed level and/or Fibonacci confluence."
        if (ev["support_resistance"].value or ev["fib_confluence"].value)
        else "Price is not at a meaningful level or Fib confluence."
    )
    seq["liquidity_event"] = EvidenceItem(
        bool(ev["liquidity_sweep_reclaim"].value),
        ev["liquidity_sweep_reclaim"].reason
    )
    seq["reclaim_or_rejection"] = EvidenceItem(
        bool(ev["liquidity_sweep_reclaim"].value),
        "Reclaim/rejection is part of the sweep test above (close-based, not wick-only)."
    )
    seq["confirmation"] = EvidenceItem(
        bool(ev["entry_confirmation_1h"].value), dir_1h_reason
    )
    seq["structural_invalidation"] = EvidenceItem(
        bool(ev["invalidation_defined"].value), ev["invalidation_defined"].reason
    )
    seq["acceptable_rr"] = EvidenceItem(
        bool(ev["rr_at_least_2"].value), ev["rr_at_least_2"].reason
    )
    seq["execution_plan"] = EvidenceItem(
        stop is not None and target is not None,
        "Entry, stop and target are all defined."
        if (stop is not None and target is not None)
        else "Entry, stop and/or target could not be derived from structure."
    )
    analysis.sequence_evidence = seq

    return analysis
