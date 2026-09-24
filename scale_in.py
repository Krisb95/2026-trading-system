"""
Scaling in: splitting an entry across several price levels instead of one.

WHEN IT IS OFFERED
  Only when the chart actually provides it — two or more CONFIRMED support
  levels below price (resistance above, for a short), far enough apart to be
  distinct rather than noise. If the structure shows one level, one entry is
  the plan. Rungs at arbitrary intervals would be averaging down dressed up as
  a strategy.

HOW RISK IS HELD CONSTANT
  Size is allocated by RISK, not by units. Each rung gets a share of the same
  total risk budget, and because deeper rungs sit closer to the stop, they buy
  more units for the same money at risk. If every rung fills, the loss at the
  stop is exactly the budget — never more.

THE HONEST TRADE-OFF
  A ladder improves the average entry when price works down through it, and
  gives a better reward:risk than a single entry at the shallowest level. But:

    * price often bounces after the first rung, leaving a smaller position than
      planned in a trade that then works — a real cost, not a hypothetical
    * when every rung fills, the trade is at full size in a market that has
      been going against you the whole way down
    * the stop must sit below the DEEPEST rung, so it is wider than a
      single-entry stop, and the whole ladder is wrong together

  Whether laddering improves results for this strategy has not been measured.
  Both outcomes are reported — all rungs filled, and only the first — so the
  trade-off is visible rather than assumed.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from technical import find_swing_points
from formatting import format_price as fp

EQUAL = "Equal"
DEEPER_HEAVIER = "Heavier deeper"
SHALLOWER_HEAVIER = "Heavier at the first level"
WEIGHTINGS = [EQUAL, DEEPER_HEAVIER, SHALLOWER_HEAVIER]

MIN_RUNGS = 2
MAX_RUNGS = 4
MIN_GAP_ATR = 0.5          # rungs closer than this are the same level twice
MAX_SPAN_ATR = 6.0         # a rung this far away is unlikely to fill


@dataclass
class Rung:
    level: int              # 1 = first to fill
    price: float
    weight: float           # share of the risk budget
    units: float
    notional: float
    risk_amount: float
    distance_pct: float     # from the live price
    cumulative_units: float
    average_entry: float    # if filled up to and including this rung


@dataclass
class Ladder:
    rungs: List[Rung]
    stop: float
    target: float
    average_entry_all: float
    rr_all_filled: Optional[float]
    rr_first_only: Optional[float]
    rr_single_entry: Optional[float]     # the plain one-level plan, for comparison
    total_units: float
    total_notional: float
    risk_budget: float
    notes: List[str] = field(default_factory=list)

    @property
    def deepest(self) -> float:
        return self.rungs[-1].price if self.rungs else 0.0


def find_levels(df_5m, direction: str, price: float, atr: float,
                max_rungs: int = MAX_RUNGS, min_gap_atr: float = MIN_GAP_ATR,
                max_span_atr: float = MAX_SPAN_ATR, lookback: int = 288,
                left: int = 3, right: int = 3) -> List[float]:
    """Confirmed support levels below price (resistance above, for a short).

    Levels closer together than min_gap_atr are treated as one — two swings a
    few ticks apart are the same level, and laddering across them would just
    split the order for no benefit.
    """
    if df_5m is None or len(df_5m) < left + right + 1 or not atr or atr <= 0:
        return []
    window = df_5m.iloc[-lookback:]
    swings = [s for s in find_swing_points(window, left, right) if s.confirmed]
    kind = "low" if direction == "Long" else "high"
    if direction == "Long":
        candidates = sorted({s.price for s in swings
                             if s.kind == kind and s.price < price}, reverse=True)
    else:
        candidates = sorted({s.price for s in swings if s.kind == kind and s.price > price})

    chosen: List[float] = []
    for level in candidates:
        if abs(price - level) > atr * max_span_atr:
            break
        if chosen and abs(chosen[-1] - level) < atr * min_gap_atr:
            continue
        chosen.append(level)
        if len(chosen) >= max_rungs:
            break
    return chosen


def weights_for(style: str, n: int) -> List[float]:
    """Risk shares per rung, summing to 1."""
    if n <= 0:
        return []
    if style == DEEPER_HEAVIER:
        raw = [i + 1 for i in range(n)]              # 1, 2, 3...
    elif style == SHALLOWER_HEAVIER:
        raw = [n - i for i in range(n)]              # n, n-1, ...
    else:
        raw = [1] * n
    total = float(sum(raw))
    return [r / total for r in raw]


def build_ladder(direction: str, levels: Sequence[float], stop: float, target: float,
                 risk_budget: float, price: float,
                 weighting: str = EQUAL) -> Optional[Ladder]:
    """Build the ladder. Returns None if the levels or stop don't make sense.

    The stop must already sit beyond the DEEPEST level — a stop above the last
    rung would be hit before that rung filled, which is incoherent.
    """
    levels = [float(x) for x in levels]
    if len(levels) < MIN_RUNGS or risk_budget <= 0 or price <= 0:
        return None
    deepest = levels[-1]
    if direction == "Long" and not stop < deepest:
        return None
    if direction == "Short" and not stop > deepest:
        return None

    ws = weights_for(weighting, len(levels))
    rungs: List[Rung] = []
    cum_units = 0.0
    cum_cost = 0.0
    for i, (lvl, w) in enumerate(zip(levels, ws), start=1):
        per_unit_risk = abs(lvl - stop)
        if per_unit_risk <= 0:
            return None
        risk_amount = risk_budget * w
        units = risk_amount / per_unit_risk
        cum_units += units
        cum_cost += units * lvl
        rungs.append(Rung(
            level=i, price=lvl, weight=w, units=units, notional=units * lvl,
            risk_amount=risk_amount,
            distance_pct=(lvl - price) / price * 100,
            cumulative_units=cum_units, average_entry=cum_cost / cum_units))

    avg_all = cum_cost / cum_units
    risk_all = abs(avg_all - stop)
    rr_all = abs(target - avg_all) / risk_all if risk_all else None
    first = rungs[0]
    risk_first = abs(first.price - stop)
    rr_first = abs(target - first.price) / risk_first if risk_first else None

    notes: List[str] = []
    if rr_all and rr_first and rr_all > rr_first:
        notes.append(
            f"If every rung fills, the average entry is {fp(avg_all)} and reward:risk "
            f"improves to {rr_all:.2f} from {rr_first:.2f} at the first level alone.")
    notes.append(
        f"If price turns after the first rung — the common case — you hold "
        f"{first.weight * 100:.0f}% of the planned risk in a trade that works.")
    notes.append(
        f"If all rungs fill, you are at full size after price has moved "
        f"{abs(rungs[-1].distance_pct):.2f}% against the first entry, and the stop at "
        f"{fp(stop)} applies to the whole position.")

    return Ladder(rungs=rungs, stop=stop, target=target, average_entry_all=avg_all,
                  rr_all_filled=rr_all, rr_first_only=rr_first, rr_single_entry=rr_first,
                  total_units=cum_units, total_notional=cum_cost,
                  risk_budget=risk_budget, notes=notes)


def ladder_stop(direction: str, deepest_level: float, atr: float,
                buffer_atr: float) -> float:
    """Stop placed beyond the deepest rung by the strategy's usual buffer."""
    buf = atr * buffer_atr
    return deepest_level - buf if direction == "Long" else deepest_level + buf


def availability(levels: Sequence[float]) -> Tuple[bool, str]:
    """Whether a ladder is on offer, and why not when it isn't."""
    if len(levels) >= MIN_RUNGS:
        return True, f"{len(levels)} distinct levels found."
    if len(levels) == 1:
        return False, ("Only one confirmed level below price, so there's nothing to ladder "
                       "across — the single entry is the plan.")
    return False, "No confirmed levels found to ladder across."
