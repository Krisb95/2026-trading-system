"""
Risk calculator for linear and simple contract-based instruments.

Formulas (per project spec, section 8):
    risk_amount        = account_equity * risk_pct
    stop_distance_price = abs(entry - stop)
    stop_distance_pct   = stop_distance_price / entry
    position_notional   = risk_amount / stop_distance_pct
    margin              = position_notional / leverage

Quantity is derived from notional using the instrument's contract multiplier,
so this does NOT assume every market is 1 unit = 1 quote-currency (see
INSTRUMENT_SPECS below). This is still a simplification: it does not model
maintenance margin curves, funding rates, or exchange-specific liquidation
mechanics — see `liquidation_distance_warning` for what it actually checks.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InstrumentSpec:
    contract_multiplier: float = 1.0   # quote-currency value of 1 unit price move per 1 contract/share
    tick_size: float = 0.01
    lot_size: float = 1.0              # minimum increment of quantity
    min_order_size: float = 0.0        # minimum tradeable quantity
    instrument_type: str = "linear"    # "linear" (equity/spot) | "futures"


# Known instrument overrides. Anything not listed falls back to a plain
# linear 1:1 instrument (appropriate for equities and most spot crypto).
INSTRUMENT_SPECS = {
    "CL=F": InstrumentSpec(contract_multiplier=1000, tick_size=0.01, lot_size=1, min_order_size=1, instrument_type="futures"),   # WTI crude, 1,000 bbl/contract
    "BZ=F": InstrumentSpec(contract_multiplier=1000, tick_size=0.01, lot_size=1, min_order_size=1, instrument_type="futures"),
    "GC=F": InstrumentSpec(contract_multiplier=100, tick_size=0.10, lot_size=1, min_order_size=1, instrument_type="futures"),     # Gold, 100 troy oz/contract
    "SI=F": InstrumentSpec(contract_multiplier=5000, tick_size=0.005, lot_size=1, min_order_size=1, instrument_type="futures"),   # Silver, 5,000 troy oz/contract
    "PL=F": InstrumentSpec(contract_multiplier=50, tick_size=0.10, lot_size=1, min_order_size=1, instrument_type="futures"),
    "PA=F": InstrumentSpec(contract_multiplier=100, tick_size=0.05, lot_size=1, min_order_size=1, instrument_type="futures"),
    "HG=F": InstrumentSpec(contract_multiplier=25000, tick_size=0.0005, lot_size=1, min_order_size=1, instrument_type="futures"),
    "NG=F": InstrumentSpec(contract_multiplier=10000, tick_size=0.001, lot_size=1, min_order_size=1, instrument_type="futures"),
    "RB=F": InstrumentSpec(contract_multiplier=42000, tick_size=0.0001, lot_size=1, min_order_size=1, instrument_type="futures"),
    "HO=F": InstrumentSpec(contract_multiplier=42000, tick_size=0.0001, lot_size=1, min_order_size=1, instrument_type="futures"),
}

DEFAULT_SPEC = InstrumentSpec()  # equities, ETFs, and spot crypto


def get_instrument_spec(ticker: str) -> InstrumentSpec:
    return INSTRUMENT_SPECS.get(ticker.upper(), DEFAULT_SPEC)


class InvalidRiskInputError(ValueError):
    pass


@dataclass
class RiskCalculationResult:
    max_permitted_loss: float
    stop_distance_price: float
    stop_distance_pct: float
    position_notional: float
    quantity: float
    margin_required: float
    gross_profit_at_target: Optional[float]
    gross_loss_at_stop: float
    fees_and_slippage_cost: float
    net_profit_at_target: Optional[float]
    net_loss_at_stop: float
    net_reward_risk: Optional[float]
    exceeds_account_equity: bool
    liquidation_warning: Optional[str]
    instrument_type: str
    warnings: list = field(default_factory=list)


def calculate_risk(
    account_equity: float,
    risk_pct: float,
    entry: float,
    stop: float,
    direction: str,
    target: Optional[float] = None,
    leverage: float = 1.0,
    ticker: Optional[str] = None,
    fee_rate: float = 0.0,
    slippage_pct: float = 0.0,
) -> RiskCalculationResult:
    """Compute position sizing and P&L estimates for a single trade.

    direction: "Long" or "Short"
    fee_rate: round-trip fee as a fraction of notional (e.g. 0.0006 = 0.06%)
    slippage_pct: assumed round-trip slippage as a fraction of notional
    """
    direction = direction.capitalize()
    if direction not in ("Long", "Short"):
        raise InvalidRiskInputError(f"direction must be 'Long' or 'Short', got {direction!r}")
    if account_equity <= 0:
        raise InvalidRiskInputError("account_equity must be positive")
    if not (0 < risk_pct <= 100):
        raise InvalidRiskInputError("risk_pct must be between 0 and 100")
    if entry <= 0 or stop <= 0:
        raise InvalidRiskInputError("entry and stop must be positive")
    if leverage <= 0:
        raise InvalidRiskInputError("leverage must be positive")

    if direction == "Long" and stop >= entry:
        raise InvalidRiskInputError("For a Long, stop must be below entry")
    if direction == "Short" and stop <= entry:
        raise InvalidRiskInputError("For a Short, stop must be above entry")

    spec = get_instrument_spec(ticker) if ticker else DEFAULT_SPEC

    risk_amount = account_equity * (risk_pct / 100.0)
    stop_distance_price = abs(entry - stop)
    stop_distance_pct = stop_distance_price / entry

    position_notional = risk_amount / stop_distance_pct
    # Quantity of contracts/shares/units implied by that notional.
    quantity_raw = position_notional / (entry * spec.contract_multiplier)

    warnings = []
    # Round down to the instrument's lot size so the recommendation is
    # actually tradeable, and flag if that rounds below the exchange minimum.
    if spec.lot_size > 0:
        quantity = (quantity_raw // spec.lot_size) * spec.lot_size
    else:
        quantity = quantity_raw
    if quantity < spec.min_order_size:
        warnings.append(
            f"Rounded quantity ({quantity:g}) is below this instrument's minimum "
            f"order size ({spec.min_order_size:g}) — this trade isn't sized-in."
        )

    actual_notional = quantity * entry * spec.contract_multiplier
    margin_required = actual_notional / leverage

    gross_loss_at_stop = quantity * spec.contract_multiplier * stop_distance_price
    gross_profit_at_target = None
    if target is not None and target > 0:
        if direction == "Long":
            gross_profit_at_target = quantity * spec.contract_multiplier * (target - entry)
        else:
            gross_profit_at_target = quantity * spec.contract_multiplier * (entry - target)

    fees_and_slippage_cost = actual_notional * (fee_rate + slippage_pct)
    net_loss_at_stop = gross_loss_at_stop + fees_and_slippage_cost
    net_profit_at_target = None
    net_reward_risk = None
    if gross_profit_at_target is not None:
        net_profit_at_target = gross_profit_at_target - fees_and_slippage_cost
        if net_loss_at_stop > 0:
            net_reward_risk = net_profit_at_target / net_loss_at_stop

    exceeds_account_equity = margin_required > account_equity

    liquidation_warning = None
    if leverage > 1:
        # Simplified: ignores maintenance margin, funding, and exchange-specific
        # liquidation engines. This only checks whether your OWN stop sits
        # beyond the naive 1/leverage liquidation buffer from entry.
        naive_liquidation_pct = 1.0 / leverage
        if stop_distance_pct >= naive_liquidation_pct:
            liquidation_warning = (
                f"⚠️ Your stop ({stop_distance_pct:.1%} away) is at or beyond the naive "
                f"liquidation buffer for {leverage:g}x leverage (~{naive_liquidation_pct:.1%}). "
                f"You could be liquidated before your stop is hit. This estimate ignores "
                f"maintenance margin and funding — check your exchange's actual liquidation "
                f"price before sizing a leveraged trade this tight."
            )
        else:
            liquidation_warning = (
                f"Naive liquidation buffer at {leverage:g}x is ~{naive_liquidation_pct:.1%} "
                f"from entry — your stop is comfortably inside it, but this ignores "
                f"maintenance margin/funding. Confirm the real liquidation price on your exchange."
            )

    return RiskCalculationResult(
        max_permitted_loss=risk_amount,
        stop_distance_price=stop_distance_price,
        stop_distance_pct=stop_distance_pct,
        position_notional=actual_notional,
        quantity=quantity,
        margin_required=margin_required,
        gross_profit_at_target=gross_profit_at_target,
        gross_loss_at_stop=gross_loss_at_stop,
        fees_and_slippage_cost=fees_and_slippage_cost,
        net_profit_at_target=net_profit_at_target,
        net_loss_at_stop=net_loss_at_stop,
        net_reward_risk=net_reward_risk,
        exceeds_account_equity=exceeds_account_equity,
        liquidation_warning=liquidation_warning,
        instrument_type=spec.instrument_type,
        warnings=warnings,
    )
