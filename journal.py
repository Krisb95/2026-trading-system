"""
Trade journal: manual record-keeping with the fields specified, plus CSV export.
Kept as plain dicts/DataFrame rather than a DB, since this is a single-user
Streamlit app with in-memory/session state.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional
import pandas as pd
import io

JOURNAL_FIELDS = [
    "status", "asset", "direction", "leverage", "entry", "size_notional_usd",
    "margin", "current_pl", "tp", "sl", "updated_tp", "updated_sl",
    "potential_profit_at_tp", "potential_loss_at_sl", "realized_pnl",
    "pl_status", "entry_reason", "score_breakdown", "exit_reason",
    "timestamp_utc", "management_notes",
]


@dataclass
class JournalEntry:
    status: str = "Open"
    asset: str = ""
    direction: str = "Long"
    leverage: float = 1.0
    entry: float = 0.0
    size_notional_usd: float = 0.0
    margin: float = 0.0
    current_pl: float = 0.0
    tp: float = 0.0
    sl: float = 0.0
    updated_tp: Optional[float] = None
    updated_sl: Optional[float] = None
    potential_profit_at_tp: float = 0.0
    potential_loss_at_sl: float = 0.0
    realized_pnl: Optional[float] = None
    pl_status: str = "Open"   # "Open" / "Win" / "Loss" / "Breakeven"
    entry_reason: str = ""
    score_breakdown: str = ""
    exit_reason: str = ""
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    management_notes: str = ""


class TradeJournal:
    def __init__(self):
        self._entries: List[JournalEntry] = []

    def add(self, entry: JournalEntry) -> int:
        self._entries.append(entry)
        return len(self._entries) - 1  # index/id

    def update(self, index: int, **kwargs) -> None:
        if not (0 <= index < len(self._entries)):
            raise IndexError(f"No journal entry at index {index}")
        entry = self._entries[index]
        for k, v in kwargs.items():
            if not hasattr(entry, k):
                raise AttributeError(f"JournalEntry has no field {k!r}")
            setattr(entry, k, v)

    def close_trade(self, index: int, realized_pnl: float, exit_reason: str) -> None:
        pl_status = "Win" if realized_pnl > 0 else ("Loss" if realized_pnl < 0 else "Breakeven")
        self.update(index, status="Closed", realized_pnl=realized_pnl,
                    exit_reason=exit_reason, pl_status=pl_status)

    def all(self) -> List[JournalEntry]:
        return list(self._entries)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._entries:
            return pd.DataFrame(columns=JOURNAL_FIELDS)
        return pd.DataFrame([asdict(e) for e in self._entries])

    def to_csv_bytes(self) -> bytes:
        buf = io.StringIO()
        self.to_dataframe().to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")
