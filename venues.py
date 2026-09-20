"""
Which coins you can actually trade.

The CoinGecko top-100 is ranked by market cap, which is the wrong filter for a
trading shortlist: it includes exchange tokens (WBT, OKB), wrapped assets and
chains that are not listed as perpetuals on the venues you use. Scanning them
wastes time and, worse, produces A+ setups you cannot act on.

This module fetches the live perpetual listings from Bybit and Hyperliquid and
filters the universe to their union. Both endpoints are public and need no key.

IF A VENUE CANNOT BE REACHED its listings are simply absent from the union and
that is reported — the filter is never silently skipped, because an unfiltered
list looks identical to a working one until you try to place a trade.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import requests

BYBIT_INSTRUMENTS = "https://api.bybit.com/v5/market/instruments-info"
HYPERLIQUID_INFO = "https://api.hyperliquid.xyz/info"


@dataclass
class VenueListings:
    bybit: Set[str] = field(default_factory=set)
    hyperliquid: Set[str] = field(default_factory=set)
    problems: List[str] = field(default_factory=list)

    @property
    def union(self) -> Set[str]:
        return self.bybit | self.hyperliquid

    @property
    def both(self) -> Set[str]:
        return self.bybit & self.hyperliquid

    @property
    def any_reachable(self) -> bool:
        return bool(self.bybit or self.hyperliquid)

    def venues_for(self, asset: str) -> List[str]:
        asset = asset.upper()
        out = []
        if asset in self.bybit:
            out.append("Bybit")
        if asset in self.hyperliquid:
            out.append("HL")
        return out


def fetch_bybit_perps(timeout: int = 10) -> Tuple[Set[str], Optional[str]]:
    """Base assets with a USDT linear perpetual on Bybit."""
    try:
        resp = requests.get(BYBIT_INSTRUMENTS,
                             params={"category": "linear", "limit": 1000},
                             timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("retCode") not in (0, None):
            return set(), f"Bybit error: {payload.get('retMsg')}"
        items = (payload.get("result") or {}).get("list") or []
        assets = set()
        for item in items:
            if item.get("quoteCoin") != "USDT":
                continue
            if item.get("status") not in (None, "Trading"):
                continue
            base = str(item.get("baseCoin", "")).upper().strip()
            if base:
                assets.add(base)
        if not assets:
            return set(), "Bybit returned no tradable USDT perpetuals."
        return assets, None
    except Exception as e:
        return set(), f"Bybit unreachable: {type(e).__name__}: {e}"


def fetch_hyperliquid_perps(timeout: int = 10) -> Tuple[Set[str], Optional[str]]:
    """Base assets listed as perpetuals on Hyperliquid."""
    try:
        resp = requests.post(HYPERLIQUID_INFO, json={"type": "meta"}, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        universe = payload.get("universe") if isinstance(payload, dict) else None
        if not universe:
            return set(), "Hyperliquid returned no universe."
        assets = set()
        for item in universe:
            name = str(item.get("name", "")).upper().strip()
            # Delisted markets are flagged rather than removed.
            if item.get("isDelisted"):
                continue
            if name:
                assets.add(name)
        if not assets:
            return set(), "Hyperliquid returned no active markets."
        return assets, None
    except Exception as e:
        return set(), f"Hyperliquid unreachable: {type(e).__name__}: {e}"


def fetch_listings() -> VenueListings:
    problems: List[str] = []
    bybit, b_err = fetch_bybit_perps()
    if b_err:
        problems.append(b_err)
    hl, h_err = fetch_hyperliquid_perps()
    if h_err:
        problems.append(h_err)
    return VenueListings(bybit=bybit, hyperliquid=hl, problems=problems)


def base_asset_from_label(ticker: str) -> str:
    """'PEPE-USD' -> 'PEPE'."""
    return ticker.upper().replace("-USD", "").strip()


def filter_universe(tickers: Dict[str, str], listings: VenueListings,
                     require_both: bool = False
                     ) -> Tuple[Dict[str, str], Dict[str, List[str]], List[str]]:
    """Keep only coins listed on the chosen venues.

    Returns (filtered_tickers, venue_tags, notes). If neither venue could be
    reached the original mapping is returned unchanged, with a note saying the
    filter was NOT applied — silently returning everything would be
    indistinguishable from a working filter.
    """
    notes: List[str] = []
    if not listings.any_reachable:
        notes.append(
            "Venue filter NOT applied — neither Bybit nor Hyperliquid could be reached, "
            "so the full market-cap list is shown. Some coins may not be tradable for you."
        )
        return dict(tickers), {}, notes + listings.problems

    allowed = listings.both if require_both else listings.union
    filtered: Dict[str, str] = {}
    tags: Dict[str, List[str]] = {}
    for label, ticker in tickers.items():
        asset = base_asset_from_label(ticker)
        if asset in allowed:
            filtered[label] = ticker
            tags[label] = listings.venues_for(asset)

    dropped = len(tickers) - len(filtered)
    scope = "both Bybit and Hyperliquid" if require_both else "Bybit or Hyperliquid"
    notes.append(
        f"Filtered to {len(filtered)} coins listed on {scope} "
        f"({dropped} dropped — exchange tokens, wrapped assets and unlisted coins)."
    )
    if listings.problems:
        notes.extend(listings.problems)
        notes.append("One venue was unreachable, so the list may be narrower than it should be.")
    return filtered, tags, notes
