"""
Hyperliquid market data: 24-hour volume for every perp, plus candles.

WHY HYPERLIQUID'S OWN VOLUME: the trader executes on Hyperliquid, so its volume
is the liquidity they would actually trade into. A coin with huge volume on
other exchanges can still be thin here, and vice versa. Hyperliquid also lists
coins outside the market-cap top 100, which this reaches.

RATE LIMITS (public /info endpoint): 1,200 weight per minute per IP.
metaAndAssetCtxs costs 20; candleSnapshot costs 20 plus extra per 60 candles
returned. Scanning one coin (4H, 1H and 5m candles) costs roughly 70, so about
17 coins per minute is the ceiling. Requests are paced at one per 1.5 seconds
(about 13 coins a minute) to stay safely under it, and a 429 response triggers
a backoff rather than a failure.

"k" COINS: some low-priced coins trade on Hyperliquid in bundles of 1,000 —
kPEPE is 1,000 PEPE, so its price is 1,000x PEPE's. Prices from Hyperliquid
are internally consistent, but must not be mixed with another exchange's
per-coin price for the same token. Tickers from this module are therefore
prefixed "HL:" (e.g. "HL:kPEPE") so every later step knows where to fetch.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import time
import pandas as pd
import requests

INFO_URL = "https://api.hyperliquid.xyz/info"
PREFIX = "HL:"
INTERVAL_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000,
               "4h": 14_400_000, "1d": 86_400_000}

# ~70 weight per coin (three candle requests) at one request per 1.5s is about
# 930 weight a minute — comfortably under the 1,200 limit, with headroom for
# the volume request and the occasional retry.
_SETTINGS = {"spacing": 1.5, "max_retries": 3, "backoff": 3.0}
_last = {"t": 0.0}


def configure(spacing: float = 1.5, max_retries: int = 3, backoff: float = 3.0) -> None:
    """Tune pacing. Tests set everything to zero so nothing actually sleeps."""
    _SETTINGS.update(spacing=spacing, max_retries=max_retries, backoff=backoff)


def _post(body: Dict, timeout: int = 15):
    """POST to /info with pacing and 429 backoff. Returns (json, error)."""
    last_error = None
    for attempt in range(_SETTINGS["max_retries"] + 1):
        wait = _SETTINGS["spacing"] - (time.time() - _last["t"])
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.post(INFO_URL, json=body, timeout=timeout)
            _last["t"] = time.time()
            if resp.status_code == 429:
                last_error = "429 rate limited by Hyperliquid"
                if attempt < _SETTINGS["max_retries"]:
                    time.sleep(_SETTINGS["backoff"] * (attempt + 1))
                    continue
                return None, last_error
            resp.raise_for_status()
            return resp.json(), None
        except Exception as e:
            _last["t"] = time.time()
            last_error = f"{type(e).__name__}: {e}"
            if attempt < _SETTINGS["max_retries"]:
                time.sleep(_SETTINGS["backoff"] * (attempt + 1))
    return None, last_error


def is_hl_ticker(ticker: str) -> bool:
    return isinstance(ticker, str) and ticker.startswith(PREFIX)


def hl_name(ticker: str) -> str:
    """'HL:kPEPE' -> 'kPEPE'."""
    return ticker[len(PREFIX):] if is_hl_ticker(ticker) else ticker


def base_symbol(name: str) -> str:
    """Underlying token: 'kPEPE' -> 'PEPE', 'BTC' -> 'BTC'."""
    if len(name) > 1 and name[0] == "k" and name[1:].isupper():
        return name[1:]
    return name.upper()


def bundle_size(name: str) -> int:
    """How many tokens one Hyperliquid unit represents (1,000 for 'k' coins)."""
    return 1000 if base_symbol(name) != name.upper() else 1


@dataclass
class MarketContext:
    name: str
    day_volume_usd: float
    mark_price: float
    open_interest_usd: float
    funding_rate: float           # hourly funding rate
    change_24h_pct: Optional[float]
    max_leverage: Optional[int]

    @property
    def ticker(self) -> str:
        return PREFIX + self.name

    @property
    def base(self) -> str:
        return base_symbol(self.name)

    @property
    def label(self) -> str:
        suffix = " (per 1,000)" if bundle_size(self.name) > 1 else ""
        return f"{self.name}{suffix}"


def fetch_market_contexts() -> Tuple[List[MarketContext], Optional[str]]:
    """Every active Hyperliquid perp with its 24h volume, price, OI and funding."""
    data, err = _post({"type": "metaAndAssetCtxs"})
    if data is None:
        return [], err
    try:
        meta, ctxs = data[0], data[1]
        universe = meta.get("universe", [])
    except (IndexError, KeyError, TypeError, AttributeError):
        return [], "Unexpected response shape from Hyperliquid."
    out: List[MarketContext] = []
    for info, ctx in zip(universe, ctxs):
        if info.get("isDelisted"):
            continue
        try:
            mark = float(ctx.get("markPx") or 0)
            if mark <= 0:
                continue
            prev = float(ctx.get("prevDayPx") or 0)
            out.append(MarketContext(
                name=str(info["name"]),
                day_volume_usd=float(ctx.get("dayNtlVlm") or 0),
                mark_price=mark,
                open_interest_usd=float(ctx.get("openInterest") or 0) * mark,
                funding_rate=float(ctx.get("funding") or 0),
                change_24h_pct=((mark - prev) / prev * 100) if prev > 0 else None,
                max_leverage=info.get("maxLeverage")))
        except (TypeError, ValueError, KeyError):
            continue
    if not out:
        return [], "Hyperliquid returned no active markets."
    return out, None


def rank_by_volume(contexts: List[MarketContext], exclude_bases: Set[str],
                   min_volume_usd: float = 0.0, limit: int = 20) -> List[MarketContext]:
    """Highest 24h volume first, skipping excluded tokens and thin markets."""
    excl = {b.upper() for b in exclude_bases}
    kept = [c for c in contexts
            if c.base.upper() not in excl and c.day_volume_usd >= min_volume_usd]
    kept.sort(key=lambda c: c.day_volume_usd, reverse=True)
    return kept[:limit]


def fetch_candles(name: str, interval: str, n_bars: int,
                  end_ms: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """OHLCV candles for one Hyperliquid perp, oldest first, indexed by open
    time in UTC. Hyperliquid serves at most the most recent 5,000 candles."""
    if interval not in INTERVAL_MS:
        return None, f"Unsupported interval {interval!r}"
    end_ms = end_ms or int(time.time() * 1000)
    start_ms = end_ms - INTERVAL_MS[interval] * (n_bars + 1)
    data, err = _post({"type": "candleSnapshot",
                       "req": {"coin": hl_name(name), "interval": interval,
                               "startTime": start_ms, "endTime": end_ms}})
    if data is None:
        return None, err
    if not isinstance(data, list) or not data:
        return None, f"No {interval} candles returned for {hl_name(name)}."
    try:
        df = pd.DataFrame(data)
        df["ts"] = pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True)
        df = df.set_index("ts").rename(columns={"o": "Open", "h": "High", "l": "Low",
                                                 "c": "Close", "v": "Volume"})
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        df = df[~df.index.duplicated()].sort_index()
        df = df[df["Close"].notna()]
    except (KeyError, ValueError, TypeError) as e:
        return None, f"Malformed candle data: {type(e).__name__}"
    return (df, None) if not df.empty else (None, "All candles were empty.")
