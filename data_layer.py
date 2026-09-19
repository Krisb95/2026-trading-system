"""
Market data layer with explicit health status. Never silently substitutes a
stale price or another instrument's price — every fetch result is tagged
with a DataStatus and a UTC timestamp, and the caller decides what to do
with anything other than LIVE/DELAYED.

Yahoo Finance has tightened bot detection on its query endpoints, which
causes intermittent 401/429/empty-data responses via yfinance across ALL
instrument types (not one specific ticker). The fix the yfinance maintainers
recommend is impersonating a real browser TLS fingerprint via `curl_cffi`.
This module uses it when available and falls back gracefully if not
installed, so the app doesn't crash — it just loses the reliability boost.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import time

try:
    from curl_cffi import requests as curl_requests
    _CURL_CFFI_AVAILABLE = True
except ImportError:
    _CURL_CFFI_AVAILABLE = False


class DataStatus(str, Enum):
    LIVE = "LIVE"
    DELAYED = "DELAYED"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class PriceQuote:
    ticker: str
    price: Optional[float]
    fetched_at_utc: Optional[datetime]  # when THIS fetch completed (UTC)
    bar_time_utc: Optional[datetime]    # timestamp of the underlying price bar, if known
    status: DataStatus
    provider: str
    error: Optional[str] = None
    attempts: int = 1


# Instruments/exchanges where the underlying feed is inherently delayed
# (typical free-tier equity data), independent of freshness-by-age.
KNOWN_DELAYED_ASSET_CLASSES = {"stock"}


_session_cache = {"session": None}


def get_browser_session():
    """Return a cached curl_cffi session impersonating a real browser, or
    None if curl_cffi isn't installed (caller should fall back to plain
    yfinance behavior in that case)."""
    if not _CURL_CFFI_AVAILABLE:
        return None
    if _session_cache["session"] is None:
        _session_cache["session"] = curl_requests.Session(impersonate="chrome")
    return _session_cache["session"]


def curl_cffi_is_available() -> bool:
    return _CURL_CFFI_AVAILABLE


def classify_freshness(
    bar_time_utc: Optional[datetime],
    now_utc: datetime,
    asset_class: str,
    live_threshold_seconds: float,
    stale_threshold_seconds: float,
) -> DataStatus:
    """Pure function (easy to unit test): decide LIVE/DELAYED/STALE from age.

    - If we have no bar_time_utc at all, caller should treat as UNAVAILABLE
      (this function assumes you already have a bar time).
    - live_threshold_seconds: below this age, and NOT a known-delayed asset
      class, => LIVE.
    - Between live and stale thresholds => DELAYED (still usable, just not
      instantaneous — this is also the default for equities, since free
      equity feeds are exchange-delayed by nature, not just "old").
    - Beyond stale_threshold_seconds => STALE (block trade-readiness).
    """
    if bar_time_utc is None:
        raise ValueError("classify_freshness requires a bar_time_utc")

    age_seconds = (now_utc - bar_time_utc).total_seconds()
    if age_seconds < 0:
        # Clock skew guard — treat as freshest bucket rather than error out.
        age_seconds = 0

    if age_seconds > stale_threshold_seconds:
        return DataStatus.STALE
    if asset_class in KNOWN_DELAYED_ASSET_CLASSES or age_seconds > live_threshold_seconds:
        return DataStatus.DELAYED
    return DataStatus.LIVE


def to_local_display(dt_utc: Optional[datetime], tz) -> str:
    """Format a UTC datetime for display in the given local tzinfo."""
    if dt_utc is None:
        return "—"
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    local = dt_utc.astimezone(tz)
    return local.strftime("%Y-%m-%d %H:%M:%S %Z")


def _attempt_fetch(ticker: str, yf_module, use_browser_session: bool):
    """One fetch attempt. Returns (history_df_or_None, error_str_or_None)."""
    try:
        if use_browser_session:
            session = get_browser_session()
            if session is not None:
                hist = yf_module.Ticker(ticker, session=session).history(period="2d")
            else:
                hist = yf_module.Ticker(ticker).history(period="2d")
        else:
            hist = yf_module.Ticker(ticker).history(period="2d")
        return hist, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def fetch_quote(
    ticker: str,
    asset_class: str,
    yf_module,
    live_threshold_seconds: float = 60,
    stale_threshold_seconds: float = 900,
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.5,
) -> PriceQuote:
    """Fetch a quote via the given yfinance-like module, with explicit
    status classification, browser-session impersonation (via curl_cffi,
    if installed), and a small retry loop for transient failures
    (rate limits, momentary auth hiccups). `yf_module` is injected so this
    function is unit-testable with a fake/mock instead of hitting the network.
    """
    now_utc = datetime.now(timezone.utc)

    if not ticker or not ticker.strip():
        return PriceQuote(ticker=ticker, price=None, fetched_at_utc=now_utc,
                           bar_time_utc=None, status=DataStatus.UNAVAILABLE,
                           provider="yfinance", error="Empty ticker")

    last_error = None
    hist = None
    attempts = 0
    for attempt in range(1, max_retries + 2):  # e.g. max_retries=2 => 3 total attempts
        attempts = attempt
        hist, err = _attempt_fetch(ticker, yf_module, use_browser_session=True)
        if err is None and hist is not None and not hist.empty:
            last_error = None
            break
        last_error = err or "No rows returned"
        if attempt <= max_retries:
            time.sleep(retry_backoff_seconds * attempt)  # linear backoff

    if last_error is not None or hist is None or hist.empty:
        provider_note = "yfinance (curl_cffi)" if curl_cffi_is_available() else "yfinance"
        return PriceQuote(ticker=ticker, price=None, fetched_at_utc=now_utc,
                           bar_time_utc=None, status=DataStatus.UNAVAILABLE,
                           provider=provider_note, error=last_error, attempts=attempts)

    try:
        price = float(hist["Close"].iloc[-1])
        bar_time = hist.index[-1].to_pydatetime()
        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=timezone.utc)
        else:
            bar_time = bar_time.astimezone(timezone.utc)
    except Exception as e:
        return PriceQuote(ticker=ticker, price=None, fetched_at_utc=now_utc,
                           bar_time_utc=None, status=DataStatus.UNAVAILABLE,
                           provider="yfinance", error=f"Malformed data: {type(e).__name__}: {e}",
                           attempts=attempts)

    if price <= 0:
        return PriceQuote(ticker=ticker, price=None, fetched_at_utc=now_utc,
                           bar_time_utc=bar_time, status=DataStatus.UNAVAILABLE,
                           provider="yfinance", error=f"Non-positive price returned: {price}",
                           attempts=attempts)

    status = classify_freshness(bar_time, now_utc, asset_class,
                                 live_threshold_seconds, stale_threshold_seconds)

    provider_note = "yfinance (curl_cffi)" if curl_cffi_is_available() else "yfinance"
    return PriceQuote(ticker=ticker, price=price, fetched_at_utc=now_utc,
                       bar_time_utc=bar_time, status=status, provider=provider_note,
                       attempts=attempts)
