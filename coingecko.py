"""
CoinGecko fallback data source.

Yahoo Finance does not list every crypto in the top 100 — newer listings such
as Hyperliquid (HYPE) return nothing at any interval, which previously left
the app showing UNAVAILABLE with no way forward. CoinGecko serves its own OHLC
candles, so it is used as a fallback whenever Yahoo has no data for a coin.

CANDLE GRANULARITY IS FIXED BY THE API — you cannot request an arbitrary
interval. CoinGecko's /ohlc endpoint returns:
    days=1        -> 30-minute candles
    days=2..30    -> 4-hour candles
    days=31+      -> 4-day candles

That is genuinely useful here: days<=30 gives NATIVE 4H candles (better than
resampling 1H, which is what the Yahoo path has to do), and days=1 gives 30m
for a near-live price. But there is no 1-hour or 5-minute option, so the
"1h" frame is unavailable on this source and the daily frame is approximated
by 4-day candles. Those limits are reported rather than papered over.

The free API is rate-limited (roughly 5-15 calls/minute). Callers should cache.
"""

from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List
import pandas as pd
import requests

OHLC_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
MARKET_CHART_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"

# What each `days` value actually returns, per CoinGecko's documented behaviour.
GRANULARITY_BY_DAYS = {
    1: ("30m", 1800),
    7: ("4h", 14400),
    14: ("4h", 14400),
    30: ("4h", 14400),
    90: ("4d", 345600),
    180: ("4d", 345600),
    365: ("4d", 345600),
}


def fetch_ohlc(coin_id: str, days: int = 30, timeout: int = 10
                ) -> Tuple[Optional[pd.DataFrame], str, Optional[str]]:
    """Fetch OHLC candles for a CoinGecko coin id.

    Returns (dataframe, granularity_label, error). The dataframe has
    Open/High/Low/Close columns and a UTC DatetimeIndex, matching the shape
    the rest of the app expects from yfinance. Volume is not provided by this
    endpoint, so it is absent rather than faked.
    """
    if days not in GRANULARITY_BY_DAYS:
        days = min(GRANULARITY_BY_DAYS, key=lambda d: abs(d - days))
    label, _seconds = GRANULARITY_BY_DAYS[days]

    try:
        resp = requests.get(
            OHLC_URL.format(coin_id=coin_id),
            params={"vs_currency": "usd", "days": days},
            timeout=timeout,
        )
        resp.raise_for_status()
        rows = resp.json()
        if not isinstance(rows, list) or not rows:
            return None, label, "CoinGecko returned no candles."

        df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts")
        df = df[df["Close"].notna()]
        if df.empty:
            return None, label, "All candles had null closes."
        return df, label, None

    except Exception as e:
        return None, label, f"{type(e).__name__}: {e}"


def fetch_spot_price(coin_id: str, timeout: int = 10
                      ) -> Tuple[Optional[float], Optional[datetime], Optional[str]]:
    """Current spot price with its last-updated timestamp.

    Returns (price, updated_at_utc, error). The timestamp is CoinGecko's own
    last_updated_at, not the time of our request — so freshness can be judged
    honestly rather than assumed.
    """
    try:
        resp = requests.get(
            PRICE_URL,
            params={"ids": coin_id, "vs_currencies": "usd",
                    "include_last_updated_at": "true"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        entry = data.get(coin_id)
        if not entry or "usd" not in entry:
            return None, None, f"No price for coin id '{coin_id}'."
        price = float(entry["usd"])
        if not (price > 0):
            return None, None, f"Non-positive price returned: {price}"
        ts = entry.get("last_updated_at")
        updated = (datetime.fromtimestamp(ts, tz=timezone.utc)
                    if isinstance(ts, (int, float)) else None)
        return price, updated, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def build_frames(coin_id: str) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Assemble the 1d / 4h / 5m-equivalent frames the scanner expects.

    Mapping to what CoinGecko can actually provide:
        "1d" <- 4-day candles over a year  (coarser than true daily)
        "4h" <- native 4-hour candles over 30 days
        "5m" <- 30-minute candles over 1 day (finest available, not 5m)

    Every substitution is reported in the returned problems list so the UI can
    say what the analysis is really based on.
    """
    frames: Dict[str, pd.DataFrame] = {}
    problems: List[str] = []

    daily, label, err = fetch_ohlc(coin_id, days=365)
    if daily is None:
        problems.append(f"CoinGecko daily data unavailable: {err}")
        frames["1d"] = pd.DataFrame()
    else:
        frames["1d"] = daily
        problems.append(
            "1D regime is based on CoinGecko 4-day candles — Yahoo has no data for this "
            "coin and CoinGecko serves no true daily interval. Regime reads will be coarser."
        )

    four_h, label, err = fetch_ohlc(coin_id, days=30)
    if four_h is None:
        problems.append(f"CoinGecko 4H data unavailable: {err}")
        frames["4h"] = pd.DataFrame()
    else:
        frames["4h"] = four_h

    fine, label, err = fetch_ohlc(coin_id, days=1)
    if fine is None:
        problems.append(f"CoinGecko intraday data unavailable: {err}")
        frames["5m"] = pd.DataFrame()
        frames["1h"] = pd.DataFrame()
    else:
        frames["5m"] = fine
        frames["1h"] = fine
        problems.append(
            "1H confirmation uses CoinGecko 30-minute candles (its finest free interval); "
            "there is no 5-minute or 1-hour data on this source."
        )

    return frames, problems


# ---------------------------------------------------------------------
# market_chart: price series, which can be resampled into finer candles
# than the fixed /ohlc buckets allow.
#
# CoinGecko's price-point spacing for market_chart is:
#     days=1      -> ~5-minute points
#     days=2..90  -> hourly points
#     days=91+    -> daily points
#
# Resampling 5-minute points up to 1H produces genuine OHLC (12 points per
# bucket). Resampling to the SAME spacing as the source does not — each bucket
# holds one point, so Open=High=Low=Close. That is flagged wherever it applies,
# because swing detection reads High/Low and will simply be reading closes.
# ---------------------------------------------------------------------

def fetch_price_series(coin_id: str, days: int, timeout: int = 15
                        ) -> Tuple[Optional[pd.Series], Optional[str]]:
    """Fetch a timestamped price series. Returns (series, error)."""
    try:
        resp = requests.get(
            MARKET_CHART_URL.format(coin_id=coin_id),
            params={"vs_currency": "usd", "days": days},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        points = data.get("prices") if isinstance(data, dict) else None
        if not points:
            return None, "CoinGecko returned no price points."
        df = pd.DataFrame(points, columns=["ts", "price"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        series = df.set_index("ts")["price"].dropna()
        if series.empty:
            return None, "All price points were null."
        return series, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def series_to_ohlc(series: pd.Series, rule: str) -> pd.DataFrame:
    """Resample a price series into OHLC candles at `rule` (e.g. '1h')."""
    if series is None or series.empty:
        return pd.DataFrame()
    out = series.resample(rule).agg(["first", "max", "min", "last"]).dropna()
    out.columns = ["Open", "High", "Low", "Close"]
    return out


def build_frames_v2(coin_id: str) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Best-quality frame set CoinGecko can provide.

        "1d" <- daily closes over a year (market_chart, days=365)
        "4h" <- NATIVE 4-hour OHLC candles (ohlc, days=30)
        "1h" <- 5-minute points resampled to 1H (real OHLC within each hour)
        "5m" <- 5-minute points (finest available)

    The 4H frame is genuinely better than the Yahoo path, which has to resample
    1H into 4H. The daily frame is weaker: one close per day means
    Open=High=Low=Close, so daily swing detection reads closes only.
    """
    frames: Dict[str, pd.DataFrame] = {}
    problems: List[str] = []

    # --- 4H: native OHLC, the most important frame for this strategy ---
    four_h, _label, err = fetch_ohlc(coin_id, days=30)
    if four_h is None:
        problems.append(f"CoinGecko 4H unavailable: {err}")
        frames["4h"] = pd.DataFrame()
    else:
        frames["4h"] = four_h

    # --- Daily: one close per day ---
    daily_series, err = fetch_price_series(coin_id, days=365)
    if daily_series is None:
        problems.append(f"CoinGecko daily unavailable: {err}")
        frames["1d"] = pd.DataFrame()
    else:
        frames["1d"] = series_to_ohlc(daily_series, "1D")
        problems.append(
            "1D candles are built from one closing price per day, so Open/High/Low all "
            "equal the Close. Daily swing detection is therefore reading closes only — "
            "regime and moving averages are unaffected, but daily wicks are invisible."
        )

    # --- Intraday: 5-minute points, also resampled up to 1H ---
    fine, err = fetch_price_series(coin_id, days=1)
    if fine is None:
        problems.append(f"CoinGecko intraday unavailable: {err}")
        frames["5m"] = pd.DataFrame()
        frames["1h"] = pd.DataFrame()
    else:
        frames["5m"] = series_to_ohlc(fine, "5min")
        frames["1h"] = series_to_ohlc(fine, "1h")

    return frames, problems
