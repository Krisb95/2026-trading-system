"""
Bull Run Strategy V2 — trading analysis dashboard.

Discretionary trading support tool. It does NOT execute trades, connect to an
exchange, or place/modify orders — you enter everything manually on your own
platform. Setup scores measure checklist confluence only; they are not
win-probability estimates, edge claims, or profitability guarantees.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
import zoneinfo

from data_layer import fetch_quote, DataStatus, to_local_display, curl_cffi_is_available
from scoring import (score_setup, weakest_components, trade_policy_for_grade,
                      POSITIVE_COMPONENTS, NEGATIVE_COMPONENTS)
from readiness import (determine_readiness, ReadinessInputs, Readiness, display_label,
                        ENTRY_SEQUENCE_STAGES, missing_entry_sequence_stages,
                        READINESS_DESCRIPTIONS)
from risk_calc import calculate_risk, InvalidRiskInputError
from backtest import (run_backtest, compute_metrics, split_in_out_sample,
                       example_sma_crossover_signals)
from portfolio import OpenPosition, summarize_portfolio_risk
from stops import StopManager, suggest_management_label
from position_math import PositionSnapshot, analyse_scale_in, ScaleVerdict
from scanner import fetch_multi_timeframe, analyze_candidate, scan_universe
from strategy_backtest import run_strategy_backtest, stats_by_grade, verdict
from universe import fetch_top_cryptos
import coingecko
import exchanges
import venues as venues_mod
import storage
import tracking
from formatting import format_price

APP_BUILD = "2026-09-21-b21 (1H in universe scan + backtest)"

st.set_page_config(page_title="Bull Run Strategy V2", page_icon="📈", layout="wide")

DB_READY = True
DB_ERROR = None
try:
    storage.init_db()
except Exception as _db_exc:  # pragma: no cover - environment dependent
    DB_READY = False
    DB_ERROR = f"{type(_db_exc).__name__}: {_db_exc}"

st.title("📈 Bull Run Strategy V2")
st.caption(f"Build `{APP_BUILD}` · data persisted to SQLite")
if not DB_READY:
    st.error(
        f"Database could not be initialised, so the Journal and Positions tabs will "
        f"not save anything this session. Everything else still works.\n\n`{DB_ERROR}`"
    )
st.caption(
    "Discretionary trading support tool. Protect capital first — a no-trade decision "
    "is valid. Nothing here executes trades or connects to an exchange. Setup scores "
    "are a checklist quality measure, not a win probability or guarantee."
)

@st.cache_data(ttl=3600, show_spinner=False)
def _load_crypto_universe():
    return fetch_top_cryptos(limit=100)


@st.cache_data(ttl=120, show_spinner=False)
def _cached_exchange_frames(ticker: str):
    """Exchange candles, cached briefly. Exchanges tolerate far more traffic
    than CoinGecko, so this cache is about responsiveness, not rate limits."""
    r = exchanges.build_frames(ticker)
    return r.frames, r.source, r.problems


@st.cache_data(ttl=30, show_spinner=False)
def _cached_exchange_spot(ticker: str):
    return exchanges.fetch_spot(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_cg_frames(coin_id: str):
    """CoinGecko frames, cached for 5 minutes. Its free tier rate-limits hard
    (HTTP 429), and a scan costs three calls, so repeat scans of the same coin
    must not re-hit the API."""
    return coingecko.build_frames_v2(coin_id)


@st.cache_data(ttl=60, show_spinner=False)
def _cached_cg_spot(coin_id: str):
    """CoinGecko spot price, cached for 60 seconds."""
    return coingecko.fetch_spot_price(coin_id)


CRYPTO_TICKERS_ALL, CRYPTO_CG_IDS, CRYPTO_IS_LIVE, CRYPTO_NOTE = _load_crypto_universe()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_venue_listings():
    listings = venues_mod.fetch_listings()
    return listings.bybit, listings.hyperliquid, listings.problems

COMMODITY_TICKERS = {
    "Gold (Futures)": "GC=F", "Silver (Futures)": "SI=F", "Platinum (Futures)": "PL=F",
    "Copper (Futures)": "HG=F", "WTI Crude Oil (Futures)": "CL=F",
    "Brent Crude Oil (Futures)": "BZ=F", "Natural Gas (Futures)": "NG=F",
    "Gold ETF (GLD)": "GLD", "Silver ETF (SLV)": "SLV", "Energy Sector ETF (XLE)": "XLE",
}

# ---------------------------------------------------------------------
# Sidebar settings (shared across tabs)
# ---------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")
live_threshold = st.sidebar.number_input("LIVE threshold (seconds)", value=60, min_value=5)
stale_threshold = st.sidebar.number_input("STALE threshold (seconds)", value=900, min_value=60)
tz_name = st.sidebar.text_input("Timezone (IANA)", value="Australia/Sydney")
try:
    local_tz = zoneinfo.ZoneInfo(tz_name)
    st.sidebar.caption(f"Local now: {datetime.now(local_tz).strftime('%Y-%m-%d %H:%M %Z')}")
except Exception:
    st.sidebar.error(f"'{tz_name}' is not a valid IANA timezone — using UTC.")
    local_tz = timezone.utc

# A free CoinGecko demo key raises the rate limit a lot. Optional.
_cg_key = ""
try:
    _cg_key = st.secrets.get("COINGECKO_API_KEY", "")
except Exception:
    _cg_key = ""
if _cg_key:
    coingecko.set_api_key(_cg_key)

st.sidebar.markdown("---")
st.sidebar.subheader("Tradable universe")
venue_mode = st.sidebar.radio(
    "Restrict coins to",
    ["Bybit or Hyperliquid", "Bybit AND Hyperliquid", "All top-100"],
    key="venue_mode",
    help="Market-cap rank includes exchange tokens (WBT, OKB) and wrapped assets you "
         "cannot trade as perpetuals. Filtering to your venues keeps the scan actionable.")

if venue_mode == "All top-100":
    CRYPTO_TICKERS = dict(CRYPTO_TICKERS_ALL)
    VENUE_TAGS = {}
    st.sidebar.caption(f"{len(CRYPTO_TICKERS)} coins (unfiltered).")
else:
    _by, _hl, _vprob = _load_venue_listings()
    _listings = venues_mod.VenueListings(bybit=_by, hyperliquid=_hl, problems=_vprob)
    CRYPTO_TICKERS, VENUE_TAGS, _vnotes = venues_mod.filter_universe(
        CRYPTO_TICKERS_ALL, _listings,
        require_both=(venue_mode == "Bybit AND Hyperliquid"))
    st.sidebar.caption(f"Bybit: {len(_by)} perps · Hyperliquid: {len(_hl)} perps")
    for _n in _vnotes:
        if "NOT applied" in _n or "narrower" in _n:
            st.sidebar.warning(_n)
        else:
            st.sidebar.caption(_n)

st.sidebar.markdown("---")
st.sidebar.subheader("Data source")
crypto_source = st.sidebar.radio(
    "Crypto prices from",
    ["Exchange (recommended)", "CoinGecko", "Yahoo Finance"],
    key="crypto_source",
    help=("Exchange uses Binance, falling back to Kraken. It gives true OHLCV at every "
          "interval with no meaningful rate limit — the best option. CoinGecko covers "
          "more obscure coins but rate-limits hard and has no true daily OHLC. Yahoo "
          "misses newer listings. Sources fall back to each other automatically."),
)
CRYPTO_SOURCE = ("exchange" if crypto_source.startswith("Exchange")
                 else "coingecko" if crypto_source.startswith("CoinGecko")
                 else "yahoo")
CRYPTO_PREFERS_CG = CRYPTO_SOURCE == "coingecko"
st.sidebar.caption(
    "Stocks and commodities always use Yahoo Finance — CoinGecko has no equities "
    "or futures data."
)

st.sidebar.markdown("---")
st.sidebar.caption(f"curl_cffi installed: **{curl_cffi_is_available()}**")
st.sidebar.caption(f"Scannable crypto: {len(CRYPTO_TICKERS)} symbols")
if coingecko.has_api_key():
    st.sidebar.caption("CoinGecko API key: **set** (higher rate limit)")
else:
    st.sidebar.caption(
        "CoinGecko API key: not set — free anonymous limits apply, so rapid "
        "scanning will hit 429s. A free demo key in Streamlit secrets "
        "(`COINGECKO_API_KEY`) raises this substantially.")
if not CRYPTO_IS_LIVE:
    st.sidebar.warning(CRYPTO_NOTE)
st.sidebar.warning(
    "On free hosting the database is wiped when the Space rebuilds. "
    "Export your journal to CSV after any session that matters."
)

tab_market, tab_scan, tab_positions, tab_risk, tab_journal, tab_track, tab_backtest = st.tabs(
    ["🌍 Market", "🎯 Scanner", "📋 Positions", "🧮 Risk", "📓 Journal",
     "📈 Tracking", "🔁 Backtest"]
)

# ---------------------------------------------------------------------
# TAB: Market & data health
# ---------------------------------------------------------------------
with tab_market:
    st.subheader("Market overview & data health")

    c1, c2 = st.columns([2, 1])
    with c1:
        asset_type = st.radio("Asset type", ["Crypto", "Stock", "Commodity"],
                               horizontal=True, key="mkt_type")
        if asset_type == "Crypto":
            _label = st.selectbox("Symbol", list(CRYPTO_TICKERS), key="mkt_c")
            ticker = CRYPTO_TICKERS[_label]
            cg_id = CRYPTO_CG_IDS.get(_label)
            asset_class = "crypto"
        elif asset_type == "Commodity":
            ticker = COMMODITY_TICKERS[st.selectbox("Symbol", list(COMMODITY_TICKERS), key="mkt_o")]
            cg_id = None
            asset_class = "commodity"
        else:
            ticker = st.text_input("Stock ticker", value="AAPL", key="mkt_s").upper().strip()
            cg_id = None
            asset_class = "stock"
    with c2:
        st.write("")
        st.write("")
        do_fetch = st.button("🔄 Fetch / Retry", use_container_width=True)

    if do_fetch or st.session_state.get("quote_ticker") != ticker:
        with st.spinner(f"Fetching {ticker}…"):
            from data_layer import PriceQuote
            now = datetime.now(timezone.utc)
            q = None
            st.session_state.fallback_note = None

            def _cg_quote():
                """Build a PriceQuote from CoinGecko spot, or None."""
                price, updated, err = _cached_cg_spot(cg_id)
                if price is None:
                    return None, err
                status = DataStatus.LIVE
                age = None
                if updated is not None:
                    age = (now - updated).total_seconds()
                    status = (DataStatus.LIVE if age <= live_threshold
                              else DataStatus.DELAYED if age <= stale_threshold
                              else DataStatus.STALE)
                return PriceQuote(ticker=ticker, price=price, fetched_at_utc=now,
                                   bar_time_utc=updated, status=status,
                                   provider="CoinGecko (spot)", interval="spot",
                                   bar_interval_seconds=0,
                                   effective_age_seconds=age), None

            use_exchange_first = (asset_class == "crypto" and CRYPTO_SOURCE == "exchange")
            use_cg_first = (asset_class == "crypto" and CRYPTO_PREFERS_CG and cg_id)

            if use_exchange_first:
                px, src, ex_err = _cached_exchange_spot(ticker)
                if px is not None:
                    q = PriceQuote(ticker=ticker, price=px, fetched_at_utc=now,
                                    bar_time_utc=now, status=DataStatus.LIVE,
                                    provider=f"{src} (spot)", interval="spot",
                                    bar_interval_seconds=0, effective_age_seconds=0.0)
                else:
                    # Exchange failed — try CoinGecko, then Yahoo.
                    q, cg_err = (_cg_quote() if cg_id else (None, "no CoinGecko id"))
                    if q is None:
                        q = fetch_quote(ticker, asset_class, yf,
                                         live_threshold_seconds=live_threshold,
                                         stale_threshold_seconds=stale_threshold,
                                         max_retries=2)
                        st.session_state.fallback_note = (
                            f"Exchanges unavailable ({ex_err}) and CoinGecko failed "
                            f"({cg_err}) — fell back to Yahoo Finance.")
                    else:
                        st.session_state.fallback_note = (
                            f"Exchanges unavailable ({ex_err}) — used CoinGecko instead.")
            elif use_cg_first:
                q, cg_err = _cg_quote()
                if q is None:
                    q = fetch_quote(ticker, asset_class, yf,
                                     live_threshold_seconds=live_threshold,
                                     stale_threshold_seconds=stale_threshold, max_retries=2)
                    st.session_state.fallback_note = (
                        f"CoinGecko failed ({cg_err}) — fell back to Yahoo Finance.")
            else:
                q = fetch_quote(ticker, asset_class, yf,
                                 live_threshold_seconds=live_threshold,
                                 stale_threshold_seconds=stale_threshold, max_retries=2)
                if q.status == DataStatus.UNAVAILABLE and cg_id:
                    cg_q, cg_err = _cg_quote()
                    if cg_q is not None:
                        q = cg_q
                        st.session_state.fallback_note = (
                            f"Yahoo Finance has no data for {ticker} — used CoinGecko instead.")
                    else:
                        st.session_state.fallback_note = (
                            f"Yahoo has no data for {ticker} and CoinGecko also failed: {cg_err}")

            st.session_state.quote = q
            st.session_state.quote_ticker = ticker

    quote = st.session_state.get("quote")
    if quote is None:
        st.info("Press Fetch to load a quote.")
    else:
        icons = {DataStatus.LIVE: "🟢", DataStatus.DELAYED: "🟡",
                 DataStatus.STALE: "🟠", DataStatus.UNAVAILABLE: "🔴"}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Instrument", quote.ticker)
        m2.metric("Price", f"${format_price(quote.price)}" if quote.price else "—")
        m3.metric("Status", f"{icons[quote.status]} {quote.status.value}")
        m4.metric("Interval", quote.interval)

        age = (f"{quote.effective_age_seconds:,.0f}s past bar close"
               if quote.effective_age_seconds is not None else "—")
        st.caption(
            f"Provider: {quote.provider} · Bar: {to_local_display(quote.bar_time_utc, local_tz)} · "
            f"Fetched: {to_local_display(quote.fetched_at_utc, local_tz)} · "
            f"Effective age: {age} · Attempts: {quote.attempts}"
        )

        if st.session_state.get("fallback_note"):
            st.info(st.session_state.fallback_note)
        if quote.error:
            st.error(f"Fetch issue: {quote.error}")
        if quote.interval == "1d":
            st.info(
                "Only daily candles were available (intraday returned nothing — normal when a "
                "market is closed). Daily data is never reported as LIVE, since it cannot tell "
                "you what happened in the last few minutes."
            )
        if quote.status in (DataStatus.STALE, DataStatus.UNAVAILABLE):
            st.error("🚫 DATA ERROR — NO TRADE. This data is not reliable enough to act on.")
        elif quote.status == DataStatus.DELAYED:
            st.warning("Price is DELAYED — fine for planning, risky for timing-sensitive entries.")

# ---------------------------------------------------------------------
# TAB: Scanner
# ---------------------------------------------------------------------
with tab_scan:
    st.subheader("Candidate scanner")
    st.caption(
        "Derives the checklist from real 1D / 4H / 1H structure. Every item is "
        "overridable — ❔ means it could not be evaluated and scores zero rather than guessing."
    )

    mode = st.radio("Mode",
                     ["Rank the universe", "Auto-scan one instrument", "Manual checklist"],
                     key="scan_mode",
                     help="Rank the universe scores many coins and shortlists the best; "
                          "the other modes analyse a single instrument in full depth.")

    score_evidence, seq_evidence = {}, {}

    if mode == "Rank the universe":
        st.caption(
            "Scores many instruments and shortlists the best. To stay inside rate "
            "limits this uses **4H candles only** — so 1H entry confirmation cannot be "
            "evaluated (capping scores at 9/10) and the regime is read from 4H rather "
            "than daily. Treat results as a shortlist, then run a full single-instrument "
            "scan on anything promising."
        )

        u1, u2, u3 = st.columns(3)
        universe_size = u1.selectbox("How many coins", [10, 20, 30, 50],
                                      index=1, key="uni_size",
                                      help="More coins means more API calls and a longer wait.")
        uni_direction = u2.selectbox("Direction", ["Auto", "Long", "Short"], key="uni_dir")
        uni_min_rr = u3.number_input("Min R:R", min_value=1.0, value=2.0,
                                      step=0.5, key="uni_rr")

        per_coin = 0.6 if CRYPTO_SOURCE == "exchange" else 1.4   # 4 calls/coin: 4H, 1D, 1H, spot
        est = universe_size * per_coin
        st.caption(
            f"Roughly {est:.0f}s for {universe_size} coins using "
            f"{'exchange data (fast — no meaningful rate limit)' if CRYPTO_SOURCE == 'exchange' else 'CoinGecko (calls spaced to avoid 429s)'}."
        )

        st.checkbox("Record B-or-better setups for forward tracking",
                    value=True, key="track_signals",
                    help="Builds a real track record: each setup's plan is frozen now and "
                         "checked later against what price actually did.")

        if st.button("🔎 Scan universe", use_container_width=True):
            labels = list(CRYPTO_TICKERS)[:universe_size]
            instruments = [(lbl, CRYPTO_TICKERS[lbl],
                             (CRYPTO_TICKERS[lbl], CRYPTO_CG_IDS.get(lbl)))
                           for lbl in labels]

            bar = st.progress(0.0, text="Starting…")

            def _loader(payload):
                ticker, coin_id = payload
                if CRYPTO_SOURCE == "exchange":
                    df, err = exchanges.fetch_binance_klines(ticker, "4h")
                    fetch_1d = exchanges.fetch_binance_klines
                    if df is None:
                        df, err = exchanges.fetch_kraken_ohlc(ticker, "4h")
                        fetch_1d = exchanges.fetch_kraken_ohlc
                    if df is not None and not df.empty:
                        # Daily candles let 1D/4H alignment be judged from two
                        # genuinely independent timeframes. Without them the
                        # alignment points are withheld rather than faked.
                        d1, _e = fetch_1d(ticker, "1d")
                        # 1H lets the entry-confirmation component be judged, so
                        # the full 10 points are reachable in a universe scan.
                        h1, _e = fetch_1d(ticker, "1h")
                        return {"4h": df, "1d": d1 if d1 is not None else pd.DataFrame(),
                                "1h": h1 if h1 is not None else pd.DataFrame()}, []
                if coin_id:
                    df, _label, err = coingecko.fetch_ohlc(coin_id, days=30)
                    if df is not None and not df.empty:
                        return {"4h": df, "1d": pd.DataFrame(), "1h": pd.DataFrame()}, []
                return {}, [err or "no data"]

            def _spot(payload):
                """Live spot price for the Price column and entry maths. Without
                this, 'Price' was the last 4H close — up to four hours stale."""
                ticker, coin_id = payload
                px, _src, _err = exchanges.fetch_spot(ticker)
                if px is None and coin_id:
                    px, _upd, _e = coingecko.fetch_spot_price(coin_id)
                return px

            def _progress(i, total, label):
                bar.progress(min(i / max(total, 1), 1.0), text=f"{i}/{total} · {label}")

            with st.spinner("Scanning…"):
                ranked = scan_universe(
                    instruments, _loader, min_rr=uni_min_rr,
                    direction_override=None if uni_direction == "Auto" else uni_direction,
                    progress_callback=_progress, spot_loader=_spot)
            bar.empty()
            st.session_state.ranked = ranked
            if st.session_state.get("track_signals", True):
                added = tracking.record_from_ranked(ranked, VENUE_TAGS)
                if added:
                    st.toast(f"Recorded {added} new signal(s) for forward tracking.")

        ranked = st.session_state.get("ranked")
        if ranked:
            scored_all = [r for r in ranked if r.error is None]
            failed = [r for r in ranked if r.error is not None]

            f1, f2 = st.columns(2)
            min_score = f1.number_input(
                "Show scores of at least", min_value=0.0, max_value=10.0, value=0.0,
                step=0.5, key="uni_min_score",
                help="Scores are whole numbers, so 8.5 behaves the same as 9. "
                     "A 9 or 10 appears in only a few percent of market states.")
            a_plus_only = f2.checkbox("A+ only", key="uni_aplus_only")

            ok = [r for r in scored_all
                  if r.score >= min_score and (not a_plus_only or r.grade == "A+")]
            hidden = len(scored_all) - len(ok)

            st.markdown(f"##### Results · {len(ok)} shown"
                        + (f", {hidden} hidden by filter" if hidden else "")
                        + (f", {len(failed)} failed" if failed else ""))
            if scored_all and not ok:
                best = max(scored_all, key=lambda r: r.score)
                st.info(
                    f"Nothing meets the filter right now. The best setup scanned was "
                    f"**{best.label} at {best.score:.0f}/10 ({best.grade})**. An empty list is "
                    f"a legitimate answer — your rulebook treats no-trade as a valid decision.")
            if ok:
                def _gap(r):
                    if not (r.entry and r.price):
                        return "—"
                    g = (r.entry - r.price) / r.price * 100
                    return "at price" if abs(g) < 0.05 else f"{g:+.2f}%"

                table = pd.DataFrame([{
                    "Instrument": r.label,
                    "Venue": "/".join(VENUE_TAGS.get(r.label, [])) or "—",
                    "Score": f"{r.score:.1f}",
                    "Grade": r.grade,
                    "Dir": r.direction or "—",
                    "Live price": format_price(r.price) + ("" if r.price_is_live else " *"),
                    "Entry": format_price(r.entry),
                    "Entry vs live": _gap(r),
                    "Stop": format_price(r.stop),
                    "Target": format_price(r.target),
                    "R:R": f"{r.reward_risk:.2f}" if r.reward_risk else "—",
                    "Regime": r.regime.title(),
                } for r in ok])
                st.dataframe(table, use_container_width=True, hide_index=True)
                st.caption(
                    "**Entry** is the planned price to place your order at — a limit order "
                    "at a confluence zone, not the market price. **Entry vs live** shows how "
                    "far price must travel to fill it. Stop, target and R:R are all measured "
                    "from that entry. A price marked * is the last 4H close because the live "
                    "spot fetch failed.")

                tradeable = [r for r in ok if r.grade in ("A+", "B")]
                if tradeable:
                    st.success(
                        f"{len(tradeable)} setup(s) at grade B or better: "
                        + ", ".join(f"{r.label} ({r.grade}, {r.score:.1f})" for r in tradeable[:5])
                    )
                else:
                    st.info(
                        "Nothing reached grade B. Per the rulebook, C setups are not traded — "
                        "a no-trade decision is valid and common."
                    )
            if failed:
                with st.expander(f"⚠️ {len(failed)} instrument(s) could not be scored"):
                    for r in failed:
                        st.caption(f"**{r.label}** — {r.error}")

        score_evidence, seq_evidence = {}, {}

    elif mode == "Auto-scan one instrument":
        s1, s2, s3 = st.columns([2, 1, 1])
        with s1:
            stype = st.radio("Asset type", ["Crypto", "Stock", "Commodity"],
                              horizontal=True, key="scan_type")
            if stype == "Crypto":
                _slabel = st.selectbox("Symbol", list(CRYPTO_TICKERS), key="scan_c")
                scan_ticker = CRYPTO_TICKERS[_slabel]
                scan_cg_id = CRYPTO_CG_IDS.get(_slabel)
            elif stype == "Commodity":
                scan_ticker = COMMODITY_TICKERS[st.selectbox("Symbol", list(COMMODITY_TICKERS), key="scan_o")]
                scan_cg_id = None
            else:
                scan_ticker = st.text_input("Stock ticker", value="AAPL", key="scan_s").upper().strip()
                scan_cg_id = None
        with s2:
            dir_choice = st.selectbox("Direction", ["Auto", "Long", "Short"], key="scan_dir")
        with s3:
            min_rr = st.number_input("Min R:R", min_value=1.0, value=2.0, step=0.5, key="scan_rr")

        if st.button("🔍 Scan", use_container_width=True):
            with st.spinner(f"Analysing {scan_ticker} across 1D / 4H / 1H…"):
                use_exchange_first = (stype == "Crypto" and CRYPTO_SOURCE == "exchange")
                use_cg_first = (stype == "Crypto" and CRYPTO_PREFERS_CG and scan_cg_id)
                frames, problems = {}, []

                if use_exchange_first:
                    frames, ex_source, problems = _cached_exchange_frames(scan_ticker)
                    if ex_source:
                        problems = ([f"True OHLCV candles from {ex_source} "
                                      f"(all timeframes native, nothing resampled)."]
                                    + list(problems))
                    else:
                        # Neither exchange lists it — fall back to CoinGecko.
                        if scan_cg_id:
                            frames, cg_problems = _cached_cg_frames(scan_cg_id)
                            problems = ([f"{scan_ticker} is not listed on Binance or Kraken "
                                          f"— using CoinGecko candles instead."] + cg_problems)
                        if not frames or all(f is None or f.empty for f in frames.values()):
                            frames, y_problems = fetch_multi_timeframe(scan_ticker, yf)
                            problems = ["Exchanges and CoinGecko both failed — "
                                        "fell back to Yahoo Finance."] + y_problems
                elif use_cg_first:
                    frames, problems = _cached_cg_frames(scan_cg_id)
                    problems = ["Candles from CoinGecko (4H is native, not resampled)."] + problems
                    if all(f is None or f.empty for f in frames.values()):
                        frames, problems = fetch_multi_timeframe(scan_ticker, yf)
                        problems = ["CoinGecko returned nothing — fell back to Yahoo Finance."] + problems
                    else:
                        # Partial failure (commonly a rate-limited daily call):
                        # fill only the missing frames from Yahoo instead of
                        # discarding the good CoinGecko candles we already have.
                        missing = [k for k, f in frames.items() if f is None or f.empty]
                        if missing:
                            y_frames, _y_problems = fetch_multi_timeframe(scan_ticker, yf)
                            filled = []
                            for k in missing:
                                yf_frame = y_frames.get(k)
                                if yf_frame is not None and not yf_frame.empty:
                                    frames[k] = yf_frame
                                    filled.append(k)
                            if filled:
                                problems.append(
                                    f"Gap-filled {', '.join(filled)} from Yahoo Finance after "
                                    f"CoinGecko failed on those timeframes — this scan mixes "
                                    f"two data sources.")
                else:
                    frames, problems = fetch_multi_timeframe(scan_ticker, yf)
                    if all(f is None or f.empty for f in frames.values()) and scan_cg_id:
                        cg_frames, cg_problems = _cached_cg_frames(scan_cg_id)
                        if any(f is not None and not f.empty for f in cg_frames.values()):
                            frames = cg_frames
                            problems = ([f"Yahoo Finance has no data for {scan_ticker}; "
                                          f"using CoinGecko candles instead."] + cg_problems)

                _live = None
                if stype == "Crypto":
                    _live, _src, _err = _cached_exchange_spot(scan_ticker)
                st.session_state.analysis = analyze_candidate(
                    scan_ticker, frames,
                    direction_override=None if dir_choice == "Auto" else dir_choice,
                    min_rr=min_rr, data_problems=problems, live_price=_live)
                # Bump the scan id so the evidence checkboxes below get FRESH
                # widget keys. Streamlit ignores `value=` for a key that already
                # exists in session state, so reusing keys would freeze the
                # checkboxes (and therefore the score) at the first scan's result.
                st.session_state.scan_id = st.session_state.get("scan_id", 0) + 1

        analysis = st.session_state.get("analysis")
        if analysis is None:
            st.info("Pick an instrument and press Scan.")
        else:
            for p in analysis.data_problems:
                st.warning(f"Data gap: {p}")

            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Instrument", analysis.ticker)
            h2.metric("1D regime", analysis.regime_1d.title())
            h3.metric("Direction", analysis.direction or "—")
            h4.metric("Price", format_price(analysis.current_price))
            if analysis.price_source:
                st.caption(
                    f"Entry price taken from the latest **{analysis.price_source}** close. "
                    f"If this differs from the Market tab, one of them is a slightly older bar — "
                    f"always confirm against your broker before entering."
                )

            plan = analysis.entry_plan
            if plan is not None:
                badge = {"AT_ZONE": "🟢 AT ZONE", "APPROACHING": "🟡 APPROACHING",
                         "FAR": "⚪ TOO FAR", "MISSED": "🔴 MISSED"}[plan.status]
                st.markdown(f"##### Entry plan · {badge} · order type: **{plan.order_type}**")
                z1, z2, z3 = st.columns(3)
                z1.metric("Entry zone",
                          f"{format_price(plan.zone_low)} – {format_price(plan.zone_high)}")
                z2.metric("Distance to zone", f"{plan.distance_pct:.2f}%",
                          f"{plan.distance_atr:.1f} ATR")
                z3.metric("Confluence", f"{plan.confluence} source(s)")
                st.caption(f"{plan.rationale}  \nSources: {', '.join(plan.sources)}")
                if plan.confluence == 1:
                    st.caption("⚠️ Only one source supports this level — weaker than a "
                               "zone where a swing, a Fib level and equal highs/lows agree.")
                if analysis.entry_is_planned and analysis.current_price:
                    gap = (analysis.current_price - analysis.entry) / analysis.current_price * 100
                    st.info(
                        f"Planned entry **{format_price(analysis.entry)}** vs live "
                        f"**{format_price(analysis.current_price)}** "
                        f"({abs(gap):.1f}% {'better' if (gap > 0) == (analysis.direction == 'Long') else 'worse'}). "
                        f"Stop, target and reward:risk below are all measured from the "
                        f"planned entry, not the live price.")
                if analysis.alternative_zones:
                    with st.expander(f"Other candidate zones ({len(analysis.alternative_zones)})"):
                        for z in analysis.alternative_zones:
                            st.caption(
                                f"**{format_price(z.zone_low)} – {format_price(z.zone_high)}** · "
                                f"{z.status} · {z.distance_pct:.2f}% away · "
                                f"{z.confluence} source(s): {', '.join(z.sources)}")
            st.markdown("---")

            if analysis.stop is not None and analysis.target is not None:
                l1, l2, l3 = st.columns(3)
                l1.metric("Entry", format_price(analysis.entry))
                l2.metric("Structural stop", format_price(analysis.stop))
                l3.metric("Structural target", format_price(analysis.target))
                if analysis.reward_risk is not None:
                    ok = analysis.reward_risk >= min_rr
                    (st.success if ok else st.warning)(
                        f"Reward:risk **{analysis.reward_risk:.2f}:1** — {analysis.level_reason}")
                if st.button("📋 Send these levels to the Risk tab"):
                    st.session_state.prefill = {
                        "entry": float(analysis.entry), "stop": float(analysis.stop),
                        "target": float(analysis.target),
                        "direction": analysis.direction or "Long",
                        "ticker": analysis.ticker,
                    }
                    st.success("Levels copied — open the 🧮 Risk tab.")
            else:
                st.warning("Could not derive a full entry/stop/target from confirmed structure.")

            sid = st.session_state.get("scan_id", 0)
            auto_result = score_setup(analysis.score_dict())
            st.markdown(
                f"##### Auto-derived evidence · scanner says "
                f"**{auto_result.normalized_score:.1f}/10 ({auto_result.label})**"
            )
            st.caption(
                "Ticking or unticking below overrides the scanner; the score at the "
                "bottom reflects your edits, this one does not."
            )
            for key, points, label in POSITIVE_COMPONENTS + NEGATIVE_COMPONENTS:
                item = analysis.score_evidence.get(key)
                val = item.value if item else None
                icon = {True: "✅", False: "❌", None: "❔"}[val]
                score_evidence[key] = st.checkbox(
                    f"{icon} {label} ({points:+d})", value=bool(val),
                    key=f"as_{sid}_{key}")
                if item:
                    st.caption(f"　↳ {item.reason}")

            with st.expander("Entry sequence detail"):
                for key, desc in ENTRY_SEQUENCE_STAGES:
                    item = analysis.sequence_evidence.get(key)
                    val = item.value if item else None
                    icon = {True: "✅", False: "❌", None: "❔"}[val]
                    seq_evidence[key] = st.checkbox(
                        f"{icon} {desc}", value=bool(val), key=f"aq_{sid}_{key}")
                    if item:
                        st.caption(f"　↳ {item.reason}")
    else:
        st.markdown("##### Entry sequence")
        cols = st.columns(2)
        for i, (key, desc) in enumerate(ENTRY_SEQUENCE_STAGES):
            with cols[i % 2]:
                seq_evidence[key] = st.checkbox(desc, key=f"ms_{key}")

        st.markdown("##### Confluence evidence")
        m1, m2 = st.columns(2)
        with m1:
            for key, points, label in POSITIVE_COMPONENTS:
                score_evidence[key] = st.checkbox(f"{label} (+{points})", key=f"mp_{key}")
        with m2:
            for key, points, label in NEGATIVE_COMPONENTS:
                score_evidence[key] = st.checkbox(f"{label} ({points})", key=f"mn_{key}")

    st.markdown("---")
    result = score_setup(score_evidence)
    r1, r2 = st.columns([1, 2])
    r1.metric("Setup score", f"{result.normalized_score:.1f}/10", result.label)
    r2.info(trade_policy_for_grade(result.label))

    if result.label != "A+":
        missing = weakest_components(result)
        if missing:
            st.caption("Weakest: " + "; ".join(
                f"{li.label} (+{li.points_possible} unclaimed)" for li in missing))

    with st.expander("📊 Score breakdown — where every point went"):
        earned = sum(li.points_awarded for li in result.breakdown if li.points_awarded > 0)
        available = sum(li.points_possible for li in result.breakdown if li.points_possible > 0)
        st.caption(f"Earned **{earned}** of **{available}** positive points.")
        rows = []
        for li in result.breakdown:
            rows.append({
                "Component": li.label,
                "Worth": f"{li.points_possible:+d}",
                "Earned": f"{li.points_awarded:+d}",
                "Evidence": {True: "yes", False: "no", None: "could not evaluate"}[li.evidence],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "The two-point items (regime alignment, liquidity sweep + reclaim) are the "
            "difference between a C and an A+. If they never fire, 4–5/10 is the "
            "structural ceiling — that is the scoring working as designed, not a fault, "
            "but it is worth reading their reasons above to see whether you agree."
        )

    st.markdown("##### Readiness")
    q = st.session_state.get("quote")
    data_ok = q is not None and q.status in (DataStatus.LIVE, DataStatus.DELAYED)
    risk_ok = st.checkbox("Risk checks pass (sized, within budget)", key="rdy_risk")
    invalidated = st.checkbox("Structural invalidation has occurred", key="rdy_inval")
    entered = st.checkbox("I have manually entered this trade", key="rdy_active")

    readiness = determine_readiness(ReadinessInputs(
        data_is_valid=data_ok, invalidation_hit=invalidated, user_marked_active=entered,
        near_actionable_location=seq_evidence.get("location", False),
        confirmation_triggered=seq_evidence.get("confirmation", False),
        invalidation_defined=seq_evidence.get("structural_invalidation", False),
        risk_checks_pass=risk_ok))

    st.metric("Readiness", f"{display_label(readiness)} ({readiness.value})")
    st.caption(READINESS_DESCRIPTIONS[readiness])

    if not data_ok:
        st.info("Readiness shows DATA ERROR because no usable quote is loaded. "
                "Fetch the instrument on the 🌍 Market tab first.")

    if readiness in (Readiness.CONDITIONAL, Readiness.NOT_READY):
        gaps = missing_entry_sequence_stages(seq_evidence)
        if gaps:
            st.warning("What would confirm this:\n" + "\n".join(f"- {g}" for g in gaps))

# ---------------------------------------------------------------------
# TAB: Positions
# ---------------------------------------------------------------------
with tab_positions:
    st.subheader("Open positions")
    st.caption(
        "Live P/L, editable stop and target, scale-in checks, and one-click "
        "closing into the journal. Stored in the database, so it survives refreshes."
    )

    pos_equity = st.number_input(
        "Account equity ($) — used for risk percentages",
        min_value=0.0, value=10000.0, step=100.0, key="pos_equity")
    pos_max_risk = st.number_input(
        "Max risk per trade (% of equity)", min_value=0.1, max_value=100.0,
        value=1.0, step=0.1, key="pos_max_risk")

    with st.expander("➕ Add a position"):
        with st.form("add_pos"):
            p1, p2, p3 = st.columns(3)
            pa = p1.text_input("Asset (e.g. BTC-USD)")
            pc = p2.selectbox("Class", ["crypto", "stock", "commodity", "forex"])
            pdir = p3.selectbox("Direction", ["Long", "Short"])
            p4, p5, p6 = st.columns(3)
            pe = p4.number_input("Entry", min_value=0.0, format="%.8f")
            ps = p5.number_input("Stop", min_value=0.0, format="%.8f")
            pt = p6.number_input("Target (0 = none)", min_value=0.0, format="%.8f")
            p7, p8 = st.columns(2)
            pq = p7.number_input("Quantity", min_value=0.0, format="%.8f")
            plev = p8.number_input("Leverage", min_value=1.0, value=1.0, step=0.5)
            pnotes = st.text_area("Entry reason / notes")
            if st.form_submit_button("Add position") and pa and pe > 0 and pq > 0:
                storage.add_position(pa, pc, pdir, pe, ps, pq, target=pt or None,
                                      leverage=plev, entry_reason=pnotes)
                st.success(f"Added {pdir} {pa}")
                st.rerun()

    rows = storage.get_positions()
    if not rows:
        st.info("No open positions. Add one above, or send levels from the Scanner.")
    else:
        positions = [OpenPosition(asset=r["asset"], asset_class=r["asset_class"],
                                   direction=r["direction"], entry=r["entry"],
                                   stop=r["stop"], quantity=r["quantity"],
                                   contract_multiplier=r["contract_multiplier"] or 1.0)
                     for r in rows]
        summary = summarize_portfolio_risk(positions)
        s1, s2 = st.columns(2)
        s1.metric("Total open risk (to stops)", f"${summary.total_open_risk:,.2f}")
        pct = (summary.total_open_risk / pos_equity * 100) if pos_equity else 0
        s2.metric("As % of equity", f"{pct:.2f}%")
        if pct > pos_max_risk * 3:
            st.warning(
                f"Total open risk is {pct:.1f}% of equity across {len(rows)} positions. "
                f"Individually sized trades can still add up to concentrated exposure."
            )
        for w in summary.concentration_warnings:
            st.warning(w)

        st.markdown("---")

        for r in rows:
            pid = r["id"]
            header = (f"{r['direction']} {r['asset']} · entry {format_price(r['entry'])} "
                      f"· stop {format_price(r['stop'])}")
            with st.expander(header):
                # --- current price: auto-fetch for crypto, else manual ----
                pxkey = f"px_{pid}"
                cprice = st.session_state.get(pxkey, 0.0)
                fc1, fc2 = st.columns([2, 1])
                cprice = fc1.number_input("Current price", min_value=0.0,
                                           value=float(cprice), format="%.8f",
                                           key=f"cp_{pid}")
                if fc2.button("↻ Fetch", key=f"fetch_{pid}"):
                    px, src, err = exchanges.fetch_spot(r["asset"])
                    if px is not None:
                        st.session_state[f"cp_{pid}"] = float(px)
                        st.success(f"{src}: {format_price(px)}")
                        st.rerun()
                    else:
                        st.error(f"Could not fetch: {err}")

                snap = None
                if cprice > 0:
                    snap = PositionSnapshot(
                        direction=r["direction"], entry=r["entry"], stop=r["stop"],
                        target=r["target"], quantity=r["quantity"],
                        current_price=cprice,
                        contract_multiplier=r["contract_multiplier"] or 1.0)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Unrealised P/L", f"${snap.unrealised_pl:,.2f}",
                              f"{snap.unrealised_pct:+.2f}%")
                    m2.metric("Current R", f"{snap.r_multiple:+.2f}R")
                    m3.metric("If stop hit",
                              f"${snap.loss_at_stop:,.2f}",
                              "profit locked" if snap.loss_at_stop > 0 else "loss")
                    if snap.profit_at_target is not None:
                        m4.metric("If target hit", f"${snap.profit_at_target:,.2f}",
                                  f"R:R {snap.reward_risk:.2f}" if snap.reward_risk else None)

                    label = suggest_management_label(
                        r["direction"], invalidation_hit=False, risk_rule_breached=False,
                        structure_supports_tightening=False, r_multiple=snap.r_multiple)
                    if snap.stop_is_protecting_profit:
                        st.success(
                            f"Stop is beyond entry — at least "
                            f"${snap.loss_at_stop:,.2f} is locked in if it triggers.")
                    elif snap.r_multiple >= 1.5:
                        st.info(
                            f"Up {snap.r_multiple:.2f}R. The rulebook suggests considering a "
                            f"structure-based trailing stop past ~1.5R — only if real "
                            f"structure supports it, not to escape normal noise.")
                    st.caption(f"Suggested review label: **{label.value}** "
                               f"(a prompt to think, not an instruction)")
                else:
                    st.caption("Enter or fetch a current price to see live P/L.")

                # --- edit stop / target / size -----------------------------
                st.markdown("##### Adjust levels")
                e1, e2, e3 = st.columns(3)
                new_stop = e1.number_input("Stop", min_value=0.0,
                                            value=float(r["stop"]), format="%.8f",
                                            key=f"ns_{pid}")
                new_target = e2.number_input("Target (0 = none)", min_value=0.0,
                                              value=float(r["target"] or 0.0),
                                              format="%.8f", key=f"nt_{pid}")
                new_qty = e3.number_input("Quantity", min_value=0.0,
                                           value=float(r["quantity"]), format="%.8f",
                                           key=f"nq_{pid}")

                if st.button("Save changes", key=f"save_{pid}"):
                    widening = (new_stop < r["stop"] if r["direction"] == "Long"
                                else new_stop > r["stop"])
                    if widening:
                        st.error(
                            f"Refused: moving the stop from {format_price(r['stop'])} to "
                            f"{format_price(new_stop)} increases risk. Never move a stop "
                            f"further from entry. Close the trade instead if the thesis broke."
                        )
                    else:
                        storage.update_position(pid, stop=new_stop,
                                                 target=new_target or None,
                                                 quantity=new_qty)
                        st.success("Updated.")
                        st.rerun()

                # --- scale in ---------------------------------------------
                st.markdown("##### Add capital to this position")
                a1, a2 = st.columns(2)
                add_qty = a1.number_input("Quantity to add", min_value=0.0,
                                           format="%.8f", key=f"aq_{pid}")
                add_px = a2.number_input("At price", min_value=0.0,
                                          value=float(cprice or r["entry"]),
                                          format="%.8f", key=f"ap_{pid}")
                preplanned = st.checkbox(
                    "This addition was planned before entering the trade",
                    key=f"pp_{pid}",
                    help="The rulebook only permits adding to a losing position when "
                         "that was decided in advance.")

                if st.button("Assess adding", key=f"assess_{pid}"):
                    if snap is None:
                        st.error("Enter a current price first — the check needs to know "
                                 "whether the position is in profit.")
                    elif add_qty <= 0:
                        st.error("Enter a quantity to add.")
                    else:
                        res = analyse_scale_in(snap, add_qty, add_px, pos_equity,
                                                max_risk_pct=pos_max_risk,
                                                preplanned=preplanned)
                        box = {ScaleVerdict.ALLOWED: st.success,
                               ScaleVerdict.CAUTION: st.warning,
                               ScaleVerdict.BLOCKED: st.error}[res.verdict]
                        box(f"**{res.verdict.value}**")
                        for reason in res.reasons:
                            st.caption(f"• {reason}")
                        q1, q2, q3 = st.columns(3)
                        q1.metric("New avg entry", format_price(res.new_average_entry))
                        q2.metric("New total risk", f"${res.new_total_risk:,.2f}",
                                  f"{res.new_risk_pct_of_equity:.2f}% of equity")
                        if res.new_reward_risk is not None:
                            q3.metric("New R:R", f"{res.new_reward_risk:.2f}")

                        if res.verdict is not ScaleVerdict.BLOCKED:
                            if st.button("Apply this addition", key=f"apply_{pid}"):
                                storage.update_position(
                                    pid, entry=res.new_average_entry,
                                    quantity=res.new_quantity)
                                st.success("Position updated with the new average entry.")
                                st.rerun()

                # --- close into journal ------------------------------------
                st.markdown("##### Close this trade")
                c1, c2 = st.columns(2)
                exit_px = c1.number_input("Exit price", min_value=0.0,
                                           value=float(cprice or r["entry"]),
                                           format="%.8f", key=f"ex_{pid}")
                exit_reason = c2.text_input("Exit reason", key=f"er_{pid}",
                                             placeholder="Target hit / stopped / thesis broke")
                if exit_px > 0:
                    preview = PositionSnapshot(
                        direction=r["direction"], entry=r["entry"], stop=r["stop"],
                        target=r["target"], quantity=r["quantity"],
                        current_price=exit_px,
                        contract_multiplier=r["contract_multiplier"] or 1.0)
                    realised = preview.realised_pl(exit_px)
                    st.caption(f"Realised P/L at that exit: **${realised:,.2f}**")

                if st.button("✅ Close and log to journal", key=f"close_{pid}"):
                    if exit_px <= 0:
                        st.error("Enter an exit price.")
                    else:
                        preview = PositionSnapshot(
                            direction=r["direction"], entry=r["entry"], stop=r["stop"],
                            target=r["target"], quantity=r["quantity"],
                            current_price=exit_px,
                            contract_multiplier=r["contract_multiplier"] or 1.0)
                        realised = preview.realised_pl(exit_px)
                        jid = storage.add_journal_entry({
                            "asset": r["asset"], "direction": r["direction"],
                            "leverage": r["leverage"] or 1.0, "entry": r["entry"],
                            "sl": r["stop"], "tp": r["target"] or 0.0,
                            "size_notional_usd": preview.notional,
                            "potential_profit_at_tp": preview.profit_at_target or 0.0,
                            "potential_loss_at_sl": preview.loss_at_stop,
                            "entry_reason": r["entry_reason"] or "",
                            "status": "Closed", "realized_pnl": realised,
                            "exit_reason": exit_reason or "Closed from Positions tab",
                            "pl_status": ("Win" if realised > 0 else
                                          "Loss" if realised < 0 else "Breakeven"),
                        })
                        storage.delete_position(pid)
                        st.success(f"Logged to journal as entry #{jid} "
                                   f"(realised ${realised:,.2f}) and removed from positions.")
                        st.rerun()

                if st.button("🗑 Remove without logging", key=f"rm_{pid}"):
                    storage.delete_position(pid)
                    st.rerun()


# ---------------------------------------------------------------------
# TAB: Risk calculator
# ---------------------------------------------------------------------
with tab_risk:
    st.subheader("Position sizing & risk")
    pre = st.session_state.get("prefill", {})
    if pre:
        st.success(f"Using levels sent from the scanner for {pre.get('ticker','')}.")

    k1, k2, k3 = st.columns(3)
    equity = k1.number_input("Account equity ($)", min_value=0.0, value=10000.0, step=100.0)
    risk_pct = k2.number_input("Risk % per trade", min_value=0.1, max_value=100.0,
                                value=1.0, step=0.1)
    direction = k3.selectbox("Direction", ["Long", "Short"],
                              index=0 if pre.get("direction", "Long") == "Long" else 1)

    k4, k5, k6 = st.columns(3)
    entry = k4.number_input("Entry", min_value=0.0, value=float(pre.get("entry", 100.0)), step=0.01)
    stop = k5.number_input("Stop", min_value=0.0, value=float(pre.get("stop", 95.0)), step=0.01)
    target = k6.number_input("Target", min_value=0.0, value=float(pre.get("target", 115.0)), step=0.01)

    k7, k8, k9 = st.columns(3)
    leverage = k7.number_input("Leverage", min_value=1.0, value=1.0, step=0.5)
    fee = k8.number_input("Fee rate", min_value=0.0, value=0.0006, step=0.0001, format="%.4f")
    slip = k9.number_input("Slippage", min_value=0.0, value=0.0005, step=0.0001, format="%.4f")

    spec_ticker = st.text_input("Contract-spec ticker (optional, e.g. GC=F)",
                                 value=pre.get("ticker", ""))

    if st.button("Calculate", use_container_width=True):
        try:
            res = calculate_risk(account_equity=equity, risk_pct=risk_pct, entry=entry,
                                  stop=stop, direction=direction,
                                  target=target if target > 0 else None,
                                  leverage=leverage, ticker=spec_ticker or None,
                                  fee_rate=fee, slippage_pct=slip)
            a, b, c, d = st.columns(4)
            a.metric("Max loss", f"${res.max_permitted_loss:,.2f}")
            b.metric("Quantity", f"{res.quantity:g}")
            c.metric("Notional", f"${res.position_notional:,.2f}")
            d.metric("Margin", f"${res.margin_required:,.2f}")

            e, f, g = st.columns(3)
            e.metric("Loss at stop", f"${res.net_loss_at_stop:,.2f}")
            if res.net_profit_at_target is not None:
                f.metric("Profit at target", f"${res.net_profit_at_target:,.2f}")
            if res.net_reward_risk is not None:
                g.metric("Net R:R", f"{res.net_reward_risk:.2f}")

            if res.exceeds_account_equity:
                st.error("⚠️ Required margin exceeds account equity.")
            if res.liquidation_warning:
                st.warning(res.liquidation_warning)
            for w in res.warnings:
                st.warning(w)

            if st.button("📓 Log this to the journal"):
                storage.add_journal_entry({
                    "asset": spec_ticker or "—", "direction": direction, "leverage": leverage,
                    "entry": entry, "sl": stop, "tp": target,
                    "size_notional_usd": res.position_notional, "margin": res.margin_required,
                    "potential_loss_at_sl": res.net_loss_at_stop,
                    "potential_profit_at_tp": res.net_profit_at_target or 0.0,
                })
                st.success("Logged — see the 📓 Journal tab.")
        except InvalidRiskInputError as e:
            st.error(str(e))

# ---------------------------------------------------------------------
# TAB: Journal
# ---------------------------------------------------------------------
with tab_journal:
    st.subheader("Trade journal")

    with st.form("journal_add"):
        j1, j2, j3, j4 = st.columns(4)
        ja = j1.text_input("Asset")
        jd = j2.selectbox("Direction", ["Long", "Short"])
        jl = j3.number_input("Leverage", min_value=1.0, value=1.0)
        je = j4.number_input("Entry", min_value=0.0)
        j5, j6, j7 = st.columns(3)
        jtp = j5.number_input("TP", min_value=0.0)
        jsl = j6.number_input("SL", min_value=0.0)
        jsz = j7.number_input("Size (notional USD)", min_value=0.0)
        jr = st.text_area("Entry reason / score breakdown")
        if st.form_submit_button("Add entry") and ja:
            storage.add_journal_entry({
                "asset": ja, "direction": jd, "leverage": jl, "entry": je,
                "tp": jtp, "sl": jsl, "size_notional_usd": jsz, "entry_reason": jr})
            st.success("Added.")
            st.rerun()

    df = storage.get_journal_df()
    if df.empty:
        st.info("No journal entries yet.")
    else:
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇️ Export CSV", data=storage.journal_to_csv_bytes(),
                            file_name=f"trade_journal_{datetime.now().date()}.csv",
                            mime="text/csv", use_container_width=True)

        open_rows = df[df["status"] == "Open"]
        if not open_rows.empty:
            st.markdown("##### Close a trade")
            cid = st.selectbox("Entry", open_rows["id"].tolist(),
                                format_func=lambda i: f"#{i} {df[df['id']==i]['asset'].iloc[0]}")
            cpnl = st.number_input("Realized P&L", step=0.01)
            creason = st.text_input("Exit reason")
            if st.button("Close trade"):
                storage.close_journal_entry(int(cid), cpnl, creason)
                st.rerun()

        with st.expander("⚠️ Danger zone"):
            if st.checkbox("I understand this permanently deletes all journal entries and positions"):
                if st.button("Delete everything"):
                    storage.clear_all()
                    st.rerun()

# ---------------------------------------------------------------------
# TAB: Backtest
# ---------------------------------------------------------------------
with tab_track:
    st.subheader("Forward tracking")
    st.caption(
        "Every B-or-better setup from a universe scan is recorded with its plan frozen "
        "at that moment. Outcomes are then checked against what price actually did. "
        "Because the plan is fixed before the result is known, this is the most honest "
        "test the strategy can get."
    )
    st.info(
        "Works even while the app sleeps — pressing Update fetches every candle printed "
        "since each signal was created. But the database is **wiped on every redeploy**, "
        "so export the CSV regularly and re-import it after updating the app."
    )

    sig_df = storage.get_signals_df()
    open_count = int(sig_df["status"].isin(["PENDING", "FILLED"]).sum()) if not sig_df.empty else 0

    t1, t2 = st.columns(2)
    t1.metric("Signals tracked", len(sig_df))
    t2.metric("Still open", open_count)

    if st.button("🔄 Update outcomes", use_container_width=True, disabled=open_count == 0):
        bar = st.progress(0.0, text="Checking…")

        def _bars(ticker):
            df, _err = exchanges.fetch_binance_klines(ticker, "1h", limit=1000)
            if df is None:
                df, _err = exchanges.fetch_kraken_ohlc(ticker, "1h")
            return df

        counts = tracking.update_all(
            _bars, bar_hours=1.0,
            progress=lambda i, n, t: bar.progress(min(i / max(n, 1), 1.0),
                                                   text=f"{i}/{n} · {t}"))
        bar.empty()
        st.success(f"Checked {counts['checked']} · resolved {counts['resolved']}"
                   + (f" · {counts['failed']} could not be fetched" if counts["failed"] else ""))
        st.rerun()

    if sig_df.empty:
        st.info("No signals yet. Run a universe scan on the 🎯 Scanner tab with tracking "
                "switched on, then come back after a few days.")
    else:
        stats = tracking.tracked_stats()
        st.markdown("##### Live track record by grade")
        st.dataframe(pd.DataFrame([{
            "Grade": s_.grade, "Signals": s_.signals, "Filled": s_.filled,
            "Wins": s_.wins, "Losses": s_.losses, "Expired": s_.expired,
            "Open": s_.still_open,
            "Win rate": f"{s_.win_rate:.0%}" if s_.win_rate is not None else "—",
            "Avg R / trade": f"{s_.avg_r:+.2f}" if s_.avg_r is not None else "—",
            "Total R": f"{s_.total_r:+.1f}",
            "Enough data?": "yes" if s_.enough_data else "no (<30)",
        } for s_ in stats]), use_container_width=True, hide_index=True)
        resolved_total = sum(1 for _, r in sig_df.iterrows() if r["status"] in ("WIN", "LOSS"))
        if resolved_total < 30:
            st.caption(
                f"Only {resolved_total} resolved trade(s) so far. Under about 30 per grade, "
                f"results are dominated by luck — a few lucky wins can make any grade look "
                f"brilliant. Give it a few weeks.")

        show = sig_df.copy()
        for col in ("entry", "stop", "target"):
            show[col] = show[col].apply(format_price)
        st.dataframe(show[["created_utc", "label", "direction", "grade", "score",
                           "entry", "stop", "target", "status", "r_result"]],
                     use_container_width=True, hide_index=True)

    e1, e2 = st.columns(2)
    e1.download_button("⬇️ Export tracking CSV", data=storage.signals_to_csv_bytes(),
                       file_name=f"tracked_signals_{datetime.now().date()}.csv",
                       mime="text/csv", use_container_width=True)
    upload = e2.file_uploader("Restore from CSV", type=["csv"], key="sig_upload",
                               label_visibility="collapsed")
    if upload is not None and st.button("Import CSV"):
        try:
            added = storage.import_signals_csv(upload.getvalue())
            st.success(f"Restored {added} signal(s).")
            st.rerun()
        except ValueError as e:
            st.error(str(e))

with tab_backtest:
    st.subheader("Strategy backtest")
    st.caption(
        "Replays the real scanner over past data. At each step it sees only candles "
        "that had already closed, makes its plan exactly as it would live, then checks "
        "what price did next. The question: **do higher grades actually do better?**"
    )
    st.warning(
        "No score guarantees profit. A backtest measures whether the grades had an "
        "edge in the past — it cannot promise they will in future, and a few months of "
        "one market regime is a single sample. Ambiguous candles are resolved "
        "pessimistically, so results lean worse rather than better."
    )

    b1, b2, b3 = st.columns(3)
    bt_size = b1.selectbox("Coins", [5, 10, 20], index=1, key="bt_size",
                            help="More coins = more trades = more trustworthy, but slower.")
    bt_rr = b2.number_input("Min R:R", min_value=1.0, value=2.0, step=0.5, key="bt_rr")
    bt_expiry = b3.selectbox("Limit order valid for", ["1 day", "3 days", "5 days"],
                              index=1, key="bt_exp")
    expiry_bars = {"1 day": 6, "3 days": 18, "5 days": 30}[bt_expiry]
    bt_fee = st.number_input("Fees + slippage per trade (in R)", min_value=0.0,
                              value=0.05, step=0.01, format="%.2f", key="bt_fee",
                              help="0.05R means costs take 5% of your risk per trade.")

    st.caption(f"Uses ~160 days of 4H candles per coin. Roughly "
               f"{bt_size * 15 // 60 + 1} minute(s) for {bt_size} coins.")

    if st.button("▶ Run strategy backtest", use_container_width=True):
        labels = list(CRYPTO_TICKERS)[:bt_size]
        all_trades, failures, kraken_short_1h = [], [], []
        bar = st.progress(0.0, text="Starting…")
        for i, lbl in enumerate(labels):
            ticker = CRYPTO_TICKERS[lbl]
            bar.progress(i / len(labels), text=f"{i + 1}/{len(labels)} · {lbl}")
            df4, err = exchanges.fetch_binance_klines(ticker, "4h", limit=1000)
            fetch_d = exchanges.fetch_binance_klines
            if df4 is None:
                df4, err = exchanges.fetch_kraken_ohlc(ticker, "4h")
                fetch_d = exchanges.fetch_kraken_ohlc
            if df4 is None or df4.empty:
                failures.append(f"{lbl}: {err}")
                continue
            if fetch_d is exchanges.fetch_binance_klines:
                d1, _ = fetch_d(ticker, "1d", limit=500)
                # ~4,000 hourly candles to span the same period as 1,000 4H ones
                h1, _ = exchanges.fetch_binance_history(ticker, "1h", 4000)
            else:
                d1, _ = fetch_d(ticker, "1d")
                h1, _ = fetch_d(ticker, "1h")   # Kraken only serves ~30 days
                if h1 is not None and not h1.empty:
                    kraken_short_1h.append(lbl)
            all_trades += run_strategy_backtest(df4, d1, ticker, min_rr=bt_rr,
                                                 expiry_bars=expiry_bars, fee_r=bt_fee,
                                                 df_1h=h1)
        bar.empty()
        st.session_state.bt_trades = all_trades
        st.session_state.bt_failures = failures
        st.session_state.bt_kraken_1h = kraken_short_1h

    trades = st.session_state.get("bt_trades")
    if trades is not None:
        if not trades:
            st.info("No qualifying setups were produced over this history.")
        else:
            stats = stats_by_grade(trades)
            st.markdown("##### Results by grade")
            st.dataframe(pd.DataFrame([{
                "Grade": s_.grade,
                "Signals": s_.signals,
                "Filled": s_.filled,
                "Fill rate": f"{s_.fill_rate:.0%}" if s_.fill_rate is not None else "—",
                "Wins": s_.wins, "Losses": s_.losses, "Expired": s_.expired,
                "Win rate": f"{s_.win_rate:.0%}" if s_.win_rate is not None else "—",
                "Avg R / trade": f"{s_.avg_r:+.2f}" if s_.avg_r is not None else "—",
                "Total R": f"{s_.total_r:+.1f}",
                "Enough data?": "yes" if s_.enough_data else "no (<30)",
            } for s_ in stats]), use_container_width=True, hide_index=True)

            v = verdict(stats)
            (st.success if "separated better" in v else st.warning)(v)
            st.caption(
                "**Avg R / trade** is the key column: +0.30R means that, on average, each "
                "trade returned 30% of the amount risked. Win rate alone can mislead — a "
                "40% win rate at 2:1 is profitable, a 60% win rate at 0.5:1 is not.")
            st.caption(
                "**Baseline for comparison:** on random price data, where no edge exists, "
                "this backtester returns roughly **−0.35R per trade** before fees. That is "
                "partly the pessimistic candle handling and partly real adverse selection "
                "(limit buys tend to fill while price is falling). A grade needs to clearly "
                "beat that, not just zero, before it looks like more than luck.")

            with st.expander(f"All {len(trades)} simulated trades"):
                st.dataframe(pd.DataFrame([{
                    "Coin": t.ticker, "Signal time": t.signal_time, "Dir": t.direction,
                    "Grade": t.grade, "Score": t.score,
                    "Entry": format_price(t.entry), "Stop": format_price(t.stop),
                    "Target": format_price(t.target), "Planned R:R": f"{t.planned_rr:.2f}",
                    "Outcome": t.status,
                    "R": f"{t.r_result:+.2f}" if t.r_result is not None else "—",
                } for t in trades]), use_container_width=True, hide_index=True)
        for f in st.session_state.get("bt_failures", []):
            st.caption(f"⚠️ Could not load {f}")
        short = st.session_state.get("bt_kraken_1h") or []
        if short:
            st.caption(
                f"⚠️ {len(short)} coin(s) came from Kraken, which only serves about 30 days "
                f"of 1H candles. Earlier in their history the 1H confirmation point could "
                f"not be scored, which slightly understates their grades.")
