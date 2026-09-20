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
from scanner import fetch_multi_timeframe, analyze_candidate, scan_universe
from universe import fetch_top_cryptos
import coingecko
import exchanges
import storage

APP_BUILD = "2026-09-20-b12 (Binance/Kraken exchange data)"

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


CRYPTO_TICKERS, CRYPTO_CG_IDS, CRYPTO_IS_LIVE, CRYPTO_NOTE = _load_crypto_universe()

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
st.sidebar.caption(f"Crypto list: {len(CRYPTO_TICKERS)} symbols")
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

tab_market, tab_scan, tab_positions, tab_risk, tab_journal, tab_backtest = st.tabs(
    ["🌍 Market", "🎯 Scanner", "📋 Positions", "🧮 Risk", "📓 Journal", "🔁 Backtest"]
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
        m2.metric("Price", f"${quote.price:,.4f}" if quote.price else "—")
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

        per_coin = 0.3 if CRYPTO_SOURCE == "exchange" else 1.4
        est = universe_size * per_coin
        st.caption(
            f"Roughly {est:.0f}s for {universe_size} coins using "
            f"{'exchange data (fast — no meaningful rate limit)' if CRYPTO_SOURCE == 'exchange' else 'CoinGecko (calls spaced to avoid 429s)'}."
        )

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
                    if df is None:
                        df, err = exchanges.fetch_kraken_ohlc(ticker, "4h")
                    if df is not None and not df.empty:
                        return {"4h": df, "1d": pd.DataFrame(), "1h": pd.DataFrame()}, []
                if coin_id:
                    df, _label, err = coingecko.fetch_ohlc(coin_id, days=30)
                    if df is not None and not df.empty:
                        return {"4h": df, "1d": pd.DataFrame(), "1h": pd.DataFrame()}, []
                return {}, [err or "no data"]

            def _progress(i, total, label):
                bar.progress(min(i / max(total, 1), 1.0), text=f"{i}/{total} · {label}")

            with st.spinner("Scanning…"):
                ranked = scan_universe(
                    instruments, _loader, min_rr=uni_min_rr,
                    direction_override=None if uni_direction == "Auto" else uni_direction,
                    progress_callback=_progress)
            bar.empty()
            st.session_state.ranked = ranked

        ranked = st.session_state.get("ranked")
        if ranked:
            ok = [r for r in ranked if r.error is None]
            failed = [r for r in ranked if r.error is not None]

            st.markdown(f"##### Results · {len(ok)} scored, {len(failed)} failed")
            if ok:
                table = pd.DataFrame([{
                    "Instrument": r.label,
                    "Score": f"{r.score:.1f}",
                    "Grade": r.grade,
                    "Dir": r.direction or "—",
                    "Regime": r.regime.title(),
                    "Price": f"{r.price:,.6g}" if r.price else "—",
                    "Stop": f"{r.stop:,.6g}" if r.stop else "—",
                    "Target": f"{r.target:,.6g}" if r.target else "—",
                    "R:R": f"{r.reward_risk:.2f}" if r.reward_risk else "—",
                } for r in ok])
                st.dataframe(table, use_container_width=True, hide_index=True)

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

                st.session_state.analysis = analyze_candidate(
                    scan_ticker, frames,
                    direction_override=None if dir_choice == "Auto" else dir_choice,
                    min_rr=min_rr, data_problems=problems)
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
            h4.metric("Price", f"{analysis.current_price:,.6g}" if analysis.current_price else "—")
            if analysis.price_source:
                st.caption(
                    f"Entry price taken from the latest **{analysis.price_source}** close. "
                    f"If this differs from the Market tab, one of them is a slightly older bar — "
                    f"always confirm against your broker before entering."
                )

            if analysis.stop is not None and analysis.target is not None:
                l1, l2, l3 = st.columns(3)
                l1.metric("Entry", f"{analysis.entry:,.6g}")
                l2.metric("Structural stop", f"{analysis.stop:,.6g}")
                l3.metric("Structural target", f"{analysis.target:,.6g}")
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
    st.caption("Stored in the database — survives refreshes and app sleeps.")

    with st.form("add_pos"):
        p1, p2, p3, p4 = st.columns(4)
        pa = p1.text_input("Asset")
        pc = p2.selectbox("Class", ["crypto", "stock", "commodity", "forex"])
        pd_ = p3.selectbox("Direction", ["Long", "Short"])
        pq = p4.number_input("Quantity", min_value=0.0, step=0.01)
        p5, p6 = st.columns(2)
        pe = p5.number_input("Entry", min_value=0.0, step=0.01)
        ps = p6.number_input("Stop", min_value=0.0, step=0.01)
        if st.form_submit_button("Add position") and pa and pe > 0:
            storage.add_position(pa, pc, pd_, pe, ps, pq)
            st.success(f"Added {pd_} {pa}")
            st.rerun()

    rows = storage.get_positions()
    if not rows:
        st.info("No open positions.")
    else:
        positions = [OpenPosition(asset=r["asset"], asset_class=r["asset_class"],
                                   direction=r["direction"], entry=r["entry"],
                                   stop=r["stop"], quantity=r["quantity"],
                                   contract_multiplier=r["contract_multiplier"])
                     for r in rows]
        summary = summarize_portfolio_risk(positions)
        st.metric("Total open risk (to stops)", f"${summary.total_open_risk:,.2f}")
        for w in summary.concentration_warnings:
            st.warning(w)

        for r, pos in zip(rows, positions):
            with st.expander(f"{pos.direction} {pos.asset} · entry {pos.entry:g} · stop {pos.stop:g}"):
                px = st.number_input("Current price", min_value=0.0, key=f"px{r['id']}")
                inval = st.checkbox("Invalidation hit?", key=f"iv{r['id']}")
                breach = st.checkbox("Risk rule breached?", key=f"rb{r['id']}")
                tighten = st.checkbox("Structure supports tightening?", key=f"st{r['id']}")

                if px > 0:
                    sm = StopManager(direction=pos.direction, entry=pos.entry,
                                      current_stop=pos.stop, current_target=pos.stop)
                    r_mult = sm.current_r_multiple(px)
                    label = suggest_management_label(pos.direction, inval, breach, tighten, r_mult)
                    a, b = st.columns(2)
                    a.metric("Suggested action", label.value)
                    b.metric("Current R", f"{r_mult:+.2f}R")
                    if sm.suggest_transition_to_trailing(px):
                        st.info("Past ~1.5R — consider moving to a structure-based trailing stop.")

                new_stop = st.number_input("Move stop to", min_value=0.0,
                                            value=float(pos.stop), key=f"ns{r['id']}")
                cc1, cc2 = st.columns(2)
                if cc1.button("Update stop", key=f"us{r['id']}"):
                    tightening = (new_stop > pos.stop if pos.direction == "Long"
                                   else new_stop < pos.stop)
                    if not tightening:
                        st.error("Refused — that widens risk. Never move a stop further from entry.")
                    else:
                        storage.update_position_stop(r["id"], new_stop)
                        st.success("Stop updated.")
                        st.rerun()
                if cc2.button("Remove position", key=f"rm{r['id']}"):
                    storage.delete_position(r["id"])
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
with tab_backtest:
    st.subheader("Backtest engine")
    st.warning(
        "This validates the REPLAY ENGINE — no look-ahead, realistic fills, fees, metrics — "
        "using a simple moving-average crossover as an example signal generator. It is NOT "
        "the multi-timeframe strategy from the scanner. Results here say nothing about "
        "whether your actual strategy has an edge."
    )

    b1, b2 = st.columns(2)
    bt_ticker = b1.text_input("Ticker", value="BTC-USD")
    bt_period = b2.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1)
    b3, b4, b5 = st.columns(3)
    fast = b3.number_input("Fast MA", min_value=2, value=10)
    slow = b4.number_input("Slow MA", min_value=3, value=30)
    bt_fee = b5.number_input("Fee/side", min_value=0.0, value=0.0006, format="%.4f")

    if st.button("Run backtest", use_container_width=True):
        with st.spinner("Fetching history…"):
            hist = yf.Ticker(bt_ticker).history(period=bt_period)
        if hist is None or hist.empty:
            st.error("No historical data returned for this ticker.")
        else:
            hist = hist[hist["Close"].notna()].reset_index(drop=True)
            in_s, out_s = split_in_out_sample(hist, split_ratio=0.7)
            for label, sample in [("In-sample", in_s), ("Out-of-sample", out_s)]:
                st.markdown(f"##### {label}")
                sigs = example_sma_crossover_signals(sample, fast=int(fast), slow=int(slow))
                res = run_backtest(sample, sigs, fee_rate=bt_fee)
                mt = compute_metrics(res)
                x1, x2, x3, x4 = st.columns(4)
                x1.metric("Trades", mt.trade_count)
                x2.metric("Win rate", f"{mt.win_rate:.1%}")
                x3.metric("Expectancy", f"{mt.expectancy_r:+.2f}R")
                x4.metric("Profit factor",
                          f"{mt.profit_factor:.2f}" if mt.profit_factor != float("inf") else "∞")
                y1, y2, y3 = st.columns(3)
                y1.metric("Max drawdown", f"{mt.max_drawdown_r:.2f}R")
                y2.metric("Longest losing streak", mt.max_losing_streak)
                y3.metric("Avg MFE / MAE", f"{mt.avg_mfe_r:.2f} / {mt.avg_mae_r:.2f}R")
            st.caption(
                f"{len(hist)} bars total. A small or favourable sample is not proof of an edge — "
                f"this is a mechanics check."
            )
