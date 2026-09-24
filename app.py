"""
Bull Run Strategy V2 — trading analysis dashboard.

Discretionary trading support tool. It does NOT execute trades, connect to an
exchange, or place/modify orders — you enter everything manually on your own
platform. Setup scores measure checklist confluence only; they are not
win-probability estimates, edge claims, or profitability guarantees.
"""

import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
import zoneinfo

try:
    from data_layer import fetch_quote, DataStatus, to_local_display, curl_cffi_is_available
    from scoring import (score_setup, weakest_components, trade_policy_for_grade,
                          POSITIVE_COMPONENTS, NEGATIVE_COMPONENTS)
    from readiness import (determine_readiness, ReadinessInputs, Readiness, display_label,
                            ENTRY_SEQUENCE_STAGES, missing_entry_sequence_stages,
                            READINESS_DESCRIPTIONS)
    from risk_calc import (calculate_risk, InvalidRiskInputError,
                            stop_from_pct as calc_stop_from_pct,
                            liquidation_price, stop_is_beyond_liquidation)
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
    import trend_retrace
except ImportError as _e:
    # Almost always a part-finished upload: a new app.py alongside older module
    # files. Say so plainly instead of showing a bare ImportError.
    st.error(
        f"**Some files are out of date or missing.**\n\n`{_e}`\n\n"
        f"This happens when only some files were uploaded. Upload **every `.py` file** "
        f"from the latest download together — they're released as a set and expect each "
        f"other's newest versions."
    )
    st.stop()
import expectancy
import explain
import learning
import trade_log
import hyperliquid_data
import trade_review
import sentiment
import watchlist as watchlist_mod
import market_tools
import theme
import coin_info
import grid as grid_mod
import charting
import patterns
import uihelpers
from formatting import format_price, format_rr

_REQUIRED = {
    coingecko: ['build_frames_v2', 'fetch_description', 'fetch_ohlc', 'fetch_period_changes', 'fetch_spot_price', 'has_api_key', 'search_coins', 'set_api_key'],
    hyperliquid_data: ['MarketContext', 'fetch_candles', 'fetch_market_contexts', 'is_hl_ticker', 'rank_by_volume'],
    watchlist_mod: ['DEFAULT_WATCHLIST', '_normalise', 'parse_aliases', 'parse_list', 'resolve', 'search_markets', 'suggest'],
    sentiment: ['bucket', 'fetch_fear_greed', 'set_cmc_api_key'],
    market_tools: ['OVERBOUGHT', 'OVERSOLD', 'altcoin_season_index', 'flow_rows', 'most_crowded', 'rsi_row', 'rsi_state', 'turnover_ratio'],
    theme: ['AMBER', 'BLUE', 'CSS', 'GRADE_COLOURS', 'GREEN', 'GREY', 'PURPLE', 'RED', 'gauge', 'pill'],
    storage: ['add_journal_entry', 'add_position', 'clear_all', 'close_journal_entry', 'delete_position', 'get_journal_df', 'get_positions', 'get_signals_df', 'import_journal_csv', 'import_signals_csv', 'init_db', 'journal_to_csv_bytes', 'load_value', 'save_value', 'signals_to_csv_bytes', 'update_journal_entry', 'update_position', 'update_signal'],
    trade_log: ['OUTCOMES', 'complete_trade', 'default_exit_price', 'learning_trades', 'open_scanner_trades', 'pl_status_for', 'realized_r', 'take_trade'],
    trade_review: ['CLOSE', 'CLOSE_THESIS', 'HOLD', 'STILL_VALID', 'TIGHTEN', 'review', 'setup_still_viable'],
    learning: ['LearnedModel', 'MIN_TRADES', 'learn'],
    expectancy: ['EvidenceBook', 'PROVEN_NEGATIVE', 'PROVEN_POSITIVE', 'TOO_FEW', 'UNPROVEN', 'break_even_win_rate', 'required_rr', 'simulate_expectations'],
    explain: ['AT_ENTRY', 'NO_TREND', 'WAIT_RETRACE', 'full_plan', 'short_plan'],
    trend_retrace: ['AT_ENTRY', 'KIND_INITIAL', 'STOP_BELOW_4H_CANDLE', 'STOP_BELOW_SUPPORT', 'StrategyParams', 'WAIT_RETRACE', 'analyze', 'atr', 'backtest_stats', 'drop_forming', 'five_minute_levels', 'frames_from_5m', 'reentry_plan_text', 'run_backtest', 'scan_universe_tr'],
    tracking: ['record_from_ranked', 'tracked_stats', 'tracked_trades', 'update_all'],
    exchanges: ['build_frames', 'fetch_binance_history', 'fetch_binance_klines', 'fetch_bybit_klines', 'fetch_bybit_tickers', 'fetch_kraken_ohlc', 'fetch_spot'],
    coin_info: ['CATEGORIES', 'COINS', 'coverage', 'describe'],
    grid_mod: ['MIN_LEVELS', 'WEIGHTINGS', 'build_grid', 'usable_levels'],
    charting: ['INTERVALS', 'to_tradingview_symbol', 'widget_html'],
    patterns: ['bias_of', 'contradicts', 'detect', 'summarise'],
    uihelpers: ['scan_count'],
}

_stale = sorted({f"{m.__name__.split('.')[-1]}.py" for m, attrs in _REQUIRED.items()
                 for a in attrs if not hasattr(m, a)})
if _stale:
    # A module is present but older than app.py expects. This is always a
    # part-finished upload, and it surfaces as a confusing AttributeError deep
    # in the page, so it is caught here instead.
    st.error(
        "**These files are out of date:** " + ", ".join(f"`{f}`" for f in _stale) +
        "\n\nUpload **every `.py` file** from the latest download together — they're "
        "released as a set and expect each other's newest versions."
    )
    st.stop()

MIN_RR = 3.0   # reward:risk floor — nothing below this is shown, tracked or traded


def _methods(use_tr, params, direction):
    """How the stop and target were set, in words, for the plan text."""
    if not (use_tr and params):
        return "the nearest confirmed swing beyond entry", "the first level clearing your minimum R:R"
    beyond = "below" if direction == "Long" else "above"
    level = "support" if direction == "Long" else "resistance"
    if params.stop_mode == trend_retrace.STOP_BELOW_4H_CANDLE:
        stop_m = f"{beyond} the last closed 4H candle"
    else:
        stop_m = f"{params.stop_atr_mult:g}× the 5m ATR {beyond} the {level}"
    return stop_m, f"{params.target_r:g}× the risk"


def _plan_stage(r, use_tr):
    """The explain-module stage for a ranked row from either strategy."""
    if use_tr:
        return r.entry_status
    if r.direction is None:
        return explain.NO_TREND
    return explain.AT_ENTRY if r.entry_status == "AT_ZONE" else explain.WAIT_RETRACE


def _render_learned(model):
    """Show what the learner found, including what it rejected and why."""
    (st.success if model.rules else st.info)(model.summary)
    for r in model.rules:
        st.markdown(f"**Skip setups where:** {r.description}")
        st.caption(f"　Learning trades: {r.train_avg:+.2f}R each (n={r.train_n}) vs "
                   f"{r.train_rest_avg:+.2f}R for the rest · Unseen trades: "
                   f"{r.test_avg:+.2f}R each (n={r.test_n}) vs {r.test_rest_avg:+.2f}R")
    tested = [c for c in model.candidates if c.train_avg is not None
              and c.verdict.startswith(("Looked", "Confirmed, but"))]
    if tested:
        with st.expander(f"Lessons that looked promising but were rejected ({len(tested)})"):
            st.caption("These appeared in the learning trades but failed on unseen ones — "
                       "exactly the coincidences the hold-out check exists to catch.")
            for c in tested:
                ta = f"{c.test_avg:+.2f}R (n={c.test_n})" if c.test_avg is not None else "n/a"
                st.caption(f"**{c.description}** — learning {c.train_avg:+.2f}R "
                           f"(n={c.train_n}), unseen {ta}. {c.verdict}")


@st.cache_data(ttl=120, show_spinner=False)
def _daily_change(ticker: str, price: float):
    """(change in price, change %, previous close) over the last 24h, or Nones.

    Sourced to match wherever the price came from: Hyperliquid's own previous
    daily price for HL markets, otherwise daily candles from the exchange, and
    Yahoo for stocks and commodities.
    """
    if not price:
        return None, None, None
    prev = None
    if hyperliquid_data.is_hl_ticker(ticker):
        ctxs, _e = _cached_hl_contexts()
        row = next((c for c in ctxs if c["ticker"] == ticker), None)
        if row and row.get("change") is not None:
            prev = price / (1 + row["change"] / 100) if row["change"] != -100 else None
    if prev is None:
        df, _e = exchanges.fetch_binance_klines(ticker, "1d", limit=2)
        if df is None:
            df, _e = exchanges.fetch_kraken_ohlc(ticker, "1d")
        if df is not None and len(df) >= 2:
            prev = float(df["Close"].iloc[-2])
    if prev is None:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            hist = hist[hist["Close"].notna()]
            if len(hist) >= 2:
                prev = float(hist["Close"].iloc[-2])
        except Exception:
            prev = None
    if not prev:
        return None, None, None
    return price - prev, (price - prev) / prev * 100, prev


@st.cache_data(ttl=300, show_spinner=False)
def _yahoo_frames(ticker: str):
    """5m, 1H and 4H candles for a stock, commodity or FX pair.

    Yahoo has no 4H interval, so 4H is built from 1H bars. For anything that
    doesn't trade round the clock those 4H candles span session breaks, which
    is a real difference from crypto — see the warning shown beside the scan.
    """
    frames, problems = fetch_multi_timeframe(ticker, yf)
    out = {"4h": frames.get("4h", pd.DataFrame()),
           "1h": frames.get("1h", pd.DataFrame()),
           "5m": frames.get("5m", pd.DataFrame())}
    return out, problems


def _backtest_frames(ticker, days):
    """5m, 1H and 4H history for a backtest, from whichever source this server
    can actually reach. Bybit and Binance are blocked from US-hosted servers;
    Hyperliquid is not, but only serves its most recent 5,000 candles — about
    17 days of 5m data. Returns (m5, h1, h4, source, note)."""
    bars5 = days * 288
    m5, err = exchanges.fetch_bybit_klines(ticker, "5m", limit=1000)
    if m5 is not None and len(m5) >= 500:
        h1, _ = exchanges.fetch_bybit_klines(ticker, "1h", limit=1000)
        h4, _ = exchanges.fetch_bybit_klines(ticker, "4h", limit=1000)
        if h1 is not None and h4 is not None:
            return m5, h1, h4, "Bybit", None
    m5, err = exchanges.fetch_binance_history(ticker, "5m", bars5)
    if m5 is not None and not m5.empty:
        h1, _ = exchanges.fetch_binance_history(ticker, "1h", days * 24 + 48)
        h4, _ = exchanges.fetch_binance_history(ticker, "4h", days * 6 + 30)
        if h1 is not None and h4 is not None:
            return m5, h1, h4, "Binance", None
    base = ticker.upper().replace("-USD", "")
    m5, hl_err = hyperliquid_data.fetch_candles(f"HL:{base}", "5m", min(bars5, 5000))
    if m5 is not None and not m5.empty:
        h1, _ = hyperliquid_data.fetch_candles(f"HL:{base}", "1h", 2000)
        h4, _ = hyperliquid_data.fetch_candles(f"HL:{base}", "4h", 1000)
        if h1 is not None and h4 is not None:
            covered = len(m5) * 5 / 60 / 24
            note = (f"Hyperliquid data ({covered:.0f} days — it only keeps its most recent "
                    f"5,000 candles)") if covered < days - 1 else "Hyperliquid data"
            return m5, h1, h4, "Hyperliquid", note
    return None, None, None, None, (f"Bybit/Binance: {err} · Hyperliquid: {hl_err}")


def _size_plan(direction, entry, stop, target, ticker=None):
    """Position size for a plan, from the account settings in the sidebar.
    Returns None when the plan is incomplete or the numbers don't work."""
    if None in (direction, entry, stop) or not entry or not stop:
        return None
    try:
        return calculate_risk(
            account_equity=ACCOUNT_EQUITY, risk_pct=RISK_PCT, entry=float(entry),
            stop=float(stop), direction=direction,
            target=float(target) if target else None, leverage=LEVERAGE,
            ticker=ticker, fee_rate=FEE_RATE, slippage_pct=SLIPPAGE)
    except InvalidRiskInputError:
        return None


def _render_size(res, key=""):
    """Show the size beside a trade plan, so no tab-hopping is needed."""
    if res is None:
        return
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Buy / sell", f"{res.quantity:,.6g}".rstrip("0").rstrip(".") + " units")
    s2.metric("Position value", f"${res.position_notional:,.2f}")
    s3.metric("Risk if stopped", f"-${res.net_loss_at_stop:,.2f}")
    if res.net_profit_at_target is not None:
        s4.metric("Gain at target", f"${res.net_profit_at_target:,.2f}")
    bits = [f"Margin ${res.margin_required:,.2f}"]
    if LEVERAGE > 1:
        bits.append(f"{LEVERAGE:g}x leverage")
    bits.append(f"after ${res.fees_and_slippage_cost:,.2f} costs")
    st.caption(" · ".join(bits) + ". Sized from your sidebar settings: "
               f"${ACCOUNT_EQUITY:,.0f} equity, {RISK_PCT:g}% risk.")
    if res.exceeds_account_equity:
        st.error("Required margin exceeds your account equity.")
    for w in res.warnings:
        st.warning(w)
    if res.liquidation_warning and "could be liquidated" in res.liquidation_warning.lower():
        st.error(res.liquidation_warning)


def _render_grid(key, direction, price, levels, atr_value, target_r):
    """Offer a laddered entry, but only when the chart really shows more than
    one level. A single support means a single entry."""
    usable = grid_mod.usable_levels(levels, direction, price, atr_value or 0.0)
    if len(usable) < grid_mod.MIN_LEVELS:
        return
    if not st.toggle("Ladder the entry across several levels", key=f"gr_on_{key}",
                     help="Your strategy specifies one entry. This spreads it over the "
                          "supports below — optional, and untested by the backtest."):
        return

    c1, c2 = st.columns(2)
    n = c1.slider("Levels", min_value=2, max_value=len(usable), value=min(3, len(usable)),
                  key=f"gr_n_{key}")
    style = c2.selectbox("Sizing", grid_mod.WEIGHTINGS, key=f"gr_w_{key}")
    chosen = usable[:n]
    buffer = (atr_value or 0.0) * (TR_PARAMS.stop_atr_mult if TR_PARAMS else 3.0)
    stop = (min(chosen) - buffer) if direction == "Long" else (max(chosen) + buffer)
    plan = grid_mod.build_grid(direction, chosen, price, stop,
                               risk_amount=ACCOUNT_EQUITY * RISK_PCT / 100,
                               target_r=target_r or 3.0, weighting=style)
    if plan is None:
        st.caption("These levels don't form a valid ladder.")
        return

    st.dataframe(pd.DataFrame([{
        "#": l.index,
        "Limit price": format_price(l.price),
        "From live": f"{l.distance_pct:+.2f}%",
        "Share": f"{l.weight * 100:.0f}%",
        "Units": f"{l.units:,.6g}".rstrip("0").rstrip("."),
        "Value": f"${l.notional:,.2f}",
        "Risk if stopped here": f"${l.cumulative_risk:,.2f}",
    } for l in plan.levels]), use_container_width=True, hide_index=True)

    g1, g2, g3 = st.columns(3)
    g1.metric("Average entry", format_price(plan.average_entry),
              f"vs {format_price(plan.single_entry_price)} single")
    g2.metric("Stop (below all levels)", format_price(plan.stop))
    g3.metric("Target", format_price(plan.target),
              f"{format_rr(plan.reward_risk)}")
    st.caption(f"**All levels fill, then stopped: −${plan.total_risk:,.2f}** — the same as a "
               f"single entry, not multiplied. All fill, then target: "
               f"+${plan.reward_at_target:,.2f}.")
    st.caption(plan.partial_fill_note)
    for w in plan.warnings:
        st.caption(f"⚠️ {w}")
    st.caption("Place each line as its own limit order with the SAME stop. Log each fill "
               "separately in the journal. The backtest only models single entries, so "
               "there's no evidence yet that laddering helps this strategy.")


def _take_trade_widget(key, ticker, direction, entry, stop, target, features=None,
                       score=None, grade=None, reason="", suggested_qty=0.0):
    """'I took this trade' — records a recommended setup in the journal, with the
    setup snapshot the learner needs. Same control on every recommendation."""
    if None in (direction, entry, stop, target):
        return
    if not st.toggle("I took this trade", key=f"tk_on_{key}"):
        return
    c1, c2 = st.columns(2)
    fill = c1.number_input("Your actual entry price", value=float(entry), format="%.8f",
                           key=f"tk_fill_{key}",
                           help="What you were really filled at. Leave as-is if it matched.")
    qty = c2.number_input("Quantity", min_value=0.0, value=float(suggested_qty or 0.0),
                          format="%.8f", key=f"tk_qty_{key}",
                          help="Pre-filled from your sidebar risk settings. Change it if you "
                               "sized differently.")
    c3, c4 = st.columns(2)
    lev = c3.number_input("Leverage", min_value=1.0, value=float(LEVERAGE), step=0.5,
                          key=f"tk_lev_{key}")
    followed = c4.checkbox("I followed the rules", value=True, key=f"tk_rules_{key}",
                           help="Untick if you entered early, skipped a rule or changed the "
                                "stop. Only rule-following trades teach the strategy.")
    note = st.text_input("Note (optional)", key=f"tk_note_{key}")
    if st.button("💾 Save to journal", key=f"tk_save_{key}", use_container_width=True):
        try:
            jid = trade_log.take_trade(
                ticker=ticker, direction=direction, planned_entry=float(entry),
                stop=float(stop), target=float(target), actual_entry=fill, quantity=qty,
                leverage=lev, features=features, followed_rules=followed,
                strategy_config=_config_signature(USE_TR, TR_PARAMS),
                reason=note or reason, score=score, grade=grade)
            st.success(f"Saved as journal entry #{jid}. Complete it on the 📓 Journal tab "
                       f"when the trade closes.")
        except ValueError as e:
            st.error(str(e))


def _review_inputs(ticker, entry_time):
    """Closed candles and live price for reviewing an open trade, from the same
    source the trade came from (Hyperliquid for HL: tickers)."""
    now = pd.Timestamp.now(tz="UTC")
    hours = max(1.0, (now - entry_time).total_seconds() / 3600)
    frames = {}
    if hyperliquid_data.is_hl_ticker(ticker):
        n5 = int(min(5000, hours * 12 + 60))
        for tf, n in (("4h", 30), ("1h", 48), ("5m", n5)):
            df, _e = hyperliquid_data.fetch_candles(ticker, tf, n)
            frames[tf] = trend_retrace.drop_forming(df if df is not None else pd.DataFrame(),
                                                   tf, now)
        _cx, _e = hyperliquid_data.fetch_market_contexts()
        m = next((c for c in _cx if c.ticker == ticker), None)
        price = m.mark_price if m else None
    else:
        for tf, n in (("4h", 30), ("1h", 1000), ("5m", 1000)):
            df = None
            if CRYPTO_SOURCE == "bybit":
                df, _e = exchanges.fetch_bybit_klines(ticker, tf, limit=n)
            if df is None:
                df, _e = exchanges.fetch_binance_klines(ticker, tf, limit=n)
            if df is None:
                df, _e = exchanges.fetch_kraken_ohlc(ticker, tf)
            frames[tf] = trend_retrace.drop_forming(df if df is not None else pd.DataFrame(),
                                                   tf, now)
        price, _s, _e = exchanges.fetch_spot(ticker)
    m5 = frames["5m"]
    since = m5[m5.index >= entry_time] if not m5.empty else m5
    # Older than the 5m window? Fall back to 1H candles for the excursions.
    if not m5.empty and m5.index[0] > entry_time and not frames["1h"].empty:
        since = frames["1h"][frames["1h"].index >= entry_time]
    return frames, since, price, now


def _config_signature(use_tr, params):
    """The settings that backtest evidence depends on. Evidence measured under
    one stop/target/strategy says nothing about another."""
    if use_tr and params is not None:
        return (f"TR|n={params.trend_candles}|stop={params.stop_mode}|"
                f"atr={params.stop_atr_mult:g}|tp={params.target_r:g}")
    return "CONFLUENCE"

APP_BUILD = "2026-09-24-b51 (fix commodities scan crash)"

st.set_page_config(page_title="Bull Run Strategy V2", page_icon="📈", layout="wide")
st.markdown(theme.CSS, unsafe_allow_html=True)

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

# Only commodities that exist as Bybit perpetuals, so every setup is tradable.
# The values are the Yahoo symbols used for CANDLES: Bybit's own API is blocked
# from US-hosted servers, and spot gold/silver track the XAU/XAG perps closely.
COMMODITY_TICKERS = {
    "Gold — XAUUSDT": "XAUUSD=X",
    "Silver — XAGUSDT": "XAGUSD=X",
    "Crude Oil — CLUSDT": "CL=F",
}

# What you'd actually place the order on, for each of the above.
COMMODITY_PERPS = {
    "XAUUSD=X": "XAUUSDT (Bybit, up to 75x)",
    "XAGUSD=X": "XAGUSDT (Bybit, up to 75x)",
    "CL=F": "CLUSDT (Bybit, up to 50x)",
}

STOCK_TICKERS = {
    "Apple (AAPL)": "AAPL", "Microsoft (MSFT)": "MSFT", "Nvidia (NVDA)": "NVDA",
    "Amazon (AMZN)": "AMZN", "Alphabet (GOOGL)": "GOOGL", "Meta (META)": "META",
    "Tesla (TSLA)": "TSLA", "Broadcom (AVGO)": "AVGO", "AMD (AMD)": "AMD",
    "Netflix (NFLX)": "NFLX", "JPMorgan (JPM)": "JPM", "Visa (V)": "V",
    "Coinbase (COIN)": "COIN", "MicroStrategy (MSTR)": "MSTR", "Palantir (PLTR)": "PLTR",
    "Berkshire (BRK-B)": "BRK-B", "Eli Lilly (LLY)": "LLY", "Exxon (XOM)": "XOM",
    "S&P 500 ETF (SPY)": "SPY", "Nasdaq 100 ETF (QQQ)": "QQQ",
}

FX_TICKERS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X", "USD/CAD": "CAD=X", "USD/CHF": "CHF=X",
    "NZD/USD": "NZDUSD=X", "EUR/GBP": "EURGBP=X", "EUR/JPY": "EURJPY=X",
    "AUD/JPY": "AUDJPY=X", "GBP/JPY": "GBPJPY=X", "USD/SGD": "SGD=X",
    "Dollar Index (DX-Y.NYB)": "DX-Y.NYB",
}

# ---------------------------------------------------------------------
# Sidebar settings (shared across tabs)
# ---------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

st.sidebar.subheader("Strategy")
strategy_choice = st.sidebar.radio(
    "Scanner strategy",
    ["Trend Retrace (your strategy)", "Confluence (original)"],
    key="strategy_choice",
    help="Trend Retrace: two 4H candles of HH/HL, a bullish 1H candle, then a 5m "
         "retrace to support (reverse for shorts). Confluence is the earlier 10-point "
         "checklist, kept for comparison.")
USE_TR = strategy_choice.startswith("Trend")

if USE_TR:
    with st.sidebar.expander("Stop & take profit — please confirm", expanded=False):
        st.caption("Your rules don't specify these, so these are adjustable defaults.")
        _stop_choice = st.radio("Stop loss", ["Below the 5m support (ATR buffer)",
                                              "Below the last closed 4H candle"],
                                key="tr_stop_mode")
        _stop_atr = st.number_input(
            "Buffer below support (x 5m ATR)", min_value=0.5, max_value=10.0, value=3.0,
            step=0.5, key="tr_stop_atr",
            disabled=_stop_choice.startswith("Below the last"),
            help="At 1x, trades finished in ~12 minutes — before any 1H candle could close, "
                 "so your 'add 50% after a 1H candle' rule could never trigger. 3x lets "
                 "trades run for hours.")
        _target_r = st.number_input(
            "Take profit (multiple of risk)", min_value=0.25, max_value=10.0, value=3.0,
            step=0.25, key="tr_target_r",
            help="Sets the trade-off between how OFTEN you win and how MUCH. A big target "
                 "wins rarely and large; a small one wins often and small.")
        _be = expectancy.break_even_win_rate(_target_r)
        st.caption(f"At **{_target_r:g}:1** you break even winning **{_be:.0%}** of trades "
                   f"(about {_be * 10:.0f} in 10), and profit above that.")
        _trend_n = st.number_input("4H candles required", min_value=1, max_value=5,
                                    value=2, step=1, key="tr_trend_n")
    with st.sidebar.expander("Setup quality filters", expanded=False):
        st.caption("Extra gates beyond your three rules, because two 4H candles and one 1H "
                   "candle fire often by chance. Each halves the signal count roughly; the "
                   "backtest measures whether they help. Slide to 0 to switch one off.")
        _q_body = st.slider("1H candle body, minimum share of its range", 0.0, 0.8, 0.40,
                            0.05, key="q_body",
                            help="A doji closes level — it confirms nothing.")
        _q_trend = st.slider("4H trend move, minimum ATR", 0.0, 3.0, 0.80, 0.1,
                             key="q_trend",
                             help="Marginally higher highs happen constantly in a range.")
        _q_ext = st.slider("Max distance from the 4H average (ATR)", 0.0, 8.0, 3.0, 0.5,
                           key="q_ext",
                           help="Stops you entering after price has already run. 0 = off.")
    TR_PARAMS = trend_retrace.StrategyParams(
        trend_candles=int(_trend_n),
        stop_mode=(trend_retrace.STOP_BELOW_4H_CANDLE if _stop_choice.startswith("Below the last")
                   else trend_retrace.STOP_BELOW_SUPPORT),
        stop_atr_mult=_stop_atr, target_r=_target_r,
        min_h1_body_ratio=_q_body, min_trend_move_atr=_q_trend, max_extension_atr=_q_ext)
else:
    TR_PARAMS = None

st.sidebar.subheader("Your account")
ACCOUNT_EQUITY = st.sidebar.number_input("Account equity ($)", min_value=0.0, value=10000.0,
                                          step=100.0, key="acct_equity")
RISK_PCT = st.sidebar.number_input("Risk per trade (%)", min_value=0.1, max_value=100.0,
                                    value=1.0, step=0.1, key="acct_risk",
                                    help="The most you'll lose if the stop is hit. Used "
                                         "everywhere — plans, positions and the calculator.")
LEVERAGE = st.sidebar.number_input("Leverage", min_value=1.0, max_value=50.0, value=1.0,
                                    step=0.5, key="acct_lev")
with st.sidebar.expander("Costs"):
    FEE_RATE = st.number_input("Fee rate (round trip)", min_value=0.0, value=0.0006,
                               step=0.0001, format="%.4f", key="acct_fee")
    SLIPPAGE = st.number_input("Slippage", min_value=0.0, value=0.0005, step=0.0001,
                               format="%.4f", key="acct_slip")
st.sidebar.caption(f"Risking **${ACCOUNT_EQUITY * RISK_PCT / 100:,.2f}** per trade.")

st.sidebar.markdown("---")
with st.sidebar.expander("Advanced settings"):
    st.caption("Data-freshness rules and display timezone. The defaults rarely need "
               "changing — crypto prices come live from the exchange regardless.")
    live_threshold = st.number_input("LIVE threshold (seconds)", value=60, min_value=5)
    stale_threshold = st.number_input("STALE threshold (seconds)", value=900, min_value=60)
    tz_name = st.text_input("Timezone (IANA)", value="Australia/Sydney")
    try:
        local_tz = zoneinfo.ZoneInfo(tz_name)
        st.caption(f"Local now: {datetime.now(local_tz).strftime('%Y-%m-%d %H:%M %Z')}")
    except Exception:
        st.error(f"'{tz_name}' is not a valid IANA timezone — using UTC.")
        local_tz = timezone.utc

# A free CoinGecko demo key raises the rate limit a lot. Optional.
_cg_key = ""
try:
    _cg_key = st.secrets.get("COINGECKO_API_KEY", "")
except Exception:
    _cg_key = ""
if _cg_key:
    coingecko.set_api_key(_cg_key)

_cmc_key = ""
try:
    _cmc_key = st.secrets.get("COINMARKETCAP_API_KEY", "")
except Exception:
    _cmc_key = ""
if _cmc_key:
    sentiment.set_cmc_api_key(_cmc_key)


@st.cache_data(ttl=3600, show_spinner=False)
def _coingecko_matches(symbol: str):
    return coingecko.search_coins(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def _exchange_has(symbol: str):
    """Is this symbol tradable on Binance/Kraken? A two-candle request is the
    cheapest reliable check. Used for watchlist names Hyperliquid doesn't list."""
    ticker = f"{symbol.upper()}-USD"
    df, _e = exchanges.fetch_binance_klines(ticker, "4h", limit=2)
    if df is not None and not df.empty:
        return ticker, "Binance"
    df, _e = exchanges.fetch_kraken_ohlc(ticker, "4h")
    if df is not None and not df.empty:
        return ticker, "Kraken"
    return None, None


@st.cache_data(ttl=120, show_spinner=False)
def _cached_bybit_tickers():
    return exchanges.fetch_bybit_tickers()


@st.cache_data(ttl=300, show_spinner=False)
def _cached_hl_contexts():
    ctxs, err = hyperliquid_data.fetch_market_contexts()
    return ([{"name": c.name, "ticker": c.ticker, "label": c.label,
              "volume": c.day_volume_usd, "mark": c.mark_price,
              "oi": c.open_interest_usd, "funding": c.funding_rate,
              "change": c.change_24h_pct} for c in ctxs], err)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_period_changes():
    return coingecko.fetch_period_changes("90d", limit=100)


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_fear_greed():
    fg, err = sentiment.fetch_fear_greed()
    if fg is None:
        return None, err
    return {"value": fg.value, "classification": fg.classification, "source": fg.source,
            "previous": fg.previous, "direction": fg.direction}, err

_fg, _fg_err = _cached_fear_greed()
if _fg:
    st.sidebar.metric("Fear & Greed", f"{_fg['value']} · {_fg['classification']}",
                      (f"{_fg['direction']} from {_fg['previous']}"
                       if _fg.get("previous") is not None else None), delta_color="off")
    st.sidebar.caption(f"Source: {_fg['source']}. A whole-market mood gauge — it says "
                       f"nothing about any single setup.")
elif _fg_err:
    st.sidebar.caption(f"Fear & Greed unavailable: {_fg_err}")

st.sidebar.markdown("---")
st.sidebar.subheader("Tradable universe")
venue_mode = st.sidebar.radio(
    "Restrict coins to",
    ["Bybit only", "Hyperliquid only", "Bybit or Hyperliquid",
     "Bybit AND Hyperliquid", "All top-100"],
    key="venue_mode",
    help="Market-cap rank includes exchange tokens (WBT, OKB) and wrapped assets you "
         "cannot trade as perpetuals. Filtering to your venue keeps the scan actionable. "
         "Bybit blocks requests from the US server this app runs on, so its list may be "
         "unavailable.")
_VENUE_MODES = {"Bybit only": venues_mod.MODE_BYBIT,
                "Hyperliquid only": venues_mod.MODE_HYPERLIQUID,
                "Bybit or Hyperliquid": venues_mod.MODE_EITHER,
                "Bybit AND Hyperliquid": venues_mod.MODE_BOTH}

if venue_mode == "All top-100":
    CRYPTO_TICKERS = dict(CRYPTO_TICKERS_ALL)
    VENUE_TAGS = {}
    st.sidebar.caption(f"{len(CRYPTO_TICKERS)} coins (unfiltered).")
else:
    _by, _hl, _vprob = _load_venue_listings()
    _listings = venues_mod.VenueListings(bybit=_by, hyperliquid=_hl, problems=_vprob)
    CRYPTO_TICKERS, VENUE_TAGS, _vnotes = venues_mod.filter_universe(
        CRYPTO_TICKERS_ALL, _listings, mode=_VENUE_MODES[venue_mode])
    if venue_mode == "Hyperliquid only":
        st.sidebar.caption(f"Hyperliquid: {len(_hl)} perps")
    else:
        st.sidebar.caption(f"Bybit: {len(_by) or 'unavailable'} · "
                           f"Hyperliquid: {len(_hl) or 'unavailable'} perps")
    for _n in _vnotes:
        if "NOT applied" in _n or "missing from the list" in _n or "instead of an empty" in _n:
            st.sidebar.warning(_n)
        else:
            st.sidebar.caption(_n)

st.sidebar.markdown("---")
st.sidebar.subheader("Data source")
crypto_source = st.sidebar.radio(
    "Crypto prices from",
    ["Bybit", "Exchange (Binance/Kraken)", "CoinGecko", "Yahoo Finance"],
    key="crypto_source",
    help=("Exchange uses Binance, falling back to Kraken. It gives true OHLCV at every "
          "interval with no meaningful rate limit — the best option. CoinGecko covers "
          "more obscure coins but rate-limits hard and has no true daily OHLC. Yahoo "
          "misses newer listings. Sources fall back to each other automatically."),
)
CRYPTO_SOURCE = ("bybit" if crypto_source == "Bybit"
                 else "exchange" if crypto_source.startswith("Exchange")
                 else "coingecko" if crypto_source.startswith("CoinGecko")
                 else "yahoo")
CRYPTO_PREFERS_CG = CRYPTO_SOURCE == "coingecko"
st.sidebar.caption(
    "Stocks and commodities always use Yahoo Finance — CoinGecko has no equities "
    "or futures data."
)
BYBIT_TICKERS = {}
if CRYPTO_SOURCE == "bybit":
    BYBIT_TICKERS, _bybit_err = _cached_bybit_tickers()
    if BYBIT_TICKERS:
        st.sidebar.caption(f"Bybit: {len(BYBIT_TICKERS)} USDT perps.")
    else:
        st.sidebar.warning(
            "**Bybit can't be reached from this server** — it blocks US traffic and "
            "Streamlit is US-hosted. Binance is blocked too. Candles are coming from "
            "Hyperliquid instead; prices on majors are near-identical, so scans stay "
            "accurate and you place the orders on Bybit. Running this app on your own "
            "computer in Australia would use Bybit directly — see RUNNING_LOCALLY.md in "
            "the download.")

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
    "Streamlit Cloud wipes the database whenever the app redeploys — including every time "
    "you upload new files. Export your journal and tracking CSVs before updating, then "
    "import them back afterwards."
)

(tab_market, tab_chart, tab_tools, tab_learn, tab_scan, tab_positions, tab_risk,
 tab_journal, tab_track, tab_backtest) = st.tabs(
    ["🌍 Market", "📉 Charts", "🌡️ Market tools", "📚 What coins do", "🎯 Scanner",
     "📋 Positions", "🧮 Risk", "📓 Journal", "📈 Tracking", "🔁 Backtest"]
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

        # 24h change, from whichever source knows it.
        _chg_pct = _chg_abs = None
        _base = ticker.upper().replace("-USD", "")
        if BYBIT_TICKERS.get(_base):
            _chg_pct = BYBIT_TICKERS[_base]["change_pct"]
            _chg_abs = BYBIT_TICKERS[_base]["change_abs"]
        elif asset_class == "crypto":
            _hl, _ = _cached_hl_contexts()
            _m = next((c for c in _hl if c["name"].upper() == _base), None)
            if _m is None:
                _prev, _e = exchanges.fetch_binance_klines(ticker, "1d", limit=2)
                if _prev is not None and len(_prev) >= 2 and quote.price:
                    _y = float(_prev["Close"].iloc[-2])
                    _chg_abs = quote.price - _y
                    _chg_pct = _chg_abs / _y * 100 if _y else None

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Instrument", quote.ticker)
        m2.metric("Price", f"${format_price(quote.price)}" if quote.price else "—",
                  (f"{_chg_pct:+.2f}%  ({'+' if (_chg_abs or 0) >= 0 else '-'}$"
                   f"{format_price(abs(_chg_abs))})") if _chg_pct is not None else None)
        m3.metric("Status", f"{icons[quote.status]} {quote.status.value}")
        m4.metric("Interval", quote.interval)
        if _chg_pct is not None:
            _up = _chg_pct >= 0
            st.markdown(
                theme.pill(f"{'▲' if _up else '▼'} {abs(_chg_pct):.2f}% in 24h "
                           f"({'+' if _up else '-'}${format_price(abs(_chg_abs))})",
                           theme.GREEN if _up else theme.RED),
                unsafe_allow_html=True)

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

with tab_chart:
    st.subheader("Charts")

    c1, c2, c3 = st.columns([2, 1, 1])
    _all_lists = {"Crypto": CRYPTO_TICKERS_ALL, "Stocks": STOCK_TICKERS,
                  "Commodities": COMMODITY_TICKERS, "FX": FX_TICKERS}
    _ch_market = c1.selectbox("Market", list(_all_lists), key="ch_market")
    _ch_names = list(_all_lists[_ch_market])
    _ch_pick = c2.selectbox("Instrument", _ch_names, key="ch_pick")
    _ch_interval = c3.selectbox("Timeframe", list(charting.INTERVALS), index=3,
                                key="ch_interval")

    _default_symbol = charting.to_tradingview_symbol(
        _all_lists[_ch_market][_ch_pick], _ch_market,
        exchange=("BYBIT" if CRYPTO_SOURCE == "bybit" else "BYBIT"))
    _symbol = st.text_input("TradingView symbol", value=_default_symbol, key="ch_symbol",
                            help="Edit this for anything not in the lists, or use the "
                                 "search inside the chart itself.")
    components.html(charting.widget_html(_symbol, _ch_interval, height=640), height=660)

    st.caption(
        "Full TradingView drawing tools are in the toolbar on the left of the chart — "
        "trend lines, Fibonacci retracements, boxes and text. **Drawings are not saved**: "
        "the embedded chart has no account attached, so they disappear when the page "
        "reloads or you switch tabs. For anything you want to keep, draw it on "
        "tradingview.com itself."
    )
    st.caption(
        "This chart is TradingView's own data feed, not the candles the scanner used. "
        "Prices should agree closely, but a level read off here can differ slightly from "
        "one the scanner calculated — the scanner's numbers are the ones its plans are "
        "based on."
    )

with tab_tools:
    st.subheader("Market tools")
    st.caption(
        "Conditions across the whole market. None of these is a trade signal, and none "
        "has been shown to improve this strategy — they're context you read beside the "
        "scanner. Where a reading has a conventional meaning it's given as convention, "
        "not prediction."
    )

    # --- Fear & Greed ---------------------------------------------------
    st.markdown("##### Fear & Greed")
    if _fg:
        _v = _fg["value"]
        _col = (theme.RED if _v <= 24 else theme.AMBER if _v <= 44 else
                theme.GREY if _v <= 55 else theme.BLUE if _v <= 74 else theme.GREEN)
        g1, g2 = st.columns([1, 2])
        g1.metric("Index", f"{_v}", _fg["classification"], delta_color="off")
        with g2:
            st.markdown(theme.gauge(_v, "0 · Extreme fear", "100 · Extreme greed", _col),
                        unsafe_allow_html=True)
            st.caption(f"Source: {_fg['source']}"
                       + (f" · {_fg['direction']} from {_fg['previous']} yesterday"
                          if _fg.get("previous") is not None else ""))
        st.caption("Often read as a contrarian gauge — fear as opportunity, greed as "
                   "caution — but that's folklore until tested. The band is recorded with "
                   "every setup so the learner can check whether it matters for you.")
    else:
        st.info("Fear & Greed unavailable right now.")

    # --- Altcoin season -------------------------------------------------
    st.markdown("##### Altcoin season")
    _changes, _ch_err = _cached_period_changes()
    _alt = market_tools.altcoin_season_index(_changes) if _changes else None
    if _alt:
        a1, a2 = st.columns([1, 2])
        a1.metric("Index", f"{_alt.index}", _alt.label, delta_color="off")
        with a2:
            _acol = (theme.PURPLE if _alt.index >= 75 else
                     theme.AMBER if _alt.index >= 25 else theme.BLUE)
            st.markdown(theme.gauge(_alt.index, "0 · Bitcoin season",
                                    "100 · Altcoin season", _acol), unsafe_allow_html=True)
            st.caption(f"{_alt.outperformers} of the top {_alt.sample} coins beat Bitcoin "
                       f"over 90 days (Bitcoin itself: {_alt.btc_change_pct:+.1f}%).")
        st.caption("A description of the last 90 days, not a forecast. It matters for this "
                   "strategy mainly as context: in Bitcoin season, altcoin longs are "
                   "swimming against the tide.")
    else:
        st.info(f"Altcoin season unavailable: {_ch_err or 'no data'}")

    # --- Perp flow ------------------------------------------------------
    st.markdown("##### Funding & open interest (Hyperliquid)")
    _fl_ctxs, _fl_err = _cached_hl_contexts()
    if _fl_ctxs:
        _objs = [hyperliquid_data.MarketContext(c["name"], c["volume"], c["mark"],
                                                 c["oi"], c["funding"], c.get("change"), None)
                 for c in _fl_ctxs if c["volume"] > 5e6]
        _rows = market_tools.flow_rows(_objs)
        _longs, _shorts = market_tools.most_crowded(_rows, limit=5)
        f1, f2 = st.columns(2)
        with f1:
            st.markdown(theme.pill("Longs paying most", theme.RED), unsafe_allow_html=True)
            for r_ in _longs:
                st.caption(f"**{r_.name}** · {r_.funding_rate * 100:+.4f}%/h "
                           f"({r_.funding_annual_pct:+.0f}%/yr) · OI "
                           f"${r_.open_interest_usd / 1e6:,.0f}M")
        with f2:
            st.markdown(theme.pill("Shorts paying most", theme.GREEN), unsafe_allow_html=True)
            for r_ in _shorts:
                st.caption(f"**{r_.name}** · {r_.funding_rate * 100:+.4f}%/h "
                           f"({r_.funding_annual_pct:+.0f}%/yr) · OI "
                           f"${r_.open_interest_usd / 1e6:,.0f}M")
        st.caption("Positive funding means longs pay shorts — more traders sit long, and a "
                   "stretched reading often precedes a flush of those longs. It shows "
                   "positioning, not direction: crowded longs can stay crowded while price "
                   "keeps rising.")
        with st.expander("All markets by open interest"):
            _top_oi = sorted(_rows, key=lambda r_: r_.open_interest_usd, reverse=True)[:25]
            st.dataframe(pd.DataFrame([{
                "Market": r_.name,
                "Funding /h": f"{r_.funding_rate * 100:+.4f}%",
                "Funding /yr": f"{r_.funding_annual_pct:+.0f}%",
                "Open interest": f"${r_.open_interest_usd / 1e6:,.0f}M",
                "24h volume": f"${r_.volume_usd / 1e6:,.0f}M",
                "Turnover": (f"{market_tools.turnover_ratio(r_):.1f}x"
                             if market_tools.turnover_ratio(r_) else "—"),
                "Crowding": r_.crowded or "—",
            } for r_ in _top_oi]), use_container_width=True, hide_index=True)
    else:
        st.info(f"Hyperliquid data unavailable: {_fl_err or ''}")

    # --- RSI heatmap ----------------------------------------------------
    st.markdown("##### RSI heatmap")
    rsi_count = st.selectbox("Coins", [10, 20, 30], index=0, key="rsi_count",
                             help="Uses Hyperliquid candles; requests are paced, so more "
                                  "coins take longer.")
    if st.button("Load RSI heatmap", use_container_width=True):
        _ctxs2, _ = _cached_hl_contexts()
        _pick = sorted(_ctxs2, key=lambda c: c["volume"], reverse=True)[:rsi_count]
        bar = st.progress(0.0, text="Loading…")
        rows = []
        for i_, c in enumerate(_pick):
            bar.progress(i_ / max(len(_pick), 1), text=f"{i_ + 1}/{len(_pick)} · {c['name']}")
            frames = {}
            for tf, n in (("1h", 120), ("4h", 120), ("1d", 120)):
                df, _e = hyperliquid_data.fetch_candles(c["ticker"], tf, n)
                frames[tf] = df if df is not None else pd.DataFrame()
            vals = market_tools.rsi_row(frames)
            rows.append({"Market": c["name"],
                         **{f"RSI {tf}": (round(v, 1) if v is not None else None)
                            for tf, v in vals.items()},
                         "1d state": market_tools.rsi_state(vals.get("1d"))})
        bar.empty()
        st.session_state.rsi_rows = rows

    if st.session_state.get("rsi_rows"):
        def _mark(v):
            # Colour without a charting dependency: a dot carries the band and
            # the number stays readable on any background.
            if v is None:
                return "—"
            dot = "🔴" if v >= market_tools.OVERBOUGHT else (
                "🟢" if v <= market_tools.OVERSOLD else "⚪")
            return f"{dot} {v:.0f}"

        _df = pd.DataFrame([{
            "Market": r_["Market"],
            **{k: _mark(r_[k]) for k in ("RSI 1h", "RSI 4h", "RSI 1d") if k in r_},
            "1d state": r_["1d state"],
        } for r_ in st.session_state.rsi_rows])
        st.dataframe(_df, use_container_width=True, hide_index=True)
        st.caption("🔴 overbought (70+) · ⚪ neutral · 🟢 oversold (30 or less)")
        st.caption("Above 70 is conventionally 'overbought', below 30 'oversold'. In a strong "
                   "trend RSI can sit overbought for weeks while price keeps climbing, so "
                   "treat it as a description of momentum, not a reversal signal.")

with tab_learn:
    st.subheader("What each coin actually does")
    st.caption(
        "One line per coin, in plain English. These are written from general knowledge and "
        "don't update themselves — projects pivot and change direction, so look anything up "
        "properly before putting money into it. Nothing here is a view on whether a coin is "
        "worth owning, only what it's for."
    )

    _g1, _g2 = st.columns([2, 1])
    _q = _g1.text_input("Search", key="gloss_q", placeholder="e.g. sol, oracle, privacy")
    _cats = ["All"] + sorted(coin_info.CATEGORIES)
    _cat = _g2.selectbox("Category", _cats, key="gloss_cat")

    _order = [t.replace("-USD", "") for t in CRYPTO_TICKERS_ALL.values()]
    _ranked = [s for s in _order if coin_info.describe(s)]
    _rest = sorted(s for s in coin_info.COINS if s not in _ranked)
    _symbols = _ranked + _rest

    def _matches(sym):
        cat, text = coin_info.describe(sym)
        if _cat != "All" and cat != _cat:
            return False
        if not _q:
            return True
        needle = _q.strip().lower()
        return needle in sym.lower() or needle in text.lower() or needle in cat.lower()

    _shown = [s for s in _symbols if _matches(s)]
    _covered, _total = coin_info.coverage(_order)
    st.caption(f"{len(_shown)} shown · {len(coin_info.COINS)} coins described, covering "
               f"{_covered} of the top {_total} by market cap.")

    _colours = {"Layer 1": theme.GREEN, "Layer 2": theme.BLUE, "DeFi": theme.PURPLE,
                "Meme": theme.AMBER, "Exchange": theme.BLUE, "AI": theme.PURPLE,
                "Infrastructure": theme.GREY, "Privacy": theme.GREY,
                "Payments": theme.GREEN, "Gaming": theme.AMBER,
                "Stablecoin": theme.GREY, "RWA": theme.BLUE}
    for sym in _shown:
        cat, text = coin_info.describe(sym)
        rank = _order.index(sym) + 1 if sym in _order else None
        with st.container(border=True):
            st.markdown(
                f"**{sym}**" + (f"  ·  #{rank} by market cap" if rank else "")
                + "  " + theme.pill(cat, _colours.get(cat, theme.GREY)),
                unsafe_allow_html=True)
            st.write(text)

    if not _shown:
        st.info("Nothing matches. Try a different word, or look the coin up below.")

    st.markdown("##### Look up any other coin")
    st.caption("Fetches the project's own description from CoinGecko — one call, so it's "
               "done on request rather than for a whole list.")
    _lu = st.text_input("Ticker", key="gloss_lookup", placeholder="e.g. CARDS")
    if st.button("Look it up", key="gloss_go") and _lu.strip():
        with st.spinner("Searching…"):
            _cands, _err = _coingecko_matches(_lu.strip().upper())
        if not _cands:
            st.warning(_err or "No coin found with that ticker.")
        else:
            for c in _cands[:3]:
                with st.container(border=True):
                    st.markdown(f"**{c['name']}** ({c['symbol']})"
                                + (f"  ·  #{c['rank']} by market cap" if c["rank"] else ""))
                    _txt, _home, _derr = coingecko.fetch_description(c["id"])
                    if _txt:
                        st.write(_txt[:700] + ("…" if len(_txt) > 700 else ""))
                        if _home:
                            st.caption(f"Project site: {_home}")
                        st.caption("This is the project's own description of itself, so it "
                                   "reads as marketing. Useful for what it does, not for "
                                   "whether it's any good.")
                    else:
                        st.caption(_derr or "No description available.")

# ---------------------------------------------------------------------
# TAB: Scanner
# ---------------------------------------------------------------------
with tab_scan:
    st.subheader("Candidate scanner")
    st.caption(
        "Derives the checklist from real 1D / 4H / 1H structure. Every item is "
        "overridable — ❔ means it could not be evaluated and scores zero rather than guessing."
    )

    _modes = (["Rank the universe", "Auto-scan one instrument"] if USE_TR
              else ["Rank the universe", "Auto-scan one instrument", "Manual checklist"])
    mode = st.radio("Mode", _modes, key="scan_mode",
                    help="Rank the universe checks many coins and shortlists the best; "
                         "single-instrument mode shows every rule for one coin in detail.")

    score_evidence, seq_evidence = {}, {}

    if mode == "Rank the universe":
        market = st.radio("Market", ["Crypto", "Stocks", "Commodities", "FX"],
                          horizontal=True, key="uni_market")
        NON_CRYPTO = market != "Crypto"
        if NON_CRYPTO:
            _lists = {"Stocks": STOCK_TICKERS, "Commodities": COMMODITY_TICKERS,
                      "FX": FX_TICKERS}
            _list = _lists[market]
            st.caption(
                f"Your three rules applied to {market.lower()}, using Yahoo Finance data. "
                f"Yahoo has no 4H interval, so 4H candles are built from 1H bars.")
            if market == "Stocks":
                st.warning(
                    "**Stocks aren't crypto, and the strategy assumes they are.** They trade "
                    "about 6.5 hours a day, so a '4H candle' spans session breaks, and "
                    "overnight gaps can open straight past your stop — the loss can exceed "
                    "1R, which never happens in the backtest. Outside market hours no limit "
                    "order fills and prices are stale. Treat these setups with more caution "
                    "than crypto ones, and size smaller.")
            elif market == "FX":
                st.caption(
                    "FX trades 24 hours, five days a week, so it fits the strategy better "
                    "than stocks — but weekend gaps can still open past a stop, and Yahoo's "
                    "FX prices are indicative rather than a broker's dealable quotes.")
            else:
                st.caption(
                    "Only commodities Bybit lists as perpetuals — gold, silver and crude — "
                    "so every setup is one you can actually place. They trade 24/7 with "
                    "funding every four hours, which suits the strategy better than stocks.")
                st.caption(
                    "Candles come from Yahoo (spot gold and silver, and the crude futures "
                    "contract) because Bybit's API is blocked from this server. Those track "
                    "the perps closely but aren't identical — check the price on Bybit "
                    "before placing an order. Crude futures also pause briefly each day and "
                    "close at weekends, so crude may read STALE while CLUSDT is still "
                    "trading.")
            st.caption("None of this has been backtested — the backtest only covers crypto.")

        coin_source = st.radio(
            "Which coins",
            ["Top by market cap", "High volume, outside the top 20 (Hyperliquid)",
             "My watchlist"],
            key="uni_coin_source",
            help="The second option ranks Hyperliquid perps by their 24-hour volume on "
                 "Hyperliquid — the liquidity you'd actually trade into — and skips the "
                 "largest coins by market cap.")
        WATCHLIST_MODE = (coin_source == "My watchlist") and not NON_CRYPTO
        HL_MODE = ((coin_source.startswith("High volume") or WATCHLIST_MODE)
                   and not NON_CRYPTO)
        if WATCHLIST_MODE:
            wl_text = st.text_area(
                "Coins to scan", value=watchlist_mod.DEFAULT_WATCHLIST, height=90,
                key="wl_text",
                help="Separate with commas. Full names work too (stacks, immutable). "
                     "Anything that can't be matched is listed below rather than guessed.")
            _wl_markets = []
            _wl_ctxs, _wl_err = _cached_hl_contexts()
            if _wl_ctxs:
                _wl_markets = [c["name"] for c in _wl_ctxs]
            _alias_text, _ = storage.load_value("wl_aliases")
            _alias_text = _alias_text or ""
            _aliases, _bad_alias = watchlist_mod.parse_aliases(_alias_text)
            _wl_found, _wl_missing = watchlist_mod.resolve(wl_text, _wl_markets,
                                                            aliases=_aliases)
            _elsewhere = []      # set below; defined here so a failed market
            _via_cg = []         # list can't crash the rest of the page
            if _wl_markets:
                st.caption(f"Matched {len(_wl_found)} of "
                           f"{len(watchlist_mod.parse_list(wl_text))} entries on Hyperliquid: "
                           + ", ".join(sorted({r.market for r in _wl_found})))
                # Anything Hyperliquid doesn't list may still trade elsewhere.
                _elsewhere = []
                _still_missing = []
                for u in _wl_missing:
                    _sym = watchlist_mod._normalise(_aliases.get(
                        watchlist_mod._normalise(u), u))
                    _tk, _venue = _exchange_has(_sym)
                    if _tk:
                        _elsewhere.append((u, _sym, _tk, _venue))
                    else:
                        _still_missing.append(u)
                # Last resort: CoinGecko covers coins no exchange here lists.
                _via_cg = []
                _truly_missing = []
                _cg_choice = storage.load_value("wl_cg_ids")[0] or {}
                for u in _still_missing:
                    _sym = watchlist_mod._normalise(_aliases.get(
                        watchlist_mod._normalise(u), u))
                    _cands, _cg_err = _coingecko_matches(_sym)
                    if not _cands:
                        _truly_missing.append((u, _cg_err))
                        continue
                    _chosen = _cg_choice.get(_sym)
                    if len(_cands) == 1:
                        _chosen = _chosen or _cands[0]["id"]
                    if _chosen:
                        _pick = next((c for c in _cands if c["id"] == _chosen), _cands[0])
                        _via_cg.append((u, _sym, _pick))
                    else:
                        # Several projects share this ticker — the trader picks.
                        st.warning(f"**{u}** matches {len(_cands)} different projects with "
                                   f"the ticker {_sym}. Choose which one you mean:")
                        _opt = st.selectbox(
                            f"Which {_sym}?",
                            [f"{c['name']} ({c['id']})"
                             + (f" · rank {c['rank']}" if c["rank"] else " · unranked")
                             for c in _cands],
                            key=f"cgpick_{_sym}")
                        if st.button(f"Use this for {_sym}", key=f"cgset_{_sym}"):
                            _cg_choice[_sym] = _cands[
                                [f"{c['name']} ({c['id']})"
                                 + (f" · rank {c['rank']}" if c["rank"] else " · unranked")
                                 for c in _cands].index(_opt)]["id"]
                            storage.save_value("wl_cg_ids", _cg_choice)
                            st.rerun()

                if _via_cg:
                    st.info("Not on any exchange here, so CoinGecko's candles will be used: "
                            + "; ".join(f"**{u}** → {p['name']}" for u, _s, p in _via_cg)
                            + ". CoinGecko aggregates across exchanges and has no true daily "
                              "OHLC, so these scans are coarser — and you still need a venue "
                              "that actually lists the coin to trade it.")
                if _truly_missing:
                    _hints = []
                    for u, why in _truly_missing:
                        s_ = watchlist_mod.suggest(u, _wl_markets)
                        _hints.append(f"**{u}**" + (f" → did you mean {', '.join(s_)}?"
                                                    if s_ else f" ({why})"))
                    st.warning("Couldn't find anywhere: " + "; ".join(_hints)
                               + ". Use the tools below to find the right name, or map it "
                                 "yourself — nothing is guessed for you, because scanning "
                                 "the wrong coin looks exactly like scanning the right one.")

                with st.expander(f"🔎 Browse all {len(_wl_markets)} Hyperliquid markets"):
                    _q = st.text_input("Search", key="wl_search",
                                       placeholder="e.g. light, card, drv")
                    _hits = watchlist_mod.search_markets(_q, _wl_markets, limit=60)
                    if _hits:
                        st.write(" · ".join(f"`{m}`" for m in _hits))
                        st.caption("Copy the exact name into your list above. If a coin isn't "
                                   "here, Hyperliquid doesn't list it as a perpetual and it "
                                   "can't be scanned or traded there.")
                    else:
                        st.caption("No market contains that text.")

                with st.expander("🔗 My name mappings"):
                    st.caption("One per line, e.g. `lighter = LIT`. Use this when you know "
                               "which market a name refers to. Check the project on "
                               "Hyperliquid first — tickers are reused across projects, and "
                               "mapping to the wrong one would silently scan the wrong coin.")
                    _new_alias = st.text_area("Mappings", value=_alias_text, height=90,
                                              key="wl_alias_text")
                    if st.button("Save mappings", key="wl_alias_save"):
                        _parsed, _badlines = watchlist_mod.parse_aliases(_new_alias)
                        _unknown = [f"{k} → {v}" for k, v in _parsed.items()
                                    if v not in {watchlist_mod._normalise(m)
                                                 for m in _wl_markets}]
                        storage.save_value("wl_aliases", _new_alias)
                        if _badlines:
                            st.warning("Couldn't read: " + "; ".join(_badlines))
                        if _unknown:
                            st.warning("Saved, but these point at markets Hyperliquid doesn't "
                                       "list: " + "; ".join(_unknown))
                        st.rerun()
                    if _bad_alias:
                        st.warning("Existing lines that couldn't be read: "
                                   + "; ".join(_bad_alias))
            else:
                st.error(f"Couldn't load Hyperliquid's market list: {_wl_err}")
        elif HL_MODE:
            v1, v2, v3 = st.columns(3)
            hl_exclude_n = v1.number_input("Skip the top N by market cap", min_value=0,
                                           max_value=100, value=20, step=5, key="hl_excl")
            hl_min_vol = v2.number_input("Min 24h volume ($M)", min_value=0.0, value=20.0,
                                         step=5.0, key="hl_minvol",
                                         help="Volume on Hyperliquid only. Thinner markets "
                                              "mean wider spreads and more slippage.")
            hl_count = v3.selectbox("How many coins", [10, 20, 30, 50, 100], index=1,
                                    key="hl_count",
                                    format_func=lambda n: f"Top {n} by volume")
            st.caption(
                "⚠️ High volume outside the top coins often means a sharp move, news or a "
                "pump. These markets are usually more volatile, and your strategy hasn't "
                "been backtested on them — treat results with extra caution.")
        if USE_TR:
            st.caption(
                "Applies your three rules to each coin: **two 4H candles of higher highs and "
                "higher lows**, a **bullish 1H candle**, and a **5m retrace to previous "
                "support** (reversed for shorts). Only closed candles count. A setup meeting "
                "all three rules scores 10/10; the score is a checklist count, not a "
                "probability of profit."
            )
            if NON_CRYPTO:
                _lo, _hi, _default_n = uihelpers.scan_count(len(_list))
                if _lo is None:
                    universe_size = _default_n
                    st.caption(f"Scanning all {universe_size} — there are only a few.")
                else:
                    universe_size = st.slider(f"How many {market.lower()} to scan",
                                               _lo, _hi, _default_n, key="uni_noncrypto_n")
            elif WATCHLIST_MODE:
                universe_size = len(_wl_found) + len(_elsewhere) + len(_via_cg)
            elif HL_MODE:
                universe_size = hl_count
            else:
                universe_size = st.selectbox(
                    "Scan the top…", [10, 20, 30, 50, 100], index=1, key="uni_size",
                    format_func=lambda n: f"Top {n} by market cap",
                    help="More coins means more API calls and a longer wait.")
            uni_direction, uni_min_rr = "Auto", 3.0
        else:
            st.caption(
                "Scores many instruments on the original 10-point confluence checklist "
                "using daily, 4H and 1H candles. Treat results as a shortlist, then run a "
                "full single-instrument scan on anything promising."
            )
            u1, u2, u3 = st.columns(3)
            if NON_CRYPTO:
                _lo, _hi, _default_n = uihelpers.scan_count(len(_list))
                if _lo is None:
                    universe_size = _default_n
                    st.caption(f"Scanning all {universe_size} — there are only a few.")
                else:
                    universe_size = st.slider(f"How many {market.lower()} to scan",
                                               _lo, _hi, _default_n, key="uni_noncrypto_n")
            elif WATCHLIST_MODE:
                universe_size = len(_wl_found) + len(_elsewhere) + len(_via_cg)
            elif HL_MODE:
                universe_size = hl_count
            else:
                universe_size = u1.selectbox(
                    "Scan the top…", [10, 20, 30, 50, 100], index=1, key="uni_size",
                    format_func=lambda n: f"Top {n} by market cap",
                    help="More coins means more API calls and a longer wait.")
            uni_direction = u2.selectbox("Direction", ["Auto", "Long", "Short"], key="uni_dir")
            uni_min_rr = u3.number_input("Min R:R", min_value=3.0, value=3.0,
                                          step=0.5, key="uni_rr")

        if NON_CRYPTO:
            st.caption(f"Roughly {universe_size * 3:.0f}s for {universe_size} instruments — "
                       f"Yahoo is slower than an exchange API.")
        elif HL_MODE:
            st.caption(f"Roughly {universe_size * 1.6 + 2:.0f}s for {universe_size} coins. "
                       f"One request each: 4H and 1H candles are built from the 5m data "
                       f"rather than fetched separately.")
            if WATCHLIST_MODE and not _wl_found:
                st.info("Nothing to scan yet — fix the unmatched entries above.")
        else:
            per_coin = 0.6 if CRYPTO_SOURCE == "exchange" else 1.4   # 4 calls/coin
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
            hl_marks = {}
            if NON_CRYPTO:
                _names = list(_list)[:universe_size]
                instruments = [(n, _list[n], (_list[n], None)) for n in _names]
                st.session_state.hl_info = {}
            elif WATCHLIST_MODE:
                _all, _e = _cached_hl_contexts()
                _by_name = {c["name"]: c for c in _all}
                _sel = [_by_name[r.market] for r in _wl_found if r.market in _by_name]
                instruments = [(c["label"], c["ticker"], (c["ticker"], None)) for c in _sel]
                # Watchlist coins that live on another venue.
                instruments += [(f"{sym} ({venue})", tk, (tk, None))
                                for _u, sym, tk, venue in _elsewhere]
                instruments += [(f"{sym} (CoinGecko)", f"{sym}-USD", (f"{sym}-USD", p["id"]))
                                for _u, sym, p in _via_cg]
                hl_marks = {c["ticker"]: c["mark"] for c in _sel}
                st.session_state.hl_info = {
                    c["ticker"]: hyperliquid_data.MarketContext(
                        c["name"], c["volume"], c["mark"], c["oi"], c["funding"],
                        c.get("change"), None)
                    for c in _sel}
            elif HL_MODE:
                _ctxs, _hl_err = hyperliquid_data.fetch_market_contexts()
                if not _ctxs:
                    st.error(f"Couldn't load Hyperliquid volumes: {_hl_err}")
                _top = {v.upper().replace("-USD", "")
                        for v in list(CRYPTO_TICKERS_ALL.values())[:int(hl_exclude_n)]}
                _picks = hyperliquid_data.rank_by_volume(
                    _ctxs, exclude_bases=_top, min_volume_usd=hl_min_vol * 1e6,
                    limit=universe_size)
                if _ctxs and not _picks:
                    st.warning("No Hyperliquid coins met the volume minimum after skipping "
                               "the top coins. Lower the minimum and try again.")
                instruments = [(c.label, c.ticker, (c.ticker, None)) for c in _picks]
                hl_marks = {c.ticker: c.mark_price for c in _picks}
                st.session_state.hl_info = {c.ticker: c for c in _picks}
            else:
                labels = list(CRYPTO_TICKERS)[:universe_size]
                instruments = [(lbl, CRYPTO_TICKERS[lbl],
                                 (CRYPTO_TICKERS[lbl], CRYPTO_CG_IDS.get(lbl)))
                               for lbl in labels]
                st.session_state.hl_info = {}

            bar = st.progress(0.0, text="Starting…")

            def _loader(payload):
                ticker, coin_id = payload
                if hyperliquid_data.is_hl_ticker(ticker):
                    out = {}
                    for tf, n in (("4h", 300), ("1d", 200), ("1h", 200)):
                        df, _e = hyperliquid_data.fetch_candles(ticker, tf, n)
                        out[tf] = df if df is not None else pd.DataFrame()
                    return out, []
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
                if NON_CRYPTO:
                    q = fetch_quote(ticker, "stock" if market == "Stocks" else "commodity",
                                     yf, live_threshold_seconds=live_threshold,
                                     stale_threshold_seconds=stale_threshold, max_retries=1)
                    return q.price if q and q.price else None
                if hyperliquid_data.is_hl_ticker(ticker):
                    return hl_marks.get(ticker)        # already fetched with the volumes
                if CRYPTO_SOURCE == "bybit":
                    _b = BYBIT_TICKERS.get(ticker.upper().replace("-USD", ""))
                    if _b:
                        return _b["last"]              # already fetched with the tickers
                px, _src, _err = exchanges.fetch_spot(ticker)
                if px is None and coin_id:
                    px, _upd, _e = coingecko.fetch_spot_price(coin_id)
                return px

            def _progress(i, total, label):
                bar.progress(min(i / max(total, 1), 1.0), text=f"{i}/{total} · {label}")

            def _tr_loader(payload):
                """4H, 1H and 5m candles for the Trend Retrace rules."""
                ticker, _coin_id = payload
                if NON_CRYPTO:
                    frames, _p = _yahoo_frames(ticker)
                    return frames
                out = {}
                if hyperliquid_data.is_hl_ticker(ticker):
                    # One request for 5m candles, then build 1H and 4H from
                    # them. Identical data, a third of the requests, and the
                    # venue's rate limit is what makes scans slow.
                    df, _e = hyperliquid_data.fetch_candles(ticker, "5m", 1000)
                    if df is not None and len(df) >= 720:
                        return trend_retrace.frames_from_5m(df)
                    for tf, lim in (("4h", 30), ("1h", 48), ("5m", 400)):
                        df2, _e = hyperliquid_data.fetch_candles(ticker, tf, lim)
                        out[tf] = df2 if df2 is not None else pd.DataFrame()
                    return out
                if CRYPTO_SOURCE == "bybit":
                    df, _e = exchanges.fetch_bybit_klines(ticker, "5m", limit=1000)
                    if df is not None and len(df) >= 720:
                        return trend_retrace.frames_from_5m(df)
                for tf, lim in (("4h", 30), ("1h", 48), ("5m", 400)):
                    df = None
                    if CRYPTO_SOURCE == "bybit":
                        df, _e = exchanges.fetch_bybit_klines(ticker, tf, limit=lim)
                    if df is None:
                        df, _e = exchanges.fetch_binance_klines(ticker, tf, limit=lim)
                    if df is None:
                        df, _e = exchanges.fetch_kraken_ohlc(ticker, tf)
                    if df is None:
                        # Bybit and Binance block US-hosted servers; Hyperliquid
                        # doesn't, so it's the last exchange-quality fallback.
                        df, _e = hyperliquid_data.fetch_candles(
                            f"HL:{ticker.upper().replace('-USD', '')}", tf, lim)
                    out[tf] = df if df is not None else pd.DataFrame()
                if all(f.empty for f in out.values()) and _coin_id:
                    # Not on any exchange here — CoinGecko is the last resort.
                    cg_frames, _p = _cached_cg_frames(_coin_id)
                    return {k: (cg_frames.get(k) if cg_frames.get(k) is not None
                                else pd.DataFrame()) for k in ("4h", "1h", "5m")}
                return out

            if USE_TR:
                with st.spinner("Scanning…"):
                    _extra = {"sentiment": sentiment.bucket(_fg["value"])} if _fg else None
                    _level_cache = {}

                    def _tr_loader_capture(payload):
                        frames = _tr_loader(payload)
                        # Keep the 5m structure so a ladder can be offered later
                        # without re-fetching everything.
                        _level_cache[payload[0]] = {
                            "5m": frames.get("5m"),
                            "atr": trend_retrace.atr(frames.get("5m"))}
                        return frames

                    ranked = trend_retrace.scan_universe_tr(
                        instruments, _tr_loader_capture, spot_loader=_spot,
                        params=TR_PARAMS, progress=_progress, extra_features=_extra)
                    st.session_state.uni_levels = {
                        r.ticker: {
                            "levels": trend_retrace.five_minute_levels(
                                _level_cache.get(r.ticker, {}).get("5m"),
                                r.direction, r.price or 0, TR_PARAMS),
                            "atr": _level_cache.get(r.ticker, {}).get("atr"),
                        }
                        for r in ranked if r.direction and r.price
                        and _level_cache.get(r.ticker, {}).get("5m") is not None}
            else:
                with st.spinner("Scanning…"):
                    ranked = scan_universe(
                        instruments, _loader, min_rr=uni_min_rr,
                        direction_override=None if uni_direction == "Auto" else uni_direction,
                        progress_callback=_progress, spot_loader=_spot)
            bar.empty()
            st.session_state.ranked = ranked
            if st.session_state.get("track_signals", True):
                added = tracking.record_from_ranked(ranked, VENUE_TAGS, min_rr=MIN_RR)
                if added:
                    st.toast(f"Recorded {added} new signal(s) for forward tracking.")

        ranked = st.session_state.get("ranked")
        if ranked:
            scored_all = [r for r in ranked if r.error is None]
            # Never hide setups that meet the target the trader chose.
            _rr_floor = (min(MIN_RR, TR_PARAMS.target_r) if USE_TR and TR_PARAMS
                         else max(MIN_RR, uni_min_rr))
            failed = [r for r in ranked if r.error is not None]

            _ev_raw, _ev_when = storage.load_value("evidence")
            EVIDENCE = None
            if _ev_raw:
                if _ev_raw.get("config") == _config_signature(USE_TR, TR_PARAMS):
                    EVIDENCE = expectancy.EvidenceBook.from_dict(_ev_raw)
                else:
                    st.warning(
                        "Your backtest evidence was measured with different strategy "
                        "settings (stop, target or strategy have changed since). It no "
                        "longer applies, so it isn't used. Re-run the 🔁 Backtest.")

            f1, f2 = st.columns(2)
            min_score = f1.number_input(
                "Show scores of at least", min_value=0.0, max_value=10.0, value=0.0,
                step=0.5, key="uni_min_score",
                help="Scores are whole numbers, so 8.5 behaves the same as 9. "
                     "A 9 or 10 appears in only a few percent of market states.")
            a_plus_only = f2.checkbox("A+ only", key="uni_aplus_only")
            proven_only = st.checkbox(
                "Only show setups with proven positive expectancy",
                key="uni_proven_only", disabled=EVIDENCE is None,
                help="Hides anything the backtest hasn't shown to make money over many "
                     "trades. Run the backtest first to build the evidence.")
            if EVIDENCE is None:
                st.caption("ℹ️ No evidence yet — run the 🔁 Backtest to label setups with "
                           "whether they've made money historically.")
            else:
                st.caption(f"Evidence from: {EVIDENCE.source}.")

            def _evidence(r):
                if EVIDENCE is None:
                    return None, ""
                return EVIDENCE.evidence_for(r.ticker, r.direction)

            _lm_raw, _ = storage.load_value("learned_model")
            LEARNED = None
            if _lm_raw and _lm_raw.get("config") == _config_signature(USE_TR, TR_PARAMS):
                LEARNED = learning.LearnedModel.from_dict(_lm_raw)
            _has_lessons = LEARNED is not None and bool(LEARNED.rules)
            apply_lessons = st.checkbox(
                "Skip setups the system has learned to avoid",
                key="uni_apply_lessons", disabled=not _has_lessons,
                help="Hides setups matching a lesson confirmed on unseen trades.")
            if _has_lessons:
                st.caption("Lessons in use (from " + LEARNED.source + "): "
                           + "; ".join(f"skip when {r_.description}" for r_ in LEARNED.rules))
            elif LEARNED is not None:
                st.caption("The system has checked its trades and found no lesson worth "
                           "acting on yet — so it isn't filtering anything.")

            def _lesson(r):
                if not _has_lessons:
                    return True, []
                return LEARNED.check(r.features)

            ok = [r for r in scored_all
                  if r.score >= min_score and (not a_plus_only or r.grade == "A+")
                  and r.reward_risk is not None and r.reward_risk >= _rr_floor - 1e-9
                  and (not proven_only
                       or _evidence(r)[0] == expectancy.PROVEN_POSITIVE)
                  and (not apply_lessons or _lesson(r)[0])]
            hidden = len(scored_all) - len(ok)

            st.markdown(f"##### Results · {len(ok)} shown"
                        + (f", {hidden} hidden by filter" if hidden else "")
                        + (f", {len(failed)} failed" if failed else ""))
            st.caption(f"Setups with reward:risk below {_rr_floor:g}:1 are never shown.")
            if EVIDENCE is not None and ok:
                with st.expander("Why each setup has the evidence label it does"):
                    for r in ok:
                        v, why = _evidence(r)
                        st.caption(f"**{r.label} {r.direction or ''}** — {v}. {why}")
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

                _hlinfo = st.session_state.get("hl_info") or {}
                table = pd.DataFrame([{
                    "Instrument": r.label,
                    "Trade plan": explain.short_plan(
                        r.direction, _plan_stage(r, USE_TR), r.entry, r.stop, r.target,
                        r.reward_risk, r.price),
                    "Learned": ("—" if not _has_lessons else
                                ("✅ OK" if _lesson(r)[0] else "⚠️ Avoid")),
                    "Evidence": {expectancy.PROVEN_POSITIVE: "✅ Proven +",
                                 expectancy.PROVEN_NEGATIVE: "❌ Proven −",
                                 expectancy.UNPROVEN: "❔ Unproven",
                                 expectancy.TOO_FEW: "— Too few"}.get(_evidence(r)[0], "—"),
                    "Venue": ("—" if NON_CRYPTO else
                              "HL" if hyperliquid_data.is_hl_ticker(r.ticker)
                              else "/".join(VENUE_TAGS.get(r.label, [])) or "—"),
                    **({"24h volume": f"${_hlinfo[r.ticker].day_volume_usd / 1e6:,.0f}M",
                        "Open interest": f"${_hlinfo[r.ticker].open_interest_usd / 1e6:,.0f}M",
                        "Funding (1h)": f"{_hlinfo[r.ticker].funding_rate * 100:+.4f}%"}
                       if r.ticker in _hlinfo else {}),
                    "Score": f"{r.score:.1f}",
                    "Grade": r.grade,
                    "Dir": r.direction or "—",
                    "Live price": format_price(r.price) + ("" if r.price_is_live else " *"),
                    "Entry": format_price(r.entry),
                    "Entry vs live": _gap(r),
                    "Stop": format_price(r.stop),
                    "Target": format_price(r.target),
                    "R:R": format_rr(r.reward_risk),
                    ("Status" if USE_TR else "Regime"):
                        (r.entry_status or "—") if USE_TR else r.regime.title(),
                } for r in ok])
                _detail = st.toggle("Show every column", value=False, key="uni_detail",
                                    help="Off keeps the essentials only — easier to read on "
                                         "a phone.")
                _essential = ["Instrument", "Dir", "Live price", "Entry", "Entry vs live",
                              "Stop", "Target", "R:R", "Status", "Grade"]
                _view = table if _detail else table[[c for c in _essential
                                                     if c in table.columns]]
                st.dataframe(_view, use_container_width=True, hide_index=True,
                             column_config={"Trade plan": st.column_config.TextColumn(
                                 "Trade plan", width="large")})
                if not _detail:
                    st.caption("Showing the essentials. Turn on **Show every column** for "
                               "venue, volume, funding, evidence, the learned check and the "
                               "one-line plan.")

                _complete = [r for r in ok if None not in (r.entry, r.stop, r.target)]
                if _complete:
                    with st.expander(f"📋 Trade plans ({len(_complete)})",
                                     expanded=len(_complete) <= 5):
                        for r in _complete:
                            _sm, _tm = _methods(USE_TR, TR_PARAMS, r.direction)
                            _reasons = [x.strip() for x in (r.note or "").split(";") if x.strip()]
                            with st.container(border=True):
                                _ok_l, _why_l = _lesson(r)
                                if not _ok_l:
                                    st.warning("⚠️ The system has learned to avoid setups like "
                                               "this: " + "; ".join(_why_l) + ".")
                                for line in explain.full_plan(
                                        r.label, r.direction, _plan_stage(r, USE_TR),
                                        r.entry, r.stop, r.target, r.reward_risk, r.price,
                                        reasons=_reasons if USE_TR else None,
                                        stop_method=_sm, target_method=_tm,
                                        reentry_steps=(trend_retrace.reentry_plan_text(
                                            r.direction) if USE_TR else None)):
                                    st.markdown(line)
                                _sz = _size_plan(r.direction, r.entry, r.stop, r.target,
                                                  r.ticker)
                                _render_size(_sz)
                                if r.ticker in COMMODITY_PERPS:
                                    st.caption(f"Place this on **{COMMODITY_PERPS[r.ticker]}**.")
                                _lv = st.session_state.get("uni_levels", {}).get(r.ticker)
                                if _lv:
                                    _render_grid(f"u_{r.ticker}", r.direction, r.price,
                                                 _lv.get("levels", []), _lv.get("atr"),
                                                 r.reward_risk)
                                _take_trade_widget(
                                    f"u_{r.ticker}_{r.direction}", r.ticker, r.direction,
                                    r.entry, r.stop, r.target, features=r.features,
                                    score=r.score, grade=r.grade,
                                    reason=f"Scanner: {r.entry_status or ''}",
                                    suggested_qty=_sz.quantity if _sz else 0.0)
                st.caption(
                    "**Entry** is the planned price to place your order at — a limit order "
                    "at a confluence zone, not the market price. **Entry vs live** shows how "
                    "far price must travel to fill it. Stop, target and R:R are all measured "
                    "from that entry. A price marked * is the last 4H close because the live "
                    "spot fetch failed.")

                _counts = {g: sum(1 for r in ok if r.grade == g) for g in ("A+", "B", "C")}
                st.markdown(
                    " ".join(theme.pill(f"{g} · {n}", theme.GRADE_COLOURS[g])
                             for g, n in _counts.items() if n),
                    unsafe_allow_html=True)

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

    elif mode == "Auto-scan one instrument" and USE_TR:
        tr1, tr2 = st.columns([3, 1])
        _tl = tr1.selectbox("Coin", list(CRYPTO_TICKERS), key="tr_single")
        tr_ticker = CRYPTO_TICKERS[_tl]
        tr_dir = tr2.selectbox("Direction", ["Auto", "Long", "Short"], key="tr_single_dir",
                               help="Auto takes the direction from the 4H candles.")
        stopped_out = st.checkbox("I was just stopped out on this coin — show re-entry plan",
                                  key="tr_stopped")

        if st.button("🔍 Check the rules", use_container_width=True):
            with st.spinner(f"Checking {tr_ticker}…"):
                frames = {}
                for tf, lim in (("4h", 30), ("1h", 48), ("5m", 400)):
                    df = None
                    if CRYPTO_SOURCE == "bybit":
                        df, _e = exchanges.fetch_bybit_klines(tr_ticker, tf, limit=lim)
                    if df is None:
                        df, _e = exchanges.fetch_binance_klines(tr_ticker, tf, limit=lim)
                    if df is None:
                        df, _e = exchanges.fetch_kraken_ohlc(tr_ticker, tf)
                    if df is None:
                        df, _e = hyperliquid_data.fetch_candles(
                            f"HL:{tr_ticker.upper().replace('-USD', '')}", tf, lim)
                    frames[tf] = trend_retrace.drop_forming(
                        df if df is not None else pd.DataFrame(), tf)
                live, _src, _err = _cached_exchange_spot(tr_ticker)
                st.session_state.tr_plan = trend_retrace.analyze(
                    tr_ticker, frames["4h"], frames["1h"], frames["5m"], live_price=live,
                    params=TR_PARAMS,
                    direction_override=None if tr_dir == "Auto" else tr_dir)
                _p = st.session_state.tr_plan
                _pats = patterns.detect(frames["5m"]) + patterns.detect(frames["1h"])
                st.session_state.tr_patterns = _pats
                if _p.features is not None and _pats:
                    _p.features["pattern"] = patterns.summarise(_pats)
                    _p.features["pattern_bias"] = patterns.bias_of(_pats)
                st.session_state.tr_levels = {
                    "levels": (trend_retrace.five_minute_levels(
                        frames["5m"], _p.direction, _p.current_price or 0, TR_PARAMS)
                        if _p.direction and _p.current_price else []),
                    "atr": trend_retrace.atr(frames["5m"]),
                }

        plan = st.session_state.get("tr_plan")
        if plan is None:
            st.info("Pick a coin and press **Check the rules**.")
        else:
            h1c, h2c, h3c, h4c = st.columns(4)
            h1c.metric("Coin", plan.ticker)
            h2c.metric("Direction", plan.direction or "—")
            h3c.metric("Score", f"{plan.score:.0f}/10", plan.grade)
            h4c.metric("Live price", format_price(plan.current_price))

            st.markdown(f"##### Status: **{plan.stage}**")
            labels = {"4h": "1 · 4H trend — two candles of HH/HL (LH/LL for shorts)",
                      "1h": "2 · 1H confirmation candle",
                      "5m": "3 · 5m retrace to previous support (resistance for shorts)"}
            for key_, text in labels.items():
                rule = plan.rules.get(key_)
                if rule is None:
                    st.markdown(f"⬜ **{text}** — not checked (no direction yet)")
                else:
                    st.markdown(f"{'✅' if rule.passed else '❌'} **{text}**")
                    st.caption(f"　↳ {rule.reason}")

            if plan.entry is not None:
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Entry (limit)", format_price(plan.entry))
                e2.metric("Stop", format_price(plan.stop))
                e3.metric("Target", format_price(plan.target))
                e4.metric("R:R", format_rr(plan.reward_risk))
                if plan.current_price:
                    gap = (plan.entry - plan.current_price) / plan.current_price * 100
                    st.caption(f"Entry is {gap:+.2f}% from the live price — a limit order "
                               f"waiting for the retrace, not a market buy.")
            for p_ in plan.problems:
                st.warning(p_)
            _pats = st.session_state.get("tr_patterns") or []
            if _pats:
                st.markdown("##### Candlestick patterns right now")
                for _pt in _pats:
                    _icon = {"bullish": "🟢", "bearish": "🔴"}.get(_pt.bias, "⚪")
                    st.markdown(f"{_icon} **{_pt.name}** — {_pt.description}")
                    st.caption(f"　{_pt.convention}")
                if plan.direction and patterns.contradicts(_pats, plan.direction):
                    st.warning(f"At least one pattern points against this "
                               f"{plan.direction.lower()}.")
                st.caption("Patterns describe what just happened; they don't predict what "
                           "comes next. They're recorded with the setup so the learner can "
                           "test whether they make any difference to your results.")

            if plan.quality_fails:
                st.warning("**Weak setup — your three rules pass, but:**")
                for q_ in plan.quality_fails:
                    st.caption(f"• {q_}")

            if plan.entry is not None:
                _sm, _tm = _methods(True, TR_PARAMS, plan.direction)
                with st.container(border=True):
                    st.markdown("##### Trade plan")
                    for line in explain.full_plan(
                            plan.ticker, plan.direction, plan.stage, plan.entry, plan.stop,
                            plan.target, plan.reward_risk, plan.current_price,
                            reasons=[r_.reason for r_ in plan.rules.values()],
                            stop_method=_sm, target_method=_tm):
                        st.markdown(line)
                    _sz = _size_plan(plan.direction, plan.entry, plan.stop, plan.target,
                                      plan.ticker)
                    _render_size(_sz)
                    _tr_levels = st.session_state.get("tr_levels") or {}
                    _render_grid(f"s_{plan.ticker}", plan.direction, plan.current_price,
                                 _tr_levels.get("levels", []), _tr_levels.get("atr"),
                                 plan.reward_risk)
                    _take_trade_widget(
                        f"s_{plan.ticker}_{plan.direction}", plan.ticker, plan.direction,
                        plan.entry, plan.stop, plan.target, features=plan.features,
                        score=plan.score, grade=plan.grade,
                        reason=f"Single-coin check: {plan.stage}",
                        suggested_qty=_sz.quantity if _sz else 0.0)

            if plan.stage == trend_retrace.AT_ENTRY:
                st.success("All three rules are met and price is at the entry level.")
            elif plan.stage == trend_retrace.WAIT_RETRACE:
                st.info("All three rules are met. Place the limit order at the entry and let "
                        "the retrace come to you.")

            if stopped_out and plan.direction:
                st.markdown("##### Re-entry plan (after being stopped out)")
                for i_, step_ in enumerate(trend_retrace.reentry_plan_text(plan.direction), 1):
                    st.markdown(f"{i_}. {step_}")
                st.caption("Each re-entry half should risk half of your normal amount, so the "
                           "full re-entry risks the same as one normal trade.")

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
            min_rr = st.number_input("Min R:R", min_value=3.0, value=3.0, step=0.5, key="scan_rr")

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

    # The confluence score/readiness panel only applies to the original
    # strategy's single-coin and manual modes. Under Trend Retrace, or after a
    # universe scan, it would show a meaningless score built from no evidence.
    if not USE_TR and mode != "Rank the universe":
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

    pos_equity, pos_max_risk = ACCOUNT_EQUITY, RISK_PCT
    st.caption(f"Using your sidebar settings: **${pos_equity:,.0f}** equity, "
               f"**{pos_max_risk:g}%** risk per trade.")

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
                    if hyperliquid_data.is_hl_ticker(r["asset"]):
                        _cx, err = hyperliquid_data.fetch_market_contexts()
                        _m = next((c for c in _cx if c.ticker == r["asset"]), None)
                        px, src = (_m.mark_price, "Hyperliquid") if _m else (None, None)
                        err = err or ("not listed on Hyperliquid" if _m is None else None)
                    else:
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
    with st.expander("📊 What does long-run profitable actually look like?", expanded=False):
        st.caption(
            "Pick a win rate and see what trading it would be like over many trades. "
            "Profitable doesn't mean few losses — at 3:1 you can lose most trades and "
            "still make money, but you must be able to sit through the losing runs.")
        w1, w2 = st.columns(2)
        _wr = w1.slider("Win rate", min_value=5, max_value=70, value=30, step=1,
                        format="%d%%", key="calc_wr") / 100
        _crr = w2.number_input("Reward:risk", min_value=3.0, max_value=10.0, value=3.0,
                               step=0.5, key="calc_rr")
        w3, w4 = st.columns(2)
        _crisk = w3.number_input("Risk per trade (%)", min_value=0.1, max_value=10.0,
                                 value=1.0, step=0.1, key="calc_risk")
        _cn = w4.selectbox("Number of trades", [50, 100, 200], index=1, key="calc_n")
        _ce = expectancy.simulate_expectations(_wr, _crr, risk_pct=_crisk, n_trades=_cn)
        _be = expectancy.break_even_win_rate(_crr)
        k1, k2, k3 = st.columns(3)
        k1.metric(f"Expected over {_cn} trades", f"{_ce.expected_total_pct:+.0f}%",
                  f"{_ce.pct_per_trade:+.2f}% per trade")
        k2.metric("Longest losing streak", f"{_ce.streak_typical} typical",
                  f"up to {_ce.streak_bad}", delta_color="off")
        k3.metric("Deepest drop", f"{_ce.drawdown_typical_pct:.0f}% typical",
                  f"up to {_ce.drawdown_bad_pct:.0f}%", delta_color="off")
        if _wr < _be:
            st.error(f"Below the {_be:.0%} break-even rate at {_crr:g}:1 — this loses money "
                     f"over time however it's traded.")
        else:
            st.caption(f"Break-even at {_crr:g}:1 is {_be:.0%}. Chance of still being down "
                       f"after {_cn} trades, purely from luck: **{_ce.chance_of_loss_pct:.0f}%**.")
        st.markdown("**Want to win more often?**")
        _target_wins = st.slider("Wins out of 10 you want", 1, 9, 3, key="calc_wins")
        _p = _target_wins / 10
        _need_be = expectancy.required_rr(_p)
        _need_edge = expectancy.required_rr(_p, edge_per_trade=0.20)
        if _need_be:
            st.caption(
                f"To win **{_target_wins} in 10** you need a reward:risk of about "
                f"**{_need_be:.2f}:1** just to break even, and **{_need_edge:.2f}:1** to make "
                f"+0.20R per trade. Winning more often always means winning less each time — "
                f"that trade-off can't be avoided, only chosen.")
            _hi = expectancy.simulate_expectations(_p, max(_need_edge, 0.05),
                                                    risk_pct=_crisk, n_trades=_cn)
            st.caption(f"That style: {_hi.expected_total_pct:+.0f}% over {_cn} trades, worst "
                       f"losing streak about {_hi.streak_bad}, worst drop "
                       f"{_hi.drawdown_bad_pct:.0f}%. Smoother than a big-target style — but "
                       f"each loss undoes several wins, so discipline on the stop matters "
                       f"more, not less.")
        if _crisk > 2:
            st.warning(f"At {_crisk:g}% per trade, a bad losing run could take "
                       f"{_ce.drawdown_bad_pct:.0f}% off your account. Most traders keep risk "
                       f"at 1–2% so a normal streak is survivable.")

    st.subheader("Position sizing & risk")
    pre = st.session_state.get("prefill", {})
    if pre:
        st.success(f"Using levels sent from the scanner for {pre.get('ticker','')}.")

    st.caption("Defaults come from your sidebar settings — change them here just to try "
               "'what if' variations.")
    k1, k2, k3 = st.columns(3)
    equity = k1.number_input("Account equity ($)", min_value=0.0,
                             value=float(ACCOUNT_EQUITY), step=100.0)
    risk_pct = k2.number_input("Risk % per trade", min_value=0.1, max_value=100.0,
                                value=float(RISK_PCT), step=0.1)
    direction = k3.selectbox("Direction", ["Long", "Short"],
                              index=0 if pre.get("direction", "Long") == "Long" else 1)

    stop_style = st.radio("Set the stop as", ["A price", "A % from entry"],
                          horizontal=True, key="risk_stop_style")
    k4, k5, k6 = st.columns(3)
    entry = k4.number_input("Entry", min_value=0.0, value=float(pre.get("entry", 100.0)),
                            step=0.01, format="%.8f")
    if stop_style == "A price":
        stop = k5.number_input("Stop", min_value=0.0, value=float(pre.get("stop", 95.0)),
                               step=0.01, format="%.8f")
        stop_pct_display = (abs(entry - stop) / entry * 100) if entry else 0.0
    else:
        stop_pct = k5.number_input("Stop distance (%)", min_value=0.01, max_value=99.0,
                                   value=1.5, step=0.1, key="risk_stop_pct",
                                   help="How far against you price can go before you're wrong.")
        try:
            stop = calc_stop_from_pct(entry, stop_pct, direction)
        except InvalidRiskInputError:
            stop = 0.0
        stop_pct_display = stop_pct
        k5.caption(f"= {format_price(stop)}")
    target = k6.number_input("Target", min_value=0.0, value=float(pre.get("target", 115.0)),
                             step=0.01, format="%.8f")

    k7, k8, k9 = st.columns(3)
    leverage = k7.number_input("Leverage", min_value=1.0, value=float(LEVERAGE), step=0.5)
    fee = k8.number_input("Fee rate", min_value=0.0, value=float(FEE_RATE), step=0.0001,
                          format="%.4f")
    slip = k9.number_input("Slippage", min_value=0.0, value=float(SLIPPAGE), step=0.0001,
                           format="%.4f")

    mmr = st.number_input("Maintenance margin (%)", min_value=0.0, max_value=10.0, value=0.5,
                          step=0.1, key="risk_mmr",
                          help="Used to estimate the liquidation price. Exchanges raise this "
                               "for larger positions — check yours for the real figure.")
    spec_ticker = st.text_input("Contract-spec ticker (optional, e.g. GC=F)",
                                 value=pre.get("ticker", ""))

    if entry > 0 and leverage > 1:
        _liq = liquidation_price(entry, leverage, direction, mmr)
        if _liq:
            _gap = abs(entry - _liq) / entry * 100
            st.caption(f"Estimated liquidation around **{format_price(_liq)}** "
                       f"({_gap:.2f}% from entry) at {leverage:g}x — an estimate, not the "
                       f"exchange's figure.")
            if stop > 0 and stop_is_beyond_liquidation(entry, stop, leverage, direction, mmr):
                st.error("⚠️ Your stop sits beyond the estimated liquidation price — you'd be "
                         "liquidated before the stop protects you. Reduce leverage or tighten "
                         "the stop.")

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
                g.metric("Net R:R", format_rr(res.net_reward_risk))

            st.caption(f"Stop is {res.stop_distance_pct * 100:.2f}% from entry "
                       f"({format_price(res.stop_distance_price)} per coin). Position "
                       f"{res.quantity:g} coins = {format_price(res.position_notional)} "
                       f"notional on {format_price(res.margin_required)} margin.")
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

    # --- trades taken from the scanner, waiting to be completed ----------
    _open = trade_log.open_scanner_trades()
    st.markdown(f"##### Trades to complete ({len(_open)})")
    if _open.empty:
        st.caption("Trades you mark **I took this trade** on the 🎯 Scanner appear here. "
                   "When one closes, complete it so the system can learn from it.")
    for _, jr in _open.iterrows():
        jid = int(jr["id"])
        with st.container(border=True):
            st.markdown(f"**#{jid} · {jr['direction']} {jr['asset']}** — entry "
                        f"{format_price(jr['entry'])}, stop {format_price(jr['sl'])}, "
                        f"target {format_price(jr['tp'])}")
            outcome = st.radio("How did it end?", trade_log.OUTCOMES, horizontal=True,
                               key=f"cm_out_{jid}")
            _default = trade_log.default_exit_price(outcome, float(jr["sl"]), float(jr["tp"]),
                                                    manual=float(jr["entry"]))
            cc1, cc2 = st.columns(2)
            exit_px = cc1.number_input("Exit price achieved", value=float(_default),
                                       format="%.8f", key=f"cm_exit_{jid}_{outcome}")
            fill_px = cc2.number_input("Actual entry (correct if needed)",
                                       value=float(jr["entry"]), format="%.8f",
                                       key=f"cm_fill_{jid}")
            cc3, cc4 = st.columns(2)
            fin_sl = cc3.number_input("Final stop loss (if you moved it)",
                                      value=float(jr["sl"]), format="%.8f",
                                      key=f"cm_sl_{jid}")
            fin_tp = cc4.number_input("Final take profit (if you moved it)",
                                      value=float(jr["tp"]), format="%.8f",
                                      key=f"cm_tp_{jid}")
            fees = st.number_input("Fees paid ($, optional)", min_value=0.0, value=0.0,
                                   key=f"cm_fee_{jid}")
            _init = jr.get("initial_stop")
            _init = float(jr["sl"]) if _init is None or pd.isna(_init) else float(_init)
            _r = trade_log.realized_r(jr["direction"], fill_px, _init, exit_px)
            if _r is not None:
                st.caption(f"Result: **{_r:+.2f}R** — {trade_log.pl_status_for(_r).lower()}, "
                           f"measured against your original stop of {format_price(_init)}.")
            if st.button("🔍 Review this trade", key=f"rv_go_{jid}", use_container_width=True):
                with st.spinner("Reviewing…"):
                    _et = pd.Timestamp(jr["timestamp_utc"])
                    _et = _et.tz_localize("UTC") if _et.tzinfo is None else _et
                    _fr, _since, _px, _now = _review_inputs(jr["asset"], _et)
                    if _px is None:
                        st.error("Couldn't fetch a current price for this coin.")
                    else:
                        st.session_state[f"rv_{jid}"] = (trade_review.review(
                            direction=jr["direction"], entry=float(jr["entry"]),
                            stop=float(jr["sl"]), target=float(jr["tp"]), entry_time=_et,
                            price=float(_px), now=_now, df_4h=_fr["4h"], df_1h=_fr["1h"],
                            candles_since_entry=_since,
                            structure_candles=_fr["5m"].tail(200),
                            atr_value=trend_retrace.atr(_fr["5m"]),
                            trend_candles=(TR_PARAMS.trend_candles if TR_PARAMS else 2),
                            stale_hours=st.session_state.get("trb_stale", 12.0)), float(_px))

            _rv = st.session_state.get(f"rv_{jid}")
            if _rv is not None:
                rv, rv_px = _rv
                box = {trade_review.HOLD: st.success, trade_review.TIGHTEN: st.info,
                       trade_review.CLOSE: st.warning,
                       trade_review.CLOSE_THESIS: st.error}[rv.verdict]
                box(f"**Review: {rv.verdict}**")
                for why in rv.reasons:
                    st.caption(f"• {why}")
                for act in rv.actions:
                    st.markdown(act)
                if rv.current_r is not None:
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("Close now", f"{rv.close_now_r:+.2f}R")
                    rc2.metric("If stop hits", f"{rv.stop_r:+.2f}R")
                    rc3.metric("If target hits", f"{rv.target_r:+.2f}R")
                if rv.suggested_stop is not None:
                    widening = (rv.suggested_stop < float(jr["sl"]) if jr["direction"] == "Long"
                                else rv.suggested_stop > float(jr["sl"]))
                    if not widening and st.button(
                            f"Move stop to {format_price(rv.suggested_stop)}",
                            key=f"rv_sl_{jid}"):
                        storage.update_journal_entry(jid, sl=float(rv.suggested_stop),
                                                     updated_sl=float(rv.suggested_stop))
                        st.success("Stop updated in the journal — remember to move it on "
                                   "Hyperliquid too. Results still count against your "
                                   "original stop.")
                        st.session_state.pop(f"rv_{jid}", None)
                        st.rerun()
                if rv.verdict in (trade_review.CLOSE, trade_review.CLOSE_THESIS):
                    if st.button(f"Close now at {format_price(rv_px)} ({rv.close_now_r:+.2f}R)",
                                 key=f"rv_close_{jid}"):
                        trade_log.complete_trade(jid, exit_price=rv_px,
                                                 exit_reason=f"Closed early after review: "
                                                             f"{rv.verdict}")
                        st.success("Completed in the journal — remember to close it on "
                                   "Hyperliquid too.")
                        st.session_state.pop(f"rv_{jid}", None)
                        st.rerun()
                st.caption("The review is a second opinion based on your rules, not an "
                           "instruction. Nothing is changed on the exchange.")

            if st.button("✅ Complete trade", key=f"cm_go_{jid}", use_container_width=True):
                trade_log.complete_trade(jid, exit_price=exit_px, actual_entry=fill_px,
                                         final_stop=fin_sl, final_target=fin_tp, fees=fees,
                                         exit_reason=outcome)
                st.success(f"#{jid} completed at {_r:+.2f}R.")
                st.rerun()

    # --- learning from the trades you've actually taken -------------------
    st.markdown("##### 🧠 Learn from your trades")
    _include_disc = st.checkbox(
        "Include trades where I didn't follow the rules", key="lt_disc",
        help="Off by default: those trades test your judgement, not the strategy.")
    _mine = trade_log.learning_trades(config=_config_signature(USE_TR, TR_PARAMS),
                                      followed_only=not _include_disc)
    if _mine:
        _wins = sum(t.status == "WIN" for t in _mine)
        _avg = sum(t.r_result for t in _mine) / len(_mine)
        m1, m2, m3 = st.columns(3)
        m1.metric("Completed trades", len(_mine))
        m2.metric("Win rate", f"{_wins / len(_mine):.0%}")
        m3.metric("Average result", f"{_avg:+.2f}R")
    st.caption(
        f"{len(_mine)} completed trade(s) under your current settings; about "
        f"{learning.MIN_TRADES} are needed before lessons are trustworthy. Your real trades "
        f"are the most valuable evidence there is — they include your actual fills and exits.")
    if st.button("🧠 Learn from my trades", use_container_width=True,
                 disabled=len(_mine) < learning.MIN_TRADES, key="lt_go"):
        _m = learning.learn(_mine, source=f"your real trades, {len(_mine)} trades")
        st.session_state.mine_model = _m
        _md = _m.to_dict()
        _md["config"] = _config_signature(USE_TR, TR_PARAMS)
        storage.save_value("learned_model", _md)
    if st.session_state.get("mine_model") is not None:
        _render_learned(st.session_state.mine_model)

    st.markdown("---")
    st.markdown("##### Add a trade manually")

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

    _jup = st.file_uploader("Restore journal from an exported CSV", type=["csv"],
                            key="journal_upload")
    if _jup is not None and st.button("Import journal", key="journal_import"):
        try:
            _n = storage.import_journal_csv(_jup.getvalue())
            st.success(f"Restored {_n} journal entr{'y' if _n == 1 else 'ies'}.")
            st.rerun()
        except ValueError as e:
            st.error(str(e))

    df = storage.get_journal_df()
    if df.empty:
        st.info("No journal entries yet.")
    else:
        _hide = ["features", "strategy_config", "margin", "current_pl",
                 "potential_profit_at_tp", "potential_loss_at_sl", "management_notes"]
        _view = df.drop(columns=[c for c in _hide if c in df.columns])
        _first = [c for c in ("id", "status", "asset", "direction", "pl_status",
                              "realized_r", "entry", "sl", "tp", "exit_price",
                              "realized_pnl") if c in _view.columns]
        _view = _view[_first + [c for c in _view.columns if c not in _first]]
        st.dataframe(_view, use_container_width=True, hide_index=True,
                     column_config={"realized_r": st.column_config.NumberColumn(
                         "Result (R)", format="%+.2f")})
        st.caption("The full export below includes every column, including the setup "
                   "snapshots used for learning.")
        st.warning("**Export your journal before every app update.** Streamlit wipes the "
                   "database on redeploy — your trades, and everything the system has "
                   "learned from them, would otherwise be lost. Restore with the import "
                   "box below.")
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
            if hyperliquid_data.is_hl_ticker(ticker):
                df, _err = hyperliquid_data.fetch_candles(ticker, "1h", 1000)
                return df
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

    st.markdown("##### 🔁 Re-check saved setups")
    st.caption("A plan made yesterday describes a chart that no longer exists. This "
               "re-checks each waiting setup against the current 4H trend and price.")
    _pending = sig_df[sig_df["status"] == "PENDING"] if not sig_df.empty else sig_df
    if _pending.empty:
        st.caption("No setups waiting to fill.")
    elif st.button(f"Re-check {len(_pending)} waiting setup(s)", use_container_width=True):
        _out = []
        bar = st.progress(0.0, text="Checking…")
        for i_, (_, sg) in enumerate(_pending.iterrows()):
            bar.progress(i_ / max(len(_pending), 1), text=sg["ticker"])
            _created = pd.Timestamp(sg["created_utc"])
            _created = _created.tz_localize("UTC") if _created.tzinfo is None else _created
            try:
                _fr, _since, _px, _now = _review_inputs(sg["ticker"], _created)
                if _px is None:
                    _out.append((sg, None, "No current price available."))
                    continue
                _v = trade_review.setup_still_viable(
                    sg["direction"], float(sg["entry"]), float(sg["stop"]), _created,
                    float(_px), _now, _fr["4h"],
                    trend_candles=(TR_PARAMS.trend_candles if TR_PARAMS else 2))
                _out.append((sg, _v, None))
            except Exception as e:
                _out.append((sg, None, f"{type(e).__name__}: {e}"))
        bar.empty()
        st.session_state.viability = _out

    for sg, _v, _err in st.session_state.get("viability", []):
        with st.container(border=True):
            st.markdown(f"**{sg['direction']} {sg['label'] or sg['ticker']}** — entry "
                        f"{format_price(sg['entry'])}, stop {format_price(sg['stop'])}")
            if _err or _v is None:
                st.caption(f"Couldn't check: {_err}")
                continue
            box = {trade_review.STILL_VALID: st.success}.get(_v.verdict, st.warning)
            box(f"**{_v.verdict}** · now {format_price(_v.price)}")
            for why in _v.reasons:
                st.caption(f"• {why}")
            if _v.verdict != trade_review.STILL_VALID:
                if st.button("Cancel this setup", key=f"vcancel_{int(sg['id'])}"):
                    storage.update_signal(int(sg["id"]), status="EXPIRED",
                                          exit_utc=datetime.now(timezone.utc).isoformat())
                    st.session_state.viability = [
                        x for x in st.session_state.viability if x[0]["id"] != sg["id"]]
                    st.rerun()

    st.markdown("##### 🧠 Learn from the app's own trades")
    _tt = tracking.tracked_trades()
    _usable = [t for t in _tt if t.status in ("WIN", "LOSS") and t.features]
    st.caption(
        f"{len(_usable)} resolved trade(s) with a setup snapshot so far; about "
        f"{learning.MIN_TRADES} are needed. These are the strongest evidence available — each "
        f"plan was locked in before its result was known.")
    if st.button("🧠 Learn from tracked trades", use_container_width=True,
                 disabled=len(_usable) < learning.MIN_TRADES):
        _m = learning.learn(_tt, source=f"live tracked trades, {len(_usable)} trades")
        st.session_state.track_model = _m
        _md = _m.to_dict()
        _md["config"] = _config_signature(USE_TR, TR_PARAMS)
        storage.save_value("learned_model", _md)
    if st.session_state.get("track_model") is not None:
        _render_learned(st.session_state.track_model)
    if len(_usable) < learning.MIN_TRADES:
        st.caption("Keep scanning with tracking switched on; this unlocks once enough of the "
                   "app's own trades have resolved.")

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
  if USE_TR:
    st.subheader("Backtest — Trend Retrace (your strategy)")
    st.caption(
        "Replays your rules candle by candle on 5-minute data: two 4H HH/HL candles, a "
        "bullish 1H candle, then a limit order at 5m support. After a stop-out it follows "
        "your re-entry rules — wait for a 4H candle in your direction, re-enter 50% on a "
        "5m retrace, add 50% after a confirming 1H candle. Only closed candles are used."
    )
    st.warning(
        "No strategy guarantees profit. This shows how your rules would have performed on "
        "past data — useful evidence, not a promise. Ambiguous candles are resolved "
        "pessimistically, so results lean worse rather than better."
    )
    bc1, bc2, bc3 = st.columns(3)
    trb_size = bc1.selectbox("Coins", [5, 10, 20, 30, 50], index=1, key="trb_size",
                             format_func=lambda n: f"Top {n}")
    trb_days = bc2.selectbox("History", ["14 days", "30 days", "60 days"], index=1,
                             key="trb_days",
                             help="Longer history = more trades = more trustworthy, but slower.")
    trb_fee = bc3.number_input("Costs per trade (R)", min_value=0.0, value=0.05, step=0.01,
                               format="%.2f", key="trb_fee")
    trb_stale = st.number_input(
        "Treat a trade as stale after (hours)", min_value=2.0, max_value=72.0, value=12.0,
        step=1.0, key="trb_stale",
        help="Used to test the trade review's 'close stale trades' rule, and by the "
             "review itself on the Journal tab.")
    days = int(trb_days.split()[0])
    st.caption(f"Fetches ~{days * 288:,} five-minute candles per coin. "
               f"Allow roughly {max(1, trb_size * days // 60)} minute(s).")

    if st.button("▶ Run backtest", use_container_width=True, key="trb_run"):
        labels = list(CRYPTO_TICKERS)[:trb_size]
        results_on, results_off, results_rev, failures = [], [], [], []
        sources = set()
        bar = st.progress(0.0, text="Starting…")
        for i_, lbl in enumerate(labels):
            tk = CRYPTO_TICKERS[lbl]
            bar.progress(i_ / len(labels), text=f"{i_ + 1}/{len(labels)} · {lbl}")
            m5, h1, h4, _src, _note = _backtest_frames(tk, days)
            if m5 is None:
                failures.append(f"{lbl}: {_note}")
                continue
            if _note:
                sources.add(_note)
            else:
                sources.add(_src)
            results_on += trend_retrace.run_backtest(m5, h1, h4, tk, params=TR_PARAMS,
                                                      fee_r=trb_fee, enable_reentry=True)
            results_off += trend_retrace.run_backtest(m5, h1, h4, tk, params=TR_PARAMS,
                                                       fee_r=trb_fee, enable_reentry=False)
            results_rev += trend_retrace.run_backtest(m5, h1, h4, tk, params=TR_PARAMS,
                                                       fee_r=trb_fee, enable_reentry=True,
                                                       exit_on_reversal=True,
                                                       exit_stale_hours=trb_stale)
        bar.empty()
        st.session_state.trb = (results_on, results_off, failures)
        st.session_state.trb_sources = sorted(sources)
        st.session_state.trb_review = results_rev
        if results_on:
            _ev = expectancy.EvidenceBook.from_trades(
                results_on, source=f"Trend Retrace backtest, {trb_size} coins, "
                                   f"{days} days").to_dict()
            _ev["config"] = _config_signature(True, TR_PARAMS)
            storage.save_value("evidence", _ev)
            # Learn once per backtest run (not on every page refresh, which
            # would overwrite lessons learned from live tracked trades).
            _initial = [t for t in results_on if t.kind == trend_retrace.KIND_INITIAL]
            _model = learning.learn(_initial, source=f"backtest, {len(_initial)} trades")
            st.session_state.trb_model = _model
            _md = _model.to_dict()
            _md["config"] = _config_signature(True, TR_PARAMS)
            storage.save_value("learned_model", _md)

    trb = st.session_state.get("trb")
    if trb is not None:
        res_on, res_off, fails = trb
        if st.session_state.get("trb_sources"):
            st.caption("Candles from: " + ", ".join(st.session_state.trb_sources)
                       + ". Prices on majors are near-identical across venues, so a "
                         "backtest on one is a fair guide for trading another.")
        if not res_on:
            st.info("No trades were produced over this history.")
        else:
            stats = trend_retrace.backtest_stats(res_on)
            st.markdown("##### Results")
            st.dataframe(pd.DataFrame([{
                "Trade type": x.group, "Entries": x.entries, "Wins": x.wins,
                "Losses": x.losses, "Expired orders": x.expired,
                "Win rate": f"{x.win_rate:.0%}" if x.win_rate is not None else "—",
                "Avg R per entry": f"{x.avg_r_per_entry:+.2f}"
                                   if x.avg_r_per_entry is not None else "—",
                "Total R": f"{x.total_r:+.1f}",
                "Luck range": f"±{x.luck_range:.1f}" if x.luck_range is not None else "—",
                "Verdict": x.verdict,
            } for x in stats]), use_container_width=True, hide_index=True)

            on_total = next(x.total_r for x in stats if x.group == "All")
            off_total = next(x.total_r for x in trend_retrace.backtest_stats(res_off)
                             if x.group == "All")
            st.markdown("##### Do the re-entry rules help?")
            cA, cB = st.columns(2)
            cA.metric("With re-entry", f"{on_total:+.1f}R")
            cB.metric("Without re-entry", f"{off_total:+.1f}R",
                      f"{on_total - off_total:+.1f}R difference")
            resolved = next(x.entries for x in stats if x.group == "All")
            if resolved < 30:
                st.caption(f"Only {resolved} resolved entries — too few to trust. Add coins "
                           f"or lengthen the history before drawing conclusions.")
            _luck = next(x.luck_range for x in stats if x.group == "All")
            if _luck is not None:
                st.caption(
                    f"A difference smaller than about **±{_luck:.0f}R** between the two could "
                    f"easily be chance, so don't read much into it unless it's larger.")
            st.caption(
                "**Total R** is the result in units of your normal risk: +10R means you'd "
                "have made ten times the amount you risk per trade. Re-entry halves count "
                "at 50%. **Luck range** is how far chance alone commonly pushes the total "
                "over that many trades — only a total outside it says much. **Baseline:** on "
                "random price data, where no edge exists, these rules come out roughly "
                "break-even before costs (about 33% wins at 2R) and lose roughly the cost per "
                "trade after costs. So an edge has to show up as clearly positive, after "
                "costs, across plenty of trades.")
            _rev = st.session_state.get("trb_review")
            if _rev:
                _rs = next(x for x in trend_retrace.backtest_stats(_rev) if x.group == "All")
                st.markdown("##### Does reviewing trades help?")
                st.caption(
                    f"Replays the same history applying the trade review's two early exits: "
                    f"close when the 4H trend reverses, and close a trade that has gone "
                    f"nowhere for {st.session_state.get('trb_stale', 12):g}h while the 1H "
                    f"turns against it.")
                rA, rB = st.columns(2)
                rA.metric("Following the plan", f"{on_total:+.1f}R")
                rB.metric("With review exits", f"{_rs.total_r:+.1f}R",
                          f"{_rs.total_r - on_total:+.1f}R")
                _early = [t for t in _rev if t.exit_reason in ("reversal", "stale")]
                _diff = _rs.total_r - on_total
                if _luck is not None and abs(_diff) < _luck:
                    st.info(f"The difference is inside the ±{_luck:.0f}R luck range — no "
                            f"evidence either way yet. Following the plan is the safer default.")
                elif _diff > 0:
                    st.success("Closing stale and reversed trades improved results by more "
                               "than luck explains — the review's 'close' advice is worth "
                               "taking seriously for this strategy.")
                else:
                    st.warning("Closing early made results worse by more than luck explains — "
                               "sideways trades often recover in this strategy. Lean towards "
                               "holding to the stop.")
                st.caption(f"{len(_early)} trade(s) were closed early by the review rules.")

            _all = next(x for x in stats if x.group == "All")
            if _all.entries >= 10 and _all.win_rate is not None:
                st.markdown("##### What trading this would feel like")
                _rr = TR_PARAMS.target_r if TR_PARAMS else 3.0
                _risk = st.number_input("Risk per trade (% of account)", min_value=0.1,
                                        max_value=10.0, value=1.0, step=0.1, key="exp_risk")
                ex = expectancy.simulate_expectations(_all.win_rate, _rr, risk_pct=_risk,
                                                       n_trades=100)
                x1, x2, x3 = st.columns(3)
                x1.metric("Expected over 100 trades", f"{ex.expected_total_pct:+.0f}%",
                          f"{ex.pct_per_trade:+.2f}% per trade")
                x2.metric("Longest losing streak", f"{ex.streak_typical} typical",
                          f"up to {ex.streak_bad} in a bad run", delta_color="off")
                x3.metric("Deepest drop from a peak", f"{ex.drawdown_typical_pct:.0f}% typical",
                          f"up to {ex.drawdown_bad_pct:.0f}% in a bad run", delta_color="off")
                be = expectancy.break_even_win_rate(_rr)
                st.caption(
                    f"Based on the backtest's **{_all.win_rate:.0%} win rate** at {_rr:g}:1 "
                    f"(break-even is {be:.0%}). Even if that edge is real, there's a "
                    f"**{ex.chance_of_loss_pct:.0f}% chance of being down after 100 trades** "
                    f"purely from luck. A losing streak of the length above is normal — it is "
                    f"not a sign the strategy has stopped working, and it's the moment people "
                    f"most often abandon a good system or raise their risk to win it back.")
                if _all.verdict != "Clearly positive":
                    st.warning(
                        "These projections assume the backtest's win rate is the true one. "
                        "Your result is not yet distinguishable from luck, so treat them as "
                        "illustrative until the evidence is stronger.")

            st.markdown("##### 🧠 What the system learned")
            st.caption(
                "Looks for kinds of setup that lost — by trend strength, retrace depth, stop "
                "width, 1H candle strength, volatility, session and direction — using the "
                "earlier 70% of trades, then keeps only lessons that also held on the later "
                "30% it never saw. Uses initial entries, so each trade stands on its own.")
            _model = st.session_state.get("trb_model")
            if _model is not None:
                _render_learned(_model)
            if _model is not None and _model.rules:
                st.caption("These lessons are now available as a filter on the 🎯 Scanner.")

            with st.expander(f"All {len(res_on)} trade records"):
                st.dataframe(pd.DataFrame([{
                    "Coin": t.ticker, "Type": t.kind, "Dir": t.direction,
                    "Signal": t.signal_time, "Entry": format_price(t.entry),
                    "Stop": format_price(t.stop), "Target": format_price(t.target),
                    "Outcome": t.status,
                    "R": f"{t.weighted_r:+.2f}" if t.weighted_r is not None else "—",
                } for t in res_on]), use_container_width=True, hide_index=True)
        for f_ in fails:
            st.caption(f"⚠️ Could not load {f_}")
        if fails and not res_on:
            st.error(
                "No source could supply history for these coins. Bybit and Binance block "
                "US-hosted servers, and Hyperliquid only lists a few hundred perps — so a "
                "coin it doesn't list can't be backtested here. Running the app on your own "
                "machine in Australia would reach Bybit and Binance directly.")

  else:
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
      bt_rr = b2.number_input("Min R:R", min_value=3.0, value=3.0, step=0.5, key="bt_rr")
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
          if all_trades:
              _ev = expectancy.EvidenceBook.from_trades(
                  all_trades, source=f"Confluence backtest, {bt_size} coins").to_dict()
              _ev["config"] = _config_signature(False, None)
              storage.save_value("evidence", _ev)
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
                  "30% win rate at 3:1 is profitable, a 60% win rate at 0.5:1 is not.")
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
