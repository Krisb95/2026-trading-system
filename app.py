"""
Bull Run Strategy V2 — integrated trading analysis dashboard.

This is the UI layer that ties together data_layer, scoring, readiness,
risk_calc, technical, backtest, portfolio, journal, and stops. Stages 1-5
(data health, score/readiness, risk calculator, journal, stop management)
are fully wired to live data. Stages 6-8 (multi-timeframe structure,
backtesting, portfolio) are here as working v1 tools per the honest scope
notes in each module's docstring — treat them as a first pass, not a
finished discretionary-trading replacement.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
import zoneinfo

from data_layer import fetch_quote, DataStatus, to_local_display, curl_cffi_is_available
from scoring import score_setup, weakest_components, trade_policy_for_grade, POSITIVE_COMPONENTS, NEGATIVE_COMPONENTS
from readiness import (determine_readiness, ReadinessInputs, Readiness, display_label,
                        ENTRY_SEQUENCE_STAGES, missing_entry_sequence_stages, READINESS_DESCRIPTIONS)
from risk_calc import calculate_risk, InvalidRiskInputError, get_instrument_spec
from technical import find_swing_points, label_structure, fib_levels
from backtest import (run_backtest, compute_metrics, split_in_out_sample,
                       example_sma_crossover_signals)
from portfolio import OpenPosition, summarize_portfolio_risk
from scanner import fetch_multi_timeframe, analyze_candidate
from journal import TradeJournal, JournalEntry
from stops import StopManager, StopManagementError, ManagementLabel, suggest_management_label

import plotly.graph_objects as go

st.set_page_config(page_title="Bull Run Strategy V2", page_icon="📈", layout="wide")

APP_BUILD = "2026-09-19-b4 (intraday freshness + NaN guard + auto-scanner)"

# ---------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------
if "journal" not in st.session_state:
    st.session_state.journal = TradeJournal()
if "open_positions" not in st.session_state:
    st.session_state.open_positions = []  # list of OpenPosition
if "stop_managers" not in st.session_state:
    st.session_state.stop_managers = {}  # keyed by a label the user picks

st.title("📈 Bull Run Strategy V2")
st.caption(f"Build: `{APP_BUILD}`")
st.caption(
    "Discretionary trading support tool. Protect capital first — a no-trade "
    "decision is valid. Nothing here executes trades or connects to an "
    "exchange; you enter everything manually. Setup scores are a checklist "
    "quality measure, not a win-probability or profitability guarantee."
)

CRYPTO_TICKERS = {
    "Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD", "Solana (SOL)": "SOL-USD",
    "Sui (SUI)": "SUI-USD", "XRP": "XRP-USD", "Cardano (ADA)": "ADA-USD",
    "Avalanche (AVAX)": "AVAX-USD", "Near (NEAR)": "NEAR-USD", "Dogecoin (DOGE)": "DOGE-USD",
}
COMMODITY_TICKERS = {
    "Gold (Futures)": "GC=F", "Silver (Futures)": "SI=F", "WTI Crude Oil (Futures)": "CL=F",
    "Brent Crude Oil (Futures)": "BZ=F", "Natural Gas (Futures)": "NG=F",
}

tab_market, tab_candidate, tab_positions, tab_risk, tab_journal, tab_backtest, tab_settings = st.tabs(
    ["🌍 Market & Data Health", "🎯 Candidate Scanner", "📋 Open Positions", "🧮 Risk Calculator",
     "📓 Trade Journal", "🔁 Backtesting", "⚙️ Settings"]
)

# ---------------------------------------------------------------------
# Settings (freshness thresholds, timezone) — referenced by other tabs
# ---------------------------------------------------------------------
with tab_settings:
    st.subheader("Data freshness thresholds")
    live_threshold = st.number_input("LIVE threshold (seconds)", value=60, min_value=5)
    stale_threshold = st.number_input("STALE threshold (seconds)", value=900, min_value=60)
    st.subheader("Timezone for displayed timestamps")
    tz_name = st.text_input("IANA timezone (e.g. Australia/Sydney, America/New_York)", value="Australia/Sydney")
    try:
        local_tz = zoneinfo.ZoneInfo(tz_name)
        st.caption(f"✅ Valid timezone. Current local time: {datetime.now(local_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    except Exception:
        st.error(f"'{tz_name}' isn't a recognized IANA timezone — falling back to UTC.")
        local_tz = timezone.utc
    st.markdown("---")
    st.caption(f"curl_cffi (Yahoo Finance reliability layer) installed: **{curl_cffi_is_available()}**")

# ---------------------------------------------------------------------
# TAB: Market & Data Health
# ---------------------------------------------------------------------
with tab_market:
    st.subheader("Market overview & data-health panel")
    st.caption("Fetch a symbol and see its explicit data status before trusting it for anything downstream.")

    col1, col2 = st.columns([2, 1])
    with col1:
        asset_type = st.radio("Asset type", ["Stock", "Crypto", "Commodity"], horizontal=True, key="market_asset_type")
        if asset_type == "Crypto":
            choice = st.selectbox("Symbol", list(CRYPTO_TICKERS.keys()))
            ticker = CRYPTO_TICKERS[choice]
            asset_class = "crypto"
        elif asset_type == "Commodity":
            choice = st.selectbox("Symbol", list(COMMODITY_TICKERS.keys()))
            ticker = COMMODITY_TICKERS[choice]
            asset_class = "commodity"
        else:
            ticker = st.text_input("Stock ticker", value="AAPL").upper().strip()
            asset_class = "stock"
    with col2:
        st.write("")
        st.write("")
        retry = st.button("🔄 Fetch / Retry", use_container_width=True)

    if "last_quote" not in st.session_state:
        st.session_state.last_quote = None

    if retry or st.session_state.last_quote is None or st.session_state.get("last_quote_ticker") != ticker:
        with st.spinner(f"Fetching {ticker}..."):
            quote = fetch_quote(ticker, asset_class, yf, live_threshold_seconds=live_threshold,
                                 stale_threshold_seconds=stale_threshold, max_retries=2)
            st.session_state.last_quote = quote
            st.session_state.last_quote_ticker = ticker

    quote = st.session_state.last_quote

    status_colors = {
        DataStatus.LIVE: "🟢", DataStatus.DELAYED: "🟡",
        DataStatus.STALE: "🟠", DataStatus.UNAVAILABLE: "🔴",
    }
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Instrument", quote.ticker)
    m2.metric("Live Price", f"${quote.price:,.4f}" if quote.price else "—")
    m3.metric("Data Status", f"{status_colors[quote.status]} {quote.status.value}")
    m4.metric("Provider", quote.provider)

    age_txt = (f"{quote.effective_age_seconds:,.0f}s past bar close"
               if quote.effective_age_seconds is not None else "—")
    st.caption(
        f"Bar interval: **{quote.interval}**  |  "
        f"Bar timestamp: {to_local_display(quote.bar_time_utc, local_tz)}  |  "
        f"Fetched at: {to_local_display(quote.fetched_at_utc, local_tz)}  |  "
        f"Effective age: {age_txt}  |  Attempts: {quote.attempts}"
    )
    if quote.interval == "1d":
        st.info(
            "Only daily candles were available for this instrument right now "
            "(intraday returned nothing — common when a market is closed). "
            "Daily data is never reported as LIVE, since it can't tell you what "
            "happened in the last few minutes."
        )
    if quote.error:
        st.error(f"Fetch issue: {quote.error}")

    if quote.status in (DataStatus.STALE, DataStatus.UNAVAILABLE):
        st.error("🚫 DATA ERROR — NO TRADE. This instrument's data is not reliable enough to trade on right now.")
    elif quote.status == DataStatus.DELAYED:
        st.warning("This price is DELAYED (typical for free equity feeds) — treat timing-sensitive decisions with caution.")

# ---------------------------------------------------------------------
# TAB: Candidate Scanner (score + readiness + entry sequence checklist)
# ---------------------------------------------------------------------
with tab_candidate:
    st.subheader("Candidate scanner")
    st.caption(
        "Analyses real 1D / 4H / 1H structure and fills the checklist for you. "
        "Score measures checklist confluence only — not a win probability. "
        "Readiness is judged separately: a high score never auto-promotes a setup to READY."
    )

    scan_mode = st.radio(
        "Evidence source",
        ["Auto-scan from market data", "Manual checklist"],
        horizontal=True, key="scan_mode",
        help="Auto-scan derives every item from actual candles. You can still override anything.",
    )

    if scan_mode == "Auto-scan from market data":
        sc1, sc2, sc3 = st.columns([2, 1, 1])
        with sc1:
            scan_type = st.radio("Asset type", ["Crypto", "Stock", "Commodity"],
                                  horizontal=True, key="scan_asset_type")
            if scan_type == "Crypto":
                scan_choice = st.selectbox("Symbol", list(CRYPTO_TICKERS.keys()), key="scan_crypto")
                scan_ticker = CRYPTO_TICKERS[scan_choice]
            elif scan_type == "Commodity":
                scan_choice = st.selectbox("Symbol", list(COMMODITY_TICKERS.keys()), key="scan_comm")
                scan_ticker = COMMODITY_TICKERS[scan_choice]
            else:
                scan_ticker = st.text_input("Stock ticker", value="AAPL", key="scan_stock").upper().strip()
        with sc2:
            dir_choice = st.selectbox("Direction", ["Auto", "Long", "Short"], key="scan_dir")
        with sc3:
            min_rr = st.number_input("Min R:R", min_value=1.0, value=2.0, step=0.5, key="scan_minrr")

        if st.button("🔍 Scan this instrument", use_container_width=True):
            with st.spinner(f"Analysing {scan_ticker} across 1D / 4H / 1H..."):
                frames, problems = fetch_multi_timeframe(scan_ticker, yf)
                analysis = analyze_candidate(
                    scan_ticker, frames,
                    direction_override=None if dir_choice == "Auto" else dir_choice,
                    min_rr=min_rr, data_problems=problems,
                )
                st.session_state.analysis = analysis

        analysis = st.session_state.get("analysis")
        if analysis is None:
            st.info("Pick an instrument and hit Scan to auto-derive the checklist.")
            score_evidence, seq_evidence = {}, {}
        else:
            for p in analysis.data_problems:
                st.warning(f"Data gap: {p}")

            hc1, hc2, hc3, hc4 = st.columns(4)
            hc1.metric("Instrument", analysis.ticker)
            hc2.metric("1D regime", analysis.regime_1d.title())
            hc3.metric("Direction", analysis.direction or "—")
            hc4.metric("Price", f"{analysis.current_price:,.4g}" if analysis.current_price else "—")

            if analysis.stop is not None and analysis.target is not None:
                lc1, lc2, lc3 = st.columns(3)
                lc1.metric("Entry (current)", f"{analysis.entry:,.4g}")
                lc2.metric("Structural stop", f"{analysis.stop:,.4g}")
                lc3.metric("Structural target", f"{analysis.target:,.4g}")
                if analysis.reward_risk is not None:
                    st.caption(f"Derived reward:risk = **{analysis.reward_risk:.2f}:1** "
                               f"(levels taken from confirmed 4H swings, not invented)")
            else:
                st.warning("Could not derive a full entry/stop/target from confirmed structure.")

            st.markdown("##### Auto-derived evidence (override any of it)")
            score_evidence, seq_evidence = {}, {}
            for key, points, label in POSITIVE_COMPONENTS + NEGATIVE_COMPONENTS:
                item = analysis.score_evidence.get(key)
                auto_val = item.value if item else None
                icon = {True: "✅", False: "❌", None: "❔"}[auto_val]
                default = bool(auto_val)
                score_evidence[key] = st.checkbox(
                    f"{icon} {label} ({points:+d})", value=default, key=f"auto_score_{key}",
                    help=item.reason if item else "Not evaluated.",
                )
                if item:
                    st.caption(f"　↳ {item.reason}")

            with st.expander("Entry sequence detail"):
                for key, description in ENTRY_SEQUENCE_STAGES:
                    item = analysis.sequence_evidence.get(key)
                    auto_val = item.value if item else None
                    icon = {True: "✅", False: "❌", None: "❔"}[auto_val]
                    seq_evidence[key] = st.checkbox(
                        f"{icon} {description}", value=bool(auto_val), key=f"auto_seq_{key}",
                        help=item.reason if item else "Not evaluated.",
                    )
                    if item:
                        st.caption(f"　↳ {item.reason}")
    else:
        st.markdown("##### Entry sequence checklist")
        seq_evidence = {}
        seq_cols = st.columns(2)
        for i, (key, description) in enumerate(ENTRY_SEQUENCE_STAGES):
            with seq_cols[i % 2]:
                seq_evidence[key] = st.checkbox(description, key=f"seq_{key}")

        st.markdown("##### Confluence scoring evidence")
        score_evidence = {}
        mc1, mc2 = st.columns(2)
        with mc1:
            for key, points, label in POSITIVE_COMPONENTS:
                score_evidence[key] = st.checkbox(f"{label} (+{points})", key=f"score_{key}")
        with mc2:
            for key, points, label in NEGATIVE_COMPONENTS:
                score_evidence[key] = st.checkbox(f"{label} ({points})", key=f"score_{key}")

    # --- score + readiness (shared by both modes) ---------------------
    result = score_setup(score_evidence)
    st.markdown("---")
    rc1, rc2 = st.columns([1, 2])
    rc1.metric("Setup Score", f"{result.normalized_score:.1f}/10", result.label)
    rc2.info(trade_policy_for_grade(result.label))

    if result.label != "A+":
        missing = weakest_components(result)
        if missing:
            st.caption("What's weakest: " + "; ".join(
                f"{li.label} (+{li.points_possible} unclaimed)" for li in missing))

    st.markdown("##### Readiness")
    data_ok = st.session_state.get("last_quote") is not None and \
        st.session_state.last_quote.status in (DataStatus.LIVE, DataStatus.DELAYED)
    risk_checks_pass = st.checkbox("Risk checks pass (position sized, within risk budget)", key="risk_checks_pass")
    invalidation_hit = st.checkbox("Structural invalidation has occurred", key="invalidation_hit")
    user_marked_active = st.checkbox("I have manually entered this trade", key="user_marked_active")

    readiness_inputs = ReadinessInputs(
        data_is_valid=data_ok,
        invalidation_hit=invalidation_hit,
        user_marked_active=user_marked_active,
        near_actionable_location=seq_evidence.get("location", False),
        confirmation_triggered=seq_evidence.get("confirmation", False),
        invalidation_defined=seq_evidence.get("structural_invalidation", False),
        risk_checks_pass=risk_checks_pass,
    )
    readiness = determine_readiness(readiness_inputs)
    st.metric("Readiness", f"{display_label(readiness)} ({readiness.value})")
    st.caption(READINESS_DESCRIPTIONS[readiness])

    if not data_ok:
        st.caption(
            "Note: readiness shows DATA ERROR because no usable quote has been fetched yet. "
            "Fetch the instrument on the Market & Data Health tab first."
        )

    if readiness in (Readiness.CONDITIONAL, Readiness.NOT_READY):
        missing_stages = missing_entry_sequence_stages(seq_evidence)
        if missing_stages:
            st.warning("What would confirm this setup:\n" + "\n".join(f"- {m}" for m in missing_stages))

# ---------------------------------------------------------------------
# TAB: Open Positions (management labels)
# ---------------------------------------------------------------------
with tab_positions:
    st.subheader("Open-position review")
    st.caption("Add positions manually; labels are suggestions to review, never auto-executed.")

    with st.form("add_position_form"):
        pc1, pc2, pc3, pc4 = st.columns(4)
        p_asset = pc1.text_input("Asset (e.g. BTC-USD)")
        p_class = pc2.selectbox("Asset class", ["crypto", "stock", "commodity", "forex"])
        p_direction = pc3.selectbox("Direction", ["Long", "Short"])
        p_qty = pc4.number_input("Quantity", min_value=0.0, step=0.01)
        pc5, pc6 = st.columns(2)
        p_entry = pc5.number_input("Entry", min_value=0.0, step=0.01)
        p_stop = pc6.number_input("Current stop", min_value=0.0, step=0.01)
        submitted = st.form_submit_button("Add position")
        if submitted and p_asset and p_entry > 0:
            st.session_state.open_positions.append(
                OpenPosition(asset=p_asset, asset_class=p_class, direction=p_direction,
                             entry=p_entry, stop=p_stop, quantity=p_qty)
            )
            st.success(f"Added {p_direction} {p_asset}")

    if st.session_state.open_positions:
        summary = summarize_portfolio_risk(st.session_state.open_positions)
        st.metric("Total open risk (to stops)", f"${summary.total_open_risk:,.2f}")
        for warning in summary.concentration_warnings:
            st.warning(warning)

        for i, pos in enumerate(st.session_state.open_positions):
            with st.expander(f"{pos.direction} {pos.asset} — entry {pos.entry}, stop {pos.stop}"):
                current_px = st.number_input(f"Current price for {pos.asset}", min_value=0.0, key=f"px_{i}")
                invalidation = st.checkbox("Invalidation hit?", key=f"inv_{i}")
                risk_breach = st.checkbox("Risk rule breached?", key=f"riskbreach_{i}")
                structure_ok = st.checkbox("Structure supports tightening stop?", key=f"structok_{i}")
                if current_px > 0:
                    sm = StopManager(direction=pos.direction, entry=pos.entry,
                                      current_stop=pos.stop, current_target=pos.stop)
                    r_mult = sm.current_r_multiple(current_px)
                    label = suggest_management_label(
                        pos.direction, invalidation, risk_breach, structure_ok, r_mult
                    )
                    st.metric("Suggested action", label.value)
                    st.caption(f"Current R multiple: {r_mult:+.2f}R")
                if st.button("Remove position", key=f"remove_{i}"):
                    st.session_state.open_positions.pop(i)
                    st.rerun()
    else:
        st.info("No open positions logged yet.")

# ---------------------------------------------------------------------
# TAB: Risk Calculator
# ---------------------------------------------------------------------
with tab_risk:
    st.subheader("Position sizing & risk calculator")

    rc1, rc2, rc3 = st.columns(3)
    equity = rc1.number_input("Account equity ($)", min_value=0.0, value=10000.0, step=100.0)
    risk_pct = rc2.number_input("Risk % per trade", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
    direction_calc = rc3.selectbox("Direction", ["Long", "Short"], key="risk_calc_direction")

    rc4, rc5, rc6 = st.columns(3)
    entry_calc = rc4.number_input("Entry price", min_value=0.0, value=100.0, step=0.01)
    stop_calc = rc5.number_input("Stop price", min_value=0.0, value=95.0, step=0.01)
    target_calc = rc6.number_input("Target price", min_value=0.0, value=115.0, step=0.01)

    rc7, rc8, rc9 = st.columns(3)
    leverage_calc = rc7.number_input("Leverage", min_value=1.0, value=1.0, step=0.5)
    fee_calc = rc8.number_input("Round-trip fee rate (e.g. 0.001 = 0.1%)", min_value=0.0, value=0.0, step=0.0005, format="%.4f")
    slippage_calc = rc9.number_input("Slippage (fraction, e.g. 0.0005)", min_value=0.0, value=0.0, step=0.0005, format="%.4f")

    ticker_for_spec = st.text_input("Instrument ticker (for contract specs, optional — e.g. GC=F, CL=F)", value="")

    if st.button("Calculate"):
        try:
            result = calculate_risk(
                account_equity=equity, risk_pct=risk_pct, entry=entry_calc, stop=stop_calc,
                direction=direction_calc, target=target_calc if target_calc > 0 else None,
                leverage=leverage_calc, ticker=ticker_for_spec or None,
                fee_rate=fee_calc, slippage_pct=slippage_calc,
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Max permitted loss", f"${result.max_permitted_loss:,.2f}")
            m2.metric("Quantity", f"{result.quantity:g}")
            m3.metric("Position notional", f"${result.position_notional:,.2f}")
            m4.metric("Margin required", f"${result.margin_required:,.2f}")

            m5, m6, m7 = st.columns(3)
            m5.metric("Gross loss at stop", f"${result.gross_loss_at_stop:,.2f}")
            if result.gross_profit_at_target is not None:
                m6.metric("Gross profit at target", f"${result.gross_profit_at_target:,.2f}")
            if result.net_reward_risk is not None:
                m7.metric("Net R:R (after costs)", f"{result.net_reward_risk:.2f}")

            if result.exceeds_account_equity:
                st.error("⚠️ Required margin exceeds your account equity.")
            if result.liquidation_warning:
                st.warning(result.liquidation_warning)
            for w in result.warnings:
                st.warning(w)
        except InvalidRiskInputError as e:
            st.error(str(e))

# ---------------------------------------------------------------------
# TAB: Trade Journal
# ---------------------------------------------------------------------
with tab_journal:
    st.subheader("Trade journal")

    with st.form("journal_add"):
        jc1, jc2, jc3, jc4 = st.columns(4)
        j_asset = jc1.text_input("Asset")
        j_direction = jc2.selectbox("Direction", ["Long", "Short"], key="j_direction")
        j_leverage = jc3.number_input("Leverage", min_value=1.0, value=1.0, key="j_leverage")
        j_entry = jc4.number_input("Entry", min_value=0.0, key="j_entry")
        jc5, jc6, jc7 = st.columns(3)
        j_tp = jc5.number_input("TP", min_value=0.0, key="j_tp")
        j_sl = jc6.number_input("SL", min_value=0.0, key="j_sl")
        j_size = jc7.number_input("Size (notional USD)", min_value=0.0, key="j_size")
        j_reason = st.text_area("Entry reason / score breakdown", key="j_reason")
        if st.form_submit_button("Add to journal") and j_asset:
            st.session_state.journal.add(JournalEntry(
                asset=j_asset, direction=j_direction, leverage=j_leverage, entry=j_entry,
                tp=j_tp, sl=j_sl, size_notional_usd=j_size, entry_reason=j_reason,
            ))
            st.success("Added.")

    df = st.session_state.journal.to_dataframe()
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇️ Export CSV", data=st.session_state.journal.to_csv_bytes(),
                            file_name="trade_journal.csv", mime="text/csv")

        close_idx = st.number_input("Row index to close", min_value=0, max_value=max(len(df) - 1, 0), step=1)
        close_pnl = st.number_input("Realized P&L", key="close_pnl")
        close_reason = st.text_input("Exit reason", key="close_reason")
        if st.button("Close this trade"):
            st.session_state.journal.close_trade(int(close_idx), close_pnl, close_reason)
            st.rerun()
    else:
        st.info("No journal entries yet.")

# ---------------------------------------------------------------------
# TAB: Backtesting
# ---------------------------------------------------------------------
with tab_backtest:
    st.subheader("Backtesting (engine v1 — example strategy only)")
    st.warning(
        "This validates the REPLAY ENGINE (no look-ahead, fees, metrics) using a simple "
        "moving-average crossover as an example signal generator. It is NOT the "
        "multi-timeframe regime/structure/liquidity strategy — wiring that in is future work."
    )

    bt_ticker = st.text_input("Ticker for backtest", value="BTC-USD")
    bt_period = st.selectbox("History period", ["6mo", "1y", "2y", "5y"], index=1)
    fast = st.number_input("Fast MA period", min_value=2, value=10)
    slow = st.number_input("Slow MA period", min_value=3, value=30)
    fee_bt = st.number_input("Fee rate (per side)", min_value=0.0, value=0.0006, format="%.4f")

    if st.button("Run backtest"):
        with st.spinner("Fetching history..."):
            hist = yf.Ticker(bt_ticker).history(period=bt_period)
        if hist is None or hist.empty:
            st.error("No historical data returned for this ticker.")
        else:
            hist = hist.reset_index(drop=True)
            in_sample, out_sample = split_in_out_sample(hist, split_ratio=0.7)

            for label, sample in [("In-sample", in_sample), ("Out-of-sample", out_sample)]:
                st.markdown(f"##### {label}")
                signals = example_sma_crossover_signals(sample, fast=int(fast), slow=int(slow))
                results = run_backtest(sample, signals, fee_rate=fee_bt)
                metrics = compute_metrics(results)
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Trades", metrics.trade_count)
                mc2.metric("Win rate", f"{metrics.win_rate:.1%}")
                mc3.metric("Expectancy", f"{metrics.expectancy_r:+.2f}R")
                mc4.metric("Profit factor", f"{metrics.profit_factor:.2f}" if metrics.profit_factor != float("inf") else "∞")
                mc5, mc6, mc7 = st.columns(3)
                mc5.metric("Max drawdown", f"{metrics.max_drawdown_r:.2f}R")
                mc6.metric("Max losing streak", metrics.max_losing_streak)
                mc7.metric("Avg MFE / MAE", f"{metrics.avg_mfe_r:.2f}R / {metrics.avg_mae_r:.2f}R")

            st.caption(
                f"Sample size: {len(hist)} bars total. Small samples should not be read as "
                f"proof of an edge — this is a mechanics check, not a profitability claim."
            )
