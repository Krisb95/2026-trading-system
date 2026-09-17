import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="2026 Active System Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Active System Setup & Trade Grade Scanner")
st.markdown("---")

# ---------------------------------------------------------
# ASSET DATABASE WITH PRE-SET TRADE PARAMETERS
# ---------------------------------------------------------
PRESET_ASSETS = {
    "Apple (AAPL)": {
        "symbol": "AAPL",
        "entry": 225.00,
        "target": 255.00,
        "stop": 215.00,
        "confluence": True,
        "regime": True,
        "volume": True,
        "notes": "Tech mega-cap breakout pattern."
    },
    "Google (GOOGL)": {
        "symbol": "GOOGL",
        "entry": 175.00,
        "target": 195.00,
        "stop": 165.00,
        "confluence": True,
        "regime": True,
        "volume": False,
        "notes": "Consolidation retest at key support."
    },
    "Gold Futures (GC=F)": {
        "symbol": "GC=F",
        "entry": 2500.00,
        "target": 2700.00,
        "stop": 2420.00,
        "confluence": True,
        "regime": True,
        "volume": True,
        "notes": "Macro momentum hedge setup."
    },
    "Brent Crude Oil (BZ=F)": {
        "symbol": "BZ=F",
        "entry": 78.00,
        "target": 88.00,
        "stop": 73.00,
        "confluence": False,
        "regime": True,
        "volume": False,
        "notes": "Range trade bounce off demand zone."
    },
    "Zcash (ZEC-USD)": {
        "symbol": "ZEC-USD",
        "entry": 32.00,
        "target": 45.00,
        "stop": 28.50,
        "confluence": True,
        "regime": False,
        "volume": True,
        "notes": "Volatile crypto recovery attempt."
    },
    "Sui (SUI-USD)": {
        "symbol": "SUI-USD",
        "entry": 1.80,
        "target": 2.50,
        "stop": 1.55,
        "confluence": True,
        "regime": True,
        "volume": True,
        "notes": "Layer-1 expansion structure."
    },
    "Hyperliquid (HYPE-USD)": {
        "symbol": "HYPE-USD",
        "entry": 12.50,
        "target": 18.00,
        "stop": 10.50,
        "confluence": True,
        "regime": True,
        "volume": True,
        "notes": "High-conviction DEX setup."
    }
}

# ---------------------------------------------------------
# SIDEBAR TICKER SELECTION
# ---------------------------------------------------------
st.sidebar.header("🎯 Watchlist & Ticker Selector")

selection_mode = st.sidebar.radio("Selection Mode", ["Watchlist Presets", "Custom Ticker Input"])

if selection_mode == "Watchlist Presets":
    chosen_preset = st.sidebar.selectbox("Select Asset", options=list(PRESET_ASSETS.keys()))
    asset_info = PRESET_ASSETS[chosen_preset]
    ticker_symbol = asset_info["symbol"]
    default_entry = asset_info["entry"]
    default_target = asset_info["target"]
    default_stop = asset_info["stop"]
    default_confluence = asset_info["confluence"]
    default_regime = asset_info["regime"]
    default_volume = asset_info["volume"]
    asset_notes = asset_info["notes"]
else:
    ticker_symbol = st.sidebar.text_input("Enter Ticker", value="AAPL").upper().strip()
    default_entry = 100.00
    default_target = 115.00
    default_stop = 93.00
    default_confluence = True
    default_regime = True
    default_volume = False
    asset_notes = "Custom user setup."

# Fetch market data
current_price = 0.0
data_fetched = False

if ticker_symbol:
    try:
        stock = yf.Ticker(ticker_symbol)
        history = stock.history(period="1d")
        if not history.empty:
            current_price = float(history['Close'].iloc[-1])
            prev_close = float(history['Open'].iloc[0])
            delta = current_price - prev_close
            pct_change = (delta / prev_close) * 100 if prev_close > 0 else 0
            
            st.sidebar.metric(
                label=f"Live Price ({ticker_symbol})",
                value=f"${current_price:,.2f}",
                delta=f"{delta:+.2f} ({pct_change:+.2f}%)"
            )
            data_fetched = True
        else:
            st.sidebar.warning(f"No price feed for '{ticker_symbol}'. Inputs using preset defaults.")
    except Exception as e:
        st.sidebar.error(f"Quote error: {e}")

# ---------------------------------------------------------
# MAIN NAVIGATION TABS
# ---------------------------------------------------------
tab_setup, tab_trailing = st.tabs([
    "📊 Trade Setup, Parameters & Rating", 
    "📈 Trailing Stop Manager"
])

# ---------------------------------------------------------
# TAB 1: TRADE SETUP, PARAMETERS & RATING (PAGE 1)
# ---------------------------------------------------------
with tab_setup:
    st.subheader(f"⚙️ Trade Setup Parameters: {ticker_symbol}")
    st.caption(f"Asset Context: {asset_notes}")

    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        entry_price = st.number_input("Entry Price ($)", value=float(default_entry), step=0.50, format="%.2f")
    with col_p2:
        take_profit = st.number_input("Target Price ($)", value=float(default_target), step=0.50, format="%.2f")
    with col_p3:
        stop_loss = st.number_input("Hard Stop-Loss ($)", value=float(default_stop), step=0.50, format="%.2f")

    st.markdown("---")
    
    # SETUP SCORING & RATING MATRIX
    st.subheader("⭐ Trade Setup Checklist & Rating Score")
    
    c_check1, c_check2 = st.columns(2)
    
    with c_check1:
        check_regime = st.checkbox("Market Regime Alignment (+3 Pts)", value=default_regime)
        check_confluence = st.checkbox("Key Level Confluence Zone (+2 Pts)", value=default_confluence)
        check_volume = st.checkbox("Volume / Liquidity Expansion (+1 Pt)", value=default_volume)
        
    risk_per_unit = entry_price - stop_loss
    reward_per_unit = take_profit - entry_price
    rr_ratio = (reward_per_unit / risk_per_unit) if risk_per_unit > 0 else 0

    with c_check2:
        check_rr = rr_ratio >= 2.0
        st.write(f"**Reward / Risk Profile Check (+2 Pts):**")
        if check_rr:
            st.success(f"✅ R:R Ratio is **{rr_ratio:.2f}:1** (≥ 2.0 Benchmark Met)")
        else:
            st.error(f"❌ R:R Ratio is **{rr_ratio:.2f}:1** (Sub-optimal, need ≥ 2.0)")

        check_trigger = st.checkbox("Price Action Confirmation Trigger (+2 Pts)", value=True)

    # Score Calculation
    score = 0
    if check_regime: score += 3
    if check_confluence: score += 2
    if check_volume: score += 1
    if check_rr: score += 2
    if check_trigger: score += 2

    st.markdown("---")
    
    # GRADE DISPLAY
    score_col1, score_col2 = st.columns([1, 2])
    
    with score_col1:
        st.metric("Setup Matrix Score", f"{score} / 10 Pts")
        
    with score_col2:
        if score >= 8:
            st.success("🌟 **GRADE: A+ SETUP (Full Position Allocation)**\n\nHigh conviction trade meeting all primary structural and risk metrics.")
        elif 6 <= score <= 7:
            st.warning("⚠️ **GRADE: B SETUP (50% Half-Sizing Allocation)**\n\nAcceptable setup but lacks full alignment. Reduce risk capital.")
        else:
            st.error("🚫 **GRADE: C SETUP (NO TRADE / PASS)**\n\nScore is below 6 points. Do not execute this trade.")

    st.markdown("---")

    # ACCOUNT SIZING CALCULATOR
    st.subheader("💼 Position Sizing Engine")
    
    ac1, ac2, ac3 = st.columns(3)
    account_balance = ac1.number_input("Portfolio Capital ($)", value=10000.0, step=500.0)
    risk_pct = ac2.slider("Risk Limit (%)", min_value=0.5, max_value=3.0, value=1.0, step=0.5)
    leverage = ac3.slider("Isolated Leverage Cap", min_value=1.0, max_value=5.0, value=3.0, step=0.5)

    if risk_per_unit > 0:
        max_risk_dollars = account_balance * (risk_pct / 100)
        units = max_risk_dollars / risk_per_unit
        total_val = units * entry_price
        required_margin = total_val / leverage
        potential_gain = units * reward_per_unit
        est_liq = entry_price * (1 - (1 / leverage))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Position Units", f"{units:,.2f}")
        m2.metric("Margin Capital Needed", f"${required_margin:,.2f}")
        m3.metric("Max Dollar Risk", f"${max_risk_dollars:,.2f}")
        m4.metric("Potential Profit", f"${potential_gain:,.2f}")

        if est_liq >= stop_loss:
            st.error(f"🚨 **LEVERAGE WARNING:** Liquidation Price (${est_liq:,.2f}) sits above Stop Loss (${stop_loss:,.2f}). Lower leverage!")
        else:
            st.caption(f"Estimated Liquidation Level: **${est_liq:,.2f}** (Safely below Hard Stop of **${stop_loss:,.2f}**)")

# ---------------------------------------------------------
# TAB 2: ACTIVE TRAILING STOP MANAGER (ISOLATED TAB)
# ---------------------------------------------------------
with tab_trailing:
    st.subheader("📈 Active Trailing Stop & Profit Lock Manager")
    st.caption("Use this tab to simulate or manage trailing stops for active positions in profit.")

    t_col1, t_col2, t_col3 = st.columns(3)
    
    active_entry = t_col1.number_input("Active Fill Price ($)", value=float(entry_price), step=0.50, format="%.2f")
    
    manual_mode = st.checkbox("Manual Price Simulation", value=False)
    
    if manual_mode or not data_fetched or current_price == 0:
        active_live = t_col2.number_input("Current/Simulated Price ($)", value=float(active_entry * 1.08), step=0.50, format="%.2f")
    else:
        active_live = t_col2.number_input("Live Market Price ($)", value=float(current_price), disabled=True, format="%.2f")
        
    trail_dist_pct = t_col3.slider("Trailing Distance Offset (%)", min_value=2.0, max_value=15.0, value=5.0, step=0.5)

    if active_live > active_entry:
        pnl_pct = ((active_live - active_entry) / active_entry) * 100
        pnl_dollars = (active_live - active_entry) * (units if 'units' in locals() else 0)
        trailing_stop_val = active_live * (1 - (trail_dist_pct / 100))

        st.success(f"🔥 **Position Running in Profit:** Gain: **+{pnl_pct:.2f}%** | P&L: **+${pnl_dollars:,.2f}**")

        st.markdown("---")
        st.subheader("🛡️ Recommended Order Adjustments")

        ts1, ts2 = st.columns(2)
        ts1.info(f"**Option 1: Lock Breakeven**\n\nMove Hard Stop to Entry: `${active_entry:,.2f}`")
        ts2.success(f"**Option 2: Active Trailing Stop ({trail_dist_pct}% Offset)**\n\nSet Dynamic Stop Order to: `${trailing_stop_val:,.2f}`")
    else:
        st.warning("Position is flat or in draw-down. Maintain original hard stop-loss.")
        
