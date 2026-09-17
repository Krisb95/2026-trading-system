import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="Active Trading & Risk Manager",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Active System Scanner & Position Manager")
st.markdown("---")

# ---------------------------------------------------------
# SIDEBAR: TICKER AUTO-COMPLETE & LIVE MARKET FEED
# ---------------------------------------------------------
st.sidebar.header("🔍 Asset Search & Selection")

# Popular Ticker Shortcuts
TICKER_PRESETS = {
    "Custom Search": "",
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Nvidia (NVDA)": "NVDA",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Gold Futures (GC=F)": "GC=F",
    "S&P 500 Index (^GSPC)": "^GSPC",
}

selected_preset = st.sidebar.selectbox("Preset Watchlist", options=list(TICKER_PRESETS.keys()))

if selected_preset != "Custom Search":
    active_ticker = TICKER_PRESETS[selected_preset]
    st.sidebar.text_input("Selected Ticker", value=active_ticker, disabled=True)
else:
    active_ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g. AAPL, TSLA, BTC-USD)", value="AAPL").upper().strip()

current_price = 0.0
data_fetched = False

if active_ticker:
    try:
        stock = yf.Ticker(active_ticker)
        history = stock.history(period="1d")
        
        if not history.empty:
            current_price = float(history['Close'].iloc[-1])
            open_price = float(history['Open'].iloc[0])
            price_change = current_price - open_price
            pct_change = (price_change / open_price) * 100 if open_price > 0 else 0.0

            st.sidebar.metric(
                label=f"Live Price ({active_ticker})",
                value=f"${current_price:,.2f}",
                delta=f"{price_change:+.2f} ({pct_change:+.2f}%)"
            )
            data_fetched = True
        else:
            st.sidebar.error(f"No active quote found for '{active_ticker}'. Verify ticker symbol.")
    except Exception as e:
        st.sidebar.error(f"Failed to fetch market data: {e}")

# Default base price for inputs
entry_price_default = float(current_price) if data_fetched and current_price > 0 else 100.0

# ---------------------------------------------------------
# MAIN NAVIGATION TABS
# ---------------------------------------------------------
tab_params, tab_risk, tab_trailing = st.tabs([
    "⚙️ Trade Parameters", 
    "🧮 Risk & Position Sizing", 
    "📈 Trailing Stop Manager"
])

# ---------------------------------------------------------
# TAB 1: TRADE PARAMETERS
# ---------------------------------------------------------
with tab_params:
    st.subheader("⚙️ Active Trade Entry & Targets")
    st.caption("Configure core trade parameters and view automated price action alerts.")

    col1, col2, col3 = st.columns(3)

    with col1:
        entry_price = st.number_input(
            "Entry Price ($)", 
            value=entry_price_default, 
            step=0.50,
            format="%.2f"
        )
    with col2:
        take_profit_target = st.number_input(
            "Take-Profit Target ($)", 
            value=float(entry_price * 1.10), 
            step=0.50,
            format="%.2f"
        )
    with col3:
        stop_loss_price = st.number_input(
            "Hard Stop-Loss ($)", 
            value=float(entry_price * 0.95), 
            step=0.50,
            format="%.2f"
        )

    st.markdown("---")
    st.subheader("📊 Target & Alert Monitor")

    if data_fetched and current_price > 0:
        if current_price >= take_profit_target:
            st.success(f"🚨 **TARGET REACHED!** Market price (${current_price:,.2f}) hit or surpassed target (${take_profit_target:,.2f}).")
        elif current_price <= stop_loss_price:
            st.error(f"⚠️ **STOP-LOSS TRIGGERED!** Market price (${current_price:,.2f}) fell below stop level (${stop_loss_price:,.2f}).")
        else:
            dist_tp = ((take_profit_target - current_price) / current_price) * 100
            dist_sl = ((current_price - stop_loss_price) / current_price) * 100
            
            p_col1, p_col2 = st.columns(2)
            p_col1.metric("Distance to Take Profit", f"+{dist_tp:.2f}%", delta=f"${take_profit_target - current_price:,.2f}")
            p_col2.metric("Distance to Hard Stop", f"-{dist_sl:.2f}%", delta=f"-${current_price - stop_loss_price:,.2f}", delta_color="inverse")
    else:
        st.info("Enter a valid ticker on the sidebar to enable dynamic distance alerts.")

# ---------------------------------------------------------
# TAB 2: RISK & POSITION SIZING CALCULATOR
# ---------------------------------------------------------
with tab_risk:
    st.subheader("🧮 Portfolio Risk & Leverage Sizing Engine")

    col_acc1, col_acc2, col_acc3 = st.columns(3)

    with col_acc1:
        account_balance = st.number_input(
            "Total Portfolio Capital ($)", 
            value=10000.0, 
            step=500.0,
            format="%.2f"
        )
    with col_acc2:
        risk_percentage = st.slider(
            "Account Risk Limit (%)", 
            min_value=0.25, 
            max_value=3.0, 
            value=1.0, 
            step=0.25
        )
    with col_acc3:
        leverage = st.slider(
            "Leverage Multiple", 
            min_value=1.0, 
            max_value=5.0, 
            value=3.0, 
            step=0.5
        )

    st.markdown("---")

    risk_amount = account_balance * (risk_percentage / 100)
    risk_per_unit = entry_price - stop_loss_price

    if risk_per_unit > 0:
        position_units = risk_amount / risk_per_unit
        total_position_val = position_units * entry_price
        margin_required = total_position_val / leverage
        potential_profit = position_units * (take_profit_target - entry_price)
        rr_ratio = (take_profit_target - entry_price) / risk_per_unit
        est_liq_price = entry_price * (1 - (1 / leverage))

        res1, res2, res3, res4 = st.columns(4)
        res1.metric("Position Units", f"{position_units:,.2f}")
        res2.metric("Margin Required", f"${margin_required:,.2f}", help=f"Total Exposure: ${total_position_val:,.2f}")
        res3.metric("Max Dollar Risk", f"${risk_amount:,.2f}")
        res4.metric("Potential Profit", f"${potential_profit:,.2f}", delta=f"R:R Ratio {rr_ratio:.2f}")

        if est_liq_price >= stop_loss_price:
            st.error(f"🚨 **HIGH LEVERAGE WARNING:** Liquidation Price (${est_liq_price:,.2f}) triggers before Stop Loss (${stop_loss_price:,.2f}). Reduce leverage!")
        else:
            st.caption(f"Estimated Liquidation Price: **${est_liq_price:,.2f}** (Safely below Hard Stop of **${stop_loss_price:,.2f}**)")
    else:
        st.error("Stop Loss must be set lower than Entry Price to perform position calculation.")

# ---------------------------------------------------------
# TAB 3: TRAILING STOP MANAGER
# ---------------------------------------------------------
with tab_trailing:
    st.subheader("📈 Dynamic Trailing Stop & Profit Lock")

    use_manual_price = st.checkbox("Simulate Custom Market Price", value=False)

    tr_col1, tr_col2, tr_col3 = st.columns(3)
    active_entry_val = tr_col1.number_input("Trade Entry ($)", value=float(entry_price), step=0.50, format="%.2f")

    if use_manual_price or current_price == 0.0:
        active_market_val = tr_col2.number_input("Current/Simulated Price ($)", value=float(active_entry_val * 1.08), step=0.50, format="%.2f")
    else:
        active_market_val = tr_col2.number_input("Live Feed Price ($)", value=float(current_price), disabled=True, format="%.2f")

    trail_offset_pct = tr_col3.slider("Trailing Distance (%)", min_value=2.0, max_value=15.0, value=5.0, step=0.5)

    if active_market_val > active_entry_val:
        unrealized_pct = ((active_market_val - active_entry_val) / active_entry_val) * 100
        unrealized_dollars = (active_market_val - active_entry_val) * (position_units if 'position_units' in locals() and risk_per_unit > 0 else 0)
        
        dynamic_stop = active_market_val * (1 - (trail_offset_pct / 100))

        st.success(f"🔥 **Position Running in Gain:** Unrealized Gain: **+{unrealized_pct:.2f}%** | P&L: **+${unrealized_dollars:,.2f}**")

        st.markdown("---")
        st.subheader("🛡️ Recommended Order Adjustments")

        s1, s2 = st.columns(2)
        s1.info(f"**Move to Breakeven:**\n\nSet Stop-Loss Order at: `${active_entry_val:,.2f}`")
        s2.success(f"**Trailing Stop Level ({trail_offset_pct}%):**\n\nSet Stop-Loss Order at: `${dynamic_stop:,.2f}`")
    else:
        st.warning("Position is currently flat or below entry. Keep default hard stop-loss enabled.")
