import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="2026 Active Trading & Risk Manager",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Active System Scanner & Position Manager")
st.markdown("---")

# ---------------------------------------------------------
# WATCHLIST PRESETS & TICKER MAPPING
# ---------------------------------------------------------
WATCHLIST = {
    "Apple (AAPL)": "AAPL",
    "Google (GOOGL)": "GOOGL",
    "SpaceX (SPCX)": "SPCX",
    "Gold Futures (GC=F)": "GC=F",
    "Brent Crude Oil (BZ=F)": "BZ=F",
    "Zcash Crypto (ZEC-USD)": "ZEC-USD",
    "Sui Crypto (SUI-USD)": "SUI-USD",
    "Hyperliquid Crypto (HYPE-USD)": "HYPE-USD",
}

st.sidebar.header("Asset Selection")
selected_preset = st.sidebar.selectbox("Choose Watchlist Asset", options=["Custom Input"] + list(WATCHLIST.keys()))

if selected_preset != "Custom Input":
    ticker_symbol = WATCHLIST[selected_preset]
    st.sidebar.text_input("Active Ticker Symbol", value=ticker_symbol, disabled=True)
else:
    ticker_symbol = st.sidebar.text_input("Enter Custom Ticker", value="AAPL").upper().strip()

current_price = 0.0
data_fetched = False

if ticker_symbol:
    try:
        stock = yf.Ticker(ticker_symbol)
        history = stock.history(period="1d")
        
        if not history.empty:
            current_price = float(history['Close'].iloc[-1])
            prev_close = float(history['Close'].iloc[0]) if len(history) > 1 else current_price
            delta_price = current_price - prev_close
            
            st.sidebar.metric(
                label=f"Live Price ({ticker_symbol})",
                value=f"${current_price:.2f}",
                delta=f"{delta_price:+.2f}"
            )
            data_fetched = True
        else:
            st.sidebar.error(f"No price data found for '{ticker_symbol}'.")
    except Exception as e:
        st.sidebar.error(f"Error fetching ticker data: {e}")

# ---------------------------------------------------------
# TABBED NAVIGATION: SETUP CALCULATOR vs ACTIVE TRADE MANAGER
# ---------------------------------------------------------
tab_setup, tab_active = st.tabs(["📊 Trade Setup & Risk Calculator", "📈 Active Trade & Trailing Stop Manager"])

# Pre-calculate baseline metrics to pass state across tabs
entry_price_default = float(current_price) if data_fetched and current_price > 0 else 100.0

with tab_setup:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("⚙️ Trade Setup Parameters")
        entry_price = st.number_input(
            "Entry Price ($)", 
            value=entry_price_default, 
            step=1.0,
            format="%.2f"
        )
        take_profit_target = st.number_input(
            "Target Price ($)", 
            value=float(entry_price * 1.10), 
            step=1.0,
            format="%.2f"
        )
        stop_loss_price = st.number_input(
            "Hard Stop-Loss ($)", 
            value=float(entry_price * 0.95), 
            step=1.0,
            format="%.2f"
        )

    with col_right:
        st.subheader("💼 Account Risk & Leverage Controls")
        account_balance = st.number_input(
            "Total Portfolio Balance ($)", 
            value=10000.0, 
            step=500.0,
            format="%.2f"
        )
        risk_percentage = st.slider(
            "Max Risk Per Trade (%)", 
            min_value=0.5, 
            max_value=3.0, 
            value=1.0, 
            step=0.5
        )
        leverage = st.slider(
            "Isolated Margin Leverage (Max 5x Cap)", 
            min_value=1.0, 
            max_value=5.0, 
            value=3.0, 
            step=0.5
        )

    st.markdown("---")

    # Target & Alert Status
    st.subheader("🎯 Target & Alert Status")
    if data_fetched and current_price > 0:
        if current_price >= take_profit_target:
            st.success(f"🚨 **TAKE-PROFIT TRIGGERED!** Current price (${current_price:.2f}) reached or passed target (${take_profit_target:.2f}).")
        elif current_price <= stop_loss_price:
            st.error(f"⚠️ **STOP-LOSS TRIGGERED!** Current price (${current_price:.2f}) dropped below stop loss (${stop_loss_price:.2f}).")
        else:
            dist_tp = ((take_profit_target - current_price) / current_price) * 100
            dist_sl = ((current_price - stop_loss_price) / current_price) * 100
            st.info(f"📊 **In Range:** Live Price: **${current_price:.2f}** | TP Target: **+{dist_tp:.2f}%** away | Stop-Loss: **-{dist_sl:.2f}%** below")

    # Sizing Engine
    st.subheader("🧮 Calculated Position Sizing (1% Risk Engine)")
    risk_amount = account_balance * (risk_percentage / 100)
    risk_per_unit = entry_price - stop_loss_price

    if risk_per_unit > 0:
        position_units = risk_amount / risk_per_unit
        total_position_val = position_units * entry_price
        margin_required = total_position_val / leverage
        potential_profit = position_units * (take_profit_target - entry_price)
        rr_ratio = (take_profit_target - entry_price) / risk_per_unit
        
        # Estimated Liquidation Price (Longs)
        est_liq_price = entry_price * (1 - (1 / leverage))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Position Units", f"{position_units:,.2f}")
        m2.metric("Margin Required", f"${margin_required:,.2f}", help=f"Total Value: ${total_position_val:,.2f} @ {leverage}x Leverage")
        m3.metric("Max Dollar Risk", f"${risk_amount:,.2f}")
        m4.metric("Potential Gain", f"${potential_profit:,.2f}", delta=f"R:R {rr_ratio:.2f}")

        if est_liq_price >= stop_loss_price:
            st.error(f"🚨 **LEVERAGE RISK WARNING:** Estimated Liquidation Price (${est_liq_price:.2f}) is higher than or equal to your Stop Loss (${stop_loss_price:.2f}). Lower leverage or widen stop distance.")
        else:
            st.caption(f"Estimated Liquidation Price: **${est_liq_price:.2f}** (Safely below Hard Stop of **${stop_loss_price:.2f}**)")
    else:
        st.error("Stop-Loss price must be set below Entry Price to calculate position sizing.")

# ---------------------------------------------------------
# TAB 2: ACTIVE TRADE & TRAILING STOP MANAGER
# ---------------------------------------------------------
with tab_active:
    st.subheader("🔄 Trailing Stop & Profit Lock Manager")
    st.markdown("Automates trailing stop loss calculations using real-time market feeds or custom simulation scenarios.")

    override_live = st.checkbox("Simulate Target Scenario (Manual Price Input)", value=False)

    c1, c2, c3 = st.columns(3)
    active_entry = c1.number_input("Active Entry Price ($)", value=float(entry_price), step=1.0, format="%.2f")

    if override_live or current_price == 0.0:
        active_live_price = c2.number_input("Simulated Market Price ($)", value=float(active_entry * 1.05), step=1.0, format="%.2f")
    else:
        active_live_price = c2.number_input("Current Live Market Price ($)", value=float(current_price), disabled=True, format="%.2f")

    trail_pct = c3.slider("Trailing Stop Distance (%)", min_value=3.0, max_value=20.0, value=8.0, step=0.5)

    if active_live_price > active_entry:
        gain_pct = ((active_live_price - active_entry) / active_entry) * 100
        unrealized_pnl = (active_live_price - active_entry) * (position_units if 'position_units' in locals() and risk_per_unit > 0 else 0)
        
        trailing_sl_price = active_live_price * (1 - (trail_pct / 100))
        breakeven_price = active_entry

        st.success(f"🔥 **Position in Profit:** Live Gain: **+{gain_pct:.2f}%** | Unrealized P&L: **+${unrealized_pnl:,.2f}**")

        st.markdown("---")
        st.subheader("🛡️ Recommended Updated Stop-Loss Orders")

        col_sl1, col_sl2 = st.columns(2)
        
        with col_sl1:
            st.info(f"**1. Move Stop to Breakeven:**\n\n**New SL:** `${breakeven_price:,.2f}`\n\n*Action:* Guarantees a risk-free trade once initial profit target is approached.")

        with col_sl2:
            st.success(f"**2. Dynamic Trailing Stop ({trail_pct}% Offset):**\n\n**New SL:** `${trailing_sl_price:,.2f}`\n\n*Action:* Adjust exchange stop order to `${trailing_sl_price:,.2f}` to lock in profits.")

    else:
        st.warning("Position is currently at or below entry price. Keep your original Hard Stop-Loss active.")
        
