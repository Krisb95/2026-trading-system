import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="Trading Dashboard & Risk Calculator",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Mobile Trading Dashboard")
st.markdown("---")

# ---------------------------------------------------------
# 1. SIDEBAR - Ticker Watchlist & Dynamic Fetch
# ---------------------------------------------------------
st.sidebar.header("Ticker Settings")

# Requested Watchlist Mapping
TICKER_PRESETS = {
    "Apple (AAPL)": "AAPL",
    "Google (GOOGL)": "GOOGL",
    "eBay (EBAY)": "EBAY",
    "Gold Futures (GC=F)": "GC=F",
    "Brent Crude Oil (BZ=F)": "BZ=F",
    "Zcash (ZEC-USD)": "ZEC-USD",
    "Sui (SUI-USD)": "SUI-USD",
    "Avalanche (AVAX-USD)": "AVAX-USD",
    "SpaceX (Private - No Public Ticker)": "SPACEX",
    "Custom Ticker Input": "CUSTOM"
}

selected_option = st.sidebar.selectbox("Select Asset Watchlist", options=list(TICKER_PRESETS.keys()))

if TICKER_PRESETS[selected_option] == "CUSTOM":
    ticker_symbol = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper().strip()
elif TICKER_PRESETS[selected_option] == "SPACEX":
    ticker_symbol = ""
    st.sidebar.info("SpaceX is a private company and does not have a public ticker symbol on market feeds.")
else:
    ticker_symbol = TICKER_PRESETS[selected_option]

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
                value=f"${current_price:,.2f}",
                delta=f"{delta_price:+.2f}"
            )
            data_fetched = True
        else:
            st.sidebar.error(f"No pricing data found for '{ticker_symbol}'.")
    except Exception as e:
        st.sidebar.error(f"Error fetching ticker data: {e}")

# ---------------------------------------------------------
# 2. MAIN PANEL - Inputs & Parameters
# ---------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("⚙️ Trade Parameters")
    entry_price = st.number_input(
        "Entry Price ($)", 
        value=float(current_price) if data_fetched else 150.0, 
        step=1.0,
        format="%.2f"
    )
    take_profit_target = st.number_input(
        "Take-Profit Target ($)", 
        value=float(entry_price * 1.10), 
        step=1.0,
        format="%.2f"
    )
    stop_loss_price = st.number_input(
        "Stop-Loss Price ($)", 
        value=float(entry_price * 0.95), 
        step=1.0,
        format="%.2f"
    )

with col_right:
    st.subheader("💼 Account & Risk Limits")
    account_balance = st.number_input(
        "Total Portfolio Balance ($)", 
        value=10000.0, 
        step=500.0,
        format="%.2f"
    )
    risk_percentage = st.slider(
        "Max Risk Per Trade (%)", 
        min_value=0.5, 
        max_value=5.0, 
        value=2.0, 
        step=0.5
    )

st.markdown("---")

# ---------------------------------------------------------
# 3. TAKE-PROFIT & ALERTS MONITORING
# ---------------------------------------------------------
st.subheader("🎯 Target & Alert Status")

if data_fetched and current_price > 0:
    if current_price >= take_profit_target:
        st.success(
            f"🚨 **TAKE-PROFIT TRIGGERED!** Current price (${current_price:,.2f}) "
            f"has reached or passed your target (${take_profit_target:,.2f})."
        )
    elif current_price <= stop_loss_price:
        st.error(
            f"⚠️ **STOP-LOSS TRIGGERED!** Current price (${current_price:,.2f}) "
            f"has dropped to or below your stop loss (${stop_loss_price:,.2f})."
        )
    else:
        dist_tp = ((take_profit_target - current_price) / current_price) * 100
        dist_sl = ((current_price - stop_loss_price) / current_price) * 100
        st.info(
            f"📊 **In Range:** Current price is **${current_price:,.2f}** | "
            f"Take-Profit is **{dist_tp:.2f}%** away | Stop-Loss is **{dist_sl:.2f}%** below."
        )
else:
    st.warning("Select or enter a valid ticker in the sidebar to run live status alerts.")

st.markdown("---")

# ---------------------------------------------------------
# 4. POSITION SIZING & CALCULATIONS
# ---------------------------------------------------------
st.subheader("🧮 Calculated Position Sizing & P&L")

risk_amount = account_balance * (risk_percentage / 100)
risk_per_share = entry_price - stop_loss_price

if risk_per_share > 0:
    position_shares = int(risk_amount / risk_per_share)
    total_cost = position_shares * entry_price
    potential_profit = position_shares * (take_profit_target - entry_price)
    reward_risk_ratio = (take_profit_target - entry_price) / risk_per_share
    
    # Render layout metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label="Suggested Units/Shares", value=f"{position_shares:,} qty")
    m2.metric(label="Capital Required", value=f"${total_cost:,.2f}")
    m3.metric(label="Max Loss Risk", value=f"${risk_amount:,.2f}")
    m4.metric(label="Potential Gain", value=f"${potential_profit:,.2f}", delta=f"R:R {reward_risk_ratio:.2f}")

    if total_cost > account_balance:
        st.warning(
            f"⚠️ Capital required (${total_cost:,.2f}) exceeds total account balance "
            f"(${account_balance:,.2f}). Consider adjusting leverage or lowering risk."
        )
else:
    st.error("Stop-Loss price must be set below the Entry Price to calculate position size.")
    
