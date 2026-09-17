import streamlit as st
import yfinance as yf
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Trading Dashboard & Dynamic Setup Generator",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Mobile Trading Dashboard & Setup Generator")
st.markdown("---")

# ---------------------------------------------------------
# 1. WATCHLIST DEFINITION (Including BTC, ETH, SOL)
# ---------------------------------------------------------
TICKER_MAP = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Solana (SOL-USD)": "SOL-USD",
    "Zcash (ZEC-USD)": "ZEC-USD",
    "Sui (SUI-USD)": "SUI-USD",
    "Avalanche (AVAX-USD)": "AVAX-USD",
    "Gold Futures (GC=F)": "GC=F",
    "Brent Crude Oil (BZ=F)": "BZ=F",
    "Apple (AAPL)": "AAPL",
    "Google (GOOGL)": "GOOGL",
    "eBay (EBAY)": "EBAY",
    "SpaceX (Private - No Live Feed)": "SPACEX",
    "Custom Ticker Input": "CUSTOM"
}

# ---------------------------------------------------------
# 2. HELPER FUNCTION: Calculate Dynamic Setup via ATR
# ---------------------------------------------------------
def generate_live_setup(ticker_symbol):
    """
    Fetches recent price history to dynamically compute ATR-based trade setups.
    Returns: (current_price, delta_price, entry, take_profit, stop_loss)
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period="1mo")
        
        if df.empty or len(df) < 2:
            return None, None, None, None, None

        current_price = float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2]) if len(df) >= 2 else current_price
        delta_price = current_price - prev_close

        # Calculate 14-period ATR (Average True Range)
        if len(df) >= 14:
            df['H-L'] = df['High'] - df['Low']
            df['H-PC'] = np.abs(df['High'] - df['Close'].shift(1))
            df['L-PC'] = np.abs(df['Low'] - df['Close'].shift(1))
            df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            atr = df['TR'].rolling(window=14).mean().iloc[-1]
            if np.isnan(atr) or atr <= 0:
                atr = current_price * 0.02
        else:
            atr = current_price * 0.02

        # Dynamic Setup Logic (1:2 Risk-to-Reward Ratio)
        entry_price = current_price
        stop_loss = max(0.01, current_price - (1.5 * atr))
        take_profit = current_price + (3.0 * atr)

        return current_price, delta_price, entry_price, take_profit, stop_loss
    except Exception:
        return None, None, None, None, None


# ---------------------------------------------------------
# 3. SIDEBAR - Watchlist Selection & Live Metric Display
# ---------------------------------------------------------
st.sidebar.header("Ticker Settings")

selected_preset = st.sidebar.selectbox("Select Asset Watchlist", options=list(TICKER_MAP.keys()))
selected_symbol = TICKER_MAP[selected_preset]

if selected_symbol == "CUSTOM":
    ticker_symbol = st.sidebar.text_input("Enter Custom Ticker", value="BTC-USD").upper().strip()
elif selected_symbol == "SPACEX":
    ticker_symbol = ""
    st.sidebar.info("SpaceX is a private company and does not have a public market ticker feed.")
else:
    ticker_symbol = selected_symbol

current_price, delta_price, auto_entry, auto_tp, auto_sl = None, None, None, None, None
data_fetched = False

if ticker_symbol:
    with st.spinner(f"Fetching live prices & computing setup for {ticker_symbol}..."):
        current_price, delta_price, auto_entry, auto_tp, auto_sl = generate_live_setup(ticker_symbol)
        
        if current_price is not None:
            data_fetched = True
            st.sidebar.metric(
                label=f"Live Market Price ({ticker_symbol})",
                value=f"${current_price:,.2f}",
                delta=f"{delta_price:+.2f}"
            )
            st.sidebar.success("✅ Dynamic setup levels auto-populated!")
        else:
            st.sidebar.error(f"Could not fetch live pricing for '{ticker_symbol}'.")

# ---------------------------------------------------------
# 4. MAIN PANEL - Inputs & Auto-Calculated Setups
# ---------------------------------------------------------
col_left, col_right = st.columns(2)

default_entry = auto_entry if data_fetched else 100.0
default_tp = auto_tp if data_fetched else 110.0
default_sl = auto_sl if data_fetched else 95.0

with col_left:
    st.subheader("⚙️ Active Trade Setup (Live Pricing)")
    entry_price = st.number_input(
        "Entry Price ($)", 
        value=float(default_entry), 
        step=0.10,
        format="%.2f",
        help="Defaults to current market price."
    )
    take_profit_target = st.number_input(
        "Take-Profit Target ($)", 
        value=float(default_tp), 
        step=0.10,
        format="%.2f",
        help="Auto-calculated target based on market volatility (2x Risk)."
    )
    stop_loss_price = st.number_input(
        "Stop-Loss Price ($)", 
        value=float(default_sl), 
        step=0.10,
        format="%.2f",
        help="Auto-calculated stop loss based on recent volatility."
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
        value=1.0, 
        step=0.5
    )

st.markdown("---")

# ---------------------------------------------------------
# 5. TAKE-PROFIT & ALERTS MONITORING
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
# 6. POSITION SIZING & CALCULATIONS
# ---------------------------------------------------------
st.subheader("🧮 Calculated Position Sizing & P&L")

risk_amount = account_balance * (risk_percentage / 100)
risk_per_share = entry_price - stop_loss_price

if risk_per_share > 0:
    position_shares = risk_amount / risk_per_share
    total_cost = position_shares * entry_price
    potential_profit = position_shares * (take_profit_target - entry_price)
    reward_risk_ratio = (take_profit_target - entry_price) / risk_per_share
    
    m1, m2, m3, m4 = st.columns(4)
    # Crypto support fractional units
    is_crypto = "-USD" in ticker_symbol
    qty_format = f"{position_shares:,.4f} units" if is_crypto else f"{int(position_shares):,} shares"
    
    m1.metric(label="Suggested Units/Shares", value=qty_format)
    m2.metric(label="Capital Required", value=f"${total_cost:,.2f}")
    m3.metric(label="Max Loss Risk", value=f"${risk_amount:,.2f}")
    m4.metric(label="Potential Gain", value=f"${potential_profit:,.2f}", delta=f"R:R {reward_risk_ratio:.2f}")

    if total_cost > account_balance:
        st.warning(
            f"⚠️ Capital required (${total_cost:,.2f}) exceeds total account balance "
            f"(${account_balance:,.2f}). Consider applying leverage or adjusting position size."
        )
else:
    st.error("Stop-Loss price must be set below the Entry Price to calculate position size.")
    
