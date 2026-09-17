import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Trade Setup & Rating Scanner", layout="wide")

st.title("⚡ Active System Setup & Trade Rating")

# ---------------------------------------------------------
# WATCHLIST & TICKER SELECTION
# ---------------------------------------------------------
TICKER_PRESETS = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Nvidia (NVDA)": "NVDA",
    "Google (GOOGL)": "GOOGL",
    "Gold Futures (GC=F)": "GC=F",
    "Brent Crude Oil (BZ=F)": "BZ=F",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Zcash (ZEC-USD)": "ZEC-USD",
    "Sui (SUI-USD)": "SUI-USD",
    "Hyperliquid (HYPE-USD)": "HYPE-USD",
    "Custom Ticker": "CUSTOM"
}

st.sidebar.header("🎯 Watchlist Selection")
selected_preset = st.sidebar.selectbox("Select Asset", options=list(TICKER_PRESETS.keys()))

if TICKER_PRESETS[selected_preset] == "CUSTOM":
    ticker_symbol = st.sidebar.text_input("Enter Symbol", value="AAPL").upper().strip()
else:
    ticker_symbol = TICKER_PRESETS[selected_preset]

# Fetch Market Data
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
    except Exception as e:
        st.sidebar.error(f"Quote error: {e}")

entry_default = float(current_price) if data_fetched and current_price > 0 else 100.0

# ---------------------------------------------------------
# TRADE PARAMETERS
# ---------------------------------------------------------
st.subheader(f"⚙️ Trade Parameters: {ticker_symbol}")

col1, col2, col3 = st.columns(3)
with col1:
    entry_price = st.number_input("Entry Price ($)", value=entry_default, step=0.50, format="%.2f")
with col2:
    take_profit = st.number_input("Target Price ($)", value=float(entry_price * 1.15), step=0.50, format="%.2f")
with col3:
    stop_loss = st.number_input("Hard Stop-Loss ($)", value=float(entry_price * 0.95), step=0.50, format="%.2f")

st.markdown("---")

# ---------------------------------------------------------
# TRADE SETUP CHECKLIST & RATING SCORE
# ---------------------------------------------------------
st.subheader("⭐ Trade Setup Checklist & Rating Score")

c1, c2 = st.columns(2)

with c1:
    check_regime = st.checkbox("Market Regime Alignment (+3 Pts)", value=True)
    check_confluence = st.checkbox("Key Level Confluence Zone (+2 Pts)", value=True)
    check_volume = st.checkbox("Volume / Liquidity Expansion (+1 Pt)", value=True)

risk_per_unit = entry_price - stop_loss
reward_per_unit = take_profit - entry_price
rr_ratio = (reward_per_unit / risk_per_unit) if risk_per_unit > 0 else 0

with c2:
    check_rr = rr_ratio >= 2.0
    st.write("**Reward / Risk Profile Check (+2 Pts):**")
    if check_rr:
        st.success(f"✅ R:R Ratio is **{rr_ratio:.2f}:1** (≥ 2.0 Met)")
    else:
        st.error(f"❌ R:R Ratio is **{rr_ratio:.2f}:1** (< 2.0 Target)")

    check_trigger = st.checkbox("Price Action Confirmation Trigger (+2 Pts)", value=True)

# Total Score Calculation
score = 0
if check_regime: score += 3
if check_confluence: score += 2
if check_volume: score += 1
if check_rr: score += 2
if check_trigger: score += 2

st.markdown("---")

# Grade Output Display
sc1, sc2 = st.columns([1, 2])

with sc1:
    st.metric("Setup Matrix Score", f"{score} / 10 Pts")

with sc2:
    if score >= 8:
        st.success("🌟 **GRADE: A+ SETUP (Full Position Allocation)**\n\nHigh conviction trade meeting core parameters.")
    elif 6 <= score <= 7:
        st.warning("⚠️ **GRADE: B SETUP (50% Half-Sizing Allocation)**\n\nAcceptable setup. Reduce risk capital.")
    else:
        st.error("🚫 **GRADE: C SETUP (NO TRADE / PASS)**\n\nScore is below 6 points. Do not execute.")

st.markdown("---")

# ---------------------------------------------------------
# POSITION SIZING ENGINE
# ---------------------------------------------------------
st.subheader("💼 Position Sizing Engine")

ac1, ac2, ac3 = st.columns(3)
account_balance = ac1.number_input("Portfolio Capital ($)", value=10000.0, step=500.0)
risk_pct = ac2.slider("Risk Limit (%)", min_value=0.5, max_value=3.0, value=1.0, step=0.5)
leverage = ac3.slider("Leverage Cap", min_value=1.0, max_value=5.0, value=3.0, step=0.5)

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
else:
    st.error("Stop loss must be lower than entry price.")
    
