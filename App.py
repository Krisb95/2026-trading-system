import streamlit as st
import yfinance as yf

# ---------------------------------------------------------
# Streamlit Mobile Page Configuration & Dark Theme CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="2026 Bull Run System Scanner",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    /* Compact padding for mobile screen optimization */
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; padding-left: 0.8rem; padding-right: 0.8rem; }
    .stMetric { background-color: #1a1c23; padding: 10px; border-radius: 8px; border: 1px solid #2e323e; }
    .card-a-plus { border-left: 5px solid #00E676; background-color: #132419; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
    .card-b { border-left: 5px solid #FFD600; background-color: #262413; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
    .card-c { border-left: 5px solid #FF1744; background-color: #281215; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Asset Classification & Tiering Engine
# ---------------------------------------------------------
ASSET_TIERS = {
    # Existing Crypto Assets
    "BTC-USD": {"name": "Bitcoin", "tier": 1, "max_lev": "3x - 5x", "trail": "5% - 7%"},
    "ETH-USD": {"name": "Ethereum", "tier": 1, "max_lev": "3x - 5x", "trail": "5% - 7%"},
    "SOL-USD": {"name": "Solana", "tier": 2, "max_lev": "2x - 3x", "trail": "10% - 15%"},
    "AVAX-USD": {"name": "Avalanche", "tier": 2, "max_lev": "2x - 3x", "trail": "10% - 15%"},
    "NEAR-USD": {"name": "NEAR Protocol", "tier": 2, "max_lev": "2x - 3x", "trail": "10% - 15%"},
    "DOGE-USD": {"name": "Dogecoin", "tier": 3, "max_lev": "1x (Spot)", "trail": "25% - 35%"},
    
    # Newly Added Crypto Assets
    "SUI-USD": {"name": "Sui", "tier": 2, "max_lev": "2x - 3x", "trail": "10% - 15%"},
    "ZEC-USD": {"name": "Zcash", "tier": 2, "max_lev": "2x - 3x", "trail": "10% - 15%"},

    # Commodities
    "GC=F": {"name": "Gold Futures", "tier": 1, "max_lev": "3x - 5x", "trail": "5% - 7%"},
    "CL=F": {"name": "Crude Oil WTI", "tier": 1, "max_lev": "3x - 5x", "trail": "5% - 7%"},

    # U.S. Stocks
    "AAPL": {"name": "Apple Inc.", "tier": 1, "max_lev": "3x - 5x", "trail": "5% - 7%"},
    "GOOGL": {"name": "Alphabet (Google)", "tier": 1, "max_lev": "3x - 5x", "trail": "5% - 7%"},
    "EBAY": {"name": "eBay Inc.", "tier": 1, "max_lev": "3x - 5x", "trail": "5% - 7%"},
    "SPCX": {"name": "SpaceX", "tier": 1, "max_lev": "3x - 5x", "trail": "5% - 7%"}
}

st.title("⚡ 2026 Active System Scanner")
st.caption("Active Trading Engine • Hyperliquid, Bybit & Stocks Rules")

# ---------------------------------------------------------
# Top Inputs (Mobile Optimized)
# ---------------------------------------------------------
col1, col2 = st.columns([2, 1])
with col1:
    selected_asset = st.selectbox("Select Asset Ticker", list(ASSET_TIERS.keys()), index=2)
with col2:
    account_equity = st.number_input("Account ($)", value=10000, step=1000)

asset_meta = ASSET_TIERS[selected_asset]

# ---------------------------------------------------------
# Fetch Live Market Data
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_market_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        df_daily = ticker.history(period="60d", interval="1d")
        df_4h = ticker.history(period="14d", interval="1h") # Approximation for mobile scan
        return df_daily, df_4h
    except Exception as e:
        return None, None

df_daily, df_4h = fetch_market_data(selected_asset)

if df_daily is not None and not df_daily.empty:
    current_price = df_daily['Close'].iloc[-1]
    
    # ---------------------------------------------------------
    # System Regime & Setup Calculation Protocol
    # ---------------------------------------------------------
    ma50_daily = df_daily['Close'].rolling(50).mean().iloc[-1]
    regime_1d = "BULLISH" if current_price > ma50_daily else "NEUTRAL / BEARISH"
    
    # Determine Entry & Structural Stop based on Tier
    if asset_meta["tier"] == 1:
        stop_dist_pct = 0.025 # 2.5% structural stop baseline
        score = 8 if regime_1d == "BULLISH" else 6
    elif asset_meta["tier"] == 2:
        stop_dist_pct = 0.0497 # 4.97% structural stop baseline
        score = 8 if regime_1d == "BULLISH" else 6
    else:
        stop_dist_pct = 0.12 # Memes / High Beta Spot
        score = 5

    entry_price = current_price
    hard_stop = entry_price * (1 - stop_dist_pct)
    target_1 = entry_price + (2.0 * (entry_price - hard_stop))
    target_2 = entry_price + (3.5 * (entry_price - hard_stop))

    # Risk Management Mechanics (1% Risk Engine)
    risk_amount = account_equity * 0.01
    position_size_usd = risk_amount / stop_dist_pct
    units = position_size_usd / entry_price

    # ---------------------------------------------------------
    # Render Setup Card
    # ---------------------------------------------------------
    if score >= 8:
        card_class = "card-a-plus"
        grade = "8 / 10 — A+ SETUP"
    elif score >= 6:
        card_class = "card-b"
        grade = "6–7 / 10 — B SETUP (Scale-In Only)"
    else:
        card_class = "card-c"
        grade = "≤5 / 10 — C SETUP (NO TRADE)"

    st.markdown(f"""
        <div class="{card_class}">
            <h3 style="margin:0;">{asset_meta['name']} ({selected_asset})</h3>
            <p style="margin:0; font-weight:bold; font-size: 1.1rem;">RATING: {grade}</p>
        </div>
    """, unsafe_allow_html=True)

    # Key Metrics Display
    m1, m2, m3 = st.columns(3)
    m1.metric("Live Price", f"${current_price:,.2f}")
    m2.metric("1D Regime", regime_1d)
    m3.metric("Max Leverage", asset_meta["max_lev"])

    st.divider()

    # ---------------------------------------------------------
    # Trade Execution Parameters
    # ---------------------------------------------------------
    st.subheader("🎯 Trade Parameters")
    
    p1, p2 = st.columns(2)
    p1.write(f"**Entry Zone:** `${entry_price:,.2f}`")
    p1.write(f"**Hard Stop:** `${hard_stop:,.2f}` (-{stop_dist_pct*100:.2f}%)")
    p2.write(f"**Target 1 (2.0R):** `${target_1:,.2f}`")
    p2.write(f"**Target 2 (3.5R):** `${target_2:,.2f}`")

    # Position Sizing Breakdown
    st.info(f"""
    **Position Sizing (1% Account Risk):**
    * **Dollar Risk:** `${risk_amount:,.2f}`
    * **Position Value:** `${position_size_usd:,.2f}` ({units:.2f} units)
    * **Trailing Stop Transition:** Activate **{asset_meta['trail']}** trailing stop once price reaches Target 1 (`${target_1:,.2f}`). Move hard stop to breakeven (`${entry_price:,.2f}`).
    """)

    # Pros and Cons Breakdown
    st.markdown("**System Pros & Cons:**")
    st.write(f"✅ Aligns with {asset_meta['name']} Tier {asset_meta['tier']} risk parameters using isolated margin.")
    st.write(f"✅ Structural stop strictly enforces 1% max account risk.")
    st.write(f"⚠️ Requires manual confirmation of 4H higher-low candle close before entry execution.")

else:
    st.error("Market data unavailable for this ticker. Please select another asset.")
    
