import streamlit as st
import yfinance as yf
import requests

# Page Configuration
st.set_page_config(
    page_title="Trading Dashboard & Risk Calculator",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Mobile Trading Dashboard")
st.caption(
    "⚠️ For educational/planning purposes only — not financial advice. "
    "Prices update only when this app reruns; there is no background monitoring "
    "or push alerting while the app is closed."
)
st.markdown("---")

# ---------------------------------------------------------
# 1. SIDEBAR - Dynamic Ticker Selection & Fetch
# ---------------------------------------------------------
st.sidebar.header("Ticker Settings")


@st.cache_data(ttl=3600)
def get_top_100_cryptos():
    """Top 100 cryptos by market cap, mapped to Yahoo Finance '-USD' tickers."""
    try:
        response = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 100,
                "page": 1,
                "sparkline": "false",
            },
            timeout=10,
        )
        response.raise_for_status()
        coins = response.json()
        return {
            f"{coin['name']} ({coin['symbol'].upper()})": f"{coin['symbol'].upper()}-USD"
            for coin in coins
        }
    except Exception:
        return {}


asset_type = st.sidebar.radio("Asset Type", ["Stock", "Crypto"], horizontal=True)

if asset_type == "Crypto":
    top_cryptos = get_top_100_cryptos()
    if top_cryptos:
        options = list(top_cryptos.keys()) + ["Custom (type below)"]
        # Default the dropdown to Solana if it's present in the live list.
        default_index = next(
            (i for i, name in enumerate(options) if name.startswith("Solana")), 0
        )
        crypto_choice = st.sidebar.selectbox(
            "Select Crypto (Top 100 by market cap)",
            options=options,
            index=default_index,
        )
        if crypto_choice == "Custom (type below)":
            ticker_symbol = st.sidebar.text_input(
                "Custom Crypto Ticker (e.g. DOGE-USD)", value="SOL-USD"
            ).upper().strip()
        else:
            ticker_symbol = top_cryptos[crypto_choice]
            st.sidebar.caption(f"Ticker: `{ticker_symbol}`")
    else:
        st.sidebar.warning(
            "Couldn't load the live top-100 list (CoinGecko unavailable) — "
            "enter a crypto ticker manually."
        )
        ticker_symbol = st.sidebar.text_input(
            "Crypto Ticker", value="SOL-USD"
        ).upper().strip()
else:
    ticker_symbol = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()

current_price = 0.0
data_fetched = False

if ticker_symbol:
    try:
        stock = yf.Ticker(ticker_symbol)
        # Fetch 2 days so we can compute a real "previous close" delta.
        history = stock.history(period="2d")

        if not history.empty:
            current_price = float(history['Close'].iloc[-1])
            if len(history) > 1:
                prev_close = float(history['Close'].iloc[-2])
            else:
                # Only one row of data available (e.g. brand-new listing,
                # or market hasn't produced a prior bar yet) — fall back to
                # today's open rather than silently showing a $0.00 delta.
                prev_close = float(history['Open'].iloc[-1])
            delta_price = current_price - prev_close

            st.sidebar.metric(
                label=f"Live Price ({ticker_symbol})",
                value=f"${current_price:.2f}",
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
        min_value=0.01,
        step=1.0,
        format="%.2f"
    )

    # --- session_state guards so manual edits to TP/SL aren't silently
    # --- overwritten by the auto-calculated default on the next rerun.
    if "tp_initialized" not in st.session_state or st.session_state.get("tp_entry_ref") != entry_price:
        st.session_state["take_profit_target"] = entry_price * 1.10
        st.session_state["stop_loss_price"] = entry_price * 0.95
        st.session_state["tp_initialized"] = True
        st.session_state["tp_entry_ref"] = entry_price

    take_profit_target = st.number_input(
        "Take-Profit Target ($)",
        step=1.0,
        format="%.2f",
        key="take_profit_target"
    )
    stop_loss_price = st.number_input(
        "Stop-Loss Price ($)",
        step=1.0,
        format="%.2f",
        key="stop_loss_price"
    )

with col_right:
    st.subheader("💼 Account & Risk Limits")
    account_balance = st.number_input(
        "Total Portfolio Balance ($)",
        value=10000.0,
        min_value=0.0,
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
# 3. POSITION SIZING & CALCULATIONS (computed up front so both
#    tabs below can use the results)
# ---------------------------------------------------------
risk_amount = account_balance * (risk_percentage / 100)
risk_per_share = entry_price - stop_loss_price
position_shares = 0
total_cost = 0.0
potential_profit = 0.0
reward_risk_ratio = 0.0
sizing_valid = entry_price > 0 and risk_per_share > 0

if sizing_valid:
    position_shares = int(risk_amount / risk_per_share)
    total_cost = position_shares * entry_price
    potential_profit = position_shares * (take_profit_target - entry_price)
    reward_risk_ratio = (take_profit_target - entry_price) / risk_per_share

# ---------------------------------------------------------
# 4. TABS - Dashboard view / Trailing Stop Manager
# ---------------------------------------------------------
tab_dashboard, tab_trailing = st.tabs(["🎯 Dashboard", "🔒 Trailing Stop Manager"])

with tab_dashboard:
    st.subheader("🎯 Target & Alert Status")

    if entry_price <= 0:
        st.warning("Entry price must be greater than $0.")
    elif data_fetched and current_price > 0:
        if current_price >= take_profit_target:
            st.success(
                f"🚨 **TAKE-PROFIT TRIGGERED!** Current price (${current_price:.2f}) "
                f"has reached or passed your target (${take_profit_target:.2f})."
            )
        elif current_price <= stop_loss_price:
            st.error(
                f"⚠️ **STOP-LOSS TRIGGERED!** Current price (${current_price:.2f}) "
                f"has dropped to or below your stop loss (${stop_loss_price:.2f})."
            )
        else:
            dist_tp = ((take_profit_target - current_price) / current_price) * 100
            dist_sl = ((current_price - stop_loss_price) / current_price) * 100
            st.info(
                f"📊 **In Range:** Current price is **${current_price:.2f}** | "
                f"Take-Profit is **{dist_tp:.2f}%** away | Stop-Loss is **{dist_sl:.2f}%** below."
            )
    else:
        st.warning("Enter a valid ticker in the sidebar to run live status alerts.")

    st.markdown("---")
    st.subheader("🧮 Calculated Position Sizing & P&L")

    if entry_price <= 0:
        st.error("Entry price must be greater than $0 to calculate position size.")
    elif risk_per_share <= 0:
        st.error("Stop-Loss price must be set below the Entry Price to calculate position size.")
    elif position_shares == 0:
        st.warning(
            f"⚠️ Your risk budget (${risk_amount:,.2f}) divided by the per-share risk "
            f"(${risk_per_share:.2f}) rounds down to **0 shares**. This trade isn't sized-in "
            f"under your current risk %/stop distance — widen your risk % or tighten the stop."
        )
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(label="Suggested Shares", value=f"{position_shares:,} qty")
        m2.metric(label="Capital Required", value=f"${total_cost:,.2f}")
        m3.metric(label="Max Loss Risk", value=f"${risk_amount:,.2f}")
        m4.metric(label="Potential Gain", value=f"${potential_profit:,.2f}", delta=f"R:R {reward_risk_ratio:.2f}")

        if total_cost > account_balance:
            st.warning(
                f"⚠️ Capital required (${total_cost:,.2f}) exceeds total account balance "
                f"(${account_balance:,.2f}). Consider adjusting leverage or lowering risk."
            )

with tab_trailing:
    st.subheader("🔒 Trailing Stop Manager")
    st.caption(
        "Once your trade moves into profit, this tracks a trailing stop that only ever "
        "moves up (never down) and shows how much profit is currently locked in."
    )

    if entry_price <= 0:
        st.warning("Set a valid entry price to use the trailing stop manager.")
    elif not data_fetched or current_price <= 0:
        st.info("Enter a valid ticker in the sidebar to activate trailing stop management.")
    else:
        trade_active = current_price > entry_price

        if not trade_active:
            st.warning(
                f"Trade not yet active. Current price (${current_price:.2f}) is at or below "
                f"your entry price (${entry_price:.2f}). Trailing stop management kicks in "
                f"once the trade is in profit."
            )
            # Keep the stored trailing level in sync with the manual stop-loss
            # until the trade actually goes active.
            st.session_state["trailing_stop_level"] = stop_loss_price
            st.session_state["trailing_stop_ticker"] = ticker_symbol
        else:
            trailing_pct = st.slider(
                "Trailing Stop Distance (%)",
                min_value=0.5,
                max_value=20.0,
                value=5.0,
                step=0.5,
                help="How far below the current price to trail your stop-loss."
            )

            # Initialize (or reset on ticker change) the ratcheting stop level.
            if (
                "trailing_stop_level" not in st.session_state
                or st.session_state.get("trailing_stop_ticker") != ticker_symbol
            ):
                st.session_state["trailing_stop_level"] = stop_loss_price
                st.session_state["trailing_stop_ticker"] = ticker_symbol

            proposed_stop = current_price * (1 - trailing_pct / 100)
            # Ratchet mechanic: the stop only ever moves up, never down,
            # even if price pulls back and trailing_pct implies a lower level.
            if proposed_stop > st.session_state["trailing_stop_level"]:
                st.session_state["trailing_stop_level"] = proposed_stop

            active_stop = st.session_state["trailing_stop_level"]
            locked_profit_per_share = active_stop - entry_price
            locked_in = locked_profit_per_share > 0
            shares_for_calc = position_shares if sizing_valid else 0
            locked_profit_total = locked_profit_per_share * shares_for_calc if locked_in else 0.0

            t1, t2, t3 = st.columns(3)
            t1.metric("Current Price", f"${current_price:.2f}")
            t2.metric("Recommended Stop", f"${active_stop:.2f}")
            t3.metric(
                "Locked-In Profit/Share",
                f"${locked_profit_per_share:+.2f}",
                delta="Profit secured" if locked_in else "Not yet in profit"
            )

            if locked_in:
                st.success(
                    f"✅ **Move your stop-loss to ${active_stop:.2f}.** If price reverses and "
                    f"hits this level, you lock in at least **${locked_profit_total:,.2f}** "
                    f"in profit across {shares_for_calc:,} shares."
                )
            else:
                st.info(
                    f"Trailing stop is currently at **${active_stop:.2f}**, still below your "
                    f"entry price (${entry_price:.2f}) — no profit locked in yet, but risk is "
                    f"tightening as price rises toward breakeven."
                )

            st.caption(
                "This tool only *recommends* where to move your stop — you still need to "
                "update the actual stop-loss order with your broker."
            )

            if st.button("Reset Trailing Stop to Manual Stop-Loss"):
                st.session_state["trailing_stop_level"] = stop_loss_price
                st.rerun()
