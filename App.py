import streamlit as st
import yfinance as yf
import requests
import plotly.graph_objects as go
from datetime import datetime

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


OIL_TICKERS = {
    "WTI Crude Oil (Futures)": "CL=F",
    "Brent Crude Oil (Futures)": "BZ=F",
    "US Oil Fund ETF (USO)": "USO",
    "Brent Oil Fund ETF (BNO)": "BNO",
    "2x Leveraged Crude Oil ETF (UCO)": "UCO",
    "Inverse Crude Oil ETF (SCO)": "SCO",
    "Energy Select Sector ETF (XLE)": "XLE",
    "RBOB Gasoline (Futures)": "RB=F",
    "Heating Oil (Futures)": "HO=F",
    "Natural Gas (Futures)": "NG=F",
}

asset_type = st.sidebar.radio("Asset Type", ["Stock", "Crypto", "Oil & Energy"], horizontal=True)

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
elif asset_type == "Oil & Energy":
    oil_options = list(OIL_TICKERS.keys()) + ["Custom (type below)"]
    oil_choice = st.sidebar.selectbox("Select Oil/Energy Ticker", options=oil_options)
    if oil_choice == "Custom (type below)":
        ticker_symbol = st.sidebar.text_input(
            "Custom Oil/Energy Ticker (e.g. CL=F)", value="CL=F"
        ).upper().strip()
    else:
        ticker_symbol = OIL_TICKERS[oil_choice]
        st.sidebar.caption(f"Ticker: `{ticker_symbol}`")
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


@st.cache_data(ttl=900)
def get_trend_history(ticker, period):
    """Fetch historical daily closes for trend detection and charting."""
    try:
        data = yf.Ticker(ticker).history(period=period)
        return data
    except Exception:
        return None


def suggest_direction(hist_df):
    """Very simple momentum read: short-term vs longer-term moving average.
    Returns (direction, pct_change) or (None, None) if not enough data."""
    if hist_df is None or hist_df.empty or len(hist_df) < 5:
        return None, None
    closes = hist_df['Close']
    pct_change = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] * 100
    short_window = min(5, len(closes))
    long_window = min(20, len(closes))
    sma_short = closes.tail(short_window).mean()
    sma_long = closes.tail(long_window).mean()
    direction = "Short" if sma_short < sma_long else "Long"
    return direction, pct_change


trend_history_3mo = get_trend_history(ticker_symbol, "3mo") if ticker_symbol and data_fetched else None
suggested_direction, trend_pct_3mo = suggest_direction(trend_history_3mo)

# Reset the suggested direction default whenever the ticker changes —
# the user can still override it manually afterward.
if st.session_state.get("direction_ticker_ref") != ticker_symbol:
    st.session_state["trade_direction"] = suggested_direction or "Long"
    st.session_state["direction_ticker_ref"] = ticker_symbol


st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox(
    "🔄 Auto-refresh price",
    value=False,
    help="Reloads the page periodically so the live price and trailing stop "
         "keep updating without you having to touch anything."
)
if auto_refresh:
    refresh_choice = st.sidebar.selectbox(
        "Refresh every", ["15 seconds", "30 seconds", "60 seconds"], index=1
    )
    _interval_seconds = {"15 seconds": 15, "30 seconds": 30, "60 seconds": 60}[refresh_choice]
    st.markdown(
        f'<meta http-equiv="refresh" content="{_interval_seconds}">',
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# 2. MAIN PANEL - Inputs & Parameters
# ---------------------------------------------------------


def rate_trade_setup(entry, tp, sl, direction="Long"):
    """Score a trade setup 0-10 using reward:risk and stop-distance sanity.
    Returns None if the setup isn't valid for the given direction."""
    if direction == "Long":
        risk = entry - sl
        reward = tp - entry
    else:
        risk = sl - entry
        reward = entry - tp

    if entry <= 0 or risk <= 0 or reward <= 0:
        return None

    reward_risk = reward / risk
    stop_pct = (risk / entry) * 100

    # Reward:Risk component — out of 6 points
    if reward_risk >= 3:
        rr_score = 6.0
    elif reward_risk >= 2:
        rr_score = 4.5
    elif reward_risk >= 1.5:
        rr_score = 3.0
    elif reward_risk >= 1:
        rr_score = 1.5
    else:
        rr_score = 0.0

    # Stop-distance sanity component — out of 4 points.
    # Too tight (<0.5%) risks noise stop-outs; too wide (>15%) risks oversized loss.
    if 1.0 <= stop_pct <= 8.0:
        stop_score = 4.0
    elif 0.5 <= stop_pct < 1.0 or 8.0 < stop_pct <= 15.0:
        stop_score = 2.0
    else:
        stop_score = 0.0

    return rr_score + stop_score, reward_risk, stop_pct


def score_to_grade(score):
    if score >= 9:
        return "A+"
    if score >= 8:
        return "A"
    if score >= 7:
        return "B+"
    if score >= 6:
        return "B"
    if score >= 5:
        return "C"
    if score >= 3:
        return "D"
    return "F"


col_left, col_right = st.columns(2)

with col_left:
    st.subheader("⚙️ Trade Parameters")

    if suggested_direction and trend_pct_3mo is not None:
        trend_word = "Uptrend" if suggested_direction == "Long" else "Downtrend"
        st.caption(
            f"📊 3-month trend: {trend_pct_3mo:+.1f}% ({trend_word}) → "
            f"Suggests **{suggested_direction.upper()}**. Override below if you disagree."
        )

    direction = st.radio(
        "Direction",
        ["Long", "Short"],
        key="trade_direction",
        horizontal=True,
        help="Long = profit if price rises. Short = profit if price falls."
    )

    entry_price = st.number_input(
        "Entry Price ($)",
        value=float(current_price) if data_fetched else 150.0,
        min_value=0.01,
        step=1.0,
        format="%.2f"
    )

    # --- session_state guards so manual edits to TP/SL aren't silently
    # --- overwritten by the auto-calculated default on the next rerun.
    if (
        "tp_initialized" not in st.session_state
        or st.session_state.get("tp_entry_ref") != entry_price
        or st.session_state.get("tp_direction_ref") != direction
    ):
        if direction == "Long":
            st.session_state["take_profit_target"] = entry_price * 1.10
            st.session_state["stop_loss_price"] = entry_price * 0.95
        else:
            st.session_state["take_profit_target"] = entry_price * 0.90
            st.session_state["stop_loss_price"] = entry_price * 1.05
        st.session_state["tp_initialized"] = True
        st.session_state["tp_entry_ref"] = entry_price
        st.session_state["tp_direction_ref"] = direction

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

    st.markdown("##### 📋 Trade Setup Rating")
    rating = rate_trade_setup(entry_price, take_profit_target, stop_loss_price, direction)
    if rating is None:
        st.warning(
            f"⚠️ For a **{direction}**, Take-Profit and Stop-Loss must be on the correct "
            f"sides of Entry ({'TP above / SL below' if direction == 'Long' else 'TP below / SL above'})."
        )
    else:
        setup_score, reward_risk_setup, stop_pct_setup = rating
        setup_grade = score_to_grade(setup_score)
        rc1, rc2 = st.columns([1, 2])
        rc1.metric("Rating", f"{setup_score:.1f}/10", setup_grade)
        rc2.caption(
            f"Reward:Risk = {reward_risk_setup:.2f} : 1  \n"
            f"Stop distance = {stop_pct_setup:.1f}% from entry"
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
risk_per_share = (entry_price - stop_loss_price) if direction == "Long" else (stop_loss_price - entry_price)
position_shares = 0
total_cost = 0.0
potential_profit = 0.0
reward_risk_ratio = 0.0
sizing_valid = entry_price > 0 and risk_per_share > 0

if sizing_valid:
    position_shares = int(risk_amount / risk_per_share)
    total_cost = position_shares * entry_price
    if direction == "Long":
        potential_profit = position_shares * (take_profit_target - entry_price)
        reward_risk_ratio = (take_profit_target - entry_price) / risk_per_share
    else:
        potential_profit = position_shares * (entry_price - take_profit_target)
        reward_risk_ratio = (entry_price - take_profit_target) / risk_per_share


# ---------------------------------------------------------
# 4. TABS - Dashboard view / Trailing Stop Manager
# ---------------------------------------------------------
tab_dashboard, tab_trailing = st.tabs(["🎯 Dashboard", "🔒 Trailing Stop Manager"])

with tab_trailing:
    st.subheader("🔒 Trailing Stop Manager")
    st.caption(
        "Once your trade moves into profit, this tracks a trailing stop that only ever "
        "moves in your favor (up for Longs, down for Shorts — never back the other way) "
        "and shows how much profit is currently locked in. It re-evaluates on every "
        "rerun — turn on auto-refresh in the sidebar to have it keep adjusting on its "
        "own as the price moves, without you touching anything."
    )

    st.markdown("#### 📌 Active Trade Details")
    st.caption(
        "This is independent from the Trade Parameters calculator above — set these once "
        "to the trade you actually entered, and they won't get overwritten if you tweak "
        "the planning calculator or reload the ticker's live price later."
    )

    if "active_entry_price" not in st.session_state:
        st.session_state["active_entry_price"] = entry_price if entry_price > 0 else 100.0
    if "active_original_stop" not in st.session_state:
        st.session_state["active_original_stop"] = stop_loss_price if stop_loss_price > 0 else 95.0
    if "active_shares" not in st.session_state:
        st.session_state["active_shares"] = position_shares if sizing_valid else 0
    if "active_direction" not in st.session_state:
        st.session_state["active_direction"] = direction

    active_direction = st.radio(
        "Trade Direction",
        ["Long", "Short"],
        key="active_direction",
        horizontal=True,
        help="Long = you profit as price rises. Short = you profit as price falls."
    )

    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        trade_entry_price = st.number_input(
            "Your Entry Price ($)",
            min_value=0.01,
            step=1.0,
            format="%.2f",
            key="active_entry_price",
            help="The price you actually got filled at when you entered this trade."
        )
    with ac2:
        trade_original_stop = st.number_input(
            "Original Stop-Loss ($)",
            min_value=0.0,
            step=1.0,
            format="%.2f",
            key="active_original_stop",
            help="Your initial protective stop before any trailing adjustments."
        )
    with ac3:
        trade_shares = st.number_input(
            "Shares / Units Held",
            min_value=0,
            step=1,
            key="active_shares",
            help="How many shares/coins you're actually holding in this trade."
        )

    def _load_active_trade_from_calculator():
        st.session_state["active_entry_price"] = entry_price if entry_price > 0 else 100.0
        st.session_state["active_original_stop"] = stop_loss_price if stop_loss_price > 0 else 95.0
        st.session_state["active_shares"] = position_shares if sizing_valid else 0
        st.session_state["active_direction"] = direction

    st.button(
        "↺ Load these from the Trade Parameters calculator above",
        on_click=_load_active_trade_from_calculator
    )

    st.markdown("---")

# ---------------------------------------------------------
# 3b. TRAILING STOP STATE — driven by the Active Trade Details above,
#     NOT by the planning calculator's entry/stop-loss. Computed once
#     here (outside any tab) so it stays current everywhere, including
#     the persistent sidebar summary below.
# ---------------------------------------------------------
trade_active = (
    data_fetched and current_price > 0 and trade_entry_price > 0
    and (
        (active_direction == "Long" and current_price > trade_entry_price)
        or (active_direction == "Short" and current_price < trade_entry_price)
    )
)
active_stop = None
locked_profit_per_share = 0.0
locked_in = False
locked_profit_total = 0.0

if "stop_history" not in st.session_state:
    st.session_state["stop_history"] = []
if "trailing_pct_setting" not in st.session_state:
    st.session_state["trailing_pct_setting"] = 5.0

# Reset the trailing baseline (and history) whenever the ticker, direction, or
# the entered trade's entry price changes — i.e. a genuinely different trade.
needs_reset = (
    "trailing_stop_level" not in st.session_state
    or st.session_state.get("trailing_stop_ticker") != ticker_symbol
    or st.session_state.get("trailing_stop_entry_ref") != trade_entry_price
    or st.session_state.get("trailing_stop_direction_ref") != active_direction
)
if needs_reset:
    st.session_state["trailing_stop_level"] = trade_original_stop
    st.session_state["trailing_stop_ticker"] = ticker_symbol
    st.session_state["trailing_stop_entry_ref"] = trade_entry_price
    st.session_state["trailing_stop_direction_ref"] = active_direction
    st.session_state["stop_history"] = []

if trade_entry_price > 0 and data_fetched and current_price > 0:
    if not trade_active:
        # Not yet in profit — keep the trailing level pinned to the original stop.
        st.session_state["trailing_stop_level"] = trade_original_stop
    else:
        trailing_pct = st.session_state.get("trailing_pct_setting", 5.0)
        previous_stop = st.session_state["trailing_stop_level"]

        if active_direction == "Long":
            proposed_stop = current_price * (1 - trailing_pct / 100)
            ratchets = proposed_stop > previous_stop
        else:
            proposed_stop = current_price * (1 + trailing_pct / 100)
            ratchets = proposed_stop < previous_stop

        if ratchets:
            st.session_state["trailing_stop_level"] = proposed_stop
            locked_at_move = (
                max(proposed_stop - trade_entry_price, 0) if active_direction == "Long"
                else max(trade_entry_price - proposed_stop, 0)
            )
            # Log every time the stop actually ratchets in the favorable direction.
            st.session_state["stop_history"].append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Price": f"${current_price:.2f}",
                "New Stop": f"${proposed_stop:.2f}",
                "Locked Profit/Share": f"${locked_at_move:+.2f}",
            })

        active_stop = st.session_state["trailing_stop_level"]
        if active_direction == "Long":
            locked_profit_per_share = active_stop - trade_entry_price
        else:
            locked_profit_per_share = trade_entry_price - active_stop
        locked_in = locked_profit_per_share > 0
        locked_profit_total = locked_profit_per_share * trade_shares if locked_in else 0.0

# Persistent sidebar summary — always visible, regardless of which tab is open.
if trade_entry_price > 0 and data_fetched and current_price > 0:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔒 Trailing Stop Status")
    if not trade_active:
        st.sidebar.caption(
            f"Not yet in profit (entry ${trade_entry_price:.2f}, {active_direction}). "
            f"Trailing starts once price moves in your favor."
        )
    else:
        st.sidebar.metric("Recommended Stop", f"${active_stop:.2f}")
        if locked_in:
            st.sidebar.success(f"✅ Locked in: ${locked_profit_total:,.2f} ({trade_shares:,} shares)")
        else:
            st.sidebar.info("In profit, but stop hasn't cleared entry yet — no profit locked in.")

with tab_dashboard:
    st.subheader("🎯 Target & Alert Status")

    if entry_price <= 0:
        st.warning("Entry price must be greater than $0.")
    elif data_fetched and current_price > 0:
        if direction == "Long":
            tp_hit = current_price >= take_profit_target
            sl_hit = current_price <= stop_loss_price
        else:
            tp_hit = current_price <= take_profit_target
            sl_hit = current_price >= stop_loss_price

        if tp_hit:
            st.success(
                f"🚨 **TAKE-PROFIT TRIGGERED!** Current price (${current_price:.2f}) "
                f"has reached or passed your {direction} target (${take_profit_target:.2f})."
            )
        elif sl_hit:
            st.error(
                f"⚠️ **STOP-LOSS TRIGGERED!** Current price (${current_price:.2f}) "
                f"has hit your {direction} stop loss (${stop_loss_price:.2f})."
            )
        else:
            dist_tp = abs(take_profit_target - current_price) / current_price * 100
            dist_sl = abs(current_price - stop_loss_price) / current_price * 100
            st.info(
                f"📊 **In Range ({direction}):** Current price is **${current_price:.2f}** | "
                f"Take-Profit is **{dist_tp:.2f}%** away | Stop-Loss is **{dist_sl:.2f}%** away."
            )
    else:
        st.warning("Enter a valid ticker in the sidebar to run live status alerts.")

    st.markdown("---")
    st.subheader("🧮 Calculated Position Sizing & P&L")

    if entry_price <= 0:
        st.error("Entry price must be greater than $0 to calculate position size.")
    elif risk_per_share <= 0:
        st.error(
            f"For a {direction}, your Stop-Loss must be "
            f"{'below' if direction == 'Long' else 'above'} the Entry Price to size the position."
        )
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

    st.markdown("---")
    st.subheader("📈 Price Trend")
    chart_period_choice = st.selectbox(
        "Chart Timeframe",
        ["1 Month", "3 Months", "6 Months", "1 Year"],
        index=1,
        key="chart_period_choice"
    )
    _period_map = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y"}
    chart_data = get_trend_history(ticker_symbol, _period_map[chart_period_choice]) if ticker_symbol else None

    if chart_data is None or chart_data.empty:
        st.info("No historical data available to chart for this ticker.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_data.index, y=chart_data['Close'],
            mode='lines', name='Close Price', line=dict(color='#1f77b4', width=2)
        ))

        if entry_price > 0:
            fig.add_hline(
                y=entry_price, line_dash="dash", line_color="gray",
                annotation_text="Entry", annotation_position="top left"
            )
        if take_profit_target > 0:
            fig.add_hline(
                y=take_profit_target, line_dash="dash", line_color="green",
                annotation_text="Take-Profit", annotation_position="top left"
            )
        if stop_loss_price > 0:
            fig.add_hline(
                y=stop_loss_price, line_dash="dash", line_color="red",
                annotation_text="Stop-Loss", annotation_position="bottom left"
            )
        if trade_active and active_stop:
            fig.add_hline(
                y=active_stop, line_dash="dot", line_color="orange",
                annotation_text="Trailing Stop", annotation_position="bottom left"
            )

        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title=None,
            yaxis_title="Price ($)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Gray = Entry (calculator) · Green = Take-Profit · Red = Stop-Loss · "
            "Orange dotted = current trailing stop (if a trade is active)."
        )

with tab_trailing:
    if trade_entry_price <= 0:
        st.warning("Set a valid entry price above to use the trailing stop manager.")
    elif not data_fetched or current_price <= 0:
        st.info("Enter a valid ticker in the sidebar to activate trailing stop management.")
    elif not trade_active:
        st.warning(
            f"Trade not yet active. Current price (${current_price:.2f}) hasn't moved in "
            f"your favor relative to your {active_direction} entry (${trade_entry_price:.2f}). "
            f"Trailing stop management kicks in once the trade is in profit."
        )
    else:
        trailing_pct = st.slider(
            "Trailing Stop Distance (%)",
            min_value=0.5,
            max_value=20.0,
            step=0.5,
            help="How far from the current price to trail your stop-loss.",
            key="trailing_pct_setting"
        )

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
                f"in profit across {trade_shares:,} shares."
            )
        else:
            st.info(
                f"Trailing stop is currently at **${active_stop:.2f}**, still on the losing "
                f"side of your entry price (${trade_entry_price:.2f}) — no profit locked in "
                f"yet, but risk is tightening as price moves toward breakeven."
            )

        st.caption(
            "This tool only *recommends* where to move your stop — you still need to "
            "update the actual stop-loss order with your broker."
        )

        history = [h for h in st.session_state["stop_history"]]
        if history:
            with st.expander(f"📜 Stop Adjustment History ({len(history)} moves)", expanded=False):
                st.table(list(reversed(history)))

        if st.button("Reset Trailing Stop to Original Stop-Loss"):
            st.session_state["trailing_stop_level"] = trade_original_stop
            st.session_state["stop_history"] = []
            st.rerun()
