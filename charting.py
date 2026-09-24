"""
TradingView charts embedded in the app.

TradingView publish free embeddable chart widgets. The Advanced Chart widget
includes their drawing tools — trend lines, Fibonacci retracements, boxes,
text — and all their indicators, which is far beyond anything worth rebuilding
here.

TWO LIMITS WORTH KNOWING BEFORE RELYING ON IT:

1. DRAWINGS DON'T SAVE. The embedded widget has no account attached, so trend
   lines and Fibs disappear when the page reloads or you switch tabs. Saved
   drawings need TradingView's own site with an account. Use this for looking,
   and TradingView proper for anything you want to keep.

2. IT IS A SEPARATE DATA SOURCE. The chart comes from TradingView's feed, not
   from the candles this app scanned. Prices should agree closely, but a level
   read off the chart and a level the scanner calculated can differ slightly,
   especially on thin markets.

Symbols are exchange-prefixed, e.g. BYBIT:BTCUSDT.P for a Bybit perpetual.
Where a mapping is uncertain the symbol box is editable, and TradingView's own
search inside the widget is the reliable way to find anything unusual.
"""

from typing import Dict, Optional

# Commodity perps on Bybit map to widely-followed spot symbols on TradingView,
# which have far more history and liquidity in the chart feed.
COMMODITY_CHART_SYMBOLS: Dict[str, str] = {
    "XAUUSD=X": "OANDA:XAUUSD",
    "XAGUSD=X": "OANDA:XAGUSD",
    "CL=F": "TVC:USOIL",
}

FX_CHART_PREFIX = "FX:"
STOCK_DEFAULT_EXCHANGE = ""       # TradingView resolves most US tickers unaided

INTERVALS = {"5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D", "1w": "W"}


def to_tradingview_symbol(ticker: str, market: str = "Crypto",
                          exchange: str = "BYBIT") -> str:
    """Best-guess TradingView symbol for one of our tickers.

    Crypto goes to the perpetual (.P) on the chosen exchange, since that is
    what gets traded — its price can differ slightly from spot.
    """
    if not ticker:
        return ""
    raw = ticker.strip().upper()

    if market == "Commodities":
        return COMMODITY_CHART_SYMBOLS.get(ticker, raw)

    if market == "FX":
        pair = raw.replace("=X", "")
        if pair == "JPY":
            pair = "USDJPY"
        elif pair in ("CAD", "CHF", "SGD"):
            pair = "USD" + pair
        if raw.startswith("DX-Y"):
            return "TVC:DXY"
        return FX_CHART_PREFIX + pair

    if market == "Stocks":
        return raw.replace("-", ".")          # BRK-B -> BRK.B

    # Crypto
    base = raw.replace("HL:", "").replace("-USD", "")
    if base.startswith("K") and base[1:].isupper() and len(base) > 2:
        base = "1000" + base[1:]              # kPEPE -> 1000PEPE on most venues
    if exchange.upper() == "HYPERLIQUID":
        return f"HYPERLIQUID:{base}USD.P"
    return f"{exchange.upper()}:{base}USDT.P"


def widget_html(symbol: str, interval: str = "4h", height: int = 620,
                theme: str = "dark", studies: Optional[list] = None) -> str:
    """The TradingView Advanced Chart widget, with drawing tools enabled.

    withdateranges and details are off to keep the chart itself as large as
    possible on a phone.
    """
    tv_interval = INTERVALS.get(interval, "240")
    studies = studies or []
    import json
    config = {
        "autosize": True,
        "symbol": symbol,
        "interval": tv_interval,
        "timezone": "Australia/Sydney",
        "theme": theme,
        "style": "1",
        "locale": "en",
        "enable_publishing": False,
        "allow_symbol_change": True,
        "hide_side_toolbar": False,      # this is the drawing-tools toolbar
        "withdateranges": False,
        "details": False,
        "studies": studies,
        "support_host": "https://www.tradingview.com",
    }
    return f"""
<div class="tradingview-widget-container" style="height:{height}px;width:100%">
  <div class="tradingview-widget-container__widget"
       style="height:calc(100% - 32px);width:100%"></div>
  <div class="tradingview-widget-copyright">
    <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
      <span style="color:#9FB0D0;font:12px sans-serif">Charts by TradingView</span></a>
  </div>
  <script type="text/javascript"
          src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js"
          async>
  {json.dumps(config)}
  </script>
</div>
"""
