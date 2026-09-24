import unittest
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from charting import to_tradingview_symbol, widget_html, INTERVALS


class TestSymbolMapping(unittest.TestCase):
    def test_crypto_maps_to_a_bybit_perpetual(self):
        self.assertEqual(to_tradingview_symbol("BTC-USD"), "BYBIT:BTCUSDT.P")

    def test_hyperliquid_ticker_prefix_is_stripped(self):
        self.assertEqual(to_tradingview_symbol("HL:SOL"), "BYBIT:SOLUSDT.P")

    def test_bundled_k_coin_becomes_the_1000_form(self):
        self.assertEqual(to_tradingview_symbol("HL:kPEPE"), "BYBIT:1000PEPEUSDT.P")

    def test_exchange_can_be_changed(self):
        self.assertEqual(to_tradingview_symbol("BTC-USD", exchange="HYPERLIQUID"),
                         "HYPERLIQUID:BTCUSD.P")

    def test_commodities_map_to_spot_symbols(self):
        self.assertEqual(to_tradingview_symbol("XAUUSD=X", "Commodities"), "OANDA:XAUUSD")
        self.assertEqual(to_tradingview_symbol("CL=F", "Commodities"), "TVC:USOIL")

    def test_fx_pairs(self):
        self.assertEqual(to_tradingview_symbol("EURUSD=X", "FX"), "FX:EURUSD")
        self.assertEqual(to_tradingview_symbol("JPY=X", "FX"), "FX:USDJPY")
        self.assertEqual(to_tradingview_symbol("CAD=X", "FX"), "FX:USDCAD")

    def test_dollar_index(self):
        self.assertEqual(to_tradingview_symbol("DX-Y.NYB", "FX"), "TVC:DXY")

    def test_stocks_pass_through_with_dot_notation(self):
        self.assertEqual(to_tradingview_symbol("AAPL", "Stocks"), "AAPL")
        self.assertEqual(to_tradingview_symbol("BRK-B", "Stocks"), "BRK.B")

    def test_empty_input(self):
        self.assertEqual(to_tradingview_symbol(""), "")


class TestWidget(unittest.TestCase):
    def _config(self, html):
        match = re.search(r"async>\s*(\{.*?\})\s*</script>", html, re.S)
        return json.loads(match.group(1))

    def test_drawing_tools_are_enabled(self):
        cfg = self._config(widget_html("BYBIT:BTCUSDT.P"))
        self.assertFalse(cfg["hide_side_toolbar"])     # the drawing toolbar

    def test_symbol_and_interval_passed_through(self):
        cfg = self._config(widget_html("BYBIT:ETHUSDT.P", "1h"))
        self.assertEqual(cfg["symbol"], "BYBIT:ETHUSDT.P")
        self.assertEqual(cfg["interval"], "60")

    def test_four_hour_is_the_default_interval(self):
        self.assertEqual(self._config(widget_html("X"))["interval"], "240")

    def test_unknown_interval_falls_back_to_four_hour(self):
        self.assertEqual(self._config(widget_html("X", "3y"))["interval"], "240")

    def test_all_intervals_map_to_something(self):
        for name in INTERVALS:
            self.assertEqual(self._config(widget_html("X", name))["interval"],
                             INTERVALS[name])

    def test_timezone_is_the_traders(self):
        self.assertEqual(self._config(widget_html("X"))["timezone"], "Australia/Sydney")

    def test_attribution_is_present(self):
        self.assertIn("tradingview.com", widget_html("X"))
        self.assertIn("Charts by TradingView", widget_html("X"))

    def test_symbol_can_be_changed_inside_the_widget(self):
        self.assertTrue(self._config(widget_html("X"))["allow_symbol_change"])
