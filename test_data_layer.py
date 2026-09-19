import unittest
import sys, os
from datetime import datetime, timezone, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from data_layer import fetch_quote, classify_freshness, effective_age, DataStatus


class FakeTicker:
    def __init__(self, hist_df=None, raise_exc=None):
        self._hist_df = hist_df
        self._raise_exc = raise_exc

    def history(self, period="2d", interval=None):
        if self._raise_exc:
            raise self._raise_exc
        return self._hist_df


class FakeYFModule:
    """Stands in for the yfinance module so tests don't need network access."""
    def __init__(self, ticker_obj):
        self._ticker_obj = ticker_obj

    def Ticker(self, symbol, session=None):
        return self._ticker_obj


def make_hist(price, bar_time_utc):
    return pd.DataFrame(
        {"Close": [price], "Open": [price], "High": [price], "Low": [price]},
        index=pd.DatetimeIndex([bar_time_utc]),
    )


class TestClassifyFreshness(unittest.TestCase):
    def test_recent_crypto_is_live(self):
        now = datetime.now(timezone.utc)
        bar = now - timedelta(seconds=10)
        status = classify_freshness(bar, now, "crypto", live_threshold_seconds=60, stale_threshold_seconds=900)
        self.assertEqual(status, DataStatus.LIVE)

    def test_stock_is_always_delayed_even_if_recent(self):
        now = datetime.now(timezone.utc)
        bar = now - timedelta(seconds=5)
        status = classify_freshness(bar, now, "stock", live_threshold_seconds=60, stale_threshold_seconds=900)
        self.assertEqual(status, DataStatus.DELAYED)

    def test_old_bar_is_stale(self):
        now = datetime.now(timezone.utc)
        bar = now - timedelta(seconds=2000)
        status = classify_freshness(bar, now, "crypto", live_threshold_seconds=60, stale_threshold_seconds=900)
        self.assertEqual(status, DataStatus.STALE)

    def test_moderately_old_crypto_is_delayed_not_live(self):
        now = datetime.now(timezone.utc)
        bar = now - timedelta(seconds=300)
        status = classify_freshness(bar, now, "crypto", live_threshold_seconds=60, stale_threshold_seconds=900)
        self.assertEqual(status, DataStatus.DELAYED)


class TestIntervalAwareFreshness(unittest.TestCase):
    """Regression tests for the bug where a DAILY bar was flagged STALE
    purely because it is timestamped at its open (e.g. 00:00 UTC), making
    it look hours old the moment it is created."""

    def test_forming_daily_bar_is_not_stale(self):
        now = datetime.now(timezone.utc)
        bar = now - timedelta(hours=1)   # daily bar opened an hour ago
        status = classify_freshness(bar, now, "crypto",
                                     live_threshold_seconds=60,
                                     stale_threshold_seconds=900,
                                     bar_interval_seconds=86400,
                                     is_daily_fallback=True)
        self.assertEqual(status, DataStatus.DELAYED)  # usable, just not LIVE

    def test_daily_bar_more_than_a_day_past_close_is_stale(self):
        now = datetime.now(timezone.utc)
        bar = now - timedelta(days=3)
        status = classify_freshness(bar, now, "crypto",
                                     live_threshold_seconds=60,
                                     stale_threshold_seconds=900,
                                     bar_interval_seconds=86400,
                                     is_daily_fallback=True)
        self.assertEqual(status, DataStatus.STALE)

    def test_forming_5m_bar_is_live(self):
        now = datetime.now(timezone.utc)
        bar = now - timedelta(minutes=3)  # 5m bar still forming
        status = classify_freshness(bar, now, "crypto",
                                     live_threshold_seconds=60,
                                     stale_threshold_seconds=900,
                                     bar_interval_seconds=300)
        self.assertEqual(status, DataStatus.LIVE)

    def test_effective_age_floors_at_zero_for_forming_bar(self):
        now = datetime.now(timezone.utc)
        bar = now - timedelta(minutes=2)
        self.assertEqual(effective_age(bar, now, 300), 0.0)

    def test_effective_age_measures_from_bar_close(self):
        now = datetime.now(timezone.utc)
        bar = now - timedelta(minutes=20)  # 5m bar closed 15 min ago
        self.assertAlmostEqual(effective_age(bar, now, 300), 900, delta=2)

    def test_daily_fallback_never_reports_live(self):
        now = datetime.now(timezone.utc)
        bar = now - timedelta(seconds=5)
        status = classify_freshness(bar, now, "crypto",
                                     live_threshold_seconds=60,
                                     stale_threshold_seconds=900,
                                     bar_interval_seconds=86400,
                                     is_daily_fallback=True)
        self.assertNotEqual(status, DataStatus.LIVE)


class TestNaNHandling(unittest.TestCase):
    """Regression tests for the '$nan' bug: NaN fails every comparison, so
    a naive `price <= 0` check let it through and the UI rendered '$nan'."""

    def test_trailing_nan_row_is_dropped_and_real_price_used(self):
        now = datetime.now(timezone.utc)
        df = pd.DataFrame(
            {"Close": [101.5, float("nan")], "Open": [101.0, float("nan")],
             "High": [102.0, float("nan")], "Low": [100.0, float("nan")]},
            index=pd.DatetimeIndex([now - timedelta(minutes=8), now - timedelta(minutes=3)]),
        )
        yf_mod = FakeYFModule(FakeTicker(hist_df=df))
        quote = fetch_quote("AAPL", "stock", yf_mod, max_retries=0)
        self.assertIsNotNone(quote.price)
        self.assertAlmostEqual(quote.price, 101.5)
        self.assertNotEqual(quote.status, DataStatus.UNAVAILABLE)

    def test_all_nan_history_is_unavailable_not_nan_price(self):
        now = datetime.now(timezone.utc)
        df = pd.DataFrame(
            {"Close": [float("nan")], "Open": [float("nan")],
             "High": [float("nan")], "Low": [float("nan")]},
            index=pd.DatetimeIndex([now]),
        )
        yf_mod = FakeYFModule(FakeTicker(hist_df=df))
        quote = fetch_quote("AAPL", "stock", yf_mod, max_retries=0)
        self.assertEqual(quote.status, DataStatus.UNAVAILABLE)
        self.assertIsNone(quote.price)

    def test_infinite_price_rejected(self):
        now = datetime.now(timezone.utc)
        df = make_hist(float("inf"), now)
        yf_mod = FakeYFModule(FakeTicker(hist_df=df))
        quote = fetch_quote("WEIRD", "crypto", yf_mod, max_retries=0)
        self.assertEqual(quote.status, DataStatus.UNAVAILABLE)
        self.assertIsNone(quote.price)


class TestFetchQuote(unittest.TestCase):
    def test_successful_fetch_returns_live_status(self):
        now = datetime.now(timezone.utc)
        hist = make_hist(123.45, now - timedelta(seconds=5))
        yf_mod = FakeYFModule(FakeTicker(hist_df=hist))
        quote = fetch_quote("FAKE-USD", "crypto", yf_mod, live_threshold_seconds=60,
                             stale_threshold_seconds=900, max_retries=0)
        self.assertEqual(quote.status, DataStatus.LIVE)
        self.assertAlmostEqual(quote.price, 123.45)
        self.assertIsNone(quote.error)

    def test_empty_history_is_unavailable_never_faked(self):
        yf_mod = FakeYFModule(FakeTicker(hist_df=pd.DataFrame()))
        quote = fetch_quote("NOPE", "stock", yf_mod, max_retries=0)
        self.assertEqual(quote.status, DataStatus.UNAVAILABLE)
        self.assertIsNone(quote.price)
        self.assertIsNotNone(quote.error)

    def test_exception_is_surfaced_not_swallowed_into_a_price(self):
        yf_mod = FakeYFModule(FakeTicker(raise_exc=ConnectionError("429 Too Many Requests")))
        quote = fetch_quote("AAPL", "stock", yf_mod, max_retries=0)
        self.assertEqual(quote.status, DataStatus.UNAVAILABLE)
        self.assertIn("429", quote.error)

    def test_negative_price_rejected(self):
        now = datetime.now(timezone.utc)
        hist = make_hist(-5.0, now)
        yf_mod = FakeYFModule(FakeTicker(hist_df=hist))
        quote = fetch_quote("WEIRD", "crypto", yf_mod, max_retries=0)
        self.assertEqual(quote.status, DataStatus.UNAVAILABLE)

    def test_never_substitutes_another_tickers_symbol(self):
        # The returned quote always echoes back the ticker it was asked for.
        now = datetime.now(timezone.utc)
        hist = make_hist(1.5, now)
        yf_mod = FakeYFModule(FakeTicker(hist_df=hist))
        quote = fetch_quote("SUI-USD", "crypto", yf_mod, max_retries=0)
        self.assertEqual(quote.ticker, "SUI-USD")

    def test_retries_on_transient_failure_then_succeeds(self):
        # First call raises, second call (retry) succeeds.
        now = datetime.now(timezone.utc)
        hist = make_hist(2.5, now)
        call_count = {"n": 0}

        class FlakyTicker:
            def history(self, period="2d", interval=None):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise ConnectionError("429 Too Many Requests")
                return hist

        yf_mod = FakeYFModule(FlakyTicker())
        quote = fetch_quote("FLAKY-USD", "crypto", yf_mod, max_retries=2, retry_backoff_seconds=0.01)
        self.assertEqual(quote.status, DataStatus.LIVE)
        self.assertEqual(quote.attempts, 2)
        self.assertIsNone(quote.error)

    def test_exhausts_retries_and_reports_unavailable(self):
        yf_mod = FakeYFModule(FakeTicker(raise_exc=ConnectionError("429 Too Many Requests")))
        quote = fetch_quote("STILLDOWN", "crypto", yf_mod, max_retries=2, retry_backoff_seconds=0.01)
        self.assertEqual(quote.status, DataStatus.UNAVAILABLE)
        # 3 interval rungs (5m, 15m, 1d) x 3 attempts each (1 initial + 2 retries)
        self.assertEqual(quote.attempts, 9)
        self.assertIn("429", quote.error)


if __name__ == "__main__":
    unittest.main()
