# Bull Run Strategy V2 — Trading Analysis Dashboard

A discretionary trading support tool. It does **not** execute trades, connect to
any exchange, or place/modify orders. You enter every trade manually on your own
platform. Setup scores measure checklist confluence only — they are not
win-probability estimates or profitability guarantees.

## File layout

```
your-repo/
├── app.py                 ← Streamlit UI (the entry point)
├── data_layer.py          ← Price fetching + LIVE/DELAYED/STALE/UNAVAILABLE status
├── scoring.py             ← 0–10 setup score with transparent breakdown
├── readiness.py           ← READY / WAITING / NO TRADE state machine
├── risk_calc.py           ← Position sizing, leverage, margin, fees, liquidation
├── technical.py           ← Swing points, HH/HL/LH/LL, Fibonacci from real anchors
├── backtest.py            ← No-lookahead replay engine + metrics
├── portfolio.py           ← Open risk, exposure, concentration flags
├── journal.py             ← Trade journal + CSV export
├── stops.py               ← Stop management, never-widen rule, HOLD/PROTECT/EXIT
├── requirements.txt
└── tests/                 ← 89 unit tests (local dev only, not needed to deploy)
    ├── test_data_layer.py
    ├── test_scoring_readiness.py
    ├── test_risk_calc.py
    ├── test_technical.py
    ├── test_backtest.py
    └── test_portfolio_journal_stops.py
```

All modules must sit in the **same folder** as `app.py` — they import each other
by plain name (`from scoring import ...`), so a flat layout is required.

## Deploying to Streamlit Cloud

1. Put every `.py` file above in your repo root, alongside `requirements.txt`.
2. Commit and push.
3. In Streamlit Cloud, set the **main file path** to `app.py`.
   - If you keep your old dashboard as `App.py`, either delete/rename it or
     point Streamlit Cloud at the new `app.py`. Having both is fine; only the
     configured main file runs.
4. Streamlit Cloud installs `requirements.txt` automatically and redeploys.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Running the tests

```bash
python -m unittest discover -s tests
```

Expected: `Ran 89 tests ... OK`. The tests use mocks, so they need no network
access and no live market data.

## Why `curl_cffi` is in requirements.txt

Yahoo Finance tightened bot detection on its data endpoints, which caused the
intermittent "no price data" failures (SUI, AAPL, MSFT, etc. — it affected all
instrument types, not one ticker). `curl_cffi` lets `yfinance` impersonate a real
browser's TLS fingerprint, which is the fix the yfinance maintainers currently
recommend. `data_layer.py` uses it automatically when installed and falls back
gracefully when it isn't, plus retries transient failures before reporting them.

If a fetch still fails after retries, the app shows **UNAVAILABLE** and blocks
trade-readiness rather than showing a stale or substituted price.

## Tab guide

| Tab | What it does |
|---|---|
| **Market Overview** | Live prices with data-health status, timestamps, retry button |
| **Candidate** | Entry-sequence checklist → setup score + readiness (kept separate) |
| **Positions** | Open positions, portfolio risk, stop management, HOLD/PROTECT/EXIT |
| **Risk Calculator** | Position sizing with leverage, margin, fees, liquidation warning |
| **Journal** | Trade log with CSV export |
| **Backtest** | No-lookahead replay with in/out-of-sample split and metrics |
| **Settings** | Risk %, data freshness thresholds, fees, slippage |

## Honest limitations

These are real and worth reading before trusting output:

- **No exchange connection.** No order placement, no order status, no account
  balance sync, no Hyperliquid integration. Everything is manual entry.
- **Technical structure detection is a first pass.** `technical.py` uses
  rule-based fractal/pivot detection. It has no order-flow or volume-profile
  data, so "liquidity pools" means equal-highs/equal-lows clustering only.
  It is not equivalent to discretionary chart reading.
- **The backtester replays signals correctly but does not contain the full
  multi-timeframe strategy.** The included SMA crossover is an *example* used to
  validate the engine's mechanics. Wiring the real 1D/4H/1H regime-structure-
  liquidity strategy into bar-by-bar signal generation is still outstanding work.
- **Liquidation warnings are simplified.** They ignore maintenance margin curves,
  funding rates, and exchange-specific liquidation engines. Always confirm the
  real liquidation price on your exchange.
- **Correlation flags are heuristic** unless you supply actual return series.
  A hedge is never auto-classified as risk-free.
- **Free Yahoo Finance equity data is exchange-delayed** by nature. The app
  labels this as DELAYED rather than pretending it's live.
