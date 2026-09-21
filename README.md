# Bull Run Strategy V2

A discretionary trading support dashboard. It does **not** execute trades, connect
to any exchange, or place/modify orders — you enter everything manually on your own
platform. Setup scores measure checklist confluence only; they are not
win-probability estimates or profitability guarantees.

## Deploying to Streamlit Cloud

1. Put **every file below flat in your repository root** — no subfolder.
2. Commit and push.
3. In Streamlit Cloud, set **Main file path** to `app.py` (lowercase).
4. Reboot the app so it reinstalls dependencies.

```
your-repo/
├── app.py              ← main file — point Streamlit Cloud here
├── data_layer.py
├── scanner.py
├── scoring.py
├── readiness.py
├── risk_calc.py
├── technical.py
├── backtest.py
├── portfolio.py
├── stops.py
├── storage.py
├── requirements.txt
├── .gitignore
└── tests/              ← optional, local development only
```

**The modules import each other by plain name**, so they must be siblings of
`app.py`. If they end up inside a subfolder, imports break. If an old copy remains
at the root while new files sit in a subfolder, Python loads the *old* one and your
changes appear to do nothing.

### Confirming the right version is live

Under the app title you should see:

```
Build 2026-09-19-b6 ... · data persisted to SQLite
```

If that line is missing, the deployed code is not the code you just uploaded.

## The strategy — Trend Retrace

The default scanner strategy, as specified by the trader.

**Long** (short is the exact reverse):

1. **4H trend** — the last two closed 4H candles each made a higher high and a higher low.
2. **1H confirmation** — the last closed 1H candle is bullish.
3. **5m entry** — wait for price to retrace to a previous support on the 5m chart; enter there with a limit order.

**Re-entry after a stop-out:**

1. Wait for a closed 4H candle in the trade's direction.
2. On the 5m, wait for a retrace to support and re-enter **50%**.
3. Wait for another closed 1H candle in the trade's direction, then add the remaining **50%**.

**Not specified by the trader — adjustable defaults in the sidebar:**

- **Stop loss:** below the 5m support by 3x the 5m ATR (alternatively, below the last closed 4H candle). A 1x buffer was tested and rejected: trades finished in about 12 minutes, before any 1H candle could close, so the second 50% could never be added.
- **Take profit:** 2x the risk.
- **Order expiry:** an unfilled limit is cancelled after a day, or if the 4H trend reverses.

**Scoring** is a checklist count — 4H trend 4 points, 1H confirmation 3, 5m support entry 3 — so a setup meeting all three rules scores 10/10. It is not a probability of profit.

Only **closed** candles count. Exchanges return the still-forming candle as the latest row; it is dropped, because a 1H candle that is green mid-hour can close red.

The original 10-point confluence strategy remains available from the sidebar for comparison.

## Tabs

| Tab | Purpose |
|---|---|
| 🌍 Market | Live prices with explicit data-health status (LIVE / DELAYED / STALE / UNAVAILABLE), bar interval, timestamps, retry |
| 🎯 Scanner | Auto-derives the setup checklist from real 1D / 4H / 1H structure; score and readiness kept separate |
| 📋 Positions | Open positions, portfolio risk, concentration flags, stop management with never-widen enforcement |
| 🧮 Risk | Position sizing with leverage, margin, fees, slippage, liquidation warnings, contract specs |
| 📓 Journal | Persistent trade log with CSV export |
| 🔁 Backtest | No-look-ahead replay engine with in/out-of-sample split and full metrics |

## Data persistence

Journal entries and positions are stored in a local SQLite file (`trading_data.db`),
so they survive page refreshes, reruns and app sleeps.

**On Streamlit Cloud the filesystem is ephemeral** — the database is wiped whenever
the app reboots or redeploys (including every code push). Export your journal to CSV
after any session that matters. Running locally gives you genuinely permanent storage,
since the file lives on your own disk.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Local running also avoids sharing a rate-limited IP with other Streamlit Cloud users,
which was part of the original Yahoo Finance data problem.

## Tests

```bash
python -m unittest discover -s tests
```

Expected: `Ran 133 tests ... OK`. All tests use mocks and synthetic data — no network
access or live market data required.

## Why `curl_cffi` is a dependency

Yahoo Finance tightened bot detection on its data endpoints, causing intermittent
"no price data" failures across all instrument types. `curl_cffi` lets `yfinance`
impersonate a real browser's TLS fingerprint, which is the current recommended fix.
The data layer uses it when available, falls back gracefully when not, and retries
transient failures before reporting them. If a fetch still fails, the app reports
UNAVAILABLE and blocks trade-readiness rather than showing a substituted price.

## Honest limitations

- **No exchange connection.** No order placement, status, or account sync.
- **The scanner is rule-based structural analysis, not discretionary chart reading.**
  "Liquidity" means equal-high/low clustering — there is no order-flow or
  volume-profile data behind it. A "sweep" is a close-based rule.
- **4H candles are resampled from 1H**, because the provider serves no native 4H
  interval. These can differ slightly from an exchange's own 4H candles.
- **The backtester replays signals correctly but does not contain the scanner's
  multi-timeframe strategy.** The included moving-average crossover is an example
  used to validate the engine's mechanics.
- **Liquidation warnings are simplified** — they ignore maintenance margin curves,
  funding rates, and exchange-specific liquidation engines.
- **Free equity data is exchange-delayed** by nature; the app labels it DELAYED
  rather than pretending it is live.
- **Evidence marked ❔ could not be evaluated** and scores zero. It is never assumed.
