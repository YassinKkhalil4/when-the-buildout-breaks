# The live-data pipeline and dashboard

Source: [`ai_bust_live.py`](../ai_bust_live.py), [`dashboard.html`](../dashboard.html), [`manual_inputs.json`](../manual_inputs.json).

```
fetchers  ->  snapshot  ->  calibration  ->  models  ->  SQLite history  ->  FastAPI + dashboard
```

## 1. What is live and what is not

| Input | Source | Status |
|---|---|---|
| Policy and long rates | U.S. Treasury daily par yield curve CSV (fallback: FRED `fredgraph.csv`) | live |
| Equity prices, drawdowns, volatility | Stooq daily CSV (fallback: Yahoo Finance chart API) | live (Stooq currently answers with a bot-check page, so Yahoo is used) |
| Debt maturity ladders, capex, revenue, interest, cash, leases | SEC EDGAR XBRL `companyfacts` | live |
| Filing dates | SEC EDGAR `submissions` | live |
| Credit spreads and CDS, option-implied volatility, lab revenue run-rates, neocloud backlog | none free | **manual**, from `manual_inputs.json`, flagged as manual |

Model G (end-customer AI revenue against the capex plan) is **not** live: it needs lab-revenue data that no free feed provides, so it stays at its paper calibration.

## 2. Fetchers and parsers

Network access goes through one injectable function `http_get(url, headers)` so tests never touch the network. The User-Agent comes from the environment variable `AI_BUST_UA` (SEC requires a descriptive one, `"Your Name your@email"`).

- **Treasury** (`parse_treasury_csv`): newest row of the daily par yield curve for the current year; columns `3 Mo`, `2 Yr`, `10 Yr`, `30 Yr` in percent, divided by 100.
  Fallback FRED series `DGS3MO`, `DGS10`, `DGS2` (`.` means missing).
- **Prices** (`parse_stooq_csv`, `parse_yahoo_chart`, `price_metrics`): 420 days of daily closes. Symbols: SOX (`^sox`/`^SOX`), NVDA, SPX (`^spx`/`^GSPC`), MSFT, ORCL, CRWV. From the series:
  $$\text{drawdown}=1-\frac{\text{last}}{\max\{\text{close since }2026\text{-}05\text{-}01\}},\qquad \text{vol}_{30}=\text{stdev}(\ln r_{t})\sqrt{252}\ \text{over the last 30 returns}.$$
- **SEC companyfacts** (`parse_companyfacts`): for seven companies (CoreWeave, Oracle, Microsoft, Alphabet, Amazon, Meta, Nvidia) by CIK.
  - *Flows* (capex, revenue, interest) as trailing twelve months: last fiscal year plus the current year-to-date minus the prior-year year-to-date when both exist, else the last fiscal year. Several XBRL tags are tried per field, and **the tag with the most recent period end wins**
    (an earlier version took the first tag with any annual data and silently picked up tags that filers had retired years earlier).
  - *Balance sheet* items (cash, debt, lease liabilities): the latest 10-K/10-Q instant value across candidate tags.
  - *Debt maturity ladder*: six XBRL tags (next twelve months, years two to five, after year five), $B.
- **Failure handling** (`build_snapshot`): every source reports `{ok, url, error, fetched_at}`. A failed source falls back to the previous snapshot's value and is marked `stale`; if no source at all is reachable the refresh raises.

## 3. Calibration: snapshot → model inputs (`calibrate`)

Every input carries a provenance tag: `manual`, `feed`, `recorded`, or `paper default (no data)`.

**Part I inputs** (written into `ai_bust_probability.LIVE`):

| Input | Rule |
|---|---|
| `r_free` | Treasury 3-month yield |
| $\sigma$ range | $\bigl[\min(\text{NVDA implied},\text{SOX realised}),\ \max(\cdot)+0.06\bigr]$; implied vol is manual |
| SOX drawdown range | $[\max(0,\ d-0.09),\ \min(0.6,\ d+0.01)]$ around the live drawdown $d$ |
| Neocloud spread range | manual spread $\pm 0.01$ |
| Oracle CDS | manual |

**Model F overrides:** policy rate `rf0` = the 3-month yield; hyperscaler 2027 capex plan = the manual guidance if set (700 by default), else the filings-based estimate $\sum \text{TTM capex}\times(1+g)$ clipped to 0.6–1.5× the paper value;
neocloud maturity profile = CoreWeave's ladder as shares of years 1–4 and years 5+ (`nc_maturity_profile`).

**Diagnostics** (shown, not fed back): CoreWeave debt/revenue and implied interest rate against the model's NC-1 values.

## 4. Models and storage

`run_models` sets the live inputs, calls `ai_bust_probability.run_live` (10,000 Model G paths; 4,000 Model D draws; 40 Model F draws) and restores the defaults. Each run is stored with its snapshot, calibration and results in a local
**SQLite** database (`live_history.db`, table `runs`). Do not publish that file: it contains price and filing data from third-party providers.

## 5. API and dashboard

FastAPI app `app`:

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | the dashboard page |
| `/paper` | GET | the paper page (built by `build_site.py`) |
| `/api/state` | GET | latest run: sources, rates, equities, companies, manual inputs, calibration, results, history |
| `/api/history` | GET | stored runs |
| `/api/refresh?mode=live\|recorded` | POST | start a refresh in the background (409 if one is running) |

`AI_BUST_REFRESH_MIN=60` also refreshes every 60 minutes. `AI_BUST_DB`, `AI_BUST_MANUAL` override file locations.

**Security.** The app has **no authentication** and the refresh endpoint triggers computation and outbound requests. Bind it to `127.0.0.1` and reach it over an SSH tunnel; if you publish it, put a reverse proxy in front that allows only GET and HEAD
and blocks `/api/refresh`, `/docs`, `/redoc` and `/openapi.json` (see `deploy/Caddyfile.example`).

## 6. Tests

`python -m unittest test_ai_bust_live` runs ten offline tests on parsers (Treasury, FRED, Stooq, Yahoo, companyfacts), the drawdown calculation, calibration against a recorded snapshot (it reproduces the paper's ranges), stale-source fallback and a fake-server snapshot.
The payloads are synthetic, in the providers' documented layouts; they do not prove the parsers survive format changes.

## 7. What a recorded snapshot is

`fixtures/snapshot_recorded_2026-10-05.json` reassembles the figures used in the paper (5 October 2026) from news and company disclosures. It is **not** a feed pull, and its CoreWeave maturity ladder is a placeholder.
Runs from it are labelled `mode="recorded"` everywhere.
