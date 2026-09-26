# Financial News Sentiment Pipeline

A scheduled, near-real-time data pipeline that continuously ingests financial
news and market prices, scores each headline for sentiment, warehouses both as
aligned time series, and serves the result through a REST API and an
interactive dashboard.

It answers a simple question for a watchlist of tickers: **what is the mood
around each name right now, and where is it shifting sharply?**

---

## Problem

Anyone following a book of holdings faces more headlines than they can read.
Sentiment around a name can turn well before it shows up cleanly in price, but
the signal is buried in a constant, duplicated, multi-source stream of text.

This pipeline turns that stream into a compact, queryable signal: a per-ticker
sentiment time series aligned to intraday price bars, plus an alerting view that
surfaces the tickers whose sentiment moved most in the last window. It is built
as a **monitoring and early-warning tool** — not a trading system.

## Who it's for

- Traders and portfolio managers monitoring the news flow around a watchlist.
- Advisors who want a fast read on sentiment across client holdings.
- Analysts who need sentiment and price on the same time grid for research.

## What it does

- **Ingests news** from per-ticker RSS feeds, assigning each item a stable hash
  ID so the same story across outlets and polls is deduplicated on the way in.
- **Normalizes every timestamp to UTC** at ingestion, so news and market data
  never drift out of alignment.
- **Scores each new headline** for sentiment (`positive` / `neutral` /
  `negative`), a numeric score in `[-1, 1]`, and a short topic label, using an
  LLM that returns validated structured JSON. Only unscored headlines are sent
  to the model, which keeps cost and latency bounded.
- **Scores each headline a second time with FinBERT**, a finance-tuned model run
  locally, and reconciles the two scorers so their agreement can be measured — a
  general model checked against a domain specialist.
- **Ingests intraday price bars** for the watchlist and upserts them so re-runs
  never duplicate data.
- **Aligns sentiment and price** onto a shared hourly grid per ticker and
  computes an hourly return, producing a single table ready for charting and
  analysis.
- **Runs on a schedule**, re-executing the full ingest → enrich → store →
  aggregate loop on an interval, with per-stage error isolation so one failing
  source never takes down a cycle.
- **Serves the data** through a FastAPI service and a Streamlit + Plotly
  dashboard that overlays sentiment on price and highlights sharp movers.

---

## Architecture

```
  ┌──────────────┐      ┌──────────────┐
  │  News (RSS)  │      │   Prices     │
  │  feedparser  │      │  (yfinance)  │
  └──────┬───────┘      └──────┬───────┘
         │                     │
         ▼                     │
  ┌──────────────┐             │
  │  Ingest +    │  hash-dedup │
  │  UTC norm.   │             │
  └──────┬───────┘             │
         ▼                     │
  ┌──────────────┐             │
  │  Sentiment   │  LLM → JSON │
  │  enrichment  │             │
  └──────┬───────┘             │
         ▼                     ▼
  ┌─────────────────────────────────────┐
  │                DuckDB                │
  │  news · sentiment · prices           │
  │  sentiment_hourly · aligned          │
  └──────┬────────────────────────┬──────┘
         ▼                        ▼
  ┌──────────────┐        ┌──────────────┐
  │   FastAPI    │        │  Streamlit   │
  │   service    │        │  dashboard   │
  └──────────────┘        └──────────────┘
         ▲
  ┌──────┴─────────┐
  │   Scheduler    │  re-runs the whole loop every N minutes
  │  (APScheduler) │
  └────────────────┘
```

The scheduler drives a single `run_once()` cycle on an interval. Each cycle
writes to DuckDB, which acts as the time-series store. The API and dashboard are
independent **read-only** consumers of that warehouse, so they can run alongside
the writer and each other.

## Tech stack

| Concern        | Choice                                    |
|----------------|-------------------------------------------|
| Language       | Python 3.11+                              |
| News ingestion | `feedparser` (RSS, no API key)            |
| Price data     | `yfinance`                                |
| Sentiment      | OpenAI `gpt-4o-mini` (structured JSON) + FinBERT (`transformers`) |
| Storage        | DuckDB (embedded, SQL, time-series)       |
| Scheduling     | APScheduler (in-process)                  |
| API            | FastAPI + Uvicorn                         |
| Dashboard      | Streamlit + Plotly (dual-axis overlay)    |
| Data handling  | pandas                                    |

## Data model

All timestamps are stored in UTC.

| Table              | Grain                | Key columns                                                        |
|--------------------|----------------------|-------------------------------------------------------------------|
| `news`             | one row per headline | `id` (hash), `ticker`, `title`, `source`, `url`, `published_at`   |
| `sentiment`        | one row per headline | `headline_id`, `sentiment`, `score`, `topic`, `scored_at`         |
| `sentiment_finbert`| one row per headline | `headline_id`, `label`, `score`, `scored_at`                     |
| `prices`           | one intraday bar     | `ticker`, `timestamp`, `open/high/low/close`, `volume`            |
| `sentiment_hourly` | ticker × hour        | `average_score`, `article_count`, `positive_share`, `negative_share` |
| `aligned`          | ticker × hour        | sentiment metrics + `open/high/low/close`, `volume`, `hourly_return` |
| `sentiment_comparison` | view, per headline | LLM vs. FinBERT label + score, `labels_agree`, `score_gap`    |

`aligned` is built with a full outer join so nothing is dropped: market-hours
buckets keep their prices even without news, and after-hours news keeps its
sentiment even without a price bar.

## Project structure

```
news-sentiment-pipeline/
├── data/                   # DuckDB warehouse (gitignored)
├── src/
│   ├── config.py           # watchlist, feed URLs, settings
│   ├── db.py               # DuckDB connection helpers + schema
│   ├── ingest_news.py      # RSS ingestion + hash deduplication
│   ├── sentiment.py        # LLM headline scoring
│   ├── finbert.py          # FinBERT headline scoring (finance-tuned)
│   ├── ingest_prices.py    # yfinance price bars
│   ├── aggregate.py        # hourly buckets + sentiment/price alignment
│   └── pipeline.py         # one full run_once() cycle
├── api/
│   └── main.py             # FastAPI service
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── scheduler.py            # APScheduler entry point
└── requirements.txt
```

---

## Getting started

### Prerequisites

- Python 3.11 or newer
- An OpenAI API key (for sentiment scoring)

### Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

The requirements include `torch` and `transformers` for FinBERT; the FinBERT
model (~440 MB) downloads automatically on first use and is cached thereafter.

### Configure

Copy the example environment file and add your key:

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

The watchlist and feed URLs live in `src/config.py`. Adjust `WATCHLIST`, the
polling interval (`SCHEDULE_MINUTES`), the price window/interval, and the
sentiment bucket size there.

### Run the pipeline

Run a single end-to-end cycle:

```bash
python src/pipeline.py
```

Or run it continuously on a schedule (executes once immediately, then every
`SCHEDULE_MINUTES`):

```bash
python scheduler.py
```

### Run the API

```bash
uvicorn api.main:app --reload
```

Interactive documentation is served at `http://127.0.0.1:8000/docs`.

### Run the dashboard

```bash
streamlit run dashboard/app.py
```

The API and dashboard read the warehouse read-only and can run at the same time
as the scheduler.

---

## API reference

| Method | Endpoint             | Description                                                        |
|--------|----------------------|-------------------------------------------------------------------|
| GET    | `/health`            | Liveness check and warehouse reachability.                        |
| GET    | `/tickers`           | The configured watchlist.                                         |
| GET    | `/sentiment/{ticker}`| Hourly sentiment series for one ticker. Optional `?hours=N`.      |
| GET    | `/aligned/{ticker}`  | Aligned sentiment + price series for charting. Optional `?hours=N`. |
| GET    | `/signals`           | Tickers whose sentiment moved sharply vs. the prior hour. Optional `?threshold=`. |
| GET    | `/compare`           | LLM vs. FinBERT agreement across the watchlist (rate, score gap, confusion). |
| GET    | `/compare/{ticker}`  | The same comparison for one ticker.                               |

Example:

```bash
curl "http://127.0.0.1:8000/signals?threshold=0.5"
curl "http://127.0.0.1:8000/aligned/AAPL?hours=48"
```

---

## Engineering notes

- **Deduplication at ingestion.** Each headline gets a stable SHA-256 ID over
  its normalized title and URL. Without this, syndicated stories would dominate
  the sentiment average; with it, each distinct story counts once.
- **UTC everywhere.** News timestamps and market bars arrive in different zones;
  both are normalized to UTC at ingestion so every downstream join lines up.
- **Incremental enrichment.** Only headlines without a sentiment row are scored,
  bounding API cost and latency as the news table grows.
- **Resilient cycles.** Each stage of a run is isolated; a failing feed or API
  call is logged and the cycle continues with the stages that can still run.
- **Read-only concurrency.** The API and dashboard open the warehouse read-only,
  so they coexist with the scheduler and degrade gracefully (503 / a friendly
  message) if the store is briefly locked or not yet initialized.

## Two-model sentiment comparison

Every headline is scored by two independent models — a general LLM
(`gpt-4o-mini`) and FinBERT, a finance-tuned classifier run locally on CPU — and
their outputs are reconciled in the `sentiment_comparison` view, surfaced through
`GET /compare` and a dashboard panel. Both scores are mapped to the same signed
`[-1, 1]` scale so they compare directly.

On a recent sample the two agreed on the label about **54%** of the time. The
largest source of disagreement was headlines the LLM read as *positive* but
FinBERT rated *neutral*: the specialist is noticeably more conservative about
calling news positive. Running two scorers side by side both hedges against
either model's blind spots and makes that difference in behavior measurable
rather than assumed.

## Limitations

- **Near-real-time, not streaming.** Data is refreshed on a polling interval
  (default 30 minutes), not as a continuous stream.
- **Sentiment is a monitoring signal, not a trading signal.** Sentiment aligned
  to price is intended for insight and early warning; correlation is not
  causation and nothing here predicts returns.
- **`yfinance` is an unofficial data source**, suitable for this use case but
  not for mission-critical execution.
- **DuckDB is single-writer.** One process writes at a time; readers open the
  warehouse read-only.
