# Financial News Sentiment Pipeline

A scheduled, near-real-time data pipeline that continuously ingests financial
news and market prices, scores each headline for sentiment, warehouses both as
aligned time series, and serves the result through a REST API and an
interactive dashboard.

It answers a simple question for a watchlist of tickers: **what is the mood
around each name right now, and where is it shifting sharply?**

**Live dashboard:** [news-sentiment-pipeline.streamlit.app](https://news-sentiment-pipeline-ciimybpm3cgvtknu3nw5ey.streamlit.app/)
(if it has been idle, click the button to wake it; it starts in under a minute)

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
  dashboard covering 44 large caps across 7 sectors: sentiment overlaid on
  price, a sector-by-ticker market-mood map, sharp movers, and the latest
  scored headlines.

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
| Scheduling     | APScheduler (in-process) or Prefect (orchestration) |
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
| `headline_sentiment` | view, per headline | primary `label` + `score` (LLM, else FinBERT), `scorer` |
| `sentiment_hourly` | ticker × hour        | `average_score`, `article_count`, `positive_share`, `negative_share`, `scorer` |
| `aligned`          | ticker × hour        | sentiment metrics + `scorer`, `open/high/low/close`, `volume`, `hourly_return` |
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
│   ├── llm_switch.py       # owner-only on/off switch for LLM scoring
│   ├── finbert.py          # FinBERT headline scoring (finance-tuned)
│   ├── ingest_prices.py    # yfinance price bars
│   ├── aggregate.py        # hourly buckets + sentiment/price alignment
│   └── pipeline.py         # one full run_once() cycle
├── api/
│   └── main.py             # FastAPI service
├── .github/workflows/
│   └── pipeline.yml        # scheduled pipeline on GitHub Actions
├── dashboard/
│   ├── app.py              # Streamlit dashboard
│   └── requirements.txt    # lean dependencies for Streamlit Cloud
├── flows/
│   └── pipeline_flow.py    # Prefect orchestration of the pipeline
├── .streamlit/config.toml  # dashboard server hardening
├── scheduler.py            # APScheduler entry point
├── llm.sh                  # server-side LLM scoring switch
├── Dockerfile              # one image for all services
├── docker-compose.yml      # single-host deployment
├── Caddyfile               # reverse proxy + automatic HTTPS
└── requirements.txt        # pinned, audited dependencies
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

### Run with Prefect (orchestration)

For retries, run history, and a scheduling UI, run the pipeline as a Prefect
flow instead of the APScheduler loop. Run one flow immediately:

```bash
python flows/pipeline_flow.py --once
```

Or start the Prefect server (UI at `http://127.0.0.1:4200`) and serve the flow
on a schedule:

```bash
prefect server start
```

```bash
python flows/pipeline_flow.py
```

Each stage is a Prefect task; the network/API stages retry automatically, and
the flow-run timeline in the UI shows every run's success, failure, and retries.

The Prefect server has no authentication. It binds to `127.0.0.1` by default —
keep it that way, and never expose it publicly. The Docker deployment below
uses the APScheduler loop instead, so no Prefect server is deployed.

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

## Deployment

Two options: free hosting on GitHub Actions + Streamlit Community Cloud (no
server to run), or self-hosting the full stack with Docker.

### Free hosting: GitHub Actions + Streamlit Community Cloud

- **Pipeline:** `.github/workflows/pipeline.yml` runs every 30 minutes on GitHub
  Actions and publishes the warehouse to the repository's `data` branch.
- **Dashboard:** Streamlit Community Cloud runs `dashboard/app.py`, which
  downloads the latest published warehouse every 10 minutes.

Setup:

1. Add the OpenAI key as a repository secret named `OPENAI_API_KEY`
   (Settings → Secrets and variables → Actions). Only the workflow can read it.
2. On [share.streamlit.io](https://share.streamlit.io), create an app from this
   repository with main file `dashboard/app.py` and Python 3.13. In the app's
   Secrets, set `WAREHOUSE_REPO = "<owner>/<repo>"` (for a private repository,
   also `GITHUB_TOKEN` with read-only access to its contents).

LLM scoring is off by default. To turn it on, run the workflow manually
(Actions → Pipeline → Run workflow) with `llm: on` and a number of hours, or:

```bash
gh workflow run pipeline.yml -f llm=on -f hours=4
gh workflow run pipeline.yml -f llm=off
```

Only collaborators with write access can run it, and the run scores new
headlines immediately; scoring switches off automatically when the hours expire.
The hosted setup does not run the FastAPI service (run it locally or self-host).
Streamlit Community Cloud apps sleep after 12 hours without visitors and wake on
the next visit; the pipeline keeps running regardless. GitHub disables scheduled
workflows in public repositories after 60 days without activity; re-enable it
from the Actions tab if that happens.

### Self-hosting with Docker

The repository also ships a single-host deployment: one Docker image, run as four
services by `docker-compose.yml`.

| Service     | Role                                              | Reachable from       |
|-------------|---------------------------------------------------|----------------------|
| `caddy`     | Reverse proxy, automatic HTTPS                    | Internet (80/443)    |
| `dashboard` | Streamlit dashboard at `/`                        | Caddy only           |
| `api`       | FastAPI service at `/api` (docs at `/api/docs`)   | Caddy only           |
| `worker`    | Scheduled pipeline (`scheduler.py`)               | Nothing (outbound only) |

The warehouse lives on a named volume shared by the services: the worker mounts
it read-write, the API and dashboard read-only.

### Deploy

On a server with Docker installed and a domain whose DNS points at it:

```bash
git clone <your-repo-url> && cd news-sentiment-pipeline
cp .env.example .env    # set OPENAI_API_KEY and DOMAIN
docker compose up -d --build
```

Caddy obtains a TLS certificate for `DOMAIN` on first start. The worker runs a
pipeline cycle immediately and then every `SCHEDULE_MINUTES`; the dashboard shows
data once the first cycle finishes (the first run also downloads FinBERT, so it
takes a few minutes). For a local trial, leave `DOMAIN=localhost` and open
`https://localhost` (Caddy uses a locally trusted certificate).

### Memory and the FinBERT toggle

FinBERT is the only memory-heavy feature. Measured peak memory per service:

| Service              | With FinBERT | `ENABLE_FINBERT=false` |
|----------------------|--------------|------------------------|
| Worker               | ~900 MB      | ~180 MB                |
| API                  | ~130 MB      | ~130 MB                |
| Dashboard            | ~150 MB      | ~150 MB                |

On a small host, set `ENABLE_FINBERT=false` in `.env`: the pipeline skips
FinBERT scoring and never loads torch, bringing the whole stack to roughly
500 MB. The dashboard's **LLM vs. FinBERT comparison** switch lets each visitor
show or hide that panel; when FinBERT is disabled on the server the switch is
greyed out, and `GET /compare` reports `finbert_enabled: false`. Visitors can
change only their own view, never what the server computes.

### Turning LLM scoring on and off

LLM (OpenAI) scoring is controlled by an owner-only switch, and in the Docker
deployment it is **off by default**, so the API key is used only when you choose.
From the project directory on the server:

```bash
sudo ./llm.sh on        # score with the LLM for 4 hours, then switch off automatically
sudo ./llm.sh on 2      # ... for 2 hours
sudo ./llm.sh off       # stop now (takes effect even mid-cycle)
sudo ./llm.sh status
```

`on` also restarts the worker, which runs a pipeline cycle immediately, so fresh
scores appear within minutes.

While LLM scoring is off, the site stays live at no cost: news and prices keep
updating, and **FinBERT becomes the fallback scorer** for new headlines. Those
hours appear as hatched bars on the chart, headlines show which model scored
them, and a banner explains that LLM scoring is paused. When scoring is turned
back on, only headlines from the last 48 hours (`LLM_BACKFILL_HOURS`) are sent to
the LLM; older ones keep their FinBERT score. If FinBERT is also disabled, the
banner instead shows when headlines were last scored.

Each hour is scored by a single model, never a blend: the LLM's scores if any
headline in that hour has one, otherwise FinBERT's. Sharp-mover signals only
compare hours scored by the same model, because the two are calibrated
differently — switching models is not a change in sentiment.

The switch is a file on the data volume, which only the worker can write — the
public API and dashboard can read it but not change it — so only someone with
shell access to the server can flip it. If the switch file is missing or
unreadable, scoring is treated as off. Locally (outside Docker) scoring defaults
to on; `python src/llm_switch.py on|off|status` works the same way.

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
- **Market-aware prices.** The dashboard checks the NYSE session (9:30–16:00 ET,
  weekdays, DST-aware). Outside it, the latest price is labelled with its closing
  time and a dashed last-close line spans the closed period, so weekends and
  overnight read as "no trading" rather than missing data.
- **Read-only concurrency.** The API and dashboard open the warehouse read-only,
  so they coexist with the scheduler and degrade gracefully (503 / a friendly
  message) if the store is briefly locked or not yet initialized.

## Two-model sentiment comparison

Every headline is scored by two independent models — a general LLM
(`gpt-4o-mini`) and FinBERT, a finance-tuned classifier run locally on CPU — and
their outputs are reconciled in the `sentiment_comparison` view, surfaced through
`GET /compare` and a dashboard panel. Both scores are mapped to the same signed
`[-1, 1]` scale so they compare directly. FinBERT also serves as the fallback
scorer whenever LLM scoring is switched off (see
[Turning LLM scoring on and off](#turning-llm-scoring-on-and-off)).

On a recent sample the two agreed on the label about **54%** of the time. The
largest source of disagreement was headlines the LLM read as *positive* but
FinBERT rated *neutral*: the specialist is noticeably more conservative about
calling news positive. Running two scorers side by side both hedges against
either model's blind spots and makes that difference in behavior measurable
rather than assumed.

## Orchestration

The pipeline ships with two schedulers. **APScheduler** (`scheduler.py`) is the
lightweight default: an in-process loop with no extra infrastructure.
**Prefect** (`flows/pipeline_flow.py`) is the orchestration upgrade — each stage
becomes a task with automatic retries on the network/API steps, and every run
is recorded with its state, logs, and retry history in the Prefect UI. The
pipeline logic is unchanged; the tasks call the same functions in `src/`.

This is orchestration for a single-node, in-process pipeline: the win is
reliability and observability (retries, run history, a scheduling UI), not
distributed execution or horizontal scale.

## Security

The system treats everything from the internet — feed content, model output,
and API clients — as untrusted.

**Secrets**
- `OPENAI_API_KEY` is read from the environment or a gitignored `.env`; it is
  excluded from Docker images by `.dockerignore`, and in the deployment only the
  worker receives it. The public-facing services never hold the key.
- Nothing public triggers an OpenAI call, so visitors cannot spend API credit.
  LLM scoring is off by default in the deployment and can be turned on only
  from the server (`llm.sh`), with an automatic expiry.

**Untrusted feed content**
- Feeds are fetched with a timeout and a 5 MB size cap.
- Only absolute `http(s)` links are accepted, so a malicious feed cannot plant
  `javascript:` or `data:` links; the dashboard re-checks links before
  rendering them. Oversized titles are dropped.

**Untrusted model output and cost control**
- LLM responses are validated against a strict schema (label enum, score range)
  and the topic is length-capped; the prompt instructs the model to treat
  headlines as data, never instructions.
- Each call has a timeout, bounded retries, and a token cap; each pipeline cycle
  scores at most `MAX_HEADLINES_PER_RUN` headlines. Setting a monthly usage limit
  on the OpenAI project adds a hard ceiling.
- The FinBERT model is pinned to an exact upstream commit and loaded with
  `weights_only` deserialization, so upstream changes cannot silently alter or
  inject code into what runs.

**API**
- Per-client rate limiting (default 60 requests/minute, `429` with
  `Retry-After`), with client IPs taken from the proxy in a way clients cannot
  spoof.
- Every input is validated: tickers against the watchlist, `hours` and
  `threshold` against bounds; all SQL uses bound parameters.
- Responses are size-capped, error messages never expose internals, and security
  headers (`Content-Security-Policy`, `X-Frame-Options`, `nosniff`) are set.
  `ALLOWED_HOSTS` rejects unexpected `Host` headers; `ENABLE_DOCS=false` hides
  the interactive docs.

**Dashboard**
- Tracebacks and developer menus are hidden from visitors
  (`.streamlit/config.toml`); XSRF protection stays on.
- The shared-cache "Refresh" action is rate-limited server-wide.

**Infrastructure**
- HTTPS with HSTS via Caddy; only Caddy is exposed to the internet.
- Containers run as an unprivileged user with all Linux capabilities dropped,
  `no-new-privileges`, memory and process limits, and read-only root
  filesystems for the public-facing services.
- Dependencies are pinned to versions audited with `pip-audit`. Re-run
  `pip-audit` whenever you upgrade them.

The dashboard and API are intentionally public and read-only; they expose only
public news and market data. Put them behind authentication (for example, at
the proxy) before adding anything non-public.

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
