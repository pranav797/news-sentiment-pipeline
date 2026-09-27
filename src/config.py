"""Central configuration: watchlist, news feeds, and shared settings.

Everything the pipeline needs to know about *what* to pull lives here so the
watchlist and feed set can be changed in one place.
"""

import os
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

# Load secrets (OPENAI_API_KEY) from the gitignored .env file if present.
# Existing environment variables win, so a host's secret manager overrides it.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "warehouse.duckdb"

# --- Watchlist ---------------------------------------------------------------
# Start narrow: a few big names with plenty of news coverage. Finish these
# end-to-end before widening the list.
WATCHLIST = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "NVDA",   # NVIDIA
    "AMZN",   # Amazon
    "TSLA",   # Tesla
    "JPM",    # JPMorgan Chase
]

# --- News feeds --------------------------------------------------------------
# Yahoo Finance publishes a per-ticker RSS feed (no API key required). This is
# the no-key backbone; NewsAPI / Finnhub can be layered on later for volume.
YAHOO_FINANCE_RSS = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
)


def ticker_feeds(ticker: str) -> list[str]:
    """Return the list of RSS feed URLs to poll for a given ticker."""
    return [YAHOO_FINANCE_RSS.format(ticker=quote(ticker, safe=""))]


# --- Price ingestion ---------------------------------------------------------
PRICE_PERIOD = "5d"     # how far back to pull each run
PRICE_INTERVAL = "60m"  # intraday bar size

# --- Aggregation / alignment -------------------------------------------------
SENTIMENT_BUCKET = "1h"  # time grid both sentiment and prices are aligned onto

# --- Sentiment model ---------------------------------------------------------
SENTIMENT_MODEL = "gpt-4o-mini"

# Owner-only switch for LLM (OpenAI) scoring, so the API key is used only when
# you choose (e.g. for a demo). The state lives in a file on the data volume,
# writable only from the server itself (see src/llm_switch.py and llm.sh).
# LLM_SCORING_DEFAULT applies when the switch has never been set: "on" keeps
# local development unchanged; docker-compose sets "off" so a fresh deployment
# never spends until you turn it on.
LLM_SWITCH_FILE = DATA_DIR / "llm_scoring.json"
LLM_SCORING_DEFAULT = os.environ.get("LLM_SCORING_DEFAULT", "on").lower() == "on"
LLM_DEFAULT_ON_HOURS = 4  # `llm.sh on` auto-expires after this unless told otherwise

# FinBERT second-opinion scoring. It is the one memory-heavy feature (~750 MB
# for torch + the model); set ENABLE_FINBERT=false on small hosts and the
# pipeline never loads torch. Set the same value for every service so the
# dashboard and API know whether comparison data is being produced.
ENABLE_FINBERT = os.environ.get("ENABLE_FINBERT", "true").lower() == "true"

# --- Scheduling --------------------------------------------------------------
SCHEDULE_MINUTES = 30  # how often the scheduler re-runs the full pipeline

# --- Security / abuse limits -------------------------------------------------
# All overridable via environment variables at deploy time.

# Ingestion: RSS content is untrusted input from the internet.
FEED_TIMEOUT_SECONDS = _env_int("FEED_TIMEOUT_SECONDS", 15)
FEED_MAX_BYTES = _env_int("FEED_MAX_BYTES", 5 * 1024 * 1024)  # 5 MB per feed
MAX_TITLE_LENGTH = 500  # longer "headlines" are dropped as junk/abuse
ALLOWED_URL_SCHEMES = {"http", "https"}  # blocks javascript:, data:, etc.

# Enrichment: bounds OpenAI spend (and CPU for FinBERT) per pipeline cycle,
# even if a feed floods the pipeline with headlines.
MAX_HEADLINES_PER_RUN = _env_int("MAX_HEADLINES_PER_RUN", 200)
OPENAI_TIMEOUT_SECONDS = 30
OPENAI_MAX_RETRIES = 2
OPENAI_MAX_TOKENS = 100  # a sentiment JSON object needs far fewer
MAX_TOPIC_LENGTH = 60  # LLM output is untrusted too

# API: per-client request budget and response bounds.
RATE_LIMIT_PER_MINUTE = _env_int("RATE_LIMIT_PER_MINUTE", 60)
MAX_WINDOW_HOURS = 24 * 365  # upper bound for ?hours=
MAX_SERIES_ROWS = 5000  # cap on rows returned by a series endpoint
ENABLE_DOCS = os.environ.get("ENABLE_DOCS", "true").lower() == "true"
ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("ALLOWED_HOSTS", "*").split(",")
    if host.strip()
]
