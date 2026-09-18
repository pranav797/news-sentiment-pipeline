"""Central configuration: watchlist, news feeds, and shared settings.

Everything the pipeline needs to know about *what* to pull lives here so the
watchlist and feed set can be changed in one place.
"""

from pathlib import Path

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
    return [YAHOO_FINANCE_RSS.format(ticker=ticker)]


# --- Price ingestion ---------------------------------------------------------
PRICE_PERIOD = "5d"     # how far back to pull each run
PRICE_INTERVAL = "60m"  # intraday bar size

# --- Aggregation / alignment -------------------------------------------------
SENTIMENT_BUCKET = "1h"  # time grid both sentiment and prices are aligned onto

# --- Sentiment model ---------------------------------------------------------
SENTIMENT_MODEL = "gpt-4o-mini"

# --- Scheduling --------------------------------------------------------------
SCHEDULE_MINUTES = 30  # how often the scheduler re-runs the full pipeline
