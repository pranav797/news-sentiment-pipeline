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
# 200 large caps (S&P 100 plus the next most-covered names) with heavy news coverage, grouped by sector: ticker -> (name, sector).
# Iterating the dict yields tickers, so pipeline code treats it as a list.
# ponytail: ~200 is the ceiling. Each ticker is one RSS + one yfinance request
# per run, fetched one at a time: 200 tickers = ~800 Yahoo requests/hour, well
# under where Yahoo starts throttling (~2,000/hour, undocumented). Beyond that,
# space out the fetches or batch prices before adding more.
WATCHLIST: dict[str, tuple[str, str]] = {
    # Technology
    "AAPL": ("Apple", "Technology"),
    "MSFT": ("Microsoft", "Technology"),
    "NVDA": ("NVIDIA", "Technology"),
    "AVGO": ("Broadcom", "Technology"),
    "ORCL": ("Oracle", "Technology"),
    "CRM": ("Salesforce", "Technology"),
    "AMD": ("AMD", "Technology"),
    "ADBE": ("Adobe", "Technology"),
    "INTC": ("Intel", "Technology"),
    "CSCO": ("Cisco", "Technology"),
    "IBM": ("IBM", "Technology"),
    "QCOM": ("Qualcomm", "Technology"),
    "TXN": ("Texas Instruments", "Technology"),
    "INTU": ("Intuit", "Technology"),
    "NOW": ("ServiceNow", "Technology"),
    "AMAT": ("Applied Materials", "Technology"),
    "MU": ("Micron", "Technology"),
    "PLTR": ("Palantir", "Technology"),
    "ACN": ("Accenture", "Technology"),
    "ANET": ("Arista Networks", "Technology"),
    "PANW": ("Palo Alto Networks", "Technology"),
    "CRWD": ("CrowdStrike", "Technology"),
    "LRCX": ("Lam Research", "Technology"),
    "KLAC": ("KLA", "Technology"),
    "ADI": ("Analog Devices", "Technology"),
    "SNPS": ("Synopsys", "Technology"),
    "CDNS": ("Cadence Design", "Technology"),
    "MRVL": ("Marvell", "Technology"),
    "DELL": ("Dell", "Technology"),
    "HPQ": ("HP", "Technology"),
    "HPE": ("Hewlett Packard Enterprise", "Technology"),
    "FTNT": ("Fortinet", "Technology"),
    "WDAY": ("Workday", "Technology"),
    "ADSK": ("Autodesk", "Technology"),
    "SHOP": ("Shopify", "Technology"),
    "SMCI": ("Super Micro Computer", "Technology"),
    "SNOW": ("Snowflake", "Technology"),
    # Communication
    "GOOGL": ("Alphabet", "Communication"),
    "META": ("Meta", "Communication"),
    "NFLX": ("Netflix", "Communication"),
    "DIS": ("Disney", "Communication"),
    "T": ("AT&T", "Communication"),
    "VZ": ("Verizon", "Communication"),
    "CMCSA": ("Comcast", "Communication"),
    "TMUS": ("T-Mobile US", "Communication"),
    "CHTR": ("Charter", "Communication"),
    "LYV": ("Live Nation", "Communication"),
    "TTWO": ("Take-Two", "Communication"),
    "OMC": ("Omnicom", "Communication"),
    "SPOT": ("Spotify", "Communication"),
    "RBLX": ("Roblox", "Communication"),
    "PINS": ("Pinterest", "Communication"),
    # Consumer
    "AMZN": ("Amazon", "Consumer"),
    "TSLA": ("Tesla", "Consumer"),
    "WMT": ("Walmart", "Consumer"),
    "COST": ("Costco", "Consumer"),
    "HD": ("Home Depot", "Consumer"),
    "MCD": ("McDonald's", "Consumer"),
    "NKE": ("Nike", "Consumer"),
    "SBUX": ("Starbucks", "Consumer"),
    "KO": ("Coca-Cola", "Consumer"),
    "PEP": ("PepsiCo", "Consumer"),
    "LOW": ("Lowe's", "Consumer"),
    "TGT": ("Target", "Consumer"),
    "BKNG": ("Booking Holdings", "Consumer"),
    "PG": ("Procter & Gamble", "Consumer"),
    "MDLZ": ("Mondelez", "Consumer"),
    "PM": ("Philip Morris", "Consumer"),
    "MO": ("Altria", "Consumer"),
    "CL": ("Colgate-Palmolive", "Consumer"),
    "GM": ("General Motors", "Consumer"),
    "F": ("Ford", "Consumer"),
    "ABNB": ("Airbnb", "Consumer"),
    "CMG": ("Chipotle", "Consumer"),
    "MAR": ("Marriott", "Consumer"),
    "HLT": ("Hilton", "Consumer"),
    "ORLY": ("O'Reilly Automotive", "Consumer"),
    "TJX": ("TJX", "Consumer"),
    "ROST": ("Ross Stores", "Consumer"),
    "LULU": ("Lululemon", "Consumer"),
    "EBAY": ("eBay", "Consumer"),
    "DG": ("Dollar General", "Consumer"),
    "KR": ("Kroger", "Consumer"),
    "KHC": ("Kraft Heinz", "Consumer"),
    "GIS": ("General Mills", "Consumer"),
    "STZ": ("Constellation Brands", "Consumer"),
    "EL": ("Estee Lauder", "Consumer"),
    "YUM": ("Yum! Brands", "Consumer"),
    "KMB": ("Kimberly-Clark", "Consumer"),
    # Financials
    "JPM": ("JPMorgan Chase", "Financials"),
    "BAC": ("Bank of America", "Financials"),
    "GS": ("Goldman Sachs", "Financials"),
    "MS": ("Morgan Stanley", "Financials"),
    "WFC": ("Wells Fargo", "Financials"),
    "V": ("Visa", "Financials"),
    "MA": ("Mastercard", "Financials"),
    "BRK-B": ("Berkshire Hathaway", "Financials"),
    "C": ("Citigroup", "Financials"),
    "SCHW": ("Charles Schwab", "Financials"),
    "BLK": ("BlackRock", "Financials"),
    "AXP": ("American Express", "Financials"),
    "PYPL": ("PayPal", "Financials"),
    "COF": ("Capital One", "Financials"),
    "USB": ("U.S. Bancorp", "Financials"),
    "MET": ("MetLife", "Financials"),
    "AIG": ("AIG", "Financials"),
    "SPGI": ("S&P Global", "Financials"),
    "CME": ("CME Group", "Financials"),
    "ICE": ("Intercontinental Exchange", "Financials"),
    "PGR": ("Progressive", "Financials"),
    "CB": ("Chubb", "Financials"),
    "PNC": ("PNC Financial", "Financials"),
    "TFC": ("Truist", "Financials"),
    "BNY": ("BNY", "Financials"),
    "COIN": ("Coinbase", "Financials"),
    "HOOD": ("Robinhood", "Financials"),
    "KKR": ("KKR", "Financials"),
    "BX": ("Blackstone", "Financials"),
    "APO": ("Apollo", "Financials"),
    "ALL": ("Allstate", "Financials"),
    # Healthcare
    "UNH": ("UnitedHealth", "Healthcare"),
    "JNJ": ("Johnson & Johnson", "Healthcare"),
    "LLY": ("Eli Lilly", "Healthcare"),
    "PFE": ("Pfizer", "Healthcare"),
    "MRK": ("Merck", "Healthcare"),
    "ABBV": ("AbbVie", "Healthcare"),
    "TMO": ("Thermo Fisher", "Healthcare"),
    "ABT": ("Abbott", "Healthcare"),
    "DHR": ("Danaher", "Healthcare"),
    "AMGN": ("Amgen", "Healthcare"),
    "GILD": ("Gilead", "Healthcare"),
    "BMY": ("Bristol-Myers Squibb", "Healthcare"),
    "CVS": ("CVS Health", "Healthcare"),
    "MDT": ("Medtronic", "Healthcare"),
    "ISRG": ("Intuitive Surgical", "Healthcare"),
    "VRTX": ("Vertex", "Healthcare"),
    "REGN": ("Regeneron", "Healthcare"),
    "ZTS": ("Zoetis", "Healthcare"),
    "SYK": ("Stryker", "Healthcare"),
    "BSX": ("Boston Scientific", "Healthcare"),
    "ELV": ("Elevance Health", "Healthcare"),
    "CI": ("Cigna", "Healthcare"),
    "HCA": ("HCA Healthcare", "Healthcare"),
    "MRNA": ("Moderna", "Healthcare"),
    "BIIB": ("Biogen", "Healthcare"),
    "HUM": ("Humana", "Healthcare"),
    "GEHC": ("GE HealthCare", "Healthcare"),
    # Energy
    "XOM": ("ExxonMobil", "Energy"),
    "CVX": ("Chevron", "Energy"),
    "COP": ("ConocoPhillips", "Energy"),
    "SLB": ("SLB", "Energy"),
    "EOG": ("EOG Resources", "Energy"),
    "OXY": ("Occidental", "Energy"),
    "PSX": ("Phillips 66", "Energy"),
    "MPC": ("Marathon Petroleum", "Energy"),
    "VLO": ("Valero", "Energy"),
    "KMI": ("Kinder Morgan", "Energy"),
    "WMB": ("Williams", "Energy"),
    # Industrials
    "BA": ("Boeing", "Industrials"),
    "CAT": ("Caterpillar", "Industrials"),
    "GE": ("GE Aerospace", "Industrials"),
    "UPS": ("UPS", "Industrials"),
    "HON": ("Honeywell", "Industrials"),
    "RTX": ("RTX", "Industrials"),
    "LMT": ("Lockheed Martin", "Industrials"),
    "DE": ("Deere", "Industrials"),
    "UNP": ("Union Pacific", "Industrials"),
    "MMM": ("3M", "Industrials"),
    "FDX": ("FedEx", "Industrials"),
    "GD": ("General Dynamics", "Industrials"),
    "UBER": ("Uber", "Industrials"),
    "ETN": ("Eaton", "Industrials"),
    "NOC": ("Northrop Grumman", "Industrials"),
    "CSX": ("CSX", "Industrials"),
    "NSC": ("Norfolk Southern", "Industrials"),
    "WM": ("Waste Management", "Industrials"),
    "EMR": ("Emerson", "Industrials"),
    "ITW": ("Illinois Tool Works", "Industrials"),
    "DAL": ("Delta Air Lines", "Industrials"),
    "UAL": ("United Airlines", "Industrials"),
    "LUV": ("Southwest Airlines", "Industrials"),
    "GEV": ("GE Vernova", "Industrials"),
    # Utilities
    "NEE": ("NextEra Energy", "Utilities"),
    "DUK": ("Duke Energy", "Utilities"),
    "SO": ("Southern Company", "Utilities"),
    "D": ("Dominion Energy", "Utilities"),
    "AEP": ("American Electric Power", "Utilities"),
    "CEG": ("Constellation Energy", "Utilities"),
    # Materials
    "LIN": ("Linde", "Materials"),
    "SHW": ("Sherwin-Williams", "Materials"),
    "FCX": ("Freeport-McMoRan", "Materials"),
    "NEM": ("Newmont", "Materials"),
    "DOW": ("Dow", "Materials"),
    "APD": ("Air Products", "Materials"),
    # Real Estate
    "AMT": ("American Tower", "Real Estate"),
    "PLD": ("Prologis", "Real Estate"),
    "EQIX": ("Equinix", "Real Estate"),
    "SPG": ("Simon Property", "Real Estate"),
    "O": ("Realty Income", "Real Estate"),
    "CCI": ("Crown Castle", "Real Estate"),
}

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
# When LLM scoring is turned back on, only headlines this recent are sent to
# the LLM; older ones keep their FinBERT score. Keeps demo spend minimal.
LLM_BACKFILL_HOURS = 48

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
