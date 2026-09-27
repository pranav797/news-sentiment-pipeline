"""FastAPI service exposing the news-sentiment warehouse.

Endpoints:
  GET /health              — liveness + warehouse reachability
  GET /tickers             — the configured watchlist
  GET /sentiment/{ticker}  — recent hourly sentiment series for one ticker
  GET /aligned/{ticker}    — sentiment + price series for charting
  GET /signals             — tickers whose sentiment moved sharply last hour
  GET /compare             — LLM vs. FinBERT agreement across the watchlist
  GET /compare/{ticker}    — LLM vs. FinBERT agreement for one ticker

The API opens the DuckDB warehouse read-only, so it can run alongside the
scheduler and the dashboard. Run it with:

    uvicorn api.main:app --reload
"""

import logging
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

# Make the shared src/ modules importable regardless of the launch directory.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    ALLOWED_HOSTS,
    ENABLE_DOCS,
    ENABLE_FINBERT,
    MAX_SERIES_ROWS,
    MAX_WINDOW_HOURS,
    RATE_LIMIT_PER_MINUTE,
    WATCHLIST,
)
from db import get_readonly_connection  # noqa: E402

logger = logging.getLogger(__name__)

app = FastAPI(
    title="News Sentiment Pipeline API",
    description="Near-real-time financial news sentiment aligned with prices.",
    version="1.0.0",
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
)

WATCHLIST_SET = {ticker.upper() for ticker in WATCHLIST}
DOCS_PATH_NAMES = {"docs", "redoc", "openapi.json", "oauth2-redirect"}


# --- Rate limiting -----------------------------------------------------------
class RateLimiter:
    """In-memory sliding-window limiter keyed by client IP.

    Suited to a single API instance (the deployment here). Running several
    replicas would need a shared store such as Redis instead. The client table
    is bounded: stale entries are pruned, and if it is still full of active
    clients new ones are refused (fail closed) rather than growing memory.
    """

    def __init__(self, limit: int, window_seconds: float = 60, max_clients: int = 10_000):
        self.limit = limit
        self.window = window_seconds
        self.max_clients = max_clients
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> Optional[float]:
        """Record a request; return seconds to wait if over the limit, else None."""
        now = time.monotonic()
        hits = self._hits.get(key)
        if hits is None:
            if len(self._hits) >= self.max_clients:
                self._prune(now)
                if len(self._hits) >= self.max_clients:
                    return self.window
            hits = self._hits[key] = deque()

        while hits and now - hits[0] >= self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return self.window - (now - hits[0])
        hits.append(now)
        return None

    def _prune(self, now: float) -> None:
        stale = [
            key for key, hits in self._hits.items()
            if not hits or now - hits[-1] >= self.window
        ]
        for key in stale:
            del self._hits[key]


rate_limiter = RateLimiter(RATE_LIMIT_PER_MINUTE)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    client = request.client.host if request.client else "unknown"
    retry_after = rate_limiter.check(client)
    if retry_after is not None:
        return JSONResponse(
            {"detail": "Rate limit exceeded. Try again later."},
            status_code=429,
            headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
        )
    return await call_next(request)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    # Registered after rate_limit, so it wraps it and also covers 429 responses.
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    # JSON endpoints never need to load anything; Swagger UI does, so skip it.
    if request.url.path.rstrip("/").split("/")[-1] not in DOCS_PATH_NAMES:
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; frame-ancestors 'none'"
        )
    return response


# Outermost: reject requests whose Host header isn't one we serve.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)


# --- Dependencies ------------------------------------------------------------
def get_db() -> duckdb.DuckDBPyConnection:
    """Yield a per-request read-only connection, closed when the request ends."""
    try:
        connection = get_readonly_connection()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Warehouse not initialized yet. Run the pipeline first.",
        )
    except duckdb.Error:
        # Typically a running writer holding the lock. Log details server-side;
        # never return internal error text (file paths etc.) to the client.
        logger.warning("Warehouse unavailable", exc_info=True)
        raise HTTPException(
            status_code=503, detail="Warehouse temporarily unavailable. Retry shortly."
        )
    try:
        yield connection
    finally:
        connection.close()


def validate_ticker(ticker: str) -> str:
    """Normalize and confirm the ticker is on the watchlist, else 404."""
    normalized = ticker.upper()
    if normalized not in WATCHLIST_SET:
        raise HTTPException(status_code=404, detail="Unknown ticker")
    return normalized


def _query(
    connection: duckdb.DuckDBPyConnection, sql: str, params: Optional[list] = None
) -> list[dict]:
    """Run a query and return rows as dicts keyed by column name."""
    cursor = connection.execute(sql, params or [])
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


# --- Response models ---------------------------------------------------------
class Health(BaseModel):
    status: str
    warehouse_reachable: bool


class TickerList(BaseModel):
    tickers: list[str]


class SentimentPoint(BaseModel):
    time_bucket: datetime
    average_score: float
    article_count: int
    positive_share: float
    negative_share: float


class AlignedPoint(BaseModel):
    time_bucket: datetime
    average_score: Optional[float] = None
    article_count: int
    positive_share: Optional[float] = None
    negative_share: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None
    hourly_return: Optional[float] = None


class Signal(BaseModel):
    ticker: str
    time_bucket: datetime
    average_score: float
    previous_score: float
    delta: float
    direction: str
    article_count: int


class ConfusionCell(BaseModel):
    llm_label: str
    finbert_label: str
    count: int


class ModelComparison(BaseModel):
    ticker: Optional[str] = None
    finbert_enabled: bool
    compared: int
    agreement_rate: Optional[float] = None
    avg_abs_score_gap: Optional[float] = None
    confusion: list[ConfusionCell] = []


# --- Endpoints ---------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root(request: Request) -> RedirectResponse:
    # Honor the proxy prefix (e.g. /api) so the redirect stays on the API.
    prefix = request.scope.get("root_path", "")
    return RedirectResponse(url=f"{prefix}/docs" if ENABLE_DOCS else f"{prefix}/health")


@app.get("/health", response_model=Health)
def health() -> Health:
    """Liveness check that also reports whether the warehouse is reachable."""
    try:
        connection = get_readonly_connection()
        connection.close()
        reachable = True
    except Exception:
        reachable = False
    return Health(status="ok", warehouse_reachable=reachable)


@app.get("/tickers", response_model=TickerList)
def tickers() -> TickerList:
    """Return the configured watchlist."""
    return TickerList(tickers=sorted(WATCHLIST_SET))


HoursParam = Query(
    None,
    ge=1,
    le=MAX_WINDOW_HOURS,
    description=f"Limit to the last N hours (max {MAX_WINDOW_HOURS})",
)


@app.get("/sentiment/{ticker}", response_model=list[SentimentPoint])
def sentiment(
    ticker: str,
    hours: Optional[int] = HoursParam,
    connection: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[dict]:
    """Recent hourly sentiment buckets for one ticker, oldest first."""
    return _series(
        connection,
        "sentiment_hourly",
        "time_bucket, average_score, article_count, positive_share, negative_share",
        validate_ticker(ticker),
        hours,
    )


@app.get("/aligned/{ticker}", response_model=list[AlignedPoint])
def aligned(
    ticker: str,
    hours: Optional[int] = HoursParam,
    connection: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[dict]:
    """Aligned sentiment + price series for one ticker, oldest first."""
    return _series(
        connection,
        "aligned",
        "time_bucket, average_score, article_count, positive_share, "
        "negative_share, open, high, low, close, volume, hourly_return",
        validate_ticker(ticker),
        hours,
    )


def _series(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    columns: str,
    symbol: str,
    hours: Optional[int],
) -> list[dict]:
    """Return a ticker's time series, oldest first, capped at MAX_SERIES_ROWS.

    ``table`` and ``columns`` are fixed strings from the endpoints above, never
    user input; every user-supplied value is bound as a query parameter.
    """
    where = "ticker = ?"
    params: list = [symbol]
    if hours is not None:
        where += " AND time_bucket >= now() - INTERVAL (?) HOUR"
        params.append(hours)
    params.append(MAX_SERIES_ROWS)
    sql = f"""
        SELECT * FROM (
            SELECT {columns} FROM {table}
            WHERE {where}
            ORDER BY time_bucket DESC
            LIMIT ?
        ) ORDER BY time_bucket
    """
    return _safe_query(connection, sql, params)


@app.get("/signals", response_model=list[Signal])
def signals(
    threshold: float = Query(
        0.5, ge=0, le=2, description="Minimum absolute sentiment change to flag"
    ),
    connection: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[dict]:
    """Tickers whose latest hourly sentiment jumped vs. the prior bucket."""
    sql = """
        WITH ranked AS (
            SELECT
                ticker,
                time_bucket,
                average_score,
                article_count,
                LAG(average_score) OVER (
                    PARTITION BY ticker ORDER BY time_bucket
                ) AS previous_score,
                ROW_NUMBER() OVER (
                    PARTITION BY ticker ORDER BY time_bucket DESC
                ) AS recency
            FROM sentiment_hourly
        )
        SELECT
            ticker,
            time_bucket,
            average_score,
            previous_score,
            average_score - previous_score AS delta,
            CASE WHEN average_score - previous_score >= 0 THEN 'up' ELSE 'down' END
                AS direction,
            article_count
        FROM ranked
        WHERE recency = 1
          AND previous_score IS NOT NULL
          AND ABS(average_score - previous_score) >= ?
        ORDER BY ABS(average_score - previous_score) DESC
    """
    return _safe_query(connection, sql, [threshold])


@app.get("/compare", response_model=ModelComparison)
def compare_all(
    connection: duckdb.DuckDBPyConnection = Depends(get_db),
) -> ModelComparison:
    """LLM vs. FinBERT agreement across the whole watchlist."""
    return _compare(connection, None)


@app.get("/compare/{ticker}", response_model=ModelComparison)
def compare_ticker(
    ticker: str,
    connection: duckdb.DuckDBPyConnection = Depends(get_db),
) -> ModelComparison:
    """LLM vs. FinBERT agreement for one ticker."""
    symbol = validate_ticker(ticker)
    return _compare(connection, symbol)


def _compare(
    connection: duckdb.DuckDBPyConnection, ticker: Optional[str]
) -> ModelComparison:
    """Summarize agreement between the two scorers, optionally for one ticker."""
    where = ""
    params: list = []
    if ticker is not None:
        where = "WHERE ticker = ?"
        params = [ticker]

    summary = _safe_query(
        connection,
        f"""
        SELECT
            COUNT(*) AS compared,
            AVG(CASE WHEN labels_agree THEN 1.0 ELSE 0.0 END) AS agreement_rate,
            AVG(ABS(score_gap)) AS avg_abs_score_gap
        FROM sentiment_comparison
        {where}
        """,
        params,
    )
    row = summary[0] if summary else {}
    compared = int(row.get("compared") or 0)

    confusion: list[dict] = []
    if compared:
        confusion = _safe_query(
            connection,
            f"""
            SELECT llm_label, finbert_label, COUNT(*) AS count
            FROM sentiment_comparison
            {where}
            GROUP BY llm_label, finbert_label
            ORDER BY count DESC
            """,
            params,
        )

    return ModelComparison(
        ticker=ticker,
        finbert_enabled=ENABLE_FINBERT,
        compared=compared,
        agreement_rate=row.get("agreement_rate"),
        avg_abs_score_gap=row.get("avg_abs_score_gap"),
        confusion=confusion,
    )


def _safe_query(
    connection: duckdb.DuckDBPyConnection, sql: str, params: list
) -> list[dict]:
    """Run a query, treating a missing table as an empty result set."""
    try:
        return _query(connection, sql, params)
    except duckdb.CatalogException:
        # Aggregation tables don't exist until the pipeline has run once.
        return []
