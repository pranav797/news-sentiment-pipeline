"""FastAPI service exposing the news-sentiment warehouse.

Endpoints:
  GET /health              — liveness + warehouse reachability
  GET /tickers             — the configured watchlist
  GET /sentiment/{ticker}  — recent hourly sentiment series for one ticker
  GET /aligned/{ticker}    — sentiment + price series for charting
  GET /signals             — tickers whose sentiment moved sharply last hour

The API opens the DuckDB warehouse read-only, so it can run alongside the
scheduler and the dashboard. Run it with:

    uvicorn api.main:app --reload
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

# Make the shared src/ modules importable regardless of the launch directory.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import WATCHLIST  # noqa: E402
from db import get_readonly_connection  # noqa: E402

app = FastAPI(
    title="News Sentiment Pipeline API",
    description="Near-real-time financial news sentiment aligned with prices.",
    version="1.0.0",
)

WATCHLIST_SET = {ticker.upper() for ticker in WATCHLIST}


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
    except duckdb.Error as error:
        # A running writer holds an exclusive lock; surface it as unavailable.
        raise HTTPException(status_code=503, detail=f"Warehouse unavailable: {error}")
    try:
        yield connection
    finally:
        connection.close()


def validate_ticker(ticker: str) -> str:
    """Normalize and confirm the ticker is on the watchlist, else 404."""
    normalized = ticker.upper()
    if normalized not in WATCHLIST_SET:
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {ticker}")
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


# --- Endpoints ---------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


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


@app.get("/sentiment/{ticker}", response_model=list[SentimentPoint])
def sentiment(
    ticker: str,
    hours: Optional[int] = Query(None, ge=1, description="Limit to the last N hours"),
    connection: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[dict]:
    """Recent hourly sentiment buckets for one ticker, oldest first."""
    symbol = validate_ticker(ticker)
    sql = """
        SELECT time_bucket, average_score, article_count,
               positive_share, negative_share
        FROM sentiment_hourly
        WHERE ticker = ?
    """
    params: list = [symbol]
    if hours is not None:
        sql += " AND time_bucket >= now() - INTERVAL (?) HOUR"
        params.append(hours)
    sql += " ORDER BY time_bucket"
    return _safe_query(connection, sql, params)


@app.get("/aligned/{ticker}", response_model=list[AlignedPoint])
def aligned(
    ticker: str,
    hours: Optional[int] = Query(None, ge=1, description="Limit to the last N hours"),
    connection: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[dict]:
    """Aligned sentiment + price series for one ticker, oldest first."""
    symbol = validate_ticker(ticker)
    sql = """
        SELECT time_bucket, average_score, article_count, positive_share,
               negative_share, open, high, low, close, volume, hourly_return
        FROM aligned
        WHERE ticker = ?
    """
    params: list = [symbol]
    if hours is not None:
        sql += " AND time_bucket >= now() - INTERVAL (?) HOUR"
        params.append(hours)
    sql += " ORDER BY time_bucket"
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


def _safe_query(
    connection: duckdb.DuckDBPyConnection, sql: str, params: list
) -> list[dict]:
    """Run a query, treating a missing table as an empty result set."""
    try:
        return _query(connection, sql, params)
    except duckdb.CatalogException:
        # Aggregation tables don't exist until the pipeline has run once.
        return []
