"""Shared DuckDB connection and schema helpers."""

import os
import tempfile

import duckdb

from config import DB_PATH


def _initialize_schema(connection: duckdb.DuckDBPyConnection) -> None:
    """Create the pipeline tables if they do not already exist."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS news (
            id VARCHAR PRIMARY KEY,
            ticker VARCHAR NOT NULL,
            title VARCHAR NOT NULL,
            source VARCHAR NOT NULL,
            url VARCHAR NOT NULL,
            published_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sentiment (
            headline_id VARCHAR PRIMARY KEY,
            sentiment VARCHAR NOT NULL CHECK (
                sentiment IN ('positive', 'neutral', 'negative')
            ),
            score DOUBLE NOT NULL CHECK (score >= -1 AND score <= 1),
            topic VARCHAR NOT NULL,
            scored_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            ticker VARCHAR NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            PRIMARY KEY (ticker, timestamp)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sentiment_finbert (
            headline_id VARCHAR PRIMARY KEY,
            label VARCHAR NOT NULL CHECK (
                label IN ('positive', 'neutral', 'negative')
            ),
            score DOUBLE NOT NULL CHECK (score >= -1 AND score <= 1),
            scored_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # Per-headline comparison of the LLM and FinBERT scorers (headlines that
    # both models have scored).
    connection.execute(
        """
        CREATE OR REPLACE VIEW sentiment_comparison AS
        SELECT
            n.id AS headline_id,
            n.ticker,
            n.title,
            n.published_at,
            s.sentiment AS llm_label,
            s.score AS llm_score,
            f.label AS finbert_label,
            f.score AS finbert_score,
            (s.sentiment = f.label) AS labels_agree,
            (s.score - f.score) AS score_gap
        FROM news n
        JOIN sentiment s ON s.headline_id = n.id
        JOIN sentiment_finbert f ON f.headline_id = n.id
        """
    )


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return a connection to the project's file-backed DuckDB database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(DB_PATH))
    _initialize_schema(connection)
    return connection


def get_readonly_connection() -> duckdb.DuckDBPyConnection:
    """Return a read-only connection with timestamps rendered in UTC.

    Read-only openers take a shared lock, so several readers (the API, the
    dashboard) can query the warehouse at once. Raises FileNotFoundError if
    the warehouse has not been created yet (run the pipeline first).
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Warehouse not found at {DB_PATH}")
    connection = duckdb.connect(
        str(DB_PATH),
        read_only=True,
        # Readers may run on a read-only mount (see docker-compose.yml), so any
        # temp/spill files go to the system temp dir, never next to the file.
        config={"temp_directory": os.path.join(tempfile.gettempdir(), "duckdb")},
    )
    connection.execute("SET TimeZone='UTC'")
    return connection
