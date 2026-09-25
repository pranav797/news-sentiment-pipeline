import hashlib
import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import duckdb
import feedparser

from config import WATCHLIST, ticker_feeds
from db import get_connection

logger = logging.getLogger(__name__)


def fetch_feed(feed_url: str, ticker: str) -> list[dict]:
    feed = feedparser.parse(feed_url)
    records = []
    feed_source = feed.feed.get("title", feed_url)

    for entry in feed.entries:
        title = entry.get("title")
        url = entry.get("link")
        published_at = entry.get("published") or entry.get("updated")
        entry_source = entry.get("source", {})
        source = (
            entry_source.get("title")
            if isinstance(entry_source, dict)
            else entry_source
        ) or feed_source

        if not title or not url or not published_at:
            continue

        records.append(
            {
                "title": title,
                "source": source,
                "url": url,
                "published_at": published_at,
                "ticker": ticker,
            }
        )

    return records

def fetch_ticker_news(ticker: str) -> list[dict]:
    records = []

    for feed_url in ticker_feeds(ticker):
        records.extend(fetch_feed(feed_url, ticker))

    return records


def fetch_all_news() -> list[dict]:
    records = []

    for ticker in WATCHLIST:
        records.extend(fetch_ticker_news(ticker))

    return records


def _stable_news_id(title: str, url: str) -> str:
    normalized_title = re.sub(r"\s+", " ", title).strip().casefold()
    return hashlib.sha256(f"{normalized_title}\n{url.strip()}".encode("utf-8")).hexdigest()


def deduplicate_news(
    records: list[dict], connection: duckdb.DuckDBPyConnection
) -> tuple[list[dict], int]:
    """Assign stable IDs and exclude existing or repeated news records."""
    existing_ids = {
        row[0] for row in connection.execute("SELECT id FROM news").fetchall()
    }
    new_records = []
    duplicate_count = 0

    for record in records:
        record_with_id = {
            **record,
            "id": _stable_news_id(record["title"], record["url"]),
        }

        if record_with_id["id"] in existing_ids:
            duplicate_count += 1
            continue

        new_records.append(record_with_id)
        existing_ids.add(record_with_id["id"])

    return new_records, duplicate_count


def normalize_timestamp(value: str | datetime) -> datetime:
    """Convert an RSS or ISO timestamp to a timezone-aware UTC datetime."""
    if isinstance(value, datetime):
        timestamp = value
    else:
        try:
            timestamp = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    return timestamp.astimezone(timezone.utc)


def store_news(records: list[dict]) -> tuple[int, int]:
    """Normalize, deduplicate, and insert news records into DuckDB."""
    with get_connection() as connection:
        new_records, duplicate_count = deduplicate_news(records, connection)
        rows_to_insert = [
            (
                record["id"],
                record["ticker"],
                record["title"],
                record["source"],
                record["url"],
                normalize_timestamp(record["published_at"]),
            )
            for record in new_records
        ]

        if rows_to_insert:
            connection.executemany(
                """
                INSERT INTO news (
                    id, ticker, title, source, url, published_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )

    new_count = len(rows_to_insert)
    logger.info(
        "News ingestion complete: %d new records, %d duplicates",
        new_count,
        duplicate_count,
    )
    return new_count, duplicate_count