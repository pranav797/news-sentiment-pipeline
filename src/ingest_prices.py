"""Download and store market price bars for the configured watchlist."""

import logging
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from config import PRICE_INTERVAL, PRICE_PERIOD, WATCHLIST
from db import get_connection

logger = logging.getLogger(__name__)


def _to_utc(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    else:
        timestamp = timestamp.tz_convert(timezone.utc)
    return timestamp.to_pydatetime()


def fetch_prices(ticker: str) -> list[dict]:
    """Fetch recent price bars for one ticker as serializable records."""
    frame = yf.download(
        ticker,
        period=PRICE_PERIOD,
        interval=PRICE_INTERVAL,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if frame.empty:
        return []

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    records = []
    for timestamp, row in frame.iterrows():
        if pd.isna(row.get("Close")):
            continue

        records.append(
            {
                "ticker": ticker,
                "timestamp": _to_utc(timestamp),
                "open": _number_or_none(row.get("Open")),
                "high": _number_or_none(row.get("High")),
                "low": _number_or_none(row.get("Low")),
                "close": _number_or_none(row.get("Close")),
                "volume": _integer_or_none(row.get("Volume")),
            }
        )

    return records


def _number_or_none(value: object) -> float | None:
    return None if pd.isna(value) else float(value)


def _integer_or_none(value: object) -> int | None:
    return None if pd.isna(value) else int(value)


def store_prices(records: list[dict]) -> int:
    """Upsert price bars and return the number of processed records."""
    with get_connection() as connection:
        connection.executemany(
            """
            INSERT INTO prices (
                ticker, timestamp, open, high, low, close, volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (ticker, timestamp) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
            """,
            [
                (
                    record["ticker"],
                    _to_utc(record["timestamp"]),
                    record["open"],
                    record["high"],
                    record["low"],
                    record["close"],
                    record["volume"],
                )
                for record in records
            ],
        )

    logger.info("Stored %d price bars", len(records))
    return len(records)


def ingest_prices() -> int:
    """Fetch and upsert price bars for every configured ticker."""
    records = []
    for ticker in WATCHLIST:
        try:
            records.extend(fetch_prices(ticker))
        except Exception:
            logger.exception("Failed to fetch prices for %s", ticker)

    return store_prices(records)