"""Aggregate scored news sentiment into fixed time buckets and align it
with hourly price bars.

Two steps:
  1. ``aggregate_sentiment`` — bucket per-headline sentiment into a fixed
     hourly grid per ticker (average score, article count, positive/negative
     share) -> ``sentiment_hourly``.
  2. ``build_aligned`` — put sentiment and prices on the same (ticker, hour)
     grid and add an hourly return -> ``aligned``.
"""

import logging
import re

from config import SENTIMENT_BUCKET
from db import get_connection

logger = logging.getLogger(__name__)


def _bucket_interval(bucket: str) -> str:
	match = re.fullmatch(r"(\d+)([mhd])", bucket.strip().lower())
	if not match:
		raise ValueError("SENTIMENT_BUCKET must look like '15m', '1h', or '1d'")

	units = {"m": "minute", "h": "hour", "d": "day"}
	return f"{match.group(1)} {units[match.group(2)]}"


def aggregate_sentiment() -> int:
	"""Rebuild hourly sentiment aggregates and return the bucket count."""
	interval = _bucket_interval(SENTIMENT_BUCKET)

	with get_connection() as connection:
		connection.execute(
			f"""
			CREATE OR REPLACE TABLE sentiment_hourly AS
			SELECT
				news.ticker,
				time_bucket(INTERVAL '{interval}', news.published_at)
					AS time_bucket,
				AVG(sentiment.score) AS average_score,
				COUNT(*) AS article_count,
				AVG(CASE WHEN sentiment.sentiment = 'positive'
					THEN 1.0 ELSE 0.0 END) AS positive_share,
				AVG(CASE WHEN sentiment.sentiment = 'negative'
					THEN 1.0 ELSE 0.0 END) AS negative_share
			FROM news
			INNER JOIN sentiment ON sentiment.headline_id = news.id
			GROUP BY news.ticker, time_bucket
			ORDER BY news.ticker, time_bucket
			"""
		)
		bucket_count = connection.execute(
			"SELECT COUNT(*) FROM sentiment_hourly"
		).fetchone()[0]

	logger.info("Created %d sentiment buckets", bucket_count)
	return bucket_count


def build_aligned() -> int:
	"""Join bucketed sentiment to hourly price bars and return the row count.

	Uses a FULL OUTER JOIN so no data is silently dropped: market-hours
	buckets keep their prices even with no news, and after-hours news keeps
	its sentiment even with no price bar. The hourly return is computed on
	the regular price series only, so sentiment-only rows do not distort it.
	"""
	interval = _bucket_interval(SENTIMENT_BUCKET)

	with get_connection() as connection:
		connection.execute(
			f"""
			CREATE OR REPLACE TABLE aligned AS
			WITH price_hourly AS (
				SELECT
					ticker,
					time_bucket(INTERVAL '{interval}', timestamp) AS time_bucket,
					arg_min(open, timestamp) AS open,
					MAX(high) AS high,
					MIN(low) AS low,
					arg_max(close, timestamp) AS close,
					SUM(volume) AS volume
				FROM prices
				GROUP BY ticker, time_bucket
			),
			price_with_return AS (
				SELECT
					*,
					(close / LAG(close) OVER (
						PARTITION BY ticker ORDER BY time_bucket
					)) - 1 AS hourly_return
				FROM price_hourly
			)
			SELECT
				COALESCE(p.ticker, s.ticker) AS ticker,
				COALESCE(p.time_bucket, s.time_bucket) AS time_bucket,
				s.average_score,
				COALESCE(s.article_count, 0) AS article_count,
				s.positive_share,
				s.negative_share,
				p.open,
				p.high,
				p.low,
				p.close,
				p.volume,
				p.hourly_return
			FROM price_with_return p
			FULL OUTER JOIN sentiment_hourly s
				ON p.ticker = s.ticker AND p.time_bucket = s.time_bucket
			ORDER BY ticker, time_bucket
			"""
		)
		row_count = connection.execute(
			"SELECT COUNT(*) FROM aligned"
		).fetchone()[0]

	logger.info("Created %d aligned rows", row_count)
	return row_count


def run_aggregation() -> tuple[int, int]:
	"""Run both aggregation steps in order (buckets, then alignment)."""
	bucket_count = aggregate_sentiment()
	aligned_count = build_aligned()
	return bucket_count, aligned_count