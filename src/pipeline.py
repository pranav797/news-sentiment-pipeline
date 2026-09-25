"""Run one complete near-real-time pipeline cycle."""

import logging

from aggregate import run_aggregation
from ingest_news import fetch_all_news, store_news
from ingest_prices import ingest_prices
from sentiment import enrich_unscored_news

logger = logging.getLogger(__name__)


def run_once() -> dict[str, object]:
    """Run ingestion, enrichment, price loading, and aggregation in order."""
    results: dict[str, object] = {}

    try:
        results["news"] = store_news(fetch_all_news())
    except Exception:
        logger.exception("News ingestion failed; continuing pipeline run")

    try:
        results["sentiment"] = enrich_unscored_news()
    except Exception:
        logger.exception("Sentiment enrichment failed; continuing pipeline run")

    try:
        results["prices"] = ingest_prices()
    except Exception:
        logger.exception("Price ingestion failed; continuing pipeline run")

    try:
        results["aggregation"] = run_aggregation()
    except Exception:
        logger.exception("Aggregation failed; pipeline run finished with errors")

    logger.info("Pipeline run complete: %s", results)
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_once()