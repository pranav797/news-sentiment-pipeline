"""Prefect orchestration of the news-sentiment pipeline.

Wraps each stage of ``run_once`` as a Prefect task with retries on the
network/API steps, composes them in a single flow, and (when run directly)
serves the flow on a schedule with the Prefect UI for run history and
observability. This is an orchestration upgrade over ``scheduler.py``
(APScheduler) — the pipeline logic itself is unchanged and still lives in
``src/``.

Run one flow immediately:
    python flows/pipeline_flow.py --once

Serve it on a schedule (needs a Prefect server/UI — `prefect server start`):
    python flows/pipeline_flow.py
"""

import sys
from datetime import timedelta
from pathlib import Path

from prefect import flow, get_run_logger, task

# Make the shared src/ modules importable regardless of the launch directory.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aggregate import run_aggregation  # noqa: E402
from config import SCHEDULE_MINUTES  # noqa: E402
from finbert import enrich_unscored_finbert  # noqa: E402
from ingest_news import fetch_all_news, store_news  # noqa: E402
from ingest_prices import ingest_prices  # noqa: E402
from sentiment import enrich_unscored_news  # noqa: E402

# Retry policy for the tasks that hit the network or an external API — a
# transient failure retries instead of being skipped for the whole cycle.
NETWORK_RETRIES = dict(retries=3, retry_delay_seconds=30)


@task(name="ingest-news", **NETWORK_RETRIES)
def ingest_news_task() -> tuple[int, int]:
    return store_news(fetch_all_news())


@task(name="enrich-sentiment", **NETWORK_RETRIES)
def enrich_sentiment_task() -> tuple[int, int]:
    return enrich_unscored_news()


@task(name="enrich-finbert")
def enrich_finbert_task() -> tuple[int, int]:
    # Local CPU model — no network, so no retries.
    return enrich_unscored_finbert()


@task(name="ingest-prices", **NETWORK_RETRIES)
def ingest_prices_task() -> int:
    return ingest_prices()


@task(name="aggregate")
def aggregate_task() -> tuple[int, int]:
    return run_aggregation()


@flow(name="news-sentiment-pipeline")
def pipeline_flow() -> dict[str, object]:
    """Run the full pipeline once as an orchestrated flow.

    Stages run in order but are independent (each reads/writes DuckDB), so a
    stage that fails after its retries is logged and the flow continues —
    preserving the "one bad source shouldn't kill the run" behavior while
    adding retries, run history, and observability.
    """
    logger = get_run_logger()
    results: dict[str, object] = {}

    stages = [
        ("news", ingest_news_task),
        ("sentiment", enrich_sentiment_task),
        ("finbert", enrich_finbert_task),
        ("prices", ingest_prices_task),
        ("aggregation", aggregate_task),
    ]

    for key, stage_task in stages:
        state = stage_task(return_state=True)
        if state.is_completed():
            results[key] = state.result()
        else:
            logger.warning("Stage '%s' did not complete: %s", key, state.type)

    logger.info("Flow run complete: %s", results)
    return results


if __name__ == "__main__":
    if "--once" in sys.argv:
        pipeline_flow()
    else:
        pipeline_flow.serve(
            name="news-sentiment",
            interval=timedelta(minutes=SCHEDULE_MINUTES),
        )
