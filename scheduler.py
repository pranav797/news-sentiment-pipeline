"""Schedule the pipeline for repeated near-real-time runs."""

import logging
import importlib
import sys
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

config = importlib.import_module("config")
pipeline = importlib.import_module("pipeline")
SCHEDULE_MINUTES = config.SCHEDULE_MINUTES
run_once = pipeline.run_once


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    scheduler = BlockingScheduler()
    scheduler.add_job(
        run_once,
        "interval",
        minutes=SCHEDULE_MINUTES,
        id="news_sentiment_pipeline",
        max_instances=1,
        coalesce=True,
    )

    run_once()
    logging.getLogger(__name__).info(
        "Near-real-time scheduler started; running every %d minutes",
        SCHEDULE_MINUTES,
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()