"""FinBERT sentiment enrichment — a finance-tuned second opinion.

Runs headlines through the locally-hosted ProsusAI/finbert model and stores
its label + a signed score alongside the LLM's sentiment, so the two can be
compared. No API key and no per-call cost; the model downloads once and runs
on CPU.
"""

import logging

from config import MAX_HEADLINES_PER_RUN
from db import get_connection

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
# Pin an exact, owner-published commit so a compromised or updated upstream
# repo can never silently change the weights we load. To upgrade, review the
# new revision on Hugging Face and update this hash deliberately.
MODEL_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"

_classifier = None


def _get_classifier():
    """Lazily build and cache the FinBERT text-classification pipeline.

    Import and model load are deferred to first use so the rest of the
    pipeline does not pay the (slow) transformers/torch import cost unless
    FinBERT is actually used.
    """
    global _classifier

    if _classifier is None:
        from transformers import pipeline

        logger.info("Loading FinBERT model %s (first run downloads it)", MODEL_NAME)
        _classifier = pipeline(
            "text-classification",
            model=MODEL_NAME,
            revision=MODEL_REVISION,
            # Load the owner-published weights at the pinned commit rather than
            # letting transformers fetch an auto-conversion PR branch. torch>=2.6
            # loads them with weights_only=True, which blocks pickle code execution.
            model_kwargs={"use_safetensors": False},
        )

    return _classifier


def score_headline_finbert(title: str) -> dict:
    """Score one headline with FinBERT.

    Returns ``{"label": positive|neutral|negative, "score": float in [-1, 1]}``.
    FinBERT emits a label plus a confidence in [0, 1]; we map it to a signed
    score (positive -> +confidence, negative -> -confidence, neutral -> 0) so
    it lands on the same scale as the LLM score for a direct comparison.
    """
    result = _get_classifier()(title, truncation=True)[0]
    label = result["label"].lower()
    confidence = float(result["score"])

    if label == "positive":
        score = confidence
    elif label == "negative":
        score = -confidence
    else:
        label = "neutral"
        score = 0.0

    return {"label": label, "score": score}


def enrich_unscored_finbert() -> tuple[int, int]:
    """Score headlines that do not yet have a FinBERT row.

    Mirrors ``sentiment.enrich_unscored_news``: only unscored headlines are
    processed, each is wrapped in try/except, and scored/skipped counts are
    logged. At most MAX_HEADLINES_PER_RUN are scored per call (newest first)
    to bound CPU time per cycle; the remainder is picked up on later runs.
    """
    scored_count = 0
    skipped_count = 0

    with get_connection() as connection:
        headlines = connection.execute(
            """
            SELECT news.id, news.title
            FROM news
            LEFT JOIN sentiment_finbert
                ON sentiment_finbert.headline_id = news.id
            WHERE sentiment_finbert.headline_id IS NULL
            ORDER BY news.published_at DESC
            LIMIT ?
            """,
            [MAX_HEADLINES_PER_RUN],
        ).fetchall()

        for headline_id, title in headlines:
            try:
                result = score_headline_finbert(title)
                connection.execute(
                    """
                    INSERT INTO sentiment_finbert (headline_id, label, score)
                    VALUES (?, ?, ?)
                    ON CONFLICT (headline_id) DO NOTHING
                    """,
                    [headline_id, result["label"], result["score"]],
                )
                scored_count += 1
            except Exception as error:
                skipped_count += 1
                logger.warning(
                    "Skipping FinBERT sentiment for headline %s: %s",
                    headline_id,
                    error,
                )

    logger.info(
        "FinBERT enrichment complete: %d scored, %d skipped",
        scored_count,
        skipped_count,
    )
    return scored_count, skipped_count
