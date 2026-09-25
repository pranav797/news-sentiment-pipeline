import json
import logging

from openai import OpenAI

from config import SENTIMENT_MODEL
from db import get_connection

client: OpenAI | None = None
logger = logging.getLogger(__name__)


class SentimentResponseError(ValueError):
    """Raised when the model response does not match the sentiment contract."""


def _get_client() -> OpenAI:
    global client

    if client is None:
        client = OpenAI()

    return client


def _validate_result(result: object) -> dict:
    if not isinstance(result, dict):
        raise SentimentResponseError("Sentiment response must be a JSON object")

    sentiment = result.get("sentiment")
    score = result.get("score")
    topic = result.get("topic")

    if sentiment not in {"positive", "neutral", "negative"}:
        raise SentimentResponseError("Sentiment label is invalid")

    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise SentimentResponseError("Sentiment score must be numeric")

    if not -1 <= score <= 1:
        raise SentimentResponseError("Sentiment score must be between -1 and 1")

    if not isinstance(topic, str) or not topic.strip():
        raise SentimentResponseError("Sentiment topic must be a non-empty string")

    return {
        "sentiment": sentiment,
        "score": float(score),
        "topic": topic.strip(),
    }


def score_headline(title: str) -> dict:
    resp = _get_client().chat.completions.create(
        model=SENTIMENT_MODEL,
        messages=[
            {
                "role": "system",
                "content":
                "You are a financial news sentiment classifier. Given a headline, "
                "respond with ONLY a JSON object: "
                '{"sentiment": "positive|neutral|negative", "score": float in [-1,1], '
                '"topic": "short label"}. No other text.',
            },
            {"role": "user", "content": title},
        ],
    )

    content = resp.choices[0].message.content
    if not content:
        raise SentimentResponseError("Sentiment response was empty")

    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise SentimentResponseError("Sentiment response was not valid JSON") from error

    return _validate_result(result)


def enrich_unscored_news() -> tuple[int, int]:
    """Score and store news headlines that do not have sentiment results."""
    scored_count = 0
    skipped_count = 0

    with get_connection() as connection:
        headlines = connection.execute(
            """
            SELECT news.id, news.title
            FROM news
            LEFT JOIN sentiment ON sentiment.headline_id = news.id
            WHERE sentiment.headline_id IS NULL
            ORDER BY news.published_at
            """
        ).fetchall()

        for headline_id, title in headlines:
            try:
                result = score_headline(title)
                connection.execute(
                    """
                    INSERT INTO sentiment (headline_id, sentiment, score, topic)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (headline_id) DO NOTHING
                    """,
                    [
                        headline_id,
                        result["sentiment"],
                        result["score"],
                        result["topic"],
                    ],
                )
                scored_count += 1
            except Exception as error:
                skipped_count += 1
                logger.warning(
                    "Skipping sentiment for headline %s: %s",
                    headline_id,
                    error,
                )

    logger.info(
        "Sentiment enrichment complete: %d scored, %d skipped",
        scored_count,
        skipped_count,
    )
    return scored_count, skipped_count
