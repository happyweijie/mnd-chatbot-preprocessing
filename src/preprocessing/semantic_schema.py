from dataclasses import dataclass

from src.utils.columns import (
    ARTICLE_ID_KEY_COL,
    ARTICLE_TOPIC_KEY_COL,
    CHUNK_ID_COL,
    CHUNK_TOKEN_COUNT_COL,
    CONTENT_CHUNK_COL,
    CONTENT_COL,
    EXPLANATION_COL,
    EXPLANATIONS_BY_TOPIC_COL,
    NEWS_SITE_COL,
    PUBLISHED_DATE_COL,
    QUARTER_YEAR_COL,
    RETRIEVAL_TEXT_COL,
    RETRIEVAL_TOKEN_COUNT_COL,
    SENTIMENT_BY_TOPIC_COL,
    SENTIMENT_SCORE_COL,
    TITLE_COL,
    TOPIC_COL,
    TOPICS_COL,
    YEAR_COL,
)

__all__ = [
    "ARTICLE_ID_KEY_COL",
    "ARTICLE_TOPIC_KEY_COL",
    "CHUNK_ID_COL",
    "CHUNK_TOKEN_COUNT_COL",
    "CONTENT_CHUNK_COL",
    "CONTENT_COL",
    "DEFAULT_EMBEDDING_MODEL",
    "EXPLANATION_COL",
    "EXPLANATIONS_BY_TOPIC_COL",
    "NEWS_SITE_COL",
    "PUBLISHED_DATE_COL",
    "QUARTER_YEAR_COL",
    "RETRIEVAL_TEXT_COL",
    "RETRIEVAL_TOKEN_COUNT_COL",
    "SENTIMENT_BY_TOPIC_COL",
    "SENTIMENT_SCORE_COL",
    "SemanticRagConfig",
    "TITLE_COL",
    "TOPIC_COL",
    "TOPICS_COL",
    "YEAR_COL",
]

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class SemanticRagConfig:
    model: str = DEFAULT_EMBEDDING_MODEL
    target_chunk_tokens: int = 700
    max_chunk_tokens: int = 900
    overlap_tokens: int = 100
    max_batch_tokens: int = 10000         # Reduced from 200000 for slow API endpoints
    max_batch_items: int = 4              # Reduced from 64 for slow API endpoints
