from typing import Any

import pandas as pd

from src.preprocessing.data_cleaning import normalise_fields
from src.utils.columns import (
    ARTICLE_ID_COL,
    ARTICLE_ID_KEY_COL,
    ARTICLE_TOPIC_KEY_COL,
    CHUNK_ID_COL,
    CHUNK_TOKEN_COUNT_COL,
    CONTENT_CHUNK_COL,
    CONTENT_COL,
    EXPLANATION_COL,
    NEWS_SITE_COL,
    PUBLISHED_DATE_COL,
    QUARTER_YEAR_COL,
    RETRIEVAL_TEXT_COL,
    RETRIEVAL_TOKEN_COUNT_COL,
    SENTIMENT_SCORE_COL,
    TITLE_COL,
    TOPIC_COL,
    TOPICS_COL,
    URL_COL,
    YEAR_COL,
)
from src.preprocessing.semantic_schema import DEFAULT_EMBEDDING_MODEL
from src.preprocessing.token_chunking import (
    chunk_content_by_tokens,
    count_tokens,
    get_token_encoder,
)


def format_metadata_value(value: object) -> str:
    """Format dataframe values for retrieval metadata prefixes."""
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return str(value).strip()


def build_retrieval_prefix(row: Any) -> str:
    """Build the metadata prefix included in every embedded chunk."""
    fields = [
        ("Title", getattr(row, TITLE_COL, "")),
        ("Topic", getattr(row, TOPIC_COL, "")),
        ("Published date", getattr(row, PUBLISHED_DATE_COL, "")),
        ("Year", getattr(row, YEAR_COL, "")),
        ("News site", getattr(row, NEWS_SITE_COL, "")),
        ("Sentiment score", getattr(row, SENTIMENT_SCORE_COL, "")),
        ("Topic sentiment explanation", getattr(row, EXPLANATION_COL, "")),
    ]

    lines = [
        f"{label}: {formatted}"
        for label, value in fields
        if (formatted := format_metadata_value(value))
    ]

    return "\n".join(lines)


def build_retrieval_text(prefix: str, content_chunk: str) -> str:
    """Combine metadata context and source content for embedding."""
    if prefix:
        return f"{prefix}\n\nContent:\n{content_chunk}"
    return content_chunk


def prepare_article_topic_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise article-topic fields before chunking."""
    df = df.copy()
    required_for_normalise = {TITLE_COL, TOPIC_COL, SENTIMENT_SCORE_COL, PUBLISHED_DATE_COL}

    if required_for_normalise.issubset(df.columns):
        df = df.pipe(normalise_fields)

    if CONTENT_COL not in df.columns:
        raise ValueError(f"Input dataframe must include a {CONTENT_COL!r} column")

    return df


def build_semantic_chunk_dataframe(
    df: pd.DataFrame,
    model: str = DEFAULT_EMBEDDING_MODEL,
    target_chunk_tokens: int = 700,
    max_chunk_tokens: int = 900,
    overlap_tokens: int = 100,
) -> pd.DataFrame:
    """Create one row per semantic retrieval chunk."""
    print(f"[INFO] Building semantic chunks from {len(df)} articles...")
    encoder = get_token_encoder(model)
    df = prepare_article_topic_dataframe(df)
    chunk_rows = []

    for idx, row in enumerate(df.itertuples(index=False), 1):
        if idx % 100 == 0:
            print(f"[INFO] Processing article {idx}/{len(df)}...", flush=True)

        prefix = build_retrieval_prefix(row)
        content_chunks = chunk_content_by_tokens(
            getattr(row, CONTENT_COL),
            encoder=encoder,
            target_chunk_tokens=target_chunk_tokens,
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=overlap_tokens,
        )

        for chunk_id, content_chunk in enumerate(content_chunks):
            retrieval_text = build_retrieval_text(prefix, content_chunk)
            chunk_rows.append(
                {
                    ARTICLE_ID_KEY_COL: getattr(row, ARTICLE_ID_KEY_COL, ""),
                    ARTICLE_TOPIC_KEY_COL: getattr(row, ARTICLE_TOPIC_KEY_COL, ""),
                    CHUNK_ID_COL: chunk_id,
                    TITLE_COL: getattr(row, TITLE_COL, ""),
                    TOPIC_COL: getattr(row, TOPIC_COL, ""),
                    YEAR_COL: getattr(row, YEAR_COL, ""),
                    QUARTER_YEAR_COL: getattr(row, QUARTER_YEAR_COL, ""),
                    NEWS_SITE_COL: getattr(row, NEWS_SITE_COL, ""),
                    PUBLISHED_DATE_COL: getattr(row, PUBLISHED_DATE_COL, ""),
                    SENTIMENT_SCORE_COL: getattr(row, SENTIMENT_SCORE_COL, ""),
                    EXPLANATION_COL: getattr(row, EXPLANATION_COL, ""),
                    CONTENT_CHUNK_COL: content_chunk,
                    RETRIEVAL_TEXT_COL: retrieval_text,
                    CHUNK_TOKEN_COUNT_COL: count_tokens(content_chunk, encoder),
                    RETRIEVAL_TOKEN_COUNT_COL: count_tokens(retrieval_text, encoder),
                }
            )

    print(f"[INFO] Created {len(chunk_rows)} chunks from {len(df)} articles")
    return pd.DataFrame(chunk_rows)


def build_retrieval_text_for_rag(title: str, topics_list: list, content_chunk: str) -> str:
    """
    Build retrieval_text for RAG chunks.
    Includes: title, topics (comma-separated), and content excerpt.
    Does NOT include: sentiment scores, explanations, IDs, URLs.
    """
    if not topics_list or all(pd.isna(t) for t in topics_list):
        topics_str = ""
    else:
        topics_str = ", ".join(
            str(t).strip() for t in topics_list
            if pd.notna(t) and str(t).strip()
        )

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if topics_str:
        parts.append(f"Topics: {topics_str}")
    parts.append(f"\nArticle excerpt:\n{content_chunk}")

    return "\n".join(parts)


def build_rag_chunks_from_base_articles(
    df: pd.DataFrame,
    model: str = DEFAULT_EMBEDDING_MODEL,
    target_chunk_tokens: int = 700,
    max_chunk_tokens: int = 900,
    overlap_tokens: int = 100,
) -> pd.DataFrame:
    """
    Create RAG chunks from unflattened base articles.
    One row per article chunk (not per article-topic).

    Input: articles_base.parquet
    Output columns:
    - chunk_id
    - article_id
    - chunk_index
    - content_chunk
    - retrieval_text (title + topics + content excerpt)
    - embedding (added by embedding phase)
    - title, date, year, news_site, url, topics (for filtering/citation)
    """
    print(f"[INFO] Building RAG chunks from {len(df)} base articles...")
    encoder = get_token_encoder(model)
    df = df.copy()

    # Ensure required columns exist
    required_cols = {ARTICLE_ID_COL, TITLE_COL, CONTENT_COL, TOPICS_COL}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    chunk_rows = []
    chunk_counter = 0

    for idx, row in enumerate(df.itertuples(index=False), 1):
        if idx % 100 == 0:
            print(f"[INFO] Processing article {idx}/{len(df)}...", flush=True)

        article_id = getattr(row, ARTICLE_ID_COL, "")
        title = getattr(row, TITLE_COL, "")
        content = getattr(row, CONTENT_COL, "")
        topics_list = getattr(row, TOPICS_COL, [])

        if not content or not title:
            continue

        # Chunk the content
        content_chunks = chunk_content_by_tokens(
            content,
            encoder=encoder,
            target_chunk_tokens=target_chunk_tokens,
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=overlap_tokens,
        )

        for chunk_index, content_chunk in enumerate(content_chunks):
            chunk_id = f"{article_id}:chunk_{chunk_index}"
            retrieval_text = build_retrieval_text_for_rag(title, topics_list, content_chunk)

            chunk_rows.append({
                CHUNK_ID_COL: chunk_id,
                ARTICLE_ID_COL: article_id,
                "chunk_index": chunk_index,
                CONTENT_CHUNK_COL: content_chunk,
                RETRIEVAL_TEXT_COL: retrieval_text,
                TITLE_COL: title,
                PUBLISHED_DATE_COL: getattr(row, PUBLISHED_DATE_COL, None),
                YEAR_COL: getattr(row, YEAR_COL, None),
                NEWS_SITE_COL: getattr(row, NEWS_SITE_COL, ""),
                URL_COL: getattr(row, URL_COL, ""),
                TOPICS_COL: topics_list,
            })
            chunk_counter += 1

    print(f"[INFO] Created {len(chunk_rows)} RAG chunks from {len(df)} articles")
    return pd.DataFrame(chunk_rows)
