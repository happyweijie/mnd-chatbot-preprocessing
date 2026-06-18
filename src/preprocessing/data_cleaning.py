import json

import pandas as pd

from src.utils.columns import (
    ARTICLE_ID_COL,
    ARTICLE_ID_KEY_COL,
    ARTICLE_TOPIC_KEY_COL,
    CONTENT_COL,
    EXPLANATION_COL,
    EXPLANATIONS_BY_TOPIC_COL,
    NEWS_SITE_COL,
    PUBLISHED_DATE_COL,
    QUARTER_YEAR_COL,
    SENTIMENT_BY_TOPIC_COL,
    SENTIMENT_SCORE_COL,
    TITLE_COL,
    TOPIC_COL,
    TOPICS_COL,
    URL_COL,
    YEAR_COL,
)


def flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten the semistructured columns in the dataframe, including
    pipe separated and json fields.
    """
    df = df.copy()

    df[TOPICS_COL] = df[TOPICS_COL].fillna("").str.split("|")
    df[SENTIMENT_BY_TOPIC_COL] = df[SENTIMENT_BY_TOPIC_COL].apply(_parse_json_dict)
    df[EXPLANATIONS_BY_TOPIC_COL] = df[EXPLANATIONS_BY_TOPIC_COL].apply(_parse_json_dict)

    rows = []
    for row in df.itertuples(index=False):
        base = row._asdict()

        for topic in getattr(row, TOPICS_COL):
            topic = topic.strip()
            if not topic:
                continue

            new_row = base.copy()
            sentiment_by_topic = getattr(row, SENTIMENT_BY_TOPIC_COL)
            explanations_by_topic = getattr(row, EXPLANATIONS_BY_TOPIC_COL)

            new_row[TOPIC_COL] = topic
            new_row[SENTIMENT_SCORE_COL] = sentiment_by_topic.get(topic)
            new_row[EXPLANATION_COL] = explanations_by_topic.get(topic)

            new_row.pop(TOPICS_COL)
            new_row.pop(SENTIMENT_BY_TOPIC_COL)
            new_row.pop(EXPLANATIONS_BY_TOPIC_COL)

            rows.append(new_row)

    return pd.DataFrame(rows, columns=_flattened_columns(df))

def normalise_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Strip whitespace and convert to string type for text fields
    df[TITLE_COL] = df[TITLE_COL].astype("string").str.strip()
    df[TOPIC_COL] = df[TOPIC_COL].astype("string").str.strip()

    df[SENTIMENT_SCORE_COL] = pd.to_numeric(df[SENTIMENT_SCORE_COL], errors="coerce")
    df[PUBLISHED_DATE_COL] = pd.to_datetime(df[PUBLISHED_DATE_COL], errors="coerce")

    # Extract year from published date, with fallback to quarter_year if available
    df[YEAR_COL] = df[PUBLISHED_DATE_COL].dt.year

    if QUARTER_YEAR_COL in df.columns:
        fallback_year = (
            df[QUARTER_YEAR_COL]
            .astype("string")
            .str.extract(r"(\d{4})")[0]
        )
        df[YEAR_COL] = df[YEAR_COL].fillna(pd.to_numeric(fallback_year, errors="coerce"))

    # Create unique keys for article and article-topic combinations
    df[ARTICLE_ID_KEY_COL] = (
        df[TITLE_COL].fillna("")
        + " | "
        + df[PUBLISHED_DATE_COL].astype("string").fillna("")
    )

    df[ARTICLE_TOPIC_KEY_COL] = (
        df[ARTICLE_ID_KEY_COL]
        + " | "
        + df[TOPIC_COL].fillna("")
    )

    return df

# helper methods
def _parse_json_dict(value: object) -> dict:
    """Parse a JSON object value, treating missing values as an empty dict."""
    if isinstance(value, dict):
        return value
    if pd.isna(value) or value == "":
        return {}

    parsed_value = json.loads(value)
    if not isinstance(parsed_value, dict):
        raise ValueError("Expected a JSON object")
    return parsed_value


def _flattened_columns(df: pd.DataFrame) -> list[str]:
    """
    Determine the columns for the flattened dataframe
    """

    # Retain all original columns except the ones that are being flattened
    source_columns = [
        column
        for column in df.columns
        if column
        not in {
            TOPICS_COL,
            SENTIMENT_BY_TOPIC_COL,
            EXPLANATIONS_BY_TOPIC_COL,
        }
    ]

    # Add the new flattened columns
    return source_columns + [TOPIC_COL, SENTIMENT_SCORE_COL, EXPLANATION_COL]


def clean_base_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize article-level fields without flattening.
    Creates articles_base.parquet: one row per article.

    Output columns:
    - article_id: unique identifier
    - title
    - published_date: datetime
    - quarter_year
    - year: extracted from published_date
    - news_site
    - url
    - content
    - topics: list
    - sentiment_by_topic: dict
    - topic_sentiment_explanations: dict
    """
    df = df.copy()

    # Parse structured columns
    df[TOPICS_COL] = df[TOPICS_COL].fillna("").str.split("|")
    df[TOPICS_COL] = df[TOPICS_COL].apply(lambda x: [t.strip() for t in x if t.strip()])
    df[SENTIMENT_BY_TOPIC_COL] = df[SENTIMENT_BY_TOPIC_COL].apply(_parse_json_dict)
    df[EXPLANATIONS_BY_TOPIC_COL] = df[EXPLANATIONS_BY_TOPIC_COL].apply(_parse_json_dict)

    # Normalize text fields
    df[TITLE_COL] = df[TITLE_COL].astype("string").str.strip()
    df[CONTENT_COL] = df[CONTENT_COL].astype("string").str.strip()
    if NEWS_SITE_COL in df.columns:
        df[NEWS_SITE_COL] = df[NEWS_SITE_COL].astype("string").str.strip()
    if URL_COL in df.columns:
        df[URL_COL] = df[URL_COL].astype("string").str.strip()

    # Normalize date field
    df[PUBLISHED_DATE_COL] = pd.to_datetime(df[PUBLISHED_DATE_COL], errors="coerce")

    # Extract year from published date
    df[YEAR_COL] = df[PUBLISHED_DATE_COL].dt.year
    if QUARTER_YEAR_COL in df.columns:
        fallback_year = (
            df[QUARTER_YEAR_COL]
            .astype("string")
            .str.extract(r"(\d{4})")[0]
        )
        df[YEAR_COL] = df[YEAR_COL].fillna(pd.to_numeric(fallback_year, errors="coerce"))

    # Create unique article_id
    df[ARTICLE_ID_COL] = (
        df[TITLE_COL].fillna("")
        + " | "
        + df[PUBLISHED_DATE_COL].astype("string").fillna("")
    )

    # Select base article columns
    base_cols = [
        ARTICLE_ID_COL,
        TITLE_COL,
        PUBLISHED_DATE_COL,
        QUARTER_YEAR_COL,
        YEAR_COL,
        NEWS_SITE_COL,
        URL_COL,
        CONTENT_COL,
        TOPICS_COL,
        SENTIMENT_BY_TOPIC_COL,
        EXPLANATIONS_BY_TOPIC_COL,
    ]

    # Filter to columns that exist in the dataframe
    base_cols = [col for col in base_cols if col in df.columns]

    return df[base_cols].drop_duplicates(subset=[ARTICLE_ID_COL]).reset_index(drop=True)


def flatten_base_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create flattened topic-level table from base articles.
    Creates article_topics_flat.parquet: one row per article-topic pair.

    Input: articles_base.parquet
    Output columns:
    - article_id_key
    - article_topic_key
    - title
    - published_date
    - year
    - quarter_year
    - news_site
    - url
    - topic
    - sentiment_score
    - explanation
    """
    df = df.copy()

    rows = []
    for row in df.itertuples(index=False):
        topics = getattr(row, TOPICS_COL, [])
        if not topics:
            continue

        sentiment_by_topic = getattr(row, SENTIMENT_BY_TOPIC_COL, {})
        explanations_by_topic = getattr(row, EXPLANATIONS_BY_TOPIC_COL, {})
        article_id = getattr(row, ARTICLE_ID_COL, "")

        for topic in topics:
            topic = topic.strip() if isinstance(topic, str) else str(topic)
            if not topic:
                continue

            article_id_key = f"{article_id} | {topic}"

            new_row = {
                ARTICLE_ID_KEY_COL: article_id,
                ARTICLE_TOPIC_KEY_COL: article_id_key,
                TITLE_COL: getattr(row, TITLE_COL, ""),
                PUBLISHED_DATE_COL: getattr(row, PUBLISHED_DATE_COL, None),
                YEAR_COL: getattr(row, YEAR_COL, None),
                QUARTER_YEAR_COL: getattr(row, QUARTER_YEAR_COL, ""),
                NEWS_SITE_COL: getattr(row, NEWS_SITE_COL, ""),
                URL_COL: getattr(row, URL_COL, ""),
                TOPIC_COL: topic,
                SENTIMENT_SCORE_COL: sentiment_by_topic.get(topic),
                EXPLANATION_COL: explanations_by_topic.get(topic),
            }
            rows.append(new_row)

    return pd.DataFrame(rows)
