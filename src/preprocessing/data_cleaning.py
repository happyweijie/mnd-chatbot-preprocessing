import json

import pandas as pd

from src.utils.columns import (
    ARTICLE_ID_KEY_COL,
    ARTICLE_TOPIC_KEY_COL,
    EXPLANATION_COL,
    EXPLANATIONS_BY_TOPIC_COL,
    PUBLISHED_DATE_COL,
    QUARTER_YEAR_COL,
    SENTIMENT_BY_TOPIC_COL,
    SENTIMENT_SCORE_COL,
    TITLE_COL,
    TOPIC_COL,
    TOPICS_COL,
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
