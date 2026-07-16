"""Phase 2: flatten article topics into one row per article-topic pair.

Input:  articles_base.parquet (phase 1 output)
Output: articles_flat.parquet - one row per (article, topic) with
        sentiment_score and explanation as plain columns, for SQL analytics
"""

from pathlib import Path

import polars as pl

from .assertions import assert_data_correctly_flattened


def flatten_articles(base: pl.LazyFrame) -> pl.LazyFrame:
    """Explode the topics struct list into one row per article-topic."""
    flattened = base.explode("topics").unnest("topics")

    assert_data_correctly_flattened(base, flattened)

    # article_topic_key uniquely identifies each article + topic pair
    flattened = flattened.with_columns(
        article_topic_key=pl.concat_str(
            [pl.col("article_id"), pl.col("topic")],
            separator=" | ",
        )
    )

    # move article_topic_key next to article_id
    cols = flattened.collect_schema().names()
    reordered_cols = [cols[0]] + [cols[-1]] + cols[1:-1]

    # drop content (not needed for SQL queries, found in base and chunk files)
    return flattened.select(reordered_cols).drop("content")


def run(input_path: Path, output_path: Path) -> None:
    base = pl.scan_parquet(input_path)

    flattened = flatten_articles(base)
    rows = flattened.select(pl.len()).collect().item()

    flattened.sink_parquet(output_path)
    print(f"phase 2: wrote {rows} article-topic rows -> {output_path}")