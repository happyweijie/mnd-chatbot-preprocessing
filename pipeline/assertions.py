"""Data-integrity guard assertions shared across pipeline phases.

Every function fails loudly (AssertionError with a count and a hint) instead
of letting bad rows propagate silently - polars operations like strptime with
strict=False or concat_str turn problems into nulls rather than errors.
"""

import polars as pl


def assert_no_articles_with_missing_titles(lf: pl.LazyFrame) -> None:
    count = (
        lf.filter((pl.col("title").str.strip_chars() == "")
                  | pl.col("title").is_null())
        .select(pl.len())
        .collect()
        .item()
    )

    assert count == 0, f"Found {count} articles with blank or null title"


def assert_no_articles_with_missing_published_dates(lf: pl.LazyFrame) -> None:
    count = (
        lf.filter(pl.col("published_date").is_null())
        .select(pl.len())
        .collect()
        .item()
    )

    assert count == 0, (
        f"Found {count} articles with null published_date (unparseable date format?)"
    )


def assert_no_duplicate_article_ids(lf: pl.LazyFrame) -> None:
    count = (
        lf.group_by("article_id")
        .len()
        .filter(pl.col("len") > 1)
        .select(pl.len())
        .collect()
        .item()
    )

    assert count == 0, f"Found {count} duplicated article_id values"


def assert_no_articles_with_empty_topics(lf: pl.LazyFrame) -> None:
    count = (
        lf.filter(pl.col("topics").is_null()
                  | (pl.col("topics").str.strip_chars() == ""))
        .select(pl.len())
        .collect()
        .item()
    )

    assert count == 0, f"Found {count} articles with blank, null, or empty topics"


def assert_no_articles_with_duplicate_topics(lf: pl.LazyFrame) -> None:
    count = (
        lf.filter(pl.col("topics").list.len() != pl.col("topics").list.unique().list.len())
        .select(pl.len())
        .collect()
        .item()
    )

    assert count == 0, f"Found {count} articles with duplicate topics"


def assert_no_articles_with_missing_sentiments(
    lf: pl.LazyFrame,
    all_topics: list[str],
) -> None:
    missing_sentiments = (
        lf.with_columns(
            topics_with_sentiments=pl.concat_list([
                pl.when(pl.col("sentiment_by_topic").struct.field(topic).is_not_null())
                  .then(pl.lit(topic))
                for topic in all_topics
            ]).list.drop_nulls()
        )
        .filter(pl.col("topics").list.sort()
                != pl.col("topics_with_sentiments").list.sort())
        .select(pl.len())
        .collect()
        .item()
    )

    assert missing_sentiments == 0, (
        f"Found {missing_sentiments} articles with topic/sentiment mismatch"
    )


def assert_no_articles_with_missing_sentiment_reasons(
    lf: pl.LazyFrame,
    all_topics: list[str],
) -> None:
    missing_reasons = (
        lf.with_columns(
            topics_with_reasons=pl.concat_list([
                pl.when(pl.col("sentiment_reason_by_topic").struct.field(topic).is_not_null())
                  .then(pl.lit(topic))
                for topic in all_topics
            ]).list.drop_nulls()
        )
        .filter(pl.col("topics").list.sort()
                != pl.col("topics_with_reasons").list.sort())
        .select(pl.len())
        .collect()
        .item()
    )

    assert missing_reasons == 0, (
        f"Found {missing_reasons} articles with topic/explanation mismatch"
    )


def assert_all_articles_within_batch_range(
    lf: pl.LazyFrame,
    start_date,
    end_date,
    batch_id: str,
) -> None:
    # unlike the count-only assertions above, this lists the offending rows:
    # the fix is to split the delivery or rename the batch file, so the
    # operator needs to see which articles fall outside the claimed range
    offending = (
        lf.filter(~pl.col("published_date").is_between(start_date, end_date))
        .select("title", "published_date", "news_site")
        .collect()
    )

    assert offending.height == 0, (
        f"Found {offending.height} articles published outside batch "
        f"{batch_id!r} range [{start_date} .. {end_date}] - split the "
        f"delivery or fix the batch filename. Offending rows:\n{offending}"
    )


def assert_data_correctly_flattened(
    base: pl.LazyFrame,
    flattened: pl.LazyFrame,
) -> None:
    expected_rows = (
        base.select(pl.col("topics").list.len().sum())
        .collect()
        .item()
    )

    actual_rows = flattened.select(pl.len()).collect().item()

    assert expected_rows == actual_rows, (
        f"Flattening produced {actual_rows} rows, "
        f"expected {expected_rows} article-topic pairs."
    )


def assert_content_contains_no_nulls(lf: pl.LazyFrame) -> None:
    ok = lf.select(pl.col("content").is_null().not_().all()).collect().item()

    assert ok, "content contains nulls"


def assert_content_contains_only_single_paragraph(lf: pl.LazyFrame) -> None:
    ok = lf.select(pl.col("content").str.contains("\n").not_().all()).collect().item()

    assert ok, (
        "content contains newlines - the simplified chunker assumes single-line "
        "articles; restore paragraph-aware splitting before chunking this data"
    )