"""Phase 1: clean the raw articles CSV into one row per article.

Input:  raw CSV export (claude checkpoint format)
Output: articles_base.parquet - one row per article with topics as a
        list of {topic, sentiment_score, explanation} structs
"""

from pathlib import Path

import polars as pl

from .assertions import (
    assert_all_articles_within_batch_range,
    assert_no_articles_with_duplicate_topics,
    assert_no_articles_with_empty_topics,
    assert_no_articles_with_missing_published_dates,
    assert_no_articles_with_missing_sentiment_reasons,
    assert_no_articles_with_missing_sentiments,
    assert_no_articles_with_missing_titles,
    assert_no_duplicate_article_ids,
)
from .batching import parse_batch_id

COLS_TO_KEEP = [
    "title",
    "url",
    "published_date",
    "content",
    "news_site",
    "topics",
    "sentiment_by_topic",
    "sentiment_reason_by_topic",
]

OUTPUT_COLS = [
    "article_id",
    "title",
    "url",
    "published_date",
    "year",
    "quarter_year",
    "content",
    "news_site",
    "topics",
]

EXCLUDE_MASK = (
    pl.col("exclude_from_housing_analysis")
    | pl.col("housing_relevance").is_in(["low", "none"])
    | pl.col("article_type").is_in(["irrelevant", "advertisement_or_sponsored"])
)


def discover_topics(lf: pl.LazyFrame) -> list[str]:
    """Collect the sorted set of topic names present in the raw data."""
    return (
        lf.select(
            pl.col("topics")
            .fill_null("")
            .str.split("|")
            .list.eval(pl.element().str.strip_chars())
            .alias("topics")
        )
        .explode("topics")
        .filter(pl.col("topics").is_not_null() & (pl.col("topics") != ""))
        .unique()
        .sort("topics")
        .collect()
        .get_column("topics")
        .to_list()
    )


def clean_articles(lf: pl.LazyFrame, all_topics: list[str]) -> pl.LazyFrame:
    """Filter irrelevant rows and normalise into the base article schema."""
    base = lf.filter(~EXCLUDE_MASK).select(COLS_TO_KEEP)

    # Normalise the published_date field
    # the raw export is ISO ("2022-08-08T23:31:00+08:00") but excel-converted
    # copies come through as "8/8/2022 23:31" - parse both. only the calendar
    # day matters downstream, so truncate to a date; this also keeps
    # article_id identical across the two export styles
    cleaned_date = (
        pl.col("published_date")
        .str.replace("T", " ")
        .str.replace(r"\+08:00$", "")
    )
    base = base.with_columns(
        published_date=pl.coalesce(
            cleaned_date.str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
            cleaned_date.str.strptime(pl.Datetime, "%m/%d/%Y %H:%M", strict=False),
        ).dt.date()
    )

    # strict=False turns anything unparseable into null - fail loudly here
    # instead of letting nulls propagate into article_id / year / quarter_year
    assert_no_articles_with_missing_published_dates(base)

    base = base.with_columns(
        year=pl.col("published_date").dt.year(),
        quarter_year=(
            pl.col("published_date").dt.year().cast(pl.String)
            + "Q"
            + pl.col("published_date").dt.quarter().cast(pl.String)
        ),
    )

    # title is a partial key, so it must be present
    assert_no_articles_with_missing_titles(base)

    base = base.with_columns(title=pl.col("title").str.strip_chars())

    # article_id uniquely identifies each article: title + day + news_site
    # (the same story can run on two sites on the same day, so news_site is
    # needed to keep those articles distinct)
    base = base.with_columns(
        article_id=pl.concat_str(
            [
                pl.col("title"),
                pl.col("published_date").dt.strftime("%Y-%m-%d"),
                pl.col("news_site"),
            ],
            separator=" | ",
        )
    )

    assert_no_duplicate_article_ids(base)

    # all valid articles need at least one topic
    assert_no_articles_with_empty_topics(base)

    base = base.with_columns(
        topics=pl.col("topics")
        .str.split("|")
        .list.eval(pl.element().str.strip_chars())
    )

    assert_no_articles_with_duplicate_topics(base)

    # decode the per-topic sentiment JSON columns into structs keyed by topic
    sentiment_score_schema = pl.Struct(
        [pl.Field(topic, pl.Float64) for topic in all_topics]
    )
    sentiment_reason_schema = pl.Struct(
        [pl.Field(topic, pl.String) for topic in all_topics]
    )

    base = base.with_columns(
        pl.col("sentiment_by_topic").fill_null("{}")
        .str.json_decode(dtype=sentiment_score_schema),
        pl.col("sentiment_reason_by_topic").fill_null("{}")
        .str.json_decode(dtype=sentiment_reason_schema),
    )

    assert_no_articles_with_missing_sentiments(base, all_topics)
    assert_no_articles_with_missing_sentiment_reasons(base, all_topics)

    # convert topics into a list of {topic, sentiment_score, explanation}
    # structs - one entry per topic the article was tagged with
    base = base.with_columns(
        topics=pl.concat_list([
            pl.when(pl.col("sentiment_by_topic").struct.field(topic).is_not_null())
            .then(
                pl.struct([
                    pl.lit(topic).alias("topic"),
                    pl.col("sentiment_by_topic").struct.field(topic).alias("sentiment_score"),
                    pl.col("sentiment_reason_by_topic").struct.field(topic).alias("explanation"),
                ])
            )
            .otherwise(None)
            for topic in all_topics
        ]).list.drop_nulls()
    )

    return base.select(OUTPUT_COLS)


def run(
        input_path: Path, 
        output_path: Path, 
        *, 
        batch_id: str | None = None
    ) -> None:
    lf = pl.scan_csv(input_path)

    all_topics = discover_topics(lf)
    base = clean_articles(lf, all_topics)

    # add batch_id column
    if batch_id is not None:
        batch = parse_batch_id(batch_id)
        assert_all_articles_within_batch_range(
            base, batch.start_date, batch.end_date, batch_id
        )

        base = base.with_columns(batch_id=pl.lit(batch_id))
    
    raw_rows = lf.select(pl.len()).collect().item()
    excluded_rows = (
        lf.filter(EXCLUDE_MASK)
        .select(pl.len()).collect().item()
    )
    base_rows = base.select(pl.len()).collect().item()

    # no rows may be lost or invented by the cleaning itself
    assert raw_rows - excluded_rows == base_rows, (
        f"Row count mismatch: {raw_rows} raw - {excluded_rows} excluded "
        f"!= {base_rows} cleaned"
    )

    base.sink_parquet(output_path)
    print(f"phase 1: wrote {base_rows} articles -> {output_path}")