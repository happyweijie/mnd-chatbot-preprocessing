"""Phase 3: build semantic RAG chunks with retrieval text (no embeddings).

Input:  articles_base.parquet (phase 1 output)
Output: rag_chunks.parquet - one row per chunk with chunk_id, content_chunk,
        and the retrieval_text used for embedding in phase 4
"""

from pathlib import Path

import polars as pl
import tiktoken

from . import config
from .assertions import (
    assert_content_contains_no_nulls,
    assert_content_contains_only_single_paragraph,
)
from .chunking import chunk_content_by_tokens, get_token_encoder

OUTPUT_COLS = [
    "chunk_id",
    "article_id",
    "chunk_index",
    "content_chunk",
    "retrieval_text",
    "title",
    "published_date",
    "news_site",
    "url",
    "topics",
]


def build_rag_chunks(
    base: pl.LazyFrame,
    encoder: tiktoken.Encoding,
    target_chunk_tokens: int = config.TARGET_CHUNK_TOKENS,
    max_chunk_tokens: int = config.MAX_CHUNK_TOKENS,
    overlap_tokens: int = config.OVERLAP_TOKENS,
) -> pl.LazyFrame:
    """Chunk article content inside the lazy pipeline.

    The chunking UDF is opaque to the query optimiser but the plan stays
    lazy end-to-end, so sink_parquet still works.
    """
    # keep topic names only - sentiment detail stays in the flat file
    lf = base.with_columns(
        topics=pl.col("topics").list.eval(pl.element().struct.field("topic"))
    ).drop("year", "quarter_year")

    lf = (
        lf.with_columns(
            content_chunk=pl.col("content").map_elements(
                lambda content: chunk_content_by_tokens(
                    content,
                    encoder,
                    target_chunk_tokens=target_chunk_tokens,
                    max_chunk_tokens=max_chunk_tokens,
                    overlap_tokens=overlap_tokens,
                ),
                return_dtype=pl.List(pl.String),
            )
        )
        .drop("content")
        .explode("content_chunk")
        # articles with empty content explode to a single null chunk
        .filter(
            pl.col("content_chunk").is_not_null()
            & (pl.col("content_chunk") != "")
        )
    )

    # number chunks within each article, then derive chunk_id
    lf = lf.with_columns(
        chunk_index=pl.int_range(pl.len()).over("article_id")
    )
    lf = lf.with_columns(
        chunk_id=pl.format("{}:chunk_{}", pl.col("article_id"), pl.col("chunk_index"))
    )

    # retrieval_text used for embedding: title + topics + excerpt
    lf = lf.with_columns(
        retrieval_text=pl.format(
            "Title: {}\nTopics: {}\n\nArticle excerpt:\n{}",
            pl.col("title"),
            pl.col("topics").list.join(", "),
            pl.col("content_chunk"),
        )
    )

    return lf.select(OUTPUT_COLS)


def run(
    input_path: Path,
    output_path: Path,
    target_chunk_tokens: int = config.TARGET_CHUNK_TOKENS,
    max_chunk_tokens: int = config.MAX_CHUNK_TOKENS,
    overlap_tokens: int = config.OVERLAP_TOKENS,
) -> None:
    base = pl.scan_parquet(input_path)

    # the simplified chunker assumes single-line, non-null content
    assert_content_contains_no_nulls(base)
    assert_content_contains_only_single_paragraph(base)

    encoder = get_token_encoder(config.EMBEDDING_MODEL)
    chunks = build_rag_chunks(
        base,
        encoder,
        target_chunk_tokens=target_chunk_tokens,
        max_chunk_tokens=max_chunk_tokens,
        overlap_tokens=overlap_tokens,
    )

    chunks.sink_parquet(output_path)

    written = pl.scan_parquet(output_path)
    n_chunks = written.select(pl.len()).collect().item()
    n_unique = written.select(pl.col("chunk_id").n_unique()).collect().item()
    assert n_unique == n_chunks, "chunk_id values are not unique"

    print(f"phase 3: wrote {n_chunks} chunks -> {output_path}")