"""Phase 4: embed chunk retrieval_text and attach the embedding column.

Input:  rag_chunks.parquet (phase 3 output, never overwritten)
Output: rag_chunks_embedded.parquet - the chunks with an embedding column
        added (list[f32]), or a separate sample file when --limit is used

Requires OPENAI_API_KEY; OPENAI_BASE_URL is optional (gov endpoint).

Hardening for the rate-limited gov endpoint:
- requests are paced to EMBED_REQUESTS_PER_MINUTE
- transient failures retry with exponential backoff
- progress is checkpointed so an interrupted run resumes where it stopped
"""

import asyncio
import os
import time
from pathlib import Path

import polars as pl
from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingSettings
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.openai import OpenAIProvider

from . import config


def create_embedder(
    model: str = config.EMBEDDING_MODEL,
    dimensions: int | None = config.EMBEDDING_DIM,
) -> Embedder:
    # base_url must include the scheme, otherwise the openai client only
    # fails at request time with UnsupportedProtocol / "Connection error"
    base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip() or None
    if base_url and not base_url.startswith(("http://", "https://")):
        base_url = "https://" + base_url

    provider = OpenAIProvider(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=base_url,
    )

    # only request explicit dimensions when configured;
    # otherwisedimensions is set to None in config.py
    # when EMBEDDING_DIM is unset, so the model's default length is used
    settings = EmbeddingSettings(dimensions=dimensions) if dimensions else None
    
    return Embedder(
        OpenAIEmbeddingModel(model, provider=provider),
        settings=settings,
    )


async def _embed_batch_with_retry(
    embedder: Embedder,
    batch: list[str],
    max_retries: int = config.EMBED_MAX_RETRIES,
) -> list[list[float]]:
    delay = 5.0
    for attempt in range(1, max_retries + 1):
        try:
            return list(await embedder.embed_documents(batch))
        except Exception as exc:  # rate limits / transient network errors
            if attempt == max_retries:
                raise
            print(f"  batch failed ({type(exc).__name__}: {exc}), "
                  f"retry {attempt}/{max_retries - 1} in {delay:.0f}s")
            await asyncio.sleep(delay)
            delay *= 2


async def embed_texts(
    embedder: Embedder,
    chunk_ids: list[str],
    texts: list[str],
    checkpoint_path: Path,
    batch_size: int = config.EMBED_BATCH_SIZE,
    requests_per_minute: int = config.EMBED_REQUESTS_PER_MINUTE,
) -> pl.DataFrame:
    """Embed texts in paced batches, checkpointing progress to disk.

    Returns a (chunk_id, embedding) dataframe covering every input chunk_id,
    including rows recovered from an earlier interrupted run's checkpoint.
    """
    done = pl.DataFrame(
        schema={"chunk_id": pl.String, "embedding": pl.List(pl.Float32)}
    )
    if checkpoint_path.exists():
        done = pl.read_parquet(checkpoint_path)
        print(f"resuming: {len(done)} embeddings loaded from {checkpoint_path}")

    done_ids = set(done.get_column("chunk_id").to_list())
    pending = [
        (cid, text) 
        for cid, text in zip(chunk_ids, texts)
        if cid not in done_ids
    ]

    min_interval = 60.0 / requests_per_minute
    total_batches = (len(pending) + batch_size - 1) // batch_size
    last_request_start = 0.0
    new_ids: list[str] = []
    new_embeddings: list[list[float]] = []

    for batch_no, start in enumerate(range(0, len(pending), batch_size), start=1):
        batch = pending[start : start + batch_size]

        # pace request starts to stay under the endpoint's requests/min limit
        wait = min_interval - (time.monotonic() - last_request_start)
        if wait > 0:
            await asyncio.sleep(wait)
        last_request_start = time.monotonic()

        batch_texts = [text for _, text in batch]
        new_embeddings.extend(await _embed_batch_with_retry(embedder, batch_texts))
        new_ids.extend(cid for cid, _ in batch)

        print(f"batch {batch_no}/{total_batches} done, "
              f"embedded {len(done) + len(new_ids)}/{len(chunk_ids)} chunks")

        if batch_no % config.EMBED_CHECKPOINT_EVERY == 0:
            _write_checkpoint(checkpoint_path, done, new_ids, new_embeddings)

    return pl.concat([done, _to_frame(new_ids, new_embeddings)])


def _to_frame(chunk_ids: list[str], embeddings: list[list[float]]) -> pl.DataFrame:
    return pl.DataFrame({
        "chunk_id": pl.Series(chunk_ids, dtype=pl.String),
        "embedding": pl.Series(embeddings, dtype=pl.List(pl.Float32)),
    })


def _write_checkpoint(
    checkpoint_path: Path,
    done: pl.DataFrame,
    new_ids: list[str],
    new_embeddings: list[list[float]],
) -> None:
    pl.concat([done, _to_frame(new_ids, new_embeddings)]).write_parquet(checkpoint_path)


def run(
    input_path: Path,
    output_path: Path,
    batch_size: int = config.EMBED_BATCH_SIZE,
    limit: int | None = None,
) -> None:
    chunks = pl.read_parquet(input_path)
    if limit is not None:
        chunks = chunks.head(limit)
        print(f"--limit {limit}: embedding a sample, writing to {output_path}")

    embedder = create_embedder()
    checkpoint_path = output_path.with_suffix(".checkpoint.parquet")

    embeddings = asyncio.run(
        embed_texts(
            embedder,
            chunks.get_column("chunk_id").to_list(),
            chunks.get_column("retrieval_text").to_list(),
            checkpoint_path,
            batch_size=batch_size,
        )
    )

    result = chunks.join(embeddings, on="chunk_id", how="left")

    # every chunk must have an embedding; lengths must match the configured
    # dimension, or at least be uniform when using the model's default
    assert result.get_column("embedding").null_count() == 0, "missing embeddings"
    lengths = result.get_column("embedding").list.len()
    if config.EMBEDDING_DIM is not None:
        assert (lengths == config.EMBEDDING_DIM).all(), \
            f"embeddings must be {config.EMBEDDING_DIM}-dimensional"
    else:
        assert lengths.n_unique() == 1, "embeddings have inconsistent dimensions"

    result.write_parquet(output_path)
    checkpoint_path.unlink(missing_ok=True)
    print(f"phase 4: wrote {len(result)} embedded chunks -> {output_path}")
