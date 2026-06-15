from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.embeddings import embed_texts_batched
from src.utils.columns import RETRIEVAL_TEXT_COL
from src.preprocessing.semantic_chunks import build_semantic_chunk_dataframe
from src.preprocessing.semantic_schema import (
    DEFAULT_EMBEDDING_MODEL,
    SemanticRagConfig,
)
from src.preprocessing.storage import (
    load_dataframe,
    save_dataframe_copies,
    save_numpy_array_copies,
)


def preprocess_semantic_policy_rag(
    input_path: str | Path,
    chunks_output_path: str | Path,
    embeddings_output_path: str | Path | None = None,
    client: Any | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    target_chunk_tokens: int = 700,
    max_chunk_tokens: int = 900,
    overlap_tokens: int = 100,
    max_batch_tokens: int = 200000,
    max_batch_items: int = 64,
    chunks_s3_output_path: str | Path | None = None,
    embeddings_s3_output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Build semantic RAG chunks and, optionally, embeddings."""
    config = SemanticRagConfig(
        model=model,
        target_chunk_tokens=target_chunk_tokens,
        max_chunk_tokens=max_chunk_tokens,
        overlap_tokens=overlap_tokens,
        max_batch_tokens=max_batch_tokens,
        max_batch_items=max_batch_items,
    )

    df = load_dataframe(input_path)
    chunks_df = build_semantic_chunk_dataframe(
        df,
        model=config.model,
        target_chunk_tokens=config.target_chunk_tokens,
        max_chunk_tokens=config.max_chunk_tokens,
        overlap_tokens=config.overlap_tokens,
    )

    chunk_paths = [chunks_output_path]
    if chunks_s3_output_path is not None:
        chunk_paths.append(chunks_s3_output_path)
    save_dataframe_copies(chunks_df, chunk_paths)

    if embeddings_output_path is None and embeddings_s3_output_path is None:
        return chunks_df, None
    if client is None:
        raise ValueError("client is required when embeddings output is requested")

    embeddings = embed_texts_batched(
        chunks_df[RETRIEVAL_TEXT_COL].tolist(),
        client=client,
        model=config.model,
        max_batch_tokens=config.max_batch_tokens,
        max_batch_items=config.max_batch_items,
    )

    embedding_paths = []
    if embeddings_output_path is not None:
        embedding_paths.append(embeddings_output_path)
    if embeddings_s3_output_path is not None:
        embedding_paths.append(embeddings_s3_output_path)
    save_numpy_array_copies(embeddings, embedding_paths)

    return chunks_df, embeddings
