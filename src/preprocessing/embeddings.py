from typing import Any, Sequence

import numpy as np

from src.preprocessing.semantic_schema import DEFAULT_EMBEDDING_MODEL


def make_batches(
    texts: Sequence[str],
    max_batch_items: int = 64,
) -> list[list[str]]:
    """Group texts into batches by item count."""
    batches: list[list[str]] = []
    for i in range(0, len(texts), max_batch_items):
        batches.append(texts[i : i + max_batch_items])
    return batches


def embed_texts_batched(
    texts: Sequence[str],
    client: Any,
    model: str = DEFAULT_EMBEDDING_MODEL,
    max_batch_tokens: int = 200000,
    max_batch_items: int = 64,
) -> np.ndarray:
    """Create embeddings for texts using an OpenAI-compatible client."""
    all_embeddings = []

    batches = make_batches(texts, max_batch_items=max_batch_items)

    print(f"Total texts: {len(texts)}")
    print(f"Total batches: {len(batches)}")

    processed = 0
    for batch_idx, batch in enumerate(batches, start=1):
        response = client.embeddings.create(model=model, input=batch)

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        processed += len(batch)
        print(f"Batch {batch_idx}/{len(batches)} done. Processed {processed}/{len(texts)}")

    return np.array(all_embeddings, dtype=np.float32)
