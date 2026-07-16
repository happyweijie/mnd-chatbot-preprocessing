"""Single source of truth for pipeline constants.

Edit here (or use the CLI overrides) to experiment with different settings.
"""

# --- chunking ------------------------------------------------------------
# medium chunks: the median article is only ~980 tokens, so the earlier
# 700/900 settings made chunks nearly article-sized (41% of articles fit in
# one chunk) - too coarse for figure-extraction queries, which need sharp
# embeddings. overlap 75 keeps zero-overlap chunk boundaries rare (~10%)
# at ~11% corpus-wide text duplication.
TARGET_CHUNK_TOKENS = 350
MAX_CHUNK_TOKENS = 500
OVERLAP_TOKENS = 75

# --- embedding -----------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Platform AI endpoint limits: 20 requests/min, 200,000 tokens/min.
# retrieval_text averages ~390 tokens, so batch 20 at the request cap uses
# ~156k tokens/min - headroom for long chunks. throughput ~400 chunks/min,
# so the full corpus (~10.8k chunks) embeds in roughly half an hour.
EMBED_REQUESTS_PER_MINUTE = 20
EMBED_BATCH_SIZE = 20
EMBED_MAX_RETRIES = 5
EMBED_CHECKPOINT_EVERY = 25  # batches between checkpoint writes

# --- output filenames (habit/ expects these names in its tmp/) -----------
BASE_FILENAME = "articles_base.parquet"
FLAT_FILENAME = "articles_flat.parquet"
CHUNKS_FILENAME = "rag_chunks.parquet"