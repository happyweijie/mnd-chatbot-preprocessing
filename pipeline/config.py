"""Single source of truth for pipeline constants.

Edit here (or use the CLI overrides) to experiment with different settings.
The embedding model/dimension are read from the environment (EMBEDDING_MODEL,
EMBEDDING_DIM); a local .env file is loaded if present.
"""

import os

from dotenv import load_dotenv

load_dotenv()

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
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
# optional override: leave EMBEDDING_DIM unset to use the model's default
# length (1536 for text-embedding-3-small, 3072 for text-embedding-3-large)
_dim = os.environ.get("EMBEDDING_DIM", "").strip()
EMBEDDING_DIM: int | None = int(_dim) if _dim else None

# Platform AI endpoint limits: 20 requests/min, 200,000 tokens/min.
# retrieval_text averages ~390 tokens, so batch 20 at the request cap uses
# ~156k tokens/min - headroom for long chunks. throughput ~400 chunks/min,
# so the full corpus (~10.8k chunks) embeds in roughly half an hour.
EMBED_REQUESTS_PER_MINUTE = 20
EMBED_BATCH_SIZE = 20
EMBED_MAX_RETRIES = 5
EMBED_CHECKPOINT_EVERY = 25  # batches between checkpoint writes

# --- S3 batch layout (see README "S3 storage layout") ---------------------
# bucket + root prefix that `python -m pipeline batch` reads raw batches from
# and writes processed stages to.
# S3_ROOT is a bare key prefix (may contain slashes, e.g. "teams/xyz/hint"),
# never with a leading/trailing slash or s3:// scheme.
S3_BUCKET = os.environ.get("HINT_BUCKET", "")
S3_ROOT = os.environ.get("HINT_S3_PREFIX", "hint")
# local scratch for downloaded raw CSVs, stage outputs and phase-4 checkpoints;
# deliberately stable (not a tempdir) so an interrupted embed run resumes
S3_WORK_DIR = os.environ.get("HINT_WORK_DIR", "s3_work")

# --- output filenames ----------------------------------------------------
# hint/ consumes articles_flat.parquet and the *embedded* chunks file in its
# tmp/ (where the embedded file is expected under the name rag_chunks.parquet
# - rename on copy, or update hint/)
BASE_FILENAME = "articles_base.parquet"
FLAT_FILENAME = "articles_flat.parquet"
CHUNKS_FILENAME = "rag_chunks.parquet"
EMBEDDED_FILENAME = "rag_chunks_embedded.parquet"