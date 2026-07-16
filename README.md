# Article Preprocessing Pipeline

Polars-based pipeline that turns the raw housing-articles CSV into the parquet
files the H.A.B.I.T chatbot consumes. Consists of four discrete phases, each a pure
parquet-in/parquet-out step.

| Phase | Command | Input | Output |
|---|---|---|---|
| 1 clean | `clean` | raw CSV | `articles_base.parquet` (one row per article) |
| 2 flatten | `flatten` | base parquet | `articles_flat.parquet` (one row per article-topic, for SQL) |
| 3 chunk | `chunk` | base parquet | `rag_chunks.parquet` (sentence-aware token chunks + retrieval_text) |
| 4 embed | `embed` | chunks parquet | `rag_chunks.parquet` with an `embedding` column (list[f32]) |

All tunable constants (chunk sizes, embedding model/dimensions, batch size,
rate limits, filenames) live in **`pipeline/config.py`** — edit there, or use
the CLI overrides, to experiment.

## Setup

```powershell
python -m pip install -r requirements.txt
```

Environment variables (phase 4 only):

```powershell
$env:OPENAI_API_KEY = 'sk-...'
# optional, for the gov endpoint; scheme is added automatically if missing
$env:OPENAI_BASE_URL = 'https://api.ai.tech.gov.sg/platform/models'
```

Offline machines (phase 3 needs tiktoken): set `TIKTOKEN_CACHE_DIR` to an
**absolute** path (`~` is not expanded) containing the cached cl100k_base
encoder file `9b5ad71b2ce5302211f9c61530b329a4922fc6a4`.

## Usage

```powershell
# full pipeline
python -m pipeline all articles.csv out/

# without embeddings (no API access needed)
python -m pipeline all articles.csv out/ --skip-embeddings

# individual phases
python -m pipeline clean articles.csv out/
python -m pipeline flatten out/
python -m pipeline chunk out/ --target-tokens 350 --max-tokens 500 --overlap-tokens 75
python -m pipeline embed out/ --batch-size 20

# smoke-test embedding on 8 chunks (writes rag_chunks.sample.parquet)
python -m pipeline embed out/ --limit 8
```

## Notes

- **Article identity**: `article_id = title | YYYY-MM-DD | news_site`.
  `published_date` is truncated to day precision; the parser accepts both ISO
  (`2022-08-08T23:31:00+08:00`) and Excel-converted (`8/8/2022 23:31`) date
  formats, and fails loudly on anything else.
- **Chunking**: sentence-aware, targets 350 tokens with a 500-token cap and a
  75-token sentence overlap between chunks (a chunk can reach cap + overlap in
  rare oversized-sentence cases). Assumes single-line article content —
  guarded by assertions in phase 3.
- **Embedding**: `text-embedding-3-small`, 1536 dims, via Pydantic AI.
  Requests are paced to the gov endpoint's limits (20 req/min, 200k
  tokens/min), retried with backoff, and checkpointed — an interrupted run
  resumes from the last checkpoint instead of starting over.
- **Guard assertions**: every phase asserts data integrity (no null
  dates/titles, unique article/chunk ids, topic/sentiment consistency) and
  fails with a counted, actionable message instead of writing bad parquet.

## Tests

```powershell
python -m pytest tests/          # or: python tests/test_chunking.py
```

## Layout

```
pipeline/
├── __main__.py        # CLI (python -m pipeline ...)
├── config.py          # all tunable constants
├── assertions.py      # shared data-integrity guards
├── chunking.py        # pure token-chunking functions
├── phase1_clean.py    # raw CSV -> articles_base.parquet
├── phase2_flatten.py  # base -> articles_flat.parquet
├── phase3_chunk.py    # base -> rag_chunks.parquet
└── phase4_embed.py    # chunks -> chunks + embedding column
polars_notebooks/      # exploratory notebooks the pipeline was derived from
```
