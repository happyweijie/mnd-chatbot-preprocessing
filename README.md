# Article Preprocessing Pipeline

4-phase pipeline for preprocessing housing policy articles into semantic chunks and embeddings. Choose between **sentence-aware (tiktoken-based)** or **offline (character-based)** chunking. Uses Pydantic AI's Embedder for consistent embedding with the chatbot.

## What It Does

### Phase 1: Clean Base Articles (Unflattened)
- Loads Excel/CSV/Parquet articles
- Cleans and normalizes article-level fields (title, content, dates, news_site, url)
- Parses structured columns (topics list, sentiment_by_topic dict)
- Generates unique `article_id` from (title + published_date)
- Deduplicates articles

**Output:** `articles_base.parquet` — One row per article (unflattened)

### Phase 2: Create Flattened Topic-Level Table
- Explodes topics into separate rows (one row per article-topic pair)
- Maps sentiment scores and explanations to each topic
- Generates `article_id_key` and `article_topic_key` for SQL queries
- Retains article metadata for filtering and display

**Output:** `article_topics_flat.parquet` — Use this for SQL queries (top topics by year, sentiment trends, etc.)

### Phase 3: Build RAG Chunks
- Chunks article content from base articles (not from flattened data)
- Creates one row per chunk with optimal metadata
- Generates `retrieval_text` containing: title, topics, content excerpt
- Excludes sentiment scores, explanations, IDs, URLs (not semantically meaningful)
- Preserves metadata for filtering and citation

**Output:** `rag_chunks.parquet` — Chunks ready for RAG retrieval with proper `retrieval_text`

### Phase 4: Generate Embeddings
- Uses Pydantic AI's Embedder (text-embedding-3-small, 1536 dimensions)
- Creates embeddings from `retrieval_text` in RAG chunks
- Batched processing with configurable batch size
- Adds `embedding` column to RAG chunks and overwrites `rag_chunks.parquet`

**Output:** `rag_chunks.parquet` (updated with `embedding` column)

## Setup

```bash
cd preprocessing
python -m pip install -r requirements.txt
```

## Choose Your Script

| Script | Chunking Strategy | Environment | When to Use |
|--------|-------------------|-------------|------------|
| **`preprocess.py`** | Sentence-aware (tiktoken) | Requires tiktoken setup | Better chunk quality, precise token counts |
| **`simple_preprocess.py`** | Character-based | Offline, no dependencies | Restricted networks, SageMaker, no internet |

Both scripts implement the same 4-phase pipeline and use Pydantic AI's Embedder for consistent embeddings.

---

## Option 1: `preprocess.py` (Sentence-Aware + Tiktoken)

**Advantages:**
- Respects sentence boundaries — no mid-word cuts
- Accurate token counting (targets exactly 700 tokens per chunk)
- Higher quality chunks for retrieval

**Requirements:**
- Must manually download and upload tiktoken cache to Sagemaker.

### Setup for Restricted Networks (SageMaker)

**Step 1: Download tiktoken cache locally**

On your local machine with internet, run:
```bash
python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
```

This downloads encoding files to your local cache:
- **Windows:** `%LOCALAPPDATA%\tiktoken_cache\`
- **Linux/Mac:** `~/.cache/tiktoken/`

Alternatively, visit https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken and download the tiktoken file directly.

**Step 2: Upload to SageMaker**

From SageMaker terminal:
```bash
# Create cache directory
mkdir -p ~/.cache/tiktoken

# upload the contents of your local cache/the tiktoken file
```

If uploading the `.tiktoken` file directly, rename it as well:
```bash
# Rename file
mv ~/.cache/tiktoken/cl100k_base.tiktoken ~/.cache/tiktoken/9b5ad71b2ce5302211f9c61530b329a4922fc6a4
```

**Step 3: Set environment variable**

```bash
export TIKTOKEN_CACHE_DIR=~/.cache/tiktoken
export OPENAI_API_KEY='sk-...'
export OPENAI_BASE_URL='https://api.ai.tech.gov.sg/platform/models'

python preprocess.py data.xlsx output/ \
  --s3-bucket my-bucket \
  --s3-prefix processed \
  --batch-size 4
```

### Local Usage (No Cache Setup Needed)

If tiktoken is able to download and cache on its own, just run:

```bash
export OPENAI_API_KEY='sk-...'
export OPENAI_BASE_URL='https://api.ai.tech.gov.sg/platform/models'

python preprocess.py data.xlsx output/ \
  --s3-bucket my-bucket \
  --s3-prefix processed
```

---

## Option 2: `simple_preprocess.py` (Character-Based, Offline)

**Advantages:**
- Works completely offline — no internet needed
- No tiktoken setup required
- Fast chunking (~30-60 seconds for 2000+ articles)
- Works in restricted networks immediately

**Trade-off:**
- Character-based splitting (may cut mid-word/sentence)
- ~3,500 characters ≈ ~700 tokens (approximate)

### Setup

**Set environment variables:**

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = 'sk-...'
$env:OPENAI_BASE_URL = 'https://api.ai.tech.gov.sg/platform/models'

python simple_preprocess.py data.xlsx output/ `
  --s3-bucket my-bucket `
  --s3-prefix processed `
  --batch-size 4
```

**Linux/Mac/SageMaker:**
```bash
export OPENAI_API_KEY='sk-...'
export OPENAI_BASE_URL='https://api.ai.tech.gov.sg/platform/models'

python simple_preprocess.py data.xlsx output/ \
  --s3-bucket my-bucket \
  --s3-prefix processed \
  --batch-size 4
```

### Just Chunks (No Embedding)

If you only need chunks, skip embeddings:
```bash
python simple_preprocess.py data.xlsx output/ \
  --skip-embeddings \
  --s3-bucket my-bucket \
  --s3-prefix processed
```

---

## How Chunking Works

### Sentence-Aware (preprocess.py + tiktoken)

- **Target size:** 700 tokens (precise)
- **Max size:** 900 tokens (enforced)
- **Overlap:** 100 tokens between chunks
- **Strategy:** Splits on sentence boundaries, groups sentences to reach target size
- **Benefit:** No mid-word cuts, respects discourse structure

Example: "The government announced policy. This will affect housing." → one chunk (if <700 tokens)

### Character-Based (simple_preprocess.py)

- **Target size:** ~3,500 characters (≈ 700 tokens)
- **Overlap:** 500 characters between chunks
- **Strategy:** Splits at character positions, strips whitespace
- **Trade-off:** May cut mid-word, but works offline

Example: "The quick brown fox jumps over..." → splits at char 3500 (might hit mid-word)

### Retrieval Text Format

Each chunk's `retrieval_text` (used for embeddings) includes only semantically meaningful fields:

```
Title: New Housing Policy Announced
Topics: Urban Development, Public Housing

Article excerpt:
The government announced plans to build 5,000 new HDB units...
```

**Included in retrieval_text:**
- Title
- Topics (comma-separated)
- Content chunk

**Excluded from retrieval_text** (not semantically meaningful for retrieval):
- Sentiment scores and explanations (model-derived, not source evidence)
- IDs and URLs (not helpful for semantic search)
- Technical metadata

---

## API Configuration

### For Government AI Gateway (api.ai.tech.gov.sg)
```bash
export OPENAI_API_KEY='sk-...'
export OPENAI_BASE_URL='https://api.ai.tech.gov.sg/platform/models'
```

### For Standard OpenAI API
```bash
export OPENAI_API_KEY='sk-...'
# Do NOT set OPENAI_BASE_URL (uses default api.openai.com)
```

---

## Examples

### preprocess.py (Sentence-Aware)

**Full 4-phase pipeline with embeddings:**
```bash
python preprocess.py articles_2020_2025.xlsx ./output \
  --s3-bucket my-bucket \
  --s3-prefix processed
```

**All phases except embeddings:**
```bash
python preprocess.py articles_2020_2025.xlsx ./output \
  --skip-embeddings
```

### simple_preprocess.py (Offline)

**Full 4-phase pipeline with embeddings:**
```bash
python simple_preprocess.py articles_2020_2025.xlsx ./output \
  --s3-bucket my-bucket \
  --s3-prefix processed \
  --batch-size 4
```

**Phases 1-3 only (fast, no API calls):**
```bash
python simple_preprocess.py articles_2020_2025.xlsx ./output \
  --skip-embeddings
```

---

## Outputs

| File | Phase | Purpose |
|------|-------|---------|
| `articles_base.parquet` | 1 | Unflattened base articles (one row per article) |
| `article_topics_flat.parquet` | 2 | Flattened topic-level table (one row per article-topic pair) — use for SQL queries |
| `rag_chunks.parquet` | 3-4 | Article chunks with retrieval_text and (if Phase 4 completes) embeddings column |

**Local:** All files saved to `output/` directory

**S3:** All files uploaded to `s3://{bucket}/{prefix}/` when `--s3-bucket` and `--s3-prefix` are provided

**Note:** When embeddings are generated (Phase 4), `rag_chunks.parquet` is overwritten to include the `embedding` column (1536-dimensional vectors as a list per row).

---

## Using the Output

### For SQL Analytics (Topic Trends, Sentiment Analysis)
Use `article_topics_flat.parquet`:
```sql
-- Top topics by year
SELECT topic, year, COUNT(*) as count
FROM article_topics_flat
GROUP BY topic, year
ORDER BY year DESC, count DESC;

-- Average sentiment by topic
SELECT topic, AVG(sentiment_score) as avg_sentiment
FROM article_topics_flat
WHERE sentiment_score IS NOT NULL
GROUP BY topic;
```

### For RAG Retrieval (Semantic Search)
Use `rag_chunks.parquet` (with embeddings column):
1. Load chunks with embeddings: `chunks = pd.read_parquet('rag_chunks.parquet')`
2. Convert embedding column to numpy array: `embeddings = np.array([e for e in chunks['embedding']], dtype=np.float32)`
3. Generate query embedding using Pydantic AI Embedder (same model: text-embedding-3-small)
4. Vector similarity search to find top-K relevant chunks
5. Use `title`, `url`, `published_date` metadata for citation

---

## Performance Notes

### Chunking
- `preprocess.py`: ~1-2 minutes for 2000+ articles (uses tiktoken)
- `simple_preprocess.py`: ~30-60 seconds for 2000+ articles (offline)

### Embeddings
Both scripts use Pydantic AI with text-embedding-3-small (1536 dimensions):
- Batch size 4 = slower but reliable for government endpoints
- Each batch ≈ 1-3 seconds with api.ai.tech.gov.sg
- Total for 4000+ chunks at batch size 4 ≈ 2-3 hours

### Data Sizes
- Typical input: 2000+ articles with multiple topics
- Phase 1: 2000 articles → 2000 rows (articles_base ~5-10MB)
- Phase 2: 2000 articles × 3-5 topics avg → 6000-10000 rows (article_topics_flat ~10-20MB)
- Phase 3: 2000 base articles × 2-4 chunks per article → 4000-8000 chunks (rag_chunks ~50-100MB)
- Phase 4: Adds embeddings column (1536 dim per chunk) → ~100-350MB (rag_chunks with embeddings)

---

## Troubleshooting

**Embeddings hang or timeout:**
- Ensure `OPENAI_BASE_URL` is set correctly
- Try reducing batch size: `--batch-size 2`
- Check network connectivity to your API endpoint

**For `preprocess.py`: "tiktoken not found" error:**
- You forgot to set `TIKTOKEN_CACHE_DIR`
- Or cache files weren't uploaded to SageMaker
- Use `simple_preprocess.py` instead (no tiktoken needed)

**For `simple_preprocess.py`: "Name or service not known" error:**
- You're in restricted network AND trying to embed
- Use `--skip-embeddings` and embed later in unrestricted environment
- Or use `preprocess.py` with pre-cached tiktoken

**Files already exist:**
- Both scripts overwrite existing output files
- Save outputs elsewhere if you need to preserve them

**Pydantic AI Embedder errors:**
- Verify `OPENAI_API_KEY` is valid
- Check that embedder model (text-embedding-3-small) is available in your API endpoint
- For government endpoints, confirm the model name is supported

---

## Implementation Details

### Data Flow

```
Raw Input (xlsx/csv)
    ↓
[Phase 1: clean_base_articles]
    ↓ articles_base.parquet (unflattened)
    ├→ [Phase 2: flatten_base_articles]
    │     ↓ article_topics_flat.parquet (SQL table)
    │
    └→ [Phase 3: build_rag_chunks_from_base_articles]
          ↓ rag_chunks.parquet (chunks with retrieval_text)
          ↓
       [Phase 4: embed_texts_with_pydantic_ai]
          ↓ embeddings.npy (1536-dim vectors)
```

### Column Definitions

**articles_base.parquet:**
- `article_id`: Unique identifier (title + published_date)
- `title`, `content`, `published_date`, `quarter_year`, `year`
- `news_site`, `url`
- `topics`: List of topics
- `sentiment_by_topic`: Dict mapping topic → sentiment score
- `topic_sentiment_explanations`: Dict mapping topic → explanation

**article_topics_flat.parquet:**
- `article_id_key`: Original article_id
- `article_topic_key`: article_id_key + topic (unique identifier for SQL)
- `title`, `published_date`, `year`, `quarter_year`, `news_site`, `url`
- `topic`, `sentiment_score`, `explanation`

**rag_chunks.parquet:**
- `chunk_id`: Unique chunk identifier
- `article_id`: Reference to source article
- `chunk_index`: Index within article
- `content_chunk`: Actual content text
- `retrieval_text`: Text used for embedding (title + topics + content_chunk)
- `title`, `published_date`, `year`, `news_site`, `url`
- `topics`: Original topics list (for filtering)
- `embedding`: 1536-dimensional embedding vector (added in Phase 4, stored as list)
