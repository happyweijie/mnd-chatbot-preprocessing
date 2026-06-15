# Article Preprocessing Pipeline

Two-phase pipeline for preprocessing housing policy articles into embeddings. Works in restricted networks (no tiktoken dependencies).

## What It Does

### Phase 1: Flatten & Normalize
- Loads Excel/CSV/Parquet articles
- Flattens nested data (one row per article-topic pair)
- Normalizes dates, sentiment scores, and text fields
- Generates unique article IDs

**Output:** `articles.parquet`

### Phase 2: Chunk & Embed
- Splits articles into ~3,500 character chunks (equivalent to ~700 tokens)
- Adds metadata (title, topic, date, sentiment) to each chunk for context
- Calls OpenAI API to generate embeddings for retrieval

**Outputs:**
- `semantic_chunks.parquet` — Text chunks with metadata, ready for RAG retrieval
- `embeddings.npy` — Vector embeddings corresponding to each chunk

## Setup

```bash
cd preprocessing
python -m pip install -r requirements.txt
```

## Usage

**IMPORTANT:** Set environment variables first (one of two methods):

### Method 1: Shell Environment Variables

**SageMaker:**
```bash
export OPENAI_API_KEY='sk-...'
export OPENAI_BASE_URL='https://api.ai.tech.gov.sg/platform/models'

python simple_preprocess.py data.xlsx output/ \
  --s3-bucket my-bucket \
  --s3-prefix processed
```

### Method 2: Just Chunks (No Embedding)
If you only need chunks, skip embeddings entirely:
```bash
python simple_preprocess.py data.xlsx output/ \
  --skip-embeddings \
  --s3-bucket my-bucket \
  --s3-prefix processed
```

## How Chunking Works

Articles are split by **character count** (not tokens), making it work offline:

- **Target size:** ~3,500 characters (roughly 700 tokens for most articles)
- **Overlap:** 500 characters between chunks (for context continuity)
- **Sentence-aware:** Chunks try to respect sentence boundaries
- **Metadata prefix:** Each chunk includes title, topic, date, sentiment, and topic explanation before the content

Example chunk:
```
Title: New Housing Policy Announced
Topic: Urban Development
Published date: 2024-01-15
Year: 2024
News site: Housing Authority
Sentiment score: 0.75
Topic sentiment explanation: Positive response to new units

Content:
The government announced plans to build 5,000 new HDB units...
```

This metadata is embedded together with the content, so vector search retrieves relevant context, not just matching words.

## API Configuration
```bash
export OPENAI_API_KEY='sk-...'
export OPENAI_BASE_URL='https://api.ai.tech.gov.sg/platform/models'
```

## Examples

**Full pipeline with embeddings (4 items per batch for slow endpoints):**
```bash
python simple_preprocess.py articles_2020_2025.xlsx ./output \
  --s3-bucket my-bucket \
  --s3-prefix processed \
  --batch-size 4
```

**Chunks only (fast, no API calls):**
```bash
python simple_preprocess.py articles_2020_2025.xlsx ./output \
  --skip-embeddings \
  --s3-bucket my-bucket \
  --s3-prefix processed
```

**Local output only (no S3):**
```bash
python simple_preprocess.py articles_2020_2025.xlsx ./output
```

## Outputs

| File | Purpose |
|------|---------|
| `articles.parquet` | Flattened raw data (2013 article-topic pairs → 2013+ rows) |
| `semantic_chunks.parquet` | Article chunks with metadata (ready for RAG) |
| `embeddings.npy` | Vector embeddings (NumPy array, one per chunk) |

All files are saved to `output/` directory and optionally uploaded to S3.

## Performance Notes

- **Chunking Phase:** ~30-60 seconds for 2000+ articles (no network calls)
- **Embedding Phase:** Depends on API and batch size
  - Batch size 4 = slower but more reliable
  - Each batch of 4 chunks ≈ 1-3 seconds with government gateway
- **Total time:** ~2-3 hours for 2000+ articles at batch size 4

## Troubleshooting

**Embeddings hang or timeout:**
- Ensure `OPENAI_BASE_URL` is set correctly
- Try reducing batch size: `--batch-size 2`
- Check network connectivity to API endpoint

**"Name or service not known" error:**
- You're in a restricted network without internet
- Solution: Use `--skip-embeddings` and embed later in an unrestricted environment

**Files already exist:**
- Script overwrites existing output files silently
- Save outputs elsewhere if you need to preserve them
