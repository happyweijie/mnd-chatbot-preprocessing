# Article Preprocessing Pipeline

Two-phase pipeline for preprocessing housing policy articles into embeddings. Choose between **sentence-aware (tiktoken-based)** or **offline (character-based)** chunking.

## What It Does

### Phase 1: Flatten & Normalize
- Loads Excel/CSV/Parquet articles
- Flattens nested data (one row per article-topic pair)
- Normalizes dates, sentiment scores, and text fields
- Generates unique article IDs

**Output:** `articles.parquet`

### Phase 2: Chunk & Embed
- Splits articles into chunks (strategy depends on which script you use)
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

## Choose Your Script

| Script | Chunking Strategy | Environment | When to Use |
|--------|-------------------|-------------|------------|
| **`preprocess.py`** | Sentence-aware (tiktoken) | Requires tiktoken setup | Better chunk quality, precise token counts |
| **`simple_preprocess.py`** | Character-based | Offline, no dependencies | Restricted networks, SageMaker, no internet |

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

Alternatively, visit https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken and download the file directly.

**Step 2: Upload cache to SageMaker**

From SageMaker terminal:
```bash
# Create cache directory
mkdir -p ~/.cache/tiktoken

# Upload the cache files from your local machine
# (Use SageMaker file upload UI or scp if using SSH)

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

### Both: Metadata Prefix

Each chunk includes metadata before the content:

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

This metadata is embedded together with content, so vector search retrieves relevant context.

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

**Full pipeline with embeddings:**
```bash
python preprocess.py articles_2020_2025.xlsx ./output \
  --s3-bucket my-bucket \
  --s3-prefix processed
```

**Chunks only (no embeddings):**
```bash
python preprocess.py articles_2020_2025.xlsx ./output \
  --skip-embeddings
```

### simple_preprocess.py (Offline)

**Full pipeline with embeddings:**
```bash
python simple_preprocess.py articles_2020_2025.xlsx ./output \
  --s3-bucket my-bucket \
  --s3-prefix processed \
  --batch-size 4
```

**Chunks only (fast, no API calls):**
```bash
python simple_preprocess.py articles_2020_2025.xlsx ./output \
  --skip-embeddings
```

---

## Outputs

| File | Purpose |
|------|---------|
| `articles.parquet` | Flattened raw data (2013 articles → 2013+ rows) |
| `semantic_chunks.parquet` | Article chunks with metadata (ready for RAG) |
| `embeddings.npy` | Vector embeddings (NumPy array, one per chunk) |

All files saved to `output/` directory and optionally uploaded to S3.

---

## Performance Notes

### Chunking
- `preprocess.py`: ~1-2 minutes for 2000+ articles (uses tiktoken)
- `simple_preprocess.py`: ~30-60 seconds for 2000+ articles (offline)

### Embeddings
Both scripts:
- Batch size 4 = slower but reliable for government endpoints
- Each batch ≈ 1-3 seconds with api.ai.tech.gov.sg
- Total for 4000+ chunks at batch size 4 ≈ 2-3 hours

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
