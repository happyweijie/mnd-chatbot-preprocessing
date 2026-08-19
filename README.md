# Hint Article Preprocessing Pipeline

Polars-based pipeline that turns raw housing-articles CSV into the parquet
files the hint chatbot consumes. Consists of four discrete phases, each a pure
parquet-in/parquet-out step.

| Phase | Command | Input | Output |
|---|---|---|---|
| 1 clean | `clean` | raw CSV | `articles_base.parquet` (one row per article) |
| 2 flatten | `flatten` | base parquet | `articles_flat.parquet` (one row per article-topic, for SQL) |
| 3 chunk | `chunk` | base parquet | `rag_chunks.parquet` (sentence-aware token chunks + retrieval_text) |
| 4 embed | `embed` | chunks parquet | `rag_chunks_embedded.parquet` (chunks + `embedding` column, list[f32]) |

All tunable constants (chunk sizes, batch size, rate limits, filenames) live
in **`pipeline/config.py`** — edit there, or use the CLI overrides, to
experiment. The embedding model and dimensions are configured via environment
variables (`EMBEDDING_MODEL`, `EMBEDDING_DIM`) — see Setup below.

## Setup

```bash
python -m pip install -r requirements.txt
```

Setup Environment variables (phase 4 only):

```powershell
OPENAI_API_KEY = 'sk-...'
OPENAI_BASE_URL = 'xxxx`
# optional overrides (defaults: text-embedding-3-small, model's native dims)
EMBEDDING_MODEL = 'text-embedding-3-large'
EMBEDDING_DIM = '3072'
```

Alternatively, put them in a `.env` file (see `.env.example`) — it is loaded
automatically, and real environment variables take precedence. Leave
`EMBEDDING_DIM` unset to use the model's default length (1536 for
text-embedding-3-small, 3072 for text-embedding-3-large).

Offline machines (phase 3 needs tiktoken): set `TIKTOKEN_CACHE_DIR` to an
**absolute** path (`~` is not expanded) containing the cached cl100k_base
encoder file `9b5ad71b2ce5302211f9c61530b329a4922fc6a4`.

### Quick Setup using cat
```bash
cat << EOF > .env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=xxxx
EMBEDDING_MODEL=text-embedding-3-large
TIKTOKEN_CACHE_DIR=/home/sagemaker-user/.cache/tiktoken
EOF
```

### Setting up cached tiktoken encoder
Download `cl100k_base.tiktoken` and upload it to `/.cache/tiktoken`. The `.cache` folder is found in the home directory. You may need to create a folder called `tiktoken` if one does not exist.

Rename `cl100k_base.tiktoken` to `9b5ad71b2ce5302211f9c61530b329a4922fc6a4`. Note that there is no file extension.
```bash
# enter the cache directory
cd /home/sagemaker-user/.cache/tiktoken

# rename tiktoken tokenizer
mv cl100k_base.tiktoken 9b5ad71b2ce5302211f9c61530b329a4922fc6a4
```

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

# smoke-test embedding on 8 chunks (writes rag_chunks_embedded.sample.parquet)
python -m pipeline embed out/ --limit 8
```

## S3 storage layout

This is the agreed layout for raw batches and every processed
stage. `python -m pipeline batch` writes it, the hint chatbot reads it
directly (`hint/src/config.py`, S3 mode: `processed/flattened/**/*.parquet`
and `processed/embeddings/**/*.parquet`).

```
{prefix}/                              # HINT_S3_PREFIX, e.g. teams/xyz/hint (never the bucket root)
├── raw/                               # one CSV per batch (python -m pipeline upload)
│   ├── year=2020/2020-01_to_2020-12.csv
│   ├── ...
│   ├── year=2025/2025-01_to_2025-12.csv
│   └── year=2026/
│       ├── 2026-01_to_2026-05.csv
│       └── 2026-06.csv
└── processed/                         # written by the pipeline, one parquet per batch per stage
    ├── base/year=2026/2026-06.parquet          # phase 1  articles_base
    ├── flattened/year=2026/2026-06.parquet     # phase 2  articles_flat   (hint: SQL)
    ├── chunks/year=2026/2026-06.parquet        # phase 3  rag_chunks
    └── embeddings/year=2026/2026-06.parquet    # phase 4  rag_chunks_embedded (hint: RAG)
```

Conventions:

- **Hive-style `year=YYYY/` partition directories only**; filenames carry no
  `key=` prefix. Consumers read via glob (`processed/<stage>/**/*.parquet`);
  if a single merged file is ever needed, add a separate compaction step.
- **Batch id = filename stem**: `2026-06` (one month) or `2026-01_to_2026-05`
  (inclusive month range). Historical backfill uses `20XX-01_to_20XX-12`.
- **One batch = one year (enforced)**: a batch file may only contain articles
  published within the range named in its filename, so it maps 1:1 to a single
  `year=` partition in every stage. Phase 1 fails fast, listing the offending
  rows, if any `published_date` falls outside the range. Cross-year deliveries
  must be split into one file per year before upload.
- **Processed files are named after their source batch**: each run writes
  `<stage>/year=YYYY/<batch>.parquet` only — append-only and idempotent;
  re-running a batch overwrites only its own files.
- **Non-overlap (enforced)**: before uploading (`pipeline upload`) and again
  before processing (`pipeline batch`), the pipeline lists `raw/year=YYYY/`
  and parses each filename's month range; it refuses if the new batch's range
  intersects any existing file's range. Exception: an exact filename match is
  an idempotent re-run (`upload` additionally requires `--overwrite` for it).
  No data is downloaded for this check. Together with the row-range rule this
  makes overlap impossible: filenames cannot overlap, and rows cannot escape
  their filename's range.
- **Lineage**: every processed row carries `batch_id` (the filename stem)
  and `year`, so a batch can be traced or selectively reprocessed.
- **Dedup** (`article_id = title | date | news_site`) is enforced within a
  batch only; a cross-batch check can be added later without changing the
  layout.
- **Replacement rule — nothing is ever deleted automatically**; humans
  delete, the pipeline only rejects.
  - New months, old data unchanged: upload the new months as a new batch
    file (e.g. `2026-06.csv`). No deletions.
  - Same batch revised (same coverage): re-upload under the **same**
    filename (`pipeline upload --overwrite`) and reprocess — outputs are
    overwritten in place.
  - Superseding a batch with different coverage (e.g. `2026-01_to_2026-05`
    → `2026-01_to_2026-06`): delete the old batch **everywhere** first —
    1 raw CSV + its 4 identically named processed parquets — then upload
    and process the replacement. Deleting only the raw file leaves stale
    outputs in `processed/` and consumers would read those months twice.
- **Phase 4 checkpoints** (`*.checkpoint.parquet`) stay local; never under
  `processed/embeddings/`.

## Running a batch against S3 (production path)

`python -m pipeline batch` runs all four phases on one raw batch stored in
S3 and writes each stage back in the layout above:

```
s3://{bucket}/{prefix}/raw/year=2026/2026-06.csv                          <- pipeline upload
s3://{bucket}/{prefix}/processed/base/year=2026/2026-06.parquet           <- phase 1
s3://{bucket}/{prefix}/processed/flattened/year=2026/2026-06.parquet      <- phase 2
s3://{bucket}/{prefix}/processed/chunks/year=2026/2026-06.parquet         <- phase 3
s3://{bucket}/{prefix}/processed/embeddings/year=2026/2026-06.parquet     <- phase 4
```

Run it from **SageMaker Studio** on the work platform — that is where boto3
credentials (execution role), the internal PyPI mirror, the tiktoken cache
and the embedding API all work. It is a plain Python process, not a
SageMaker job — see "Handover: SageMaker Processing job" below for
turning it into one.

```bash
# 1. settings (or put them in .env)
export HINT_BUCKET=<bucket-name>          # bare name, no s3://
export HINT_S3_PREFIX=teams/xyz/hint      # bare prefix, no leading/trailing slash
export HINT_WORK_DIR=/home/sagemaker-user/hint_work   # persistent scratch (EFS)

# 2. upload the raw batch - the key raw/year=2026/2026-06.csv is derived from the id
python -m pipeline upload --batch-id 2026-06 --file ~/2026-06.csv --dry-run   # local checks only
python -m pipeline upload --batch-id 2026-06 --file ~/2026-06.csv

# 3. check the resolved keys - makes no AWS calls
python -m pipeline batch --batch-id 2026-06 --dry-run

# 4. run (phases 1-3 first if you want to inspect before spending API quota)
python -m pipeline batch --batch-id 2026-06 --skip-embeddings
python -m pipeline batch --batch-id 2026-06
```

How the uploader behaves (`pipeline upload`; use it instead of the S3
console, which is unreliable on the work platform, or `aws s3 cp`, which
lets you type the wrong key):

- Validates the batch id and that `--file` is an existing `.csv`.
- **Local validation** (default; `--skip-validation` to bypass): runs phase 1
  on the CSV into a temp dir so every phase-1 guard fires *before* upload —
  rows inside the filename's month range, parseable dates, unique article
  ids, non-empty topics. The phase-1 output is discarded; only the original
  CSV is uploaded. Without this, a bad file would sit in `raw/` until
  `batch` rejects it and a human deletes it.
- `--dry-run` stops after the local checks and prints the target key without
  any AWS call.
- Runs the **non-overlap** check against `raw/year=YYYY/` filenames; refuses
  overlapping or malformed neighbours with nothing uploaded.
- **Never silently overwrites**: if `raw/year=YYYY/<batch>.csv` already
  exists it refuses unless `--overwrite` is passed, and then warns which
  `processed/<stage>/` parquets from the previous run are now stale until
  the batch is reprocessed. It never deletes anything.

How the runner behaves:

- Validates the batch id, then runs the **non-overlap** check by listing
  `raw/year=YYYY/` filenames — before downloading anything. A missing raw
  file or an overlapping batch fails here with nothing written.
- Phase 1 enforces **rows match the filename** and stamps `batch_id`.
- **Per-stage upload**: each parquet is uploaded as soon as its phase
  finishes, so a phase-4 failure still leaves base/flattened/chunks in S3.
- **Checkpoints stay local**: phase 4 checkpoints in `HINT_WORK_DIR` and
  resumes from there if interrupted; keep that dir on persistent storage.
- **Re-runs are idempotent**: the same batch id overwrites only its own
  four parquets.

## Handover: SageMaker Processing job (not built)

The intended end state is to run `pipeline batch` as a scheduled SageMaker
Processing job instead of by hand from Studio. A phase-1-only prototype was
built and smoke-tested on the gov platform (August 2026) and then removed
from the repo, because it could not be completed without platform details
that were not available. What was learned, so the next person does not
re-discover it:

| Item | Finding |
|---|---|
| Container image | `FrameworkProcessor(estimator_cls=PyTorch, framework_version="2.2", py_version="py310")` — the stock AWS Deep Learning Container **pulls fine** from ECR (account 763104351884). The sklearn image ships Python 3.8, too old for this code. |
| VPC / subnets / KMS | Auto-injected into every job by the platform's `/etc/xdg/sagemaker/config.yaml` — do **not** set `NetworkConfig` yourself. |
| Execution role | Pass the **full role ARN** (`arn:aws:iam::<acct>:role/<name>`). A bare role name makes the SDK call the global IAM endpoint, which the VPC blocks (connect timeout). |
| SDK version | Launcher needs `sagemaker>=2.190,<3` on Studio; v3 restructured the package and breaks `FrameworkProcessor` / `sagemaker.workflow` imports. Silence v2 warnings with `SAGEMAKER_SUPPRESS_V2_WARNING=1`. |
| Code upload | Set `PipelineSession(default_bucket=..., default_bucket_prefix=HINT_S3_PREFIX)` so the SDK's code tarballs land under our prefix, not the shared bucket's root. Stage only `pipeline/` + the entry script as `source_dir` (never the repo root — `notebooks/` holds large data). |
| **Blocker: pip inside the job** | Job containers have **no pypi.org egress**; `pip install -r requirements.txt` times out. Studio's pip works because it is configured for an internal mirror. Fix = forward that mirror into the job via `env=`: `CA_REPOSITORY_ARN` if it is CodeArtifact (the DLC bootstrap logs in itself using the execution role) or `PIP_INDEX_URL` (+ `PIP_TRUSTED_HOST` if http/self-signed). Find it in a Studio terminal with `pip config list -v` / `env \| grep -i codeartifact`. Fallback: vendor wheels into `source_dir` and use `--no-index --find-links`. |
| Untested | Whether job subnets can reach the embedding endpoint (`OPENAI_BASE_URL`). Studio can; jobs may not. |

Steps to build it:

1. Get the PyPI mirror config (above) and verify with a throwaway
   `FrameworkProcessor.run()` whose script just prints `polars.__version__`.
2. Write a small entry script that calls the phase functions the way
   `pipeline/s3_batch.py:run_batch` does, but with local dirs only:
   `ProcessingInput` (raw CSV → `/opt/ml/processing/input`) and
   `ProcessingOutput` (`/opt/ml/processing/output` → `processed/<stage>/year=YYYY/`)
   replace the boto3 download/upload. Name the output file `<batch>.parquet`
   so no rename step is needed. Nothing in `pipeline/` needs to change.
3. Do the non-overlap check in the *launcher* before `pipeline.start()`
   using `pipeline.s3_batch.list_batch_ids` +
   `pipeline.batching.assert_batch_does_not_overlap`.
4. Derive the `Year` pipeline parameter from `--batch-id` in the launcher and
   cross-check it in the entry script, so a console-started execution with a
   mismatched year fails instead of writing to the wrong partition.
5. Test the embedding endpoint from inside a job before wiring phase 4.

## Notes

- **Article identity**: `article_id = title | YYYY-MM-DD | news_site`.
  `published_date` is truncated to day precision; the parser accepts both ISO
  (`2022-08-08T23:31:00+08:00`) and Excel-converted (`8/8/2022 23:31`) date
  formats, and fails loudly on anything else.
- **Chunking**: sentence-aware, targets 350 tokens with a 500-token cap and a
  75-token sentence overlap between chunks (a chunk can reach cap + overlap in
  rare oversized-sentence cases). Assumes single-line article content —
  guarded by assertions in phase 3.
- **Embedding**: `text-embedding-3-small` by default (override via
  `EMBEDDING_MODEL`/`EMBEDDING_DIM`), via Pydantic AI.
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
├── phase4_embed.py    # chunks -> chunks + embedding column
├── batching.py        # batch-id parsing + non-overlap rule (see "S3 storage layout")
├── s3_upload.py       # local raw CSV -> raw/year=YYYY/<batch>.csv (python -m pipeline upload)
└── s3_batch.py        # S3 in -> all phases -> S3 out (python -m pipeline batch)
tests/                 # pytest suite (chunking, batching, S3 uploader + batch runner with fake S3)
notebooks/             # exploratory notebooks the pipeline was derived from
```
