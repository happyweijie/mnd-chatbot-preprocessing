"""Run the full pipeline on one raw batch stored in S3.

Layout and conventions: README, "S3 storage layout".

    python -m pipeline batch --batch-id 2026-06

reads   s3://{bucket}/{root}/raw/year=2026/2026-06.csv
writes  s3://{bucket}/{root}/processed/{base,flattened,chunks,embeddings}/year=2026/2026-06.parquet

Design: the raw CSV is downloaded to a local work dir, the existing phase
functions run unchanged on local paths, and each stage's parquet is uploaded
as soon as its phase finishes (so a phase-4 failure still leaves phases 1-3
in S3). This is exactly what a SageMaker processing job does with
ProcessingInput/ProcessingOutput, so the same function can later be wrapped
in a job entry script; for now it is meant to be run from SageMaker Studio,
where boto3 credentials, the internal PyPI mirror and the embedding API all
work.

Phase-4 checkpoints stay in the local work dir and are never uploaded.
"""

from dataclasses import dataclass
from pathlib import Path

from . import config, phase1_clean, phase2_flatten, phase3_chunk, phase4_embed
from .batching import BatchRange, assert_batch_does_not_overlap, parse_batch_id


@dataclass(frozen=True)
class S3Layout:
    """S3 keys for one batch under the agreed hive-partitioned layout."""

    bucket: str
    root: str  # bare key prefix, e.g. "hint" or "teams/xyz/hint"
    batch: BatchRange

    @property
    def raw_dir(self) -> str:
        return f"{self.root}/raw/year={self.batch.year}/"

    @property
    def raw_key(self) -> str:
        return f"{self.raw_dir}{self.batch.batch_id}.csv"

    def stage_key(self, stage: str) -> str:
        return f"{self.root}/processed/{stage}/year={self.batch.year}/{self.batch.batch_id}.parquet"

    @property
    def base_key(self) -> str:
        return self.stage_key("base")

    @property
    def flattened_key(self) -> str:
        return self.stage_key("flattened")

    @property
    def chunks_key(self) -> str:
        return self.stage_key("chunks")

    @property
    def embeddings_key(self) -> str:
        return self.stage_key("embeddings")

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"


def validate_s3_settings(bucket: str, root: str) -> None:
    """Fail fast on misconfigured bucket/prefix before any AWS call."""
    assert bucket, "S3 bucket not set: pass --bucket or export HINT_BUCKET"
    assert not bucket.startswith("s3://"), (
        f"bucket must be the bare bucket name, not a URI: {bucket!r}"
    )
    assert root and not root.startswith(("s3://", "/")) and not root.endswith("/"), (
        f"prefix must be a bare key prefix without leading/trailing slashes, "
        f"e.g. 'teams/hint', got: {root!r}"
    )


# --- thin boto3 wrappers (kept separate so the rest stays testable offline) --

def _s3_client():
    import boto3  # optional dependency: only needed for S3 batch runs

    return boto3.client("s3")


def list_batch_ids(s3, bucket: str, raw_dir: str) -> list[str]:
    """Filename stems of every *.csv directly under raw/year=YYYY/."""
    stems = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=raw_dir, Delimiter="/"):
        for obj in page.get("Contents", []):
            name = obj["Key"][len(raw_dir):]
            if name.endswith(".csv"):
                stems.append(name[: -len(".csv")])
    return stems


def download(s3, bucket: str, key: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading s3://{bucket}/{key} -> {path}")
    s3.download_file(bucket, key, str(path))


def upload(s3, path: Path, bucket: str, key: str) -> None:
    print(f"uploading {path} -> s3://{bucket}/{key}")
    s3.upload_file(str(path), bucket, key)


# --- runner -----------------------------------------------------------------

def run_batch(
    batch_id: str,
    *,
    bucket: str = config.S3_BUCKET,
    root: str = config.S3_ROOT,
    work_dir: Path = Path(config.S3_WORK_DIR),
    skip_embeddings: bool = False,
    dry_run: bool = False,
    target_chunk_tokens: int = config.TARGET_CHUNK_TOKENS,
    max_chunk_tokens: int = config.MAX_CHUNK_TOKENS,
    overlap_tokens: int = config.OVERLAP_TOKENS,
    embed_batch_size: int = config.EMBED_BATCH_SIZE,
) -> S3Layout:
    batch = parse_batch_id(batch_id)  # fail fast on a malformed id
    validate_s3_settings(bucket, root)
    layout = S3Layout(bucket=bucket, root=root, batch=batch)

    local = work_dir / batch.batch_id
    raw_csv = local / "raw" / f"{batch.batch_id}.csv"
    base_path = local / config.BASE_FILENAME
    flat_path = local / config.FLAT_FILENAME
    chunks_path = local / config.CHUNKS_FILENAME
    embedded_path = local / config.EMBEDDED_FILENAME

    stages = [
        ("raw (read)", layout.raw_key, raw_csv),
        ("base", layout.base_key, base_path),
        ("flattened", layout.flattened_key, flat_path),
        ("chunks", layout.chunks_key, chunks_path),
    ]
    if not skip_embeddings:
        stages.append(("embeddings", layout.embeddings_key, embedded_path))

    print(f"batch {batch.batch_id}: {batch.start_date} .. {batch.end_date}")
    for name, key, path in stages:
        print(f"  {name:<12} {layout.uri(key)}  <->  {path}")
    if dry_run:
        print("dry run: no AWS calls made")
        return layout

    s3 = _s3_client()

    # non-overlap check: only filenames are listed, no data is downloaded
    existing = list_batch_ids(s3, bucket, layout.raw_dir)
    assert batch.batch_id in existing, (
        f"{layout.uri(layout.raw_key)} not found; raw/year={batch.year}/ "
        f"contains: {existing}"
    )
    assert_batch_does_not_overlap(batch, existing)

    local.mkdir(parents=True, exist_ok=True)
    download(s3, bucket, layout.raw_key, raw_csv)

    phase1_clean.run(raw_csv, base_path, batch_id=batch.batch_id)
    upload(s3, base_path, bucket, layout.base_key)

    phase2_flatten.run(base_path, flat_path)
    upload(s3, flat_path, bucket, layout.flattened_key)

    phase3_chunk.run(
        base_path,
        chunks_path,
        target_chunk_tokens=target_chunk_tokens,
        max_chunk_tokens=max_chunk_tokens,
        overlap_tokens=overlap_tokens,
    )
    upload(s3, chunks_path, bucket, layout.chunks_key)

    if skip_embeddings:
        print("skipping phase 4 (embeddings)")
    else:
        # checkpoint lives next to embedded_path in the work dir; phase 4
        # resumes from it if this run was interrupted, and deletes it on success
        phase4_embed.run(chunks_path, embedded_path, batch_size=embed_batch_size)
        upload(s3, embedded_path, bucket, layout.embeddings_key)

    print(f"batch {batch.batch_id} done")
    return layout
