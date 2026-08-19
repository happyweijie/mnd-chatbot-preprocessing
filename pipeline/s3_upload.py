"""Upload one raw batch CSV to S3 under the agreed layout.

Layout and conventions: README, "S3 storage layout".

    python -m pipeline upload --batch-id 2026-06 --file ~/2026-06.csv

writes  s3://{bucket}/{root}/raw/year=2026/2026-06.csv

Unlike a manual upload, the non-overlap rule is enforced *before* 
the object lands in S3 and the CSV is validated locally by running phase 1 
into a temp dir first - so a file whose rows don't match its filename 
never reaches raw/. Nothing but the original CSV is uploaded; 
the phase-1 output is discarded.

The upload never deletes and never silently overwrites: an existing file with
the same batch id needs --overwrite.
"""

import tempfile
from pathlib import Path

from . import config, phase1_clean
from .batching import assert_batch_does_not_overlap, parse_batch_id
from .s3_batch import (
    S3Layout,
    _s3_client,
    list_batch_ids,
    upload,
    validate_s3_settings,
)

STAGES = ("base", "flattened", "chunks", "embeddings")


def object_exists(s3, bucket: str, key: str) -> bool:
    """head_object wrapper: True if the key exists, False on any client error
    (404 for a missing key; a 403 on a key we can't see is treated the same,
    since either way we cannot confirm it is there)."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except Exception:  # botocore.exceptions.ClientError - kept generic so tests need no botocore
        return False
    return True


def validate_locally(csv_path: Path, batch_id: str) -> None:
    """Run phase 1 on the CSV into a throwaway dir so every phase-1 guard
    (rows within the batch's month range, parseable dates, unique article
    ids, non-empty topics, ...) fires *before* upload. The parquet it writes
    is discarded - only the original CSV is ever uploaded."""
    with tempfile.TemporaryDirectory(prefix="hint_upload_check_") as tmp:
        print(f"validating {csv_path} locally (phase 1 dry run, output discarded)")
        phase1_clean.run(csv_path, Path(tmp) / config.BASE_FILENAME, batch_id=batch_id)


def run_upload(
    batch_id: str,
    csv_path: Path,
    *,
    bucket: str = config.S3_BUCKET,
    root: str = config.S3_ROOT,
    overwrite: bool = False,
    validate: bool = True,
    dry_run: bool = False,
) -> S3Layout:
    batch = parse_batch_id(batch_id)  # fail fast on a malformed id
    validate_s3_settings(bucket, root)
    layout = S3Layout(bucket=bucket, root=root, batch=batch)

    csv_path = Path(csv_path)
    assert csv_path.is_file(), f"file not found: {csv_path}"
    assert csv_path.suffix.lower() == ".csv", (
        f"raw batches must be CSV files, got: {csv_path.name}"
    )

    if validate:
        validate_locally(csv_path, batch.batch_id)
    else:
        print("skipping local validation (--skip-validation)")

    print(f"batch {batch.batch_id}: {batch.start_date} .. {batch.end_date}")
    print(f"  {csv_path}  ->  {layout.uri(layout.raw_key)}")
    if dry_run:
        print("dry run: no AWS calls made")
        return layout

    s3 = _s3_client()

    # non-overlap check: only filenames are listed, no data is downloaded
    existing = list_batch_ids(s3, bucket, layout.raw_dir)
    assert_batch_does_not_overlap(batch, existing)

    if batch.batch_id in existing:
        assert overwrite, (
            f"{layout.uri(layout.raw_key)} already exists. Re-uploading a revised "
            f"batch under the same id is allowed but must be explicit: re-run with "
            f"--overwrite, then re-run `python -m pipeline batch --batch-id "
            f"{batch.batch_id}` so the processed outputs are refreshed too."
        )
        stale = [
            layout.stage_key(stage)
            for stage in STAGES
            if object_exists(s3, bucket, layout.stage_key(stage))
        ]
        print(f"overwriting existing raw file {layout.uri(layout.raw_key)}")
        if stale:
            print("WARNING: processed outputs from the previous upload exist and are "
                  "now stale until the batch is reprocessed:")
            for key in stale:
                print(f"  {layout.uri(key)}")

    upload(s3, csv_path, bucket, layout.raw_key)

    print(f"batch {batch.batch_id} uploaded. next: "
          f"python -m pipeline batch --batch-id {batch.batch_id} [--skip-embeddings]")
    return layout