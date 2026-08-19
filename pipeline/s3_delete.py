"""Delete one batch everywhere: raw CSV, all four processed parquets, and the
local scratch dir.

Layout and conventions: README, "S3 storage layout" / "Replacement rule".

    python -m pipeline delete --batch-id 2026-01_to_2026-05          # list only
    python -m pipeline delete --batch-id 2026-01_to_2026-05 --yes    # delete

removes  s3://{bucket}/{root}/raw/year=2026/2026-01_to_2026-05.csv
         s3://{bucket}/{root}/processed/{base,flattened,chunks,embeddings}/year=2026/2026-01_to_2026-05.parquet
         {work_dir}/2026-01_to_2026-05/            (downloaded raw, local parquets, phase-4 checkpoint)

Superseding a batch requires removing it *everywhere* first; deleting only
the raw file leaves stale processed outputs that consumers would read on top
of the replacement. This command owns that "everywhere" so it cannot be done
halfway. The pipeline itself still never deletes anything on its own - this
is a human-invoked, single-batch command:

- keys are derived from the batch id only (no wildcards, no prefix deletes)
- nothing is deleted without --yes; without it the command only reports
- a batch that exists nowhere is an error, so a typo cannot look like success
- the local {work_dir}/<batch>/ is cleared too, because phase 4 resumes from
  any checkpoint it finds there and would hand stale embeddings to a revised
  batch reprocessed under the same id later
"""

import shutil
from pathlib import Path

from . import config
from .batching import parse_batch_id
from .s3_batch import S3Layout, _s3_client, validate_s3_settings
from .s3_upload import STAGES, object_exists


def delete_object(s3, bucket: str, key: str) -> None:
    print(f"deleting s3://{bucket}/{key}")
    s3.delete_object(Bucket=bucket, Key=key)


def run_delete(
    batch_id: str,
    *,
    bucket: str = config.S3_BUCKET,
    root: str = config.S3_ROOT,
    work_dir: Path = Path(config.S3_WORK_DIR),
    yes: bool = False,
    dry_run: bool = False,
) -> S3Layout:
    batch = parse_batch_id(batch_id)  # fail fast on a malformed id
    validate_s3_settings(bucket, root)
    layout = S3Layout(bucket=bucket, root=root, batch=batch)

    keys = [layout.raw_key] + [layout.stage_key(stage) for stage in STAGES]
    local = Path(work_dir) / batch.batch_id

    print(f"batch {batch.batch_id}: {batch.start_date} .. {batch.end_date}")
    print("targets:")
    for key in keys:
        print(f"  {layout.uri(key)}")
    print(f"  {local}  (local scratch dir)")
    if dry_run:
        print("dry run: no AWS calls made")
        return layout

    s3 = _s3_client()

    present = [key for key in keys if object_exists(s3, bucket, key)]
    absent = [key for key in keys if key not in present]
    local_exists = local.is_dir()

    for key in present:
        print(f"  found    {layout.uri(key)}")
    for key in absent:
        print(f"  missing  {layout.uri(key)}")
    print(f"  {'found' if local_exists else 'missing'}    {local}")

    assert present or local_exists, (
        f"nothing to delete for batch {batch.batch_id!r}: none of its objects "
        f"exist under s3://{bucket}/{root}/ and {local} does not exist - check "
        f"the batch id, bucket and prefix"
    )

    # Use yes flag to confirm deletion
    if not yes:
        print(f"would delete {len(present)} S3 object(s)"
              f"{' + local scratch dir' if local_exists else ''}; "
              f"nothing deleted. Re-run with --yes to delete.")
        return layout

    # delete files from the batch both in the s3 bucket and locally
    for key in present:
        delete_object(s3, bucket, key)
    if local_exists:
        print(f"deleting {local}")
        shutil.rmtree(local)

    print(f"batch {batch.batch_id} deleted everywhere. next (to supersede it): "
          f"python -m pipeline upload --batch-id <new-id> --file <csv>")
    return layout