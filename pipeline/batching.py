"""Batch-id conventions for the S3 layout (see README "S3 storage layout").

A batch id is a filename stem covering whole months of one year:
  "2026-06"            a single month
  "2026-01_to_2026-05" an inclusive month range

One batch = one year is enforced here: cross-year deliveries must be split
into one file per year before upload, so a batch maps 1:1 to a single
year= partition in every processed stage.
"""

import calendar
import re
from dataclasses import dataclass
from datetime import date

_BATCH_RE = re.compile(r"^(\d{4})-(\d{2})(?:_to_(\d{4})-(\d{2}))?$")


@dataclass(frozen=True)
class BatchRange:
    batch_id: str
    year: int
    start_date: date  # first day of the start month
    end_date: date  # last day of the end month


def parse_batch_id(batch_id: str) -> BatchRange:
    m = _BATCH_RE.fullmatch(batch_id)
    if m is None:
        raise ValueError(
            f"Batch id {batch_id!r} is not of the form YYYY-MM or YYYY-MM_to_YYYY-MM"
        )

    y1, m1 = int(m.group(1)), int(m.group(2))
    y2, m2 = (int(m.group(3)), int(m.group(4))) if m.group(3) else (y1, m1)

    if not (1 <= m1 <= 12 and 1 <= m2 <= 12):
        raise ValueError(f"Batch id {batch_id!r} has an out-of-range month")
    
    # Ensure data does not span multiple years
    if y1 != y2:
        raise ValueError(
            f"Batch id {batch_id!r} spans years {y1} and {y2}; one batch = one "
            "year - split the delivery into one file per year"
        )

    # Ensure start month comes before end
    if m1 > m2:
        raise ValueError(f"Batch id {batch_id!r} has its start month after its end month")

    return BatchRange(
        batch_id=batch_id,
        year=y1,
        start_date=date(y1, m1, 1), # start date is 1st day of start month
        end_date=date(y2, m2, calendar.monthrange(y2, m2)[1]), # last date is last day of end month
    )


def assert_batch_does_not_overlap(batch: BatchRange, existing_batch_ids: list[str]) -> None:
    """
    A new batch's month range may not intersect any batch already uploaded for that year.

    The one exception is an exact batch-id match which overwrites only its own outputs. 
    Anything else that overlaps must be resolved by a human (delete the superseded
    batch everywhere first), never by the pipeline.

    `existing_batch_ids` are filename stems from raw/year=YYYY/; unparsable
    names are reported rather than skipped, since they would be invisible to
    this check forever otherwise.
    """
    overlapping = []
    malformed = []
    for other_id in existing_batch_ids:
        if other_id == batch.batch_id:
            continue  # same file, re-run and overwirte exisiting outputs
        try:
            other = parse_batch_id(other_id)
        except ValueError:
            malformed.append(other_id)
            continue

        if other.start_date <= batch.end_date and batch.start_date <= other.end_date:
            overlapping.append(other_id)

    assert not malformed, (
        f"raw/year={batch.year}/ contains files whose names are not batch ids "
        f"({malformed}) - rename or remove them before processing, otherwise "
        f"the non-overlap check cannot see what months they cover"
    )
    assert not overlapping, (
        f"Batch {batch.batch_id!r} overlaps existing batch(es) {overlapping} in "
        f"raw/year={batch.year}/. To supersede a batch, delete it everywhere first "
        f"(raw CSV + base/flattened/chunks/embeddings parquets), then re-run."
    )