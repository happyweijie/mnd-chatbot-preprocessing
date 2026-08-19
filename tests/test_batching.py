"""Tests for pipeline.batching. Run with pytest, or directly:
python tests/test_batching.py
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from pipeline.batching import assert_batch_does_not_overlap, parse_batch_id


def test_single_month():
    batch = parse_batch_id("2026-06")
    assert batch.year == 2026
    assert batch.start_date == date(2026, 6, 1)
    assert batch.end_date == date(2026, 6, 30)


def test_month_range():
    batch = parse_batch_id("2026-01_to_2026-05")
    assert batch.year == 2026
    assert batch.start_date == date(2026, 1, 1)
    assert batch.end_date == date(2026, 5, 31)


def test_full_year_backfill_form():
    batch = parse_batch_id("2020-01_to_2020-12")
    assert batch.start_date == date(2020, 1, 1)
    assert batch.end_date == date(2020, 12, 31)


def test_february_end_date_leap_year():
    assert parse_batch_id("2024-02").end_date == date(2024, 2, 29)
    assert parse_batch_id("2025-02").end_date == date(2025, 2, 28)


@pytest.mark.parametrize("bad_id", [
    "2026",              # no month
    "2026-6",            # month not zero-padded
    "2026-06.csv",       # extension included
    "2026-06_to_2026",   # incomplete range
    "jan-2026",          # not numeric
    "",
])
def test_malformed_ids_rejected(bad_id):
    with pytest.raises(ValueError, match="not of the form"):
        parse_batch_id(bad_id)


def test_out_of_range_month_rejected():
    with pytest.raises(ValueError, match="out-of-range month"):
        parse_batch_id("2026-13")


def test_cross_year_rejected():
    with pytest.raises(ValueError, match="one batch = one year"):
        parse_batch_id("2025-12_to_2026-02")


def test_start_after_end_rejected():
    with pytest.raises(ValueError, match="start month after its end month"):
        parse_batch_id("2026-05_to_2026-01")


# --- non-overlap rule -------------------------------------------------------

def test_overlap_disjoint_batches_allowed():
    assert_batch_does_not_overlap(parse_batch_id("2026-06"), ["2026-01_to_2026-05"])


def test_overlap_adjacent_months_allowed():
    # touching but not intersecting: May ends 31st, June starts 1st
    assert_batch_does_not_overlap(parse_batch_id("2026-06"),
                                  ["2026-01_to_2026-05", "2026-07"])


def test_overlap_empty_year_allowed():
    assert_batch_does_not_overlap(parse_batch_id("2026-06"), [])


def test_overlap_same_id_is_idempotent_rerun():
    assert_batch_does_not_overlap(parse_batch_id("2026-06"), ["2026-06"])


@pytest.mark.parametrize("existing", [
    "2026-01_to_2026-06",     # ends inside
    "2026-06_to_2026-12",     # starts inside
    "2026-01_to_2026-12",     # covers
    "2026-07",                # strictly inside
    "2026-06",                # same start, different coverage - NOT a rerun
])
def test_overlap_rejected(existing):
    with pytest.raises(AssertionError, match="overlaps existing batch"):
        assert_batch_does_not_overlap(parse_batch_id("2026-06_to_2026-08"), [existing])


def test_overlap_reports_all_offenders():
    with pytest.raises(AssertionError, match="2026-05_to_2026-06.*2026-08"):
        assert_batch_does_not_overlap(parse_batch_id("2026-06_to_2026-08"),
                                      ["2026-01_to_2026-04", "2026-05_to_2026-06", "2026-08"])


def test_overlap_malformed_existing_name_rejected():
    # a stray file that isn't a batch id would be invisible to the check forever
    with pytest.raises(AssertionError, match="not batch ids"):
        assert_batch_does_not_overlap(parse_batch_id("2026-06"), ["articles_2026"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))