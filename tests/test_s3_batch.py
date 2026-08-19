"""
Tests for pipeline.s3_batch, pipeline.s3_upload and pipeline.s3_delete
- key layout, settings validation, and control flow against a fake S3 client.
Run with pytest, or directly: python tests/test_s3_batch.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from pipeline import s3_batch, s3_delete, s3_upload
from pipeline.batching import parse_batch_id
from pipeline.s3_batch import S3Layout, validate_s3_settings


# --- key layout (must match README "S3 storage layout" exactly) --------------

@pytest.mark.parametrize("bucket, root, batch_id, year", [
    ("bkt", "hint", "2026-06", 2026),                          # single month, default prefix
    ("bkt", "teams/xyz/hint", "2020-01_to_2020-12", 2020),     # range, nested prefix
])
def test_layout_keys(bucket, root, batch_id, year):
    layout = S3Layout(bucket, root, parse_batch_id(batch_id))

    assert layout.raw_dir == f"{root}/raw/year={year}/"
    assert layout.raw_key == f"{root}/raw/year={year}/{batch_id}.csv"
    assert layout.base_key == f"{root}/processed/base/year={year}/{batch_id}.parquet"
    assert layout.flattened_key == f"{root}/processed/flattened/year={year}/{batch_id}.parquet"
    assert layout.chunks_key == f"{root}/processed/chunks/year={year}/{batch_id}.parquet"
    assert layout.embeddings_key == f"{root}/processed/embeddings/year={year}/{batch_id}.parquet"
    assert layout.uri(layout.raw_key) == f"s3://{bucket}/{root}/raw/year={year}/{batch_id}.csv"


# --- settings validation ----------------------------------------------------

def test_settings_ok():
    validate_s3_settings("bkt", "hint")
    validate_s3_settings("bkt", "teams/xyz/hint")


@pytest.mark.parametrize("bucket, root", [
    ("", "hint"),
    ("s3://bkt", "hint"),
    ("bkt", ""),
    ("bkt", "/hint"),
    ("bkt", "hint/"),
    ("bkt", "s3://bkt/hint"),
])
def test_settings_rejected(bucket, root):
    with pytest.raises(AssertionError):
        validate_s3_settings(bucket, root)


# --- runner control flow against a fake S3 ----------------------------------

class FakeS3:
    """Just enough of the boto3 S3 client surface for run_batch."""

    def __init__(self, keys: dict[str, Path]):
        self.objects = dict(keys)  # key -> local file holding the content
        self.uploaded: list[str] = []
        self.deleted: list[str] = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket, Prefix, Delimiter):
        contents = [{"Key": k} for k in sorted(self.objects) if k.startswith(Prefix)]
        return [{"Contents": contents}]

    def download_file(self, bucket, key, filename):
        Path(filename).write_bytes(self.objects[key].read_bytes())

    def upload_file(self, filename, bucket, key):
        self.uploaded.append(key)
        self.objects[key] = Path(filename)

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)  # stands in for botocore ClientError (404)
        return {"ContentLength": self.objects[Key].stat().st_size}

    def delete_object(self, Bucket, Key):
        self.deleted.append(Key)
        self.objects.pop(Key, None)  # like S3: deleting a missing key is not an error
        return {}


def write_raw_csv(path: Path) -> None:
    """A minimal synthetic raw export in the claude-checkpoint format: three
    articles in Aug 2022 (one with two topics) plus one excluded row."""
    import json
    import polars as pl

    def article(title, day, topics, exclude=False):
        return {
            "title": title,
            "url": f"https://example.test/{day}",
            "published_date": f"2022-08-{day:02d}T10:00:00+08:00",
            "content": " ".join(f"Sentence {i} about {title.lower()}." for i in range(1, 30)),
            "news_site": "example",
            "housing_relevance": "high",
            "exclude_from_housing_analysis": exclude,
            "article_type": "news",
            "topics": "|".join(topics),
            "sentiment_by_topic": json.dumps({t: 0.5 for t in topics}),
            "sentiment_reason_by_topic": json.dumps({t: f"reason for {t}" for t in topics}),
        }

    rows = [
        article("BTO launch", 3, ["BTO"]),
        article("Resale prices", 15, ["Resale", "Affordability"]),
        article("Rental market", 28, ["Rental"]),
        article("Sponsored content", 20, ["BTO"], exclude=True),
    ]
    pl.DataFrame(rows).write_csv(path)


@pytest.fixture
def fake_s3(monkeypatch, tmp_path):
    raw = tmp_path / "source_2022-08.csv"
    write_raw_csv(raw)
    fake = FakeS3({"hint/raw/year=2022/2022-08.csv": raw})
    monkeypatch.setattr(s3_batch, "_s3_client", lambda: fake)
    return fake


def test_dry_run_makes_no_aws_calls(monkeypatch, tmp_path, capsys):
    def boom():
        raise AssertionError("dry run must not create a client")
    monkeypatch.setattr(s3_batch, "_s3_client", boom)

    layout = s3_batch.run_batch("2026-06", bucket="bkt", root="hint",
                                work_dir=tmp_path, dry_run=True)
    out = capsys.readouterr().out
    assert layout.embeddings_key in out
    assert "dry run" in out


def test_missing_raw_file_fails_before_download(fake_s3, tmp_path):
    with pytest.raises(AssertionError, match="not found"):
        s3_batch.run_batch("2022-09", bucket="bkt", root="hint",
                           work_dir=tmp_path, skip_embeddings=True)
    assert fake_s3.uploaded == []


def test_overlapping_batch_rejected_before_download(fake_s3, tmp_path):
    raw = fake_s3.objects["hint/raw/year=2022/2022-08.csv"]
    fake_s3.objects["hint/raw/year=2022/2022-07_to_2022-09.csv"] = raw
    with pytest.raises(AssertionError, match="overlaps existing batch"):
        s3_batch.run_batch("2022-07_to_2022-09", bucket="bkt", root="hint",
                           work_dir=tmp_path, skip_embeddings=True)
    assert fake_s3.uploaded == []


def test_full_run_without_embeddings(fake_s3, tmp_path):
    import polars as pl

    work_dir = tmp_path / "work"
    layout = s3_batch.run_batch("2022-08", bucket="bkt", root="hint",
                                work_dir=work_dir, skip_embeddings=True)

    # each stage uploaded in order, embeddings skipped
    assert fake_s3.uploaded == [layout.base_key, layout.flattened_key, layout.chunks_key]
    local = work_dir / "2022-08"
    assert (local / "raw" / "2022-08.csv").exists()

    base = pl.read_parquet(local / "articles_base.parquet")
    flat = pl.read_parquet(local / "articles_flat.parquet")
    chunks = pl.read_parquet(local / "rag_chunks.parquet")
    assert base.height == 3       # excluded row dropped
    assert flat.height == 4       # one row per article-topic
    assert chunks.height >= 3     # at least one chunk per article

    # lineage columns carried through every stage
    for df in (base, flat, chunks):
        assert (df.get_column("batch_id") == "2022-08").all()
        assert (df.get_column("year") == 2022).all()


def test_article_outside_batch_range_rejected(fake_s3, tmp_path):
    # the same file uploaded under a batch id whose range doesn't cover
    # its articles must fail in phase 1 (before any upload)
    raw = fake_s3.objects.pop("hint/raw/year=2022/2022-08.csv")
    fake_s3.objects["hint/raw/year=2022/2022-07.csv"] = raw
    with pytest.raises(AssertionError, match="outside batch"):
        s3_batch.run_batch("2022-07", bucket="bkt", root="hint",
                           work_dir=tmp_path, skip_embeddings=True)
    assert fake_s3.uploaded == []


# --- uploader control flow against a fake S3 --------------------------------

@pytest.fixture
def local_csv(tmp_path):
    """A fresh copy of the synthetic Aug-2022 batch on local disk, as a user
    would have it before uploading."""
    path = tmp_path / "local" / "2022-08.csv"
    path.parent.mkdir()
    write_raw_csv(path)
    return path


@pytest.fixture
def empty_s3(monkeypatch):
    """Fake S3 with nothing under raw/ yet (unlike `fake_s3`, which already
    holds 2022-08)."""
    fake = FakeS3({})
    monkeypatch.setattr(s3_batch, "_s3_client", lambda: fake)
    monkeypatch.setattr(s3_upload, "_s3_client", lambda: fake)
    return fake


@pytest.fixture
def uploaded_s3(fake_s3, monkeypatch):
    """`fake_s3` (raw 2022-08 already present) wired into the uploader too."""
    monkeypatch.setattr(s3_upload, "_s3_client", lambda: fake_s3)
    return fake_s3


def test_upload_dry_run_makes_no_aws_calls(monkeypatch, local_csv, capsys):
    def boom():
        raise AssertionError("dry run must not create a client")
    monkeypatch.setattr(s3_upload, "_s3_client", boom)

    layout = s3_upload.run_upload("2022-08", local_csv, bucket="bkt", root="hint",
                                  dry_run=True)
    out = capsys.readouterr().out
    assert layout.raw_key == "hint/raw/year=2022/2022-08.csv"
    assert layout.uri(layout.raw_key) in out
    assert "dry run" in out


def test_upload_rejects_bad_inputs_before_any_aws_call(monkeypatch, local_csv, tmp_path):
    def boom():
        raise AssertionError("must fail before creating a client")
    monkeypatch.setattr(s3_upload, "_s3_client", boom)

    with pytest.raises(ValueError, match="not of the form"):
        s3_upload.run_upload("2022-8", local_csv, bucket="bkt", root="hint")
    with pytest.raises(AssertionError, match="file not found"):
        s3_upload.run_upload("2022-08", tmp_path / "missing.csv", bucket="bkt", root="hint")
    not_csv = tmp_path / "2022-08.xlsx"
    not_csv.write_bytes(b"")
    with pytest.raises(AssertionError, match="must be CSV"):
        s3_upload.run_upload("2022-08", not_csv, bucket="bkt", root="hint")


def test_upload_local_validation_rejects_rows_outside_batch_range(monkeypatch, local_csv):
    # the Aug-2022 file offered under a July id must fail in the local phase-1
    # dry run, before a client is even created
    def boom():
        raise AssertionError("must fail in local validation, before any AWS call")
    monkeypatch.setattr(s3_upload, "_s3_client", boom)

    with pytest.raises(AssertionError, match="outside batch"):
        s3_upload.run_upload("2022-07", local_csv, bucket="bkt", root="hint")


def test_upload_skip_validation_lets_mismatched_file_through(empty_s3, local_csv):
    # documents what --skip-validation gives up: the file lands in S3 and only
    # `batch` will reject it later
    layout = s3_upload.run_upload("2022-07", local_csv, bucket="bkt", root="hint",
                                  validate=False)
    assert empty_s3.uploaded == [layout.raw_key]


def test_upload_happy_path(empty_s3, local_csv, capsys):
    layout = s3_upload.run_upload("2022-08", local_csv, bucket="bkt", root="hint")

    assert empty_s3.uploaded == ["hint/raw/year=2022/2022-08.csv"]
    assert empty_s3.objects[layout.raw_key].read_bytes() == local_csv.read_bytes()
    assert "pipeline batch --batch-id 2022-08" in capsys.readouterr().out


def test_upload_overlapping_batch_rejected(uploaded_s3, tmp_path):
    # raw 2022-08 exists; a July-September file overlaps it. we can't use the
    # Aug-only synthetic rows under a Jul-Sep id without failing validation
    # first, so skip validation to reach the S3-side check
    csv = tmp_path / "2022-07_to_2022-09.csv"
    write_raw_csv(csv)
    with pytest.raises(AssertionError, match="overlaps existing batch"):
        s3_upload.run_upload("2022-07_to_2022-09", csv, bucket="bkt", root="hint",
                             validate=False)
    assert uploaded_s3.uploaded == []


def test_upload_existing_batch_needs_overwrite(uploaded_s3, local_csv, capsys):
    with pytest.raises(AssertionError, match="already exists"):
        s3_upload.run_upload("2022-08", local_csv, bucket="bkt", root="hint")
    assert uploaded_s3.uploaded == []

    # simulate a previous processing run so the stale-output warning fires
    layout = S3Layout("bkt", "hint", parse_batch_id("2022-08"))
    uploaded_s3.objects[layout.base_key] = local_csv
    uploaded_s3.objects[layout.flattened_key] = local_csv

    s3_upload.run_upload("2022-08", local_csv, bucket="bkt", root="hint",
                         overwrite=True)
    assert uploaded_s3.uploaded == [layout.raw_key]
    out = capsys.readouterr().out
    assert "stale" in out
    assert layout.uri(layout.base_key) in out
    assert layout.uri(layout.flattened_key) in out
    assert layout.uri(layout.chunks_key) not in out  # never processed -> not listed


def test_object_exists(uploaded_s3):
    assert s3_upload.object_exists(uploaded_s3, "bkt", "hint/raw/year=2022/2022-08.csv")
    assert not s3_upload.object_exists(uploaded_s3, "bkt", "hint/raw/year=2022/nope.csv")


# --- deleter control flow against a fake S3 ---------------------------------

@pytest.fixture
def processed_s3(fake_s3, monkeypatch, tmp_path):
    """`fake_s3` (raw 2022-08 present) plus three of its four processed
    outputs (embeddings never ran), an unrelated 2022-09 batch that must
    survive, and a local scratch dir with a stale phase-4 checkpoint."""
    monkeypatch.setattr(s3_delete, "_s3_client", lambda: fake_s3)
    raw = fake_s3.objects["hint/raw/year=2022/2022-08.csv"]
    layout = S3Layout("bkt", "hint", parse_batch_id("2022-08"))
    for key in (layout.base_key, layout.flattened_key, layout.chunks_key):
        fake_s3.objects[key] = raw
    other = S3Layout("bkt", "hint", parse_batch_id("2022-09"))
    fake_s3.objects[other.raw_key] = raw
    fake_s3.objects[other.base_key] = raw

    work_dir = tmp_path / "work"
    local = work_dir / "2022-08"
    local.mkdir(parents=True)
    (local / "rag_chunks_embedded.checkpoint.parquet").write_bytes(b"stale")
    return fake_s3, layout, work_dir


def test_delete_dry_run_makes_no_aws_calls(monkeypatch, tmp_path, capsys):
    def boom():
        raise AssertionError("dry run must not create a client")
    monkeypatch.setattr(s3_delete, "_s3_client", boom)

    layout = s3_delete.run_delete("2022-08", bucket="bkt", root="hint",
                                  work_dir=tmp_path, dry_run=True)
    out = capsys.readouterr().out
    for key in (layout.raw_key, layout.base_key, layout.flattened_key,
                layout.chunks_key, layout.embeddings_key):
        assert layout.uri(key) in out
    assert str(tmp_path / "2022-08") in out
    assert "dry run" in out


def test_delete_unknown_batch_is_an_error(processed_s3):
    fake, _, work_dir = processed_s3
    with pytest.raises(AssertionError, match="nothing to delete"):
        s3_delete.run_delete("2022-11", bucket="bkt", root="hint",
                             work_dir=work_dir, yes=True)
    assert fake.deleted == []


def test_delete_without_yes_only_reports(processed_s3, capsys):
    fake, layout, work_dir = processed_s3
    before = dict(fake.objects)

    s3_delete.run_delete("2022-08", bucket="bkt", root="hint", work_dir=work_dir)

    out = capsys.readouterr().out
    assert fake.deleted == []
    assert fake.objects == before
    assert (work_dir / "2022-08").is_dir()
    assert f"found    {layout.uri(layout.raw_key)}" in out
    assert f"missing  {layout.uri(layout.embeddings_key)}" in out
    assert "would delete 4 S3 object(s) + local scratch dir" in out
    assert "--yes" in out


def test_delete_with_yes_removes_batch_everywhere_and_nothing_else(processed_s3, capsys):
    fake, layout, work_dir = processed_s3
    other = S3Layout("bkt", "hint", parse_batch_id("2022-09"))

    s3_delete.run_delete("2022-08", bucket="bkt", root="hint",
                         work_dir=work_dir, yes=True)

    # only the four objects that existed were deleted; the never-written
    # embeddings key is reported missing, not deleted
    assert sorted(fake.deleted) == sorted([
        layout.raw_key, layout.base_key, layout.flattened_key, layout.chunks_key,
    ])
    for key in (layout.raw_key, layout.base_key, layout.flattened_key,
                layout.chunks_key, layout.embeddings_key):
        assert key not in fake.objects
    # the neighbouring batch is untouched
    assert other.raw_key in fake.objects and other.base_key in fake.objects
    # local scratch dir (with its stale checkpoint) is gone, work_dir itself stays
    assert not (work_dir / "2022-08").exists()
    assert work_dir.is_dir()
    assert "deleted everywhere" in capsys.readouterr().out


def test_delete_local_dir_only(processed_s3):
    # batch exists only locally (e.g. S3 side already cleaned by hand):
    # still worth deleting, and must not error on the S3 side
    fake, _, work_dir = processed_s3
    (work_dir / "2022-10").mkdir()
    s3_delete.run_delete("2022-10", bucket="bkt", root="hint",
                         work_dir=work_dir, yes=True)
    assert fake.deleted == []
    assert not (work_dir / "2022-10").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
