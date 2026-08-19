"""Tests for pipeline.s3_batch - key layout, settings validation, and the
runner's control flow against a fake S3 client (no AWS, no boto3 needed).
Run with pytest, or directly: python tests/test_s3_batch.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from pipeline import s3_batch
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
