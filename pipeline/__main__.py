"""CLI entrypoint: python -m pipeline <phase> ...

Phases:
  clean    raw CSV -> articles_base.parquet
  flatten  articles_base.parquet -> articles_flat.parquet
  chunk    articles_base.parquet -> rag_chunks.parquet
  embed    rag_chunks.parquet -> rag_chunks_embedded.parquet (embedding column added)
  all      run every phase in order
  upload   S3: local <batch>.csv -> raw/year=YYYY/<batch>.csv (validated, non-overlap checked)
  batch    S3: raw/year=YYYY/<batch>.csv -> processed/<stage>/year=YYYY/<batch>.parquet
  delete   S3: remove one batch everywhere (raw csv + 4 parquet outputs + local scratch dir)
"""

import argparse
from pathlib import Path

from . import (
    config,
    phase1_clean,
    phase2_flatten,
    phase3_chunk,
    phase4_embed,
    s3_batch,
    s3_delete,
    s3_upload,
)


def add_chunk_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-tokens", type=int, default=config.TARGET_CHUNK_TOKENS,
                        help=f"target chunk size in tokens (default {config.TARGET_CHUNK_TOKENS})")
    parser.add_argument("--max-tokens", type=int, default=config.MAX_CHUNK_TOKENS,
                        help=f"hard chunk size cap in tokens (default {config.MAX_CHUNK_TOKENS})")
    parser.add_argument("--overlap-tokens", type=int, default=config.OVERLAP_TOKENS,
                        help=f"overlap budget between chunks (default {config.OVERLAP_TOKENS})")


def add_embed_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=config.EMBED_BATCH_SIZE,
                        help=f"texts per embedding request (default {config.EMBED_BATCH_SIZE})")


def add_s3_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-id", required=True,
                        help='e.g. "2026-06" or "2026-01_to_2026-05"')
    parser.add_argument("--bucket", default=config.S3_BUCKET,
                        help="S3 bucket name (default: $HINT_BUCKET)")
    parser.add_argument("--prefix", default=config.S3_ROOT,
                        help=f"root key prefix inside the bucket "
                             f"(default: $HINT_S3_PREFIX or {config.S3_ROOT!r})")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved S3 keys and exit without any AWS call")


def add_work_dir_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-dir", type=Path, default=Path(config.S3_WORK_DIR),
                        help=f"local scratch dir for downloads/outputs/checkpoints "
                             f"(default: $HINT_WORK_DIR or {config.S3_WORK_DIR!r})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="phase", required=True)

    p_clean = sub.add_parser("clean", help="phase 1: clean raw CSV into base articles")
    p_clean.add_argument("input_csv", type=Path)
    p_clean.add_argument("output_dir", type=Path)

    p_flatten = sub.add_parser("flatten", help="phase 2: one row per article-topic")
    p_flatten.add_argument("data_dir", type=Path,
                           help="directory containing the base parquet; outputs land here too")

    p_chunk = sub.add_parser("chunk", help="phase 3: build RAG chunks")
    p_chunk.add_argument("data_dir", type=Path)
    add_chunk_options(p_chunk)

    p_embed = sub.add_parser("embed", help="phase 4: embed chunk retrieval text")
    p_embed.add_argument("data_dir", type=Path)
    add_embed_options(p_embed)
    p_embed.add_argument("--limit", type=int, default=None,
                         help="embed only the first N chunks into a *_sample file (smoke test)")

    p_all = sub.add_parser("all", help="run all phases in order")
    p_all.add_argument("input_csv", type=Path)
    p_all.add_argument("output_dir", type=Path)
    add_chunk_options(p_all)
    add_embed_options(p_all)
    p_all.add_argument("--skip-embeddings", action="store_true",
                       help="stop after phase 3 (no API access needed)")

    p_upload = sub.add_parser(
        "upload",
        help="upload one raw batch CSV to S3 (validated locally, non-overlap checked)",
    )
    add_s3_options(p_upload)
    p_upload.add_argument("--file", type=Path, required=True, dest="csv_path",
                          help="local raw batch CSV; uploaded as raw/year=YYYY/<batch-id>.csv")
    p_upload.add_argument("--overwrite", action="store_true",
                          help="allow replacing an existing file with the same batch id "
                               "(then re-run `batch` so processed outputs are refreshed)")
    p_upload.add_argument("--skip-validation", action="store_true",
                          help="skip the local phase-1 dry run of the CSV before uploading")

    p_batch = sub.add_parser(
        "batch",
        help="run all phases on one S3 batch",
    )
    add_s3_options(p_batch)
    add_work_dir_option(p_batch)
    p_batch.add_argument("--skip-embeddings", action="store_true",
                         help="stop after phase 3 (no API access needed)")
    add_chunk_options(p_batch)
    add_embed_options(p_batch)

    p_delete = sub.add_parser(
        "delete",
        help="delete one batch everywhere: raw CSV, 4 processed parquets, local scratch dir",
    )
    add_s3_options(p_delete)
    add_work_dir_option(p_delete)
    p_delete.add_argument("--yes", action="store_true",
                          help="actually delete; without it the command only reports what "
                               "exists and what would be removed")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.phase == "clean":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        phase1_clean.run(args.input_csv, args.output_dir / config.BASE_FILENAME)

    elif args.phase == "flatten":
        phase2_flatten.run(args.data_dir / config.BASE_FILENAME,
                           args.data_dir / config.FLAT_FILENAME)

    elif args.phase == "chunk":
        phase3_chunk.run(args.data_dir / config.BASE_FILENAME,
                         args.data_dir / config.CHUNKS_FILENAME,
                         target_chunk_tokens=args.target_tokens,
                         max_chunk_tokens=args.max_tokens,
                         overlap_tokens=args.overlap_tokens)

    elif args.phase == "embed":
        chunks_path = args.data_dir / config.CHUNKS_FILENAME
        embedded_path = args.data_dir / config.EMBEDDED_FILENAME
        # a --limit smoke test must not overwrite the full embedded file
        output_path = (embedded_path.with_suffix(".sample.parquet")
                       if args.limit is not None else embedded_path)
        phase4_embed.run(chunks_path, output_path,
                         batch_size=args.batch_size, limit=args.limit)

    elif args.phase == "all":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        base_path = args.output_dir / config.BASE_FILENAME
        chunks_path = args.output_dir / config.CHUNKS_FILENAME

        phase1_clean.run(args.input_csv, base_path)
        phase2_flatten.run(base_path, args.output_dir / config.FLAT_FILENAME)
        phase3_chunk.run(base_path, chunks_path,
                         target_chunk_tokens=args.target_tokens,
                         max_chunk_tokens=args.max_tokens,
                         overlap_tokens=args.overlap_tokens)
        if args.skip_embeddings:
            print("skipping phase 4 (embeddings)")
        else:
            phase4_embed.run(chunks_path,
                             args.output_dir / config.EMBEDDED_FILENAME,
                             batch_size=args.batch_size)

    elif args.phase == "upload":
        s3_upload.run_upload(
            args.batch_id,
            args.csv_path,
            bucket=args.bucket,
            root=args.prefix,
            overwrite=args.overwrite,
            validate=not args.skip_validation,
            dry_run=args.dry_run,
        )

    elif args.phase == "batch":
        s3_batch.run_batch(
            args.batch_id,
            bucket=args.bucket,
            root=args.prefix,
            work_dir=args.work_dir,
            skip_embeddings=args.skip_embeddings,
            dry_run=args.dry_run,
            target_chunk_tokens=args.target_tokens,
            max_chunk_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            embed_batch_size=args.batch_size,
        )

    elif args.phase == "delete":
        s3_delete.run_delete(
            args.batch_id,
            bucket=args.bucket,
            root=args.prefix,
            work_dir=args.work_dir,
            yes=args.yes,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()