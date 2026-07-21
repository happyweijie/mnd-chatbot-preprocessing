"""CLI entrypoint: python -m pipeline <phase> ...

Phases:
  clean    raw CSV -> articles_base.parquet
  flatten  articles_base.parquet -> articles_flat.parquet
  chunk    articles_base.parquet -> rag_chunks.parquet
  embed    rag_chunks.parquet -> rag_chunks_embedded.parquet (embedding column added)
  all      run every phase in order
"""

import argparse
from pathlib import Path

from . import config, phase1_clean, phase2_flatten, phase3_chunk, phase4_embed


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


if __name__ == "__main__":
    main()