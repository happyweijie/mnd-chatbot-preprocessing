"""
Simple preprocessing script based on notebook workflow.
Flattens, normalizes, chunks, and embeds article data.
"""
import argparse
import os
import sys
from pathlib import Path

import boto3
import pandas as pd
import numpy as np
from openai import OpenAI

from src.preprocessing.storage import load_dataframe
from src.preprocessing.data_cleaning import flatten_df, normalise_fields
from src.preprocessing.semantic_chunks import build_semantic_chunk_dataframe
from src.utils.columns import RETRIEVAL_TEXT_COL


def make_batches(texts, max_batch_items=64):
    """Simple batch division by item count."""
    batches = []
    for i in range(0, len(texts), max_batch_items):
        batches.append(texts[i : i + max_batch_items])
    return batches


def embed_texts_batched(texts, client, model="text-embedding-3-small", max_batch_items=64):
    """Create embeddings using notebook approach."""
    all_embeddings = []
    batches = make_batches(texts, max_batch_items=max_batch_items)

    print(f"Total texts: {len(texts)}")
    print(f"Total batches: {len(batches)}")

    processed = 0
    for batch_idx, batch in enumerate(batches, start=1):
        response = client.embeddings.create(model=model, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        processed += len(batch)
        print(f"Batch {batch_idx}/{len(batches)} done. Processed {processed}/{len(texts)}")

    return np.array(all_embeddings, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Simple preprocessing pipeline")
    parser.add_argument("input_file", help="Input CSV/Excel/Parquet file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embeddings")
    parser.add_argument("--s3-bucket", help="S3 bucket for uploading results")
    parser.add_argument("--s3-prefix", help="S3 key prefix (required with --s3-bucket)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate S3 args
    if (args.s3_bucket is None) != (args.s3_prefix is None):
        parser.error("Both --s3-bucket and --s3-prefix required for S3 upload")

    # Phase 1: Flatten and normalize
    print("\n" + "="*60)
    print("PHASE 1: Flattening and Normalizing")
    print("="*60)
    df = load_dataframe(args.input_file)
    df = df.pipe(flatten_df).pipe(normalise_fields)
    print(f"[OK] Loaded and flattened: {len(df)} rows")

    csv_path = output_dir / "articles.csv"
    parquet_path = output_dir / "articles.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    print(f"[OK] Saved to {csv_path} and {parquet_path}")

    if args.s3_bucket and args.s3_prefix:
        s3 = boto3.client("s3")
        s3.upload_file(str(parquet_path), args.s3_bucket, f"{args.s3_prefix}/articles.parquet")
        print(f"[OK] Uploaded to S3: s3://{args.s3_bucket}/{args.s3_prefix}/articles.parquet")

    # Phase 2: Chunk and embed
    print("\n" + "="*60)
    print("PHASE 2: Chunking and Embedding")
    print("="*60)
    chunks_df = build_semantic_chunk_dataframe(df)
    print(f"[OK] Created {len(chunks_df)} chunks")

    chunks_path = output_dir / "semantic_chunks.parquet"
    chunks_df.to_parquet(chunks_path, index=False)
    print(f"[OK] Saved chunks to {chunks_path}")

    if args.s3_bucket and args.s3_prefix:
        s3 = boto3.client("s3")
        s3.upload_file(str(chunks_path), args.s3_bucket, f"{args.s3_prefix}/semantic_chunks.parquet")
        print(f"[OK] Uploaded to S3: s3://{args.s3_bucket}/{args.s3_prefix}/semantic_chunks.parquet")

    if not args.skip_embeddings:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[WARNING] OPENAI_API_KEY not set, skipping embeddings")
            return

        base_url = os.getenv("OPENAI_BASE_URL")
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=300.0)

        print("[INFO] Creating embeddings...")
        embeddings = embed_texts_batched(
            chunks_df[RETRIEVAL_TEXT_COL].tolist(),
            client=client,
            max_batch_items=args.batch_size,
        )

        embeddings_path = output_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)
        print(f"[OK] Saved embeddings to {embeddings_path}")

        if args.s3_bucket and args.s3_prefix:
            s3 = boto3.client("s3")
            s3.upload_file(str(embeddings_path), args.s3_bucket, f"{args.s3_prefix}/embeddings.npy")
            print(f"[OK] Uploaded to S3: s3://{args.s3_bucket}/{args.s3_prefix}/embeddings.npy")

    print("\n" + "="*60)
    print("[OK] Preprocessing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
