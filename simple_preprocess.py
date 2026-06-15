"""
Simple preprocessing without tiktoken. No network dependencies, works offline.
"""
import argparse
import os
from pathlib import Path

import boto3
import pandas as pd
import numpy as np
from openai import OpenAI

from src.preprocessing.storage import load_dataframe
from src.preprocessing.data_cleaning import flatten_df, normalise_fields
from src.utils.columns import (
    ARTICLE_ID_KEY_COL,
    ARTICLE_TOPIC_KEY_COL,
    CHUNK_ID_COL,
    CONTENT_CHUNK_COL,
    CONTENT_COL,
    EXPLANATION_COL,
    NEWS_SITE_COL,
    PUBLISHED_DATE_COL,
    QUARTER_YEAR_COL,
    RETRIEVAL_TEXT_COL,
    SENTIMENT_SCORE_COL,
    TITLE_COL,
    TOPIC_COL,
    YEAR_COL,
)


def simple_chunk_by_chars(text, target_chars=3500, overlap_chars=500):
    """Chunk by characters (~700 tokens ≈ 3500 chars). No tiktoken needed."""
    if not text or target_chars <= 0:
        return []

    text = str(text).strip()
    if len(text) <= target_chars:
        return [text]

    chunks = []
    for i in range(0, len(text), target_chars):
        chunk = text[i : i + target_chars + overlap_chars]
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks


def build_retrieval_prefix(row):
    """Build metadata prefix for embedding."""
    fields = []
    for label, col in [
        ("Title", TITLE_COL),
        ("Topic", TOPIC_COL),
        ("Published date", PUBLISHED_DATE_COL),
        ("Year", YEAR_COL),
        ("News site", NEWS_SITE_COL),
        ("Sentiment score", SENTIMENT_SCORE_COL),
        ("Topic sentiment explanation", EXPLANATION_COL),
    ]:
        val = getattr(row, col, None)
        if pd.notna(val):
            fields.append(f"{label}: {str(val).strip()}")

    return "\n".join(fields)


def build_chunks_df(df):
    """Create chunks from articles without tiktoken."""
    print(f"[INFO] Creating chunks from {len(df)} articles...")
    chunk_rows = []

    for idx, row in enumerate(df.itertuples(index=False), 1):
        if idx % 100 == 0:
            print(f"[INFO] Processing article {idx}/{len(df)}...", flush=True)

        content = getattr(row, CONTENT_COL, "")
        if not content:
            continue

        prefix = build_retrieval_prefix(row)
        chunks = simple_chunk_by_chars(content)

        for chunk_id, chunk_text in enumerate(chunks):
            retrieval_text = f"{prefix}\n\nContent:\n{chunk_text}" if prefix else chunk_text
            chunk_rows.append({
                ARTICLE_ID_KEY_COL: getattr(row, ARTICLE_ID_KEY_COL, ""),
                ARTICLE_TOPIC_KEY_COL: getattr(row, ARTICLE_TOPIC_KEY_COL, ""),
                CHUNK_ID_COL: chunk_id,
                TITLE_COL: getattr(row, TITLE_COL, ""),
                TOPIC_COL: getattr(row, TOPIC_COL, ""),
                YEAR_COL: getattr(row, YEAR_COL, ""),
                QUARTER_YEAR_COL: getattr(row, QUARTER_YEAR_COL, ""),
                NEWS_SITE_COL: getattr(row, NEWS_SITE_COL, ""),
                PUBLISHED_DATE_COL: getattr(row, PUBLISHED_DATE_COL, ""),
                SENTIMENT_SCORE_COL: getattr(row, SENTIMENT_SCORE_COL, ""),
                EXPLANATION_COL: getattr(row, EXPLANATION_COL, ""),
                CONTENT_CHUNK_COL: chunk_text,
                RETRIEVAL_TEXT_COL: retrieval_text,
            })

    print(f"[INFO] Created {len(chunk_rows)} chunks")
    return pd.DataFrame(chunk_rows)


def embed_batched(texts, client, model="text-embedding-3-small", batch_size=64):
    """Embed texts with OpenAI."""
    all_embeddings = []
    total = len(texts)

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"[DEBUG] Batch {batch_num}/{total_batches}: {len(batch)} texts, total chars: {sum(len(str(t)) for t in batch)}")
        print(f"[DEBUG] First text sample: {str(batch[0])[:100]}...")

        try:
            print(f"[DEBUG] Calling API...", flush=True)
            response = client.embeddings.create(model=model, input=batch)
            print(f"[DEBUG] Got response, processing...")
            all_embeddings.extend([item.embedding for item in response.data])
            print(f"[INFO] Embedded {min(i + batch_size, total)}/{total}")
        except Exception as e:
            print(f"[ERROR] Batch {batch_num} failed: {e}")
            raise

    return np.array(all_embeddings, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Simple preprocessing (no tiktoken)")
    parser.add_argument("input_file", help="Input CSV/Excel/Parquet")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4, help="Embeddings batch size (use 4 for slow endpoints)")
    parser.add_argument("--s3-bucket", help="S3 bucket")
    parser.add_argument("--s3-prefix", help="S3 prefix")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Flatten and normalize
    print("\n" + "="*60)
    print("PHASE 1: Flattening and Normalizing")
    print("="*60)
    df = load_dataframe(args.input_file)
    df = df.pipe(flatten_df).pipe(normalise_fields)
    print(f"[OK] Loaded and flattened: {len(df)} rows")

    parquet_path = output_dir / "articles.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"[OK] Saved to {parquet_path}")

    if args.s3_bucket and args.s3_prefix:
        s3 = boto3.client("s3")
        s3.upload_file(str(parquet_path), args.s3_bucket, f"{args.s3_prefix}/articles.parquet")
        print(f"[OK] Uploaded to S3: s3://{args.s3_bucket}/{args.s3_prefix}/articles.parquet")

    # Phase 2: Chunk and embed
    print("\n" + "="*60)
    print("PHASE 2: Chunking and Embedding (no tiktoken)")
    print("="*60)
    chunks_df = build_chunks_df(df)

    chunks_path = output_dir / "semantic_chunks.parquet"
    chunks_df.to_parquet(chunks_path, index=False)
    print(f"[OK] Saved chunks to {chunks_path}")

    if args.s3_bucket and args.s3_prefix:
        s3 = boto3.client("s3")
        s3.upload_file(str(chunks_path), args.s3_bucket, f"{args.s3_prefix}/semantic_chunks.parquet")
        print(f"[OK] Uploaded to S3")

    if not args.skip_embeddings:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[WARNING] OPENAI_API_KEY not set, skipping embeddings")
            return

        base_url = os.getenv("OPENAI_BASE_URL")
        print(f"[DEBUG] API Key: {'***' + api_key[-10:] if api_key else 'None'}")
        print(f"[DEBUG] Base URL: {base_url or 'default (OpenAI)'}")

        client = OpenAI(api_key=api_key, base_url=base_url)

        print("[INFO] Testing API with single embedding...")
        try:
            test_response = client.embeddings.create(
                model="text-embedding-3-small",
                input=["test"]
            )
            print(f"[OK] API test successful, got embedding of size {len(test_response.data[0].embedding)}")
        except Exception as e:
            print(f"[ERROR] API test failed: {e}")
            raise

        print("[INFO] Creating embeddings...")
        embeddings = embed_batched(
            chunks_df[RETRIEVAL_TEXT_COL].tolist(),
            client=client,
            batch_size=args.batch_size,
        )

        embeddings_path = output_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)
        print(f"[OK] Saved embeddings to {embeddings_path}")

        if args.s3_bucket and args.s3_prefix:
            s3 = boto3.client("s3")
            s3.upload_file(str(embeddings_path), args.s3_bucket, f"{args.s3_prefix}/embeddings.npy")
            print(f"[OK] Uploaded to S3")

    print("\n" + "="*60)
    print("[OK] Done!")
    print("="*60)


if __name__ == "__main__":
    main()
