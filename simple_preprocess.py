"""
Simple preprocessing without tiktoken. No network dependencies, works offline.
Implements 4-phase pipeline: base articles, flattened topics, RAG chunks, embeddings.
Uses Pydantic AI's Embedder for consistency with chatbot.
"""
import argparse
import asyncio
import os
from pathlib import Path

import boto3
import pandas as pd
import numpy as np
from pydantic_ai.models.openai import OpenAIEmbeddingModel, OpenAIProvider
from pydantic_ai.embedders import Embedder, EmbeddingSettings

from src.preprocessing.storage import load_dataframe
from src.preprocessing.data_cleaning import clean_base_articles, flatten_base_articles
from src.preprocessing.semantic_chunks import build_rag_chunks_from_base_articles
from src.utils.columns import RETRIEVAL_TEXT_COL

EMBEDDING_DIM = 1536
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"




def create_embedder(api_key, base_url=None):
    """Create a Pydantic AI Embedder instance."""
    provider = OpenAIProvider(api_key=api_key, base_url=base_url)
    model = OpenAIEmbeddingModel(OPENAI_EMBEDDING_MODEL, provider=provider)
    embedder = Embedder(model, settings=EmbeddingSettings(dimensions=EMBEDDING_DIM))
    return embedder


async def embed_texts_with_pydantic_ai(texts, embedder, batch_size=64):
    """Embed texts using Pydantic AI's Embedder (async)."""
    all_embeddings = []
    total = len(texts)

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"[DEBUG] Batch {batch_num}/{total_batches}: {len(batch)} texts")

        try:
            print(f"[DEBUG] Calling Pydantic AI embedder...", flush=True)
            embeddings = await embedder.embed_documents(batch)
            all_embeddings.extend(embeddings)
            print(f"[INFO] Embedded {min(i + batch_size, total)}/{total}")
        except Exception as e:
            print(f"[ERROR] Batch {batch_num} failed: {e}")
            raise

    return np.array(all_embeddings, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="4-Phase preprocessing pipeline (no tiktoken)")
    parser.add_argument("input_file", help="Input CSV/Excel/Parquet")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation")
    parser.add_argument("--batch-size", type=int, default=4, help="Embeddings batch size")
    parser.add_argument("--s3-bucket", help="S3 bucket for upload")
    parser.add_argument("--s3-prefix", help="S3 prefix for upload")
    parser.add_argument("--output-base-name", default="articles_base.parquet", help="Output filename for Phase 1 (default: articles_base.parquet)")
    parser.add_argument("--output-flat-name", default="articles_flat.parquet", help="Output filename for Phase 2 (default: articles_flat.parquet)")
    parser.add_argument("--output-chunks-name", default="rag_chunks.parquet", help="Output filename for Phase 3-4 (default: rag_chunks.parquet)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print("\n[INFO] Loading raw data...")
    df = load_dataframe(args.input_file)
    print(f"[OK] Loaded {len(df)} rows")

    # Phase 1: Create unflattened base articles
    print("\n" + "="*60)
    print("PHASE 1: Cleaning Base Articles (unflattened)")
    print("="*60)
    articles_base = clean_base_articles(df)
    print(f"[OK] Cleaned {len(articles_base)} base articles")

    base_path = output_dir / args.output_base_name
    articles_base.to_parquet(base_path, index=False)
    print(f"[OK] Saved to {base_path}")

    if args.s3_bucket and args.s3_prefix:
        s3 = boto3.client("s3")
        s3.upload_file(str(base_path), args.s3_bucket, f"{args.s3_prefix}/{args.output_base_name}")
        print(f"[OK] Uploaded to S3: s3://{args.s3_bucket}/{args.s3_prefix}/{args.output_base_name}")

    # Phase 2: Create flattened topic-level table
    print("\n" + "="*60)
    print("PHASE 2: Creating Flattened Topic-Level Table")
    print("="*60)
    articles_flat = flatten_base_articles(articles_base)
    print(f"[OK] Created {len(articles_flat)} article-topic pairs")

    flat_path = output_dir / args.output_flat_name
    articles_flat.to_parquet(flat_path, index=False)
    print(f"[OK] Saved to {flat_path}")

    if args.s3_bucket and args.s3_prefix:
        s3 = boto3.client("s3")
        s3.upload_file(str(flat_path), args.s3_bucket, f"{args.s3_prefix}/{args.output_flat_name}")
        print(f"[OK] Uploaded to S3: s3://{args.s3_bucket}/{args.s3_prefix}/{args.output_flat_name}")

    # Phase 3: Create RAG chunks
    print("\n" + "="*60)
    print("PHASE 3: Building RAG Chunks")
    print("="*60)
    rag_chunks = build_rag_chunks_from_base_articles(articles_base)
    print(f"[OK] Created {len(rag_chunks)} RAG chunks")

    chunks_path = output_dir / args.output_chunks_name
    rag_chunks.to_parquet(chunks_path, index=False)
    print(f"[OK] Saved to {chunks_path}")

    if args.s3_bucket and args.s3_prefix:
        s3 = boto3.client("s3")
        s3.upload_file(str(chunks_path), args.s3_bucket, f"{args.s3_prefix}/{args.output_chunks_name}")
        print(f"[OK] Uploaded to S3: s3://{args.s3_bucket}/{args.s3_prefix}/{args.output_chunks_name}")

    # Phase 4: Generate embeddings
    if not args.skip_embeddings:
        print("\n" + "="*60)
        print("PHASE 4: Generating Embeddings")
        print("="*60)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[WARNING] OPENAI_API_KEY not set, skipping embeddings")
            return

        base_url = os.getenv("OPENAI_BASE_URL")
        print(f"[DEBUG] Base URL: {base_url or 'default (OpenAI)'}")
        print(f"[DEBUG] Embedding model: {OPENAI_EMBEDDING_MODEL}, dimensions: {EMBEDDING_DIM}")

        print("[INFO] Creating Pydantic AI embedder...")
        try:
            embedder = create_embedder(api_key, base_url)
            print(f"[OK] Embedder created successfully")
        except Exception as e:
            print(f"[ERROR] Failed to create embedder: {e}")
            raise

        print("[INFO] Creating embeddings...")
        embeddings = asyncio.run(embed_texts_with_pydantic_ai(
            rag_chunks[RETRIEVAL_TEXT_COL].tolist(),
            embedder=embedder,
            batch_size=args.batch_size,
        ))

        print("[INFO] Adding embeddings to chunks...")
        rag_chunks["embedding"] = [emb.tolist() for emb in embeddings]

        chunks_path = output_dir / args.output_chunks_name
        rag_chunks.to_parquet(chunks_path, index=False)
        print(f"[OK] Saved chunks with embeddings to {chunks_path}")

        if args.s3_bucket and args.s3_prefix:
            s3 = boto3.client("s3")
            s3.upload_file(str(chunks_path), args.s3_bucket, f"{args.s3_prefix}/{args.output_chunks_name}")
            print(f"[OK] Uploaded to S3: s3://{args.s3_bucket}/{args.s3_prefix}/{args.output_chunks_name}")

    print("\n" + "="*60)
    print("[OK] Preprocessing pipeline complete!")
    print("="*60)


if __name__ == "__main__":
    main()
