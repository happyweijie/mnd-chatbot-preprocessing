"""Sentence-aware, token-bounded chunking. Pure functions, no file I/O.

Note: this chunker assumes each article is a single non-null line (the source
dataset is verified newline-free); whitespace runs are collapsed to a single
space. Guard assertions in phase 3 fail loudly if that assumption breaks -
if it does, paragraph-aware splitting (split on "\\n\\n" before sentence
splitting) needs to be restored here.
"""

import os
import re
from typing import Sequence

import tiktoken

SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")

# cl100k_base cache file: SHA-1 of the tiktoken download URL
CL100K_CACHE_FILE = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"


def get_token_encoder(model: str) -> tiktoken.Encoding:
    """Return the tiktoken encoder for the embedding model.

    On offline machines set TIKTOKEN_CACHE_DIR (absolute path, ~ is NOT
    expanded) to a directory containing the cached encoder file.
    """
    cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR")
    if cache_dir and not os.path.exists(os.path.join(cache_dir, CL100K_CACHE_FILE)):
        raise FileNotFoundError(
            f"TIKTOKEN_CACHE_DIR is set to {cache_dir!r} but the cl100k_base "
            f"cache file {CL100K_CACHE_FILE!r} is not in it. Use an absolute "
            "path (tiktoken does not expand ~) and verify from the same "
            "environment the pipeline runs in."
        )

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoder: tiktoken.Encoding) -> int:
    """Count tokens using the configured tiktoken encoder."""
    return len(encoder.encode(text))


def split_into_sentences(text: str) -> list[str]:
    """Split article text into sentence-like units.

    Collapses any whitespace run to a single space. The source dataset stores
    each article as a single line, so no paragraph handling is needed.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    return [
        piece.strip()
        for piece in SENTENCE_BOUNDARY_PATTERN.split(text)
        if piece.strip()
    ]


def split_long_unit_by_tokens(
    text: str,
    max_tokens: int,
    encoder: tiktoken.Encoding,
) -> list[str]:
    """Split long sentence-like units only when they cannot fit in a chunk.

    Hard fallback to token-based splitting if a single sentence exceeds the
    max chunk size - the cut can land mid-word.
    """
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    parts = []
    for start in range(0, len(tokens), max_tokens):
        part = encoder.decode(tokens[start : start + max_tokens]).strip()
        if part:
            parts.append(part)

    return parts


def build_overlap_units(
    units: Sequence[str],
    overlap_tokens: int,
    encoder: tiktoken.Encoding,
) -> list[str]:
    """Build an overlap from the end of the previous chunk on unit boundaries.

    Only whole trailing sentences are carried; if the last sentence alone
    exceeds the budget the overlap is empty.
    """
    if overlap_tokens <= 0:
        return []

    overlap: list[str] = []
    total_tokens = 0

    for unit in reversed(units):
        unit_tokens = count_tokens(unit, encoder)
        if unit_tokens > overlap_tokens:
            break
        if total_tokens + unit_tokens > overlap_tokens:
            break
        overlap.insert(0, unit)
        total_tokens += unit_tokens

    return overlap


def chunk_content_by_tokens(
    content: str,
    encoder: tiktoken.Encoding,
    target_chunk_tokens: int,
    max_chunk_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split article content into token-aware, sentence-aware chunks.

    Prefers sentence boundaries, groups sentences around the target size, and
    seeds each new chunk with sentence-aware overlap from the previous one.
    A chunk can slightly exceed max_chunk_tokens (bounded by
    max_chunk_tokens + overlap_tokens) when an oversized-sentence part lands
    on carried overlap.
    """
    if target_chunk_tokens <= 0:
        raise ValueError("target_chunk_tokens must be greater than zero")
    if max_chunk_tokens <= 0:
        raise ValueError("max_chunk_tokens must be greater than zero")
    if target_chunk_tokens > max_chunk_tokens:
        raise ValueError("target_chunk_tokens cannot exceed max_chunk_tokens")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative")
    if overlap_tokens >= max_chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than max_chunk_tokens")

    raw_units = split_into_sentences(content)
    units: list[str] = []
    for unit in raw_units:
        units.extend(split_long_unit_by_tokens(unit, max_chunk_tokens, encoder))

    chunks: list[str] = []
    current_units: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit, encoder)

        # flush the current chunk if adding this unit would exceed the max chunk size
        # or if the current chunk has reached the target size
        should_flush = current_units and (
            current_tokens + unit_tokens > max_chunk_tokens
            or current_tokens >= target_chunk_tokens
        )

        if should_flush:
            # flush the current chunk
            chunks.append(" ".join(current_units).strip())

            # seed the next chunk with overlap from the flushed one
            current_units = build_overlap_units(current_units, overlap_tokens, encoder)
            current_tokens = sum(count_tokens(item, encoder) for item in current_units)

        current_units.append(unit)
        current_tokens += unit_tokens

    # flush any remaining units into a final chunk
    if current_units:
        chunks.append(" ".join(current_units).strip())

    return [chunk for chunk in chunks if chunk]