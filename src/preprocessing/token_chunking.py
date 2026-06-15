import re
from typing import Sequence

import tiktoken

from src.preprocessing.semantic_schema import DEFAULT_EMBEDDING_MODEL

SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def get_token_encoder(model: str = DEFAULT_EMBEDDING_MODEL) -> tiktoken.Encoding:
    """Return the tiktoken encoder for the embedding model."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: object, encoder: tiktoken.Encoding) -> int:
    """Count tokens using the configured tiktoken encoder."""
    return len(encoder.encode(str(text)))


def clean_text(text: object) -> str:
    """Normalise whitespace while preserving paragraph breaks when present."""
    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def split_into_sentences(text: object) -> list[str]:
    """Split article text into sentence-like units."""
    text = clean_text(text)
    if not text:
        return []

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    sentences: list[str] = []

    for paragraph in paragraphs:
        pieces = SENTENCE_BOUNDARY_PATTERN.split(paragraph)
        sentences.extend(piece.strip() for piece in pieces if piece.strip())

    return sentences


def split_long_unit_by_tokens(
    text: str,
    max_tokens: int,
    encoder: tiktoken.Encoding,
) -> list[str]:
    """Split a sentence-like unit only when it cannot fit in a chunk."""
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
    """Build an overlap from the end of the previous chunk on unit boundaries."""
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
    content: object,
    encoder: tiktoken.Encoding,
    target_chunk_tokens: int = 700,
    max_chunk_tokens: int = 900,
    overlap_tokens: int = 100,
) -> list[str]:
    """
    Split article content into token-aware, sentence-aware chunks.

    The function prefers sentence boundaries, groups sentences around the target
    size, and uses sentence-aware overlap between chunks.
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
        should_flush = current_units and (
            current_tokens + unit_tokens > max_chunk_tokens
            or current_tokens >= target_chunk_tokens
        )

        if should_flush:
            chunks.append(" ".join(current_units).strip())
            current_units = build_overlap_units(current_units, overlap_tokens, encoder)
            current_tokens = sum(count_tokens(item, encoder) for item in current_units)

        current_units.append(unit)
        current_tokens += unit_tokens

    if current_units:
        chunks.append(" ".join(current_units).strip())

    return [chunk for chunk in chunks if chunk]
