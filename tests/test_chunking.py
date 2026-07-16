"""Invariant tests for pipeline.chunking. Run with pytest, or directly:
python tests/test_chunking.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from pipeline.chunking import (
    build_overlap_units,
    chunk_content_by_tokens,
    count_tokens,
    get_token_encoder,
    split_into_sentences,
    split_long_unit_by_tokens,
)

ENCODER = get_token_encoder("text-embedding-3-small")

# a sentence of ~40 tokens; repeated to build long articles
SENTENCE = (
    "The Housing and Development Board announced new measures for build to "
    "order flats in Singapore, covering eligibility rules and grant amounts "
    "for first time applicants."
)


def make_article(n_sentences: int) -> str:
    return " ".join(SENTENCE for _ in range(n_sentences))


# --- split_into_sentences -------------------------------------------------

def test_split_collapses_whitespace_runs():
    # whitespace runs collapse to one space, then the boundary after "." splits
    assert split_into_sentences("Hello   world.\tAgain") == ["Hello world.", "Again"]
    assert split_into_sentences("no  boundary   here") == ["no boundary here"]


def test_split_on_sentence_boundaries():
    text = "First sentence. Second sentence! Third one?"
    assert split_into_sentences(text) == [
        "First sentence.", "Second sentence!", "Third one?",
    ]


def test_split_empty_and_blank_input():
    assert split_into_sentences("") == []
    assert split_into_sentences("   \t ") == []


def test_split_does_not_break_on_lowercase_continuation():
    # "e.g. something" style: no capital after the period, so no split
    assert split_into_sentences("This is e.g. an example.") == [
        "This is e.g. an example."
    ]


# --- split_long_unit_by_tokens --------------------------------------------

def test_short_unit_passes_through_unchanged():
    assert split_long_unit_by_tokens(SENTENCE, 100, ENCODER) == [SENTENCE]


def test_long_unit_is_split_to_token_cap():
    long_unit = " ".join(["word"] * 500)  # ~500 tokens, no sentence boundary
    parts = split_long_unit_by_tokens(long_unit, 100, ENCODER)
    assert len(parts) > 1
    assert all(count_tokens(p, ENCODER) <= 100 for p in parts)


# --- build_overlap_units ---------------------------------------------------

def test_overlap_carries_whole_trailing_sentences():
    units = ["A short one.", SENTENCE, "Tail sentence here."]
    # budget covers the tail sentence but not tail + SENTENCE, so the
    # backwards walk stops after carrying just the tail
    budget = count_tokens("Tail sentence here.", ENCODER) + 2
    assert build_overlap_units(units, budget, ENCODER) == ["Tail sentence here."]


def test_overlap_empty_when_last_sentence_exceeds_budget():
    assert build_overlap_units([SENTENCE], 10, ENCODER) == []


def test_overlap_zero_budget():
    assert build_overlap_units([SENTENCE], 0, ENCODER) == []


# --- chunk_content_by_tokens -----------------------------------------------

def test_short_article_is_single_chunk():
    chunks = chunk_content_by_tokens(SENTENCE, ENCODER, 350, 500, 75)
    assert chunks == [SENTENCE]


def test_empty_content_yields_no_chunks():
    assert chunk_content_by_tokens("", ENCODER, 350, 500, 75) == []


def test_chunks_respect_soft_cap():
    # the hard bound is max + overlap (an oversized-sentence part can land
    # on carried overlap); typical chunks stay at or under max
    article = make_article(40)
    chunks = chunk_content_by_tokens(article, ENCODER, 350, 500, 75)
    assert len(chunks) > 1
    assert all(count_tokens(c, ENCODER) <= 500 + 75 for c in chunks)


def test_consecutive_chunks_overlap():
    article = make_article(40)
    chunks = chunk_content_by_tokens(article, ENCODER, 350, 500, 75)
    for prev, nxt in zip(chunks, chunks[1:]):
        # the next chunk starts with sentences carried from the previous one
        assert nxt.split(". ")[0] in prev


def test_all_content_is_preserved():
    article = make_article(40)
    chunks = chunk_content_by_tokens(article, ENCODER, 350, 500, 75)
    # every sentence appears in at least one chunk
    assert all(SENTENCE in chunk for chunk in chunks)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        chunk_content_by_tokens("x", ENCODER, 0, 500, 75)
    with pytest.raises(ValueError):
        chunk_content_by_tokens("x", ENCODER, 350, 0, 75)
    with pytest.raises(ValueError):
        chunk_content_by_tokens("x", ENCODER, 600, 500, 75)
    with pytest.raises(ValueError):
        chunk_content_by_tokens("x", ENCODER, 350, 500, -1)
    with pytest.raises(ValueError):
        chunk_content_by_tokens("x", ENCODER, 350, 500, 500)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
