"""Spec B5/B6: chunking."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from arbiter.chunk import chunk_text

MAX = 1200
OVERLAP = 100


def test_short_text_is_one_chunk() -> None:
    assert chunk_text("Ein Satz.", MAX, OVERLAP) == ["Ein Satz."]


def test_chunks_never_exceed_max_chars() -> None:
    text = "\n\n".join("Absatz " + "wort " * 60 for _ in range(40))
    chunks = chunk_text(text, MAX, OVERLAP)
    assert len(chunks) > 1
    assert all(len(c) <= MAX for c in chunks)


def test_splits_at_paragraph_boundaries_when_possible() -> None:
    para_a = "A" * 600
    para_b = "B" * 600
    chunks = chunk_text(f"{para_a}\n\n{para_b}", MAX, OVERLAP)
    assert len(chunks) == 2
    assert chunks[0].strip().startswith("A")
    assert "B" * 500 in chunks[1]


def test_overlap_is_carried_into_the_next_chunk() -> None:
    para_a = "A" * 600
    para_b = "B" * 600
    chunks = chunk_text(f"{para_a}\n\n{para_b}", MAX, OVERLAP)
    assert chunks[1].startswith("A" * OVERLAP)


def test_paragraph_longer_than_max_is_hard_split() -> None:
    text = "X" * 3000
    chunks = chunk_text(text, MAX, OVERLAP)
    assert len(chunks) >= 3
    assert all(len(c) <= MAX for c in chunks)


def test_empty_text_gives_no_chunks() -> None:
    assert chunk_text("   \n\n  ", MAX, OVERLAP) == []


@given(st.text(alphabet="abc .\n", min_size=1, max_size=4000))
def test_property_no_chunk_exceeds_max(text: str) -> None:
    for chunk in chunk_text(text, MAX, OVERLAP):
        assert len(chunk) <= MAX
