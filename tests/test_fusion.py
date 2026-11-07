"""Spec C3/C4: reciprocal rank fusion."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from arbiter.fusion import rrf


def test_document_in_both_lists_wins() -> None:
    fused = rrf(["a", "b"], ["b", "a"], k=60)
    assert [doc for doc, _ in fused] == ["a", "b"]
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_scores_are_added_across_lists() -> None:
    fused = dict(rrf(["x"], ["x", "y"], k=60))
    assert fused["x"] == pytest.approx(1 / 61 + 1 / 61)
    assert fused["y"] == pytest.approx(1 / 62)


def test_equal_scores_sort_by_id_ascending() -> None:
    fused = rrf(["b"], ["a"], k=60)
    assert [doc for doc, _ in fused] == ["a", "b"]


def test_output_is_deterministic() -> None:
    first = rrf(["a", "b", "c"], ["c", "b", "a"], k=60)
    second = rrf(["a", "b", "c"], ["c", "b", "a"], k=60)
    assert first == second


def test_top_k_truncates_after_sorting() -> None:
    fused = rrf(["a", "b", "c"], ["d"], k=60, top_k=2)
    assert len(fused) == 2
    assert [doc for doc, _ in fused] == ["a", "d"]


@given(
    st.lists(st.text(alphabet="abcdef", min_size=1, max_size=1), max_size=8, unique=True),
    st.lists(st.text(alphabet="abcdef", min_size=1, max_size=1), max_size=8, unique=True),
)
def test_property_scores_within_bounds(left: list[str], right: list[str]) -> None:
    for _, score in rrf(left, right, k=60):
        assert 0 < score <= 2 / 61 + 1e-12
