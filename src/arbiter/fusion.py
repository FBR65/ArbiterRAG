"""Reciprocal rank fusion (spec C3, C4).

Pure function: no I/O, no model. Ties are broken by document id so the same
input always yields the same order.
"""

from __future__ import annotations

from collections.abc import Sequence


def rrf_many(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    top_k: int | None = None,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse any number of ranked id lists: ``score = sum(w * 1 / (k + rank))``."""
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights muss zu rankings passen")
    scores: dict[str, float] = {}
    for index, ranked in enumerate(rankings):
        weight = 1.0 if weights is None else weights[index]
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)

    fused = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        return fused[:top_k]
    return fused


def rrf(
    left: Sequence[str],
    right: Sequence[str],
    k: int = 60,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse two ranked id lists (the two-list case of :func:`rrf_many`)."""
    return rrf_many([left, right], k=k, top_k=top_k)
