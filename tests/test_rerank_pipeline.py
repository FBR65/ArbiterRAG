"""Spec E: reranking inside the pipeline."""

from __future__ import annotations

from pathlib import Path

from arbiter.config import Backends, Config, Rag
from arbiter.pipeline import Pipeline


class TwoChunkEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _pipeline(tmp_path: Path, reranker) -> Pipeline:
    corpus = tmp_path / "korpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Der erste Text ueber Miete.", encoding="utf-8")
    (corpus / "b.txt").write_text("Der zweite Text ueber Nebenkosten.", encoding="utf-8")
    cfg = Config(
        db_path=tmp_path / "db",
        backends=Backends(
            base_url="http://test",
            embed_model="m",
            rerank_model="r",
            chat_model="c",
        ),
        rags={
            "k": Rag(
                name="k",
                description="Test",
                corpus_path=corpus,
                extensions=(".txt",),
                fuse_top_k=5,
                rerank_top_n=2,
            )
        },
    )
    pipeline = Pipeline(cfg, cfg.rags["k"], embedder=TwoChunkEmbedder())
    pipeline.index()
    pipeline.reranker = reranker
    return pipeline


def test_reranker_order_wins_over_rrf(tmp_path: Path) -> None:
    """Score 5.0 for the chunk RRF put last: the reranker decides."""
    seen: dict[str, object] = {}

    def reranker(query: str, documents: list[str]) -> list[float]:
        seen["query"] = query
        seen["documents"] = documents
        return [5.0 if "Nebenkosten" in text else 0.1 for text in documents]

    pipeline = _pipeline(tmp_path, reranker)
    result = pipeline.search("Frage")
    assert seen["query"] == "Frage"
    assert result.rerank == "aktiv"
    assert result.passages[0].quelle == "b.txt"
    assert len(result.passages) == 2
    assert result.passages[0].rerank_score == 5.0


def test_rerank_cut_keeps_only_top_n(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, lambda query, documents: [1.0 for _ in documents])
    result = pipeline.search("Frage")
    assert len(result.passages) == 2


def test_ties_keep_the_rrf_order(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, lambda query, documents: [0.0 for _ in documents])
    result = pipeline.search("Frage")
    assert [passage.rrf_rank for passage in result.passages] == [1, 2]


def test_failing_reranker_falls_back_to_rrf_with_a_warning(tmp_path: Path) -> None:
    def reranker(query: str, documents: list[str]) -> list[float]:
        raise RuntimeError("Reranker antwortet 500")

    pipeline = _pipeline(tmp_path, reranker)
    result = pipeline.search("Frage")
    assert result.rerank == "ausgefallen"
    assert any("500" in warning for warning in result.warnungen)
    assert [passage.rrf_rank for passage in result.passages] == [1, 2]
