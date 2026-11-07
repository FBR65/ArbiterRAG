"""Spec B1-B3: chunk size, header prefix, hybrid switch, stemming."""

from __future__ import annotations

from pathlib import Path

from arbiter.config import Backends, Config, Rag
from arbiter.pipeline import Pipeline


class FlatEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _cfg(tmp_path: Path, **rag_kwargs: object) -> Config:
    corpus = tmp_path / "korpus"
    corpus.mkdir(exist_ok=True)
    (corpus / "vertrag.txt").write_text("Miete 940 Euro. " * 80, encoding="utf-8")
    return Config(
        db_path=tmp_path / "db",
        backends=Backends(
            base_url="http://test",
            embed_model="m",
            rerank_model="r",
            chat_model="c",
            laya_model="l",
        ),
        rags={
            "k": Rag(
                name="k",
                description="Test",
                corpus_path=corpus,
                extensions=(".txt",),
                hyde=False,
                rerank=False,
                **rag_kwargs,  # type: ignore[arg-type]
            )
        },
    )


def test_chunks_carry_document_and_page_prefix(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, chunk_chars=200, chunk_overlap=20)
    pipeline = Pipeline(cfg, cfg.rags["k"], embedder=FlatEmbedder())
    pipeline.index()
    hits = pipeline.store.search_dense("k", [1.0, 0.0], k=1)
    assert hits[0].text.startswith("[vertrag, S. 0]")


def test_smaller_chunks_produce_more_rows(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, chunk_chars=200, chunk_overlap=20)
    pipeline = Pipeline(cfg, cfg.rags["k"], embedder=FlatEmbedder())
    report = pipeline.index()
    assert report.chunks > 1
    assert pipeline.store.count("k") == report.chunks


def test_hybrid_off_uses_only_the_dense_half(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, hybrid=False)
    pipeline = Pipeline(cfg, cfg.rags["k"], embedder=FlatEmbedder())
    pipeline.index()
    result = pipeline.search("Miete")
    assert result.passages
    assert all(passage.rrf_rank == index for index, passage in enumerate(result.passages, 1))


def test_fts_stemming_is_configurable(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, fts_stem=True)
    pipeline = Pipeline(cfg, cfg.rags["k"], embedder=FlatEmbedder())
    pipeline.index()
    # "Mieten" must reach "Miete" when German stemming is on.
    assert pipeline.store.search_fts("k", "Mieten", k=3)
