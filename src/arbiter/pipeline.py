"""Indexing and search orchestration (spec B, C, H).

Stage 1: PDF/TXT indexing, hybrid search, RRF fusion. HyDE, reranking and the
Laya decision layer are added in later stages and stay behind explicit calls
that are not written yet — asking for them now raises instead of silently
degrading.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

from arbiter.chunk import chunk_text
from arbiter.config import Config, Rag
from arbiter.embedding import HttpEmbedder
from arbiter.fusion import rrf
from arbiter.loaders import load_corpus
from arbiter.store import Hit, Record, Store

EMBED_DIM = 1024  # bge-m3 dense vectors
MAX_FRAGE_ZEICHEN = 2000
EMBED_BATCH = 32


class QuestionTooLong(Exception):
    """Question exceeds the configured limit (spec I3)."""


@dataclass
class IndexReport:
    files: int = 0
    chunks: int = 0
    changed: int = 0
    skipped_files: int = 0
    textless_pages: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    quelle: str
    seite: int | None
    rrf_rank: int
    rrf_score: float
    dense_rank: int | None = None
    fts_rank: int | None = None


@dataclass(frozen=True)
class SearchResult:
    question: str
    passages: list[Passage]
    hyde_text: str | None = None
    rerank: str = "aus"  # "aus" | "aktiv" | "ausgefallen"


class Pipeline:
    """One RAG corpus, one store table, one embedder."""

    def __init__(
        self,
        config: Config,
        rag: Rag,
        embedder: HttpEmbedder | None = None,
    ) -> None:
        self.config = config
        self.rag = rag
        self.embedder = embedder or HttpEmbedder(
            base_url=config.backends.base_url, model=config.backends.embed_model
        )
        self.store = Store(config.db_path)

    def index(self) -> IndexReport:
        loaded = load_corpus(self.rag.corpus_path, self.rag.extensions)
        report = IndexReport(
            files=len({doc.quelle for doc in loaded.documents}),
            skipped_files=loaded.skipped_files,
            textless_pages=loaded.textless_pages,
            warnings=list(loaded.warnings),
        )
        if not loaded.documents:
            return report

        records: list[Record] = []
        texts: list[str] = []
        for document in loaded.documents:
            page = document.seite or 0
            for number, text in enumerate(chunk_text(document.text), start=1):
                chunk_id = f"{document.quelle}#{page}#{number}"
                records.append(
                    Record(
                        id=chunk_id,
                        text=text,
                        quelle=document.quelle,
                        seite=document.seite,
                        hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        vector=[],
                    )
                )
                texts.append(text)

        vectors = self._embed_all(texts)
        filled = [
            Record(
                id=record.id,
                text=record.text,
                quelle=record.quelle,
                seite=record.seite,
                hash=record.hash,
                vector=vector,
            )
            for record, vector in zip(records, vectors, strict=True)
        ]

        report.chunks = len(filled)
        report.changed = self.store.upsert(self.rag.name, filled)
        self.store.ensure_fts(self.rag.name)
        return report

    def search(self, question: str) -> SearchResult:
        if len(question) > MAX_FRAGE_ZEICHEN:
            raise QuestionTooLong(
                f"Frage zu lang: {len(question)} Zeichen, erlaubt sind {MAX_FRAGE_ZEICHEN}"
            )

        vector = self.embedder.embed([question])[0]
        dense = self.store.search_dense(self.rag.name, vector, self.rag.top_k_dense)
        fts = self.store.search_fts(self.rag.name, question, self.rag.top_k_fts)

        dense_ids = [hit.id for hit in dense]
        fts_ids = [hit.id for hit in fts]
        fused = rrf(dense_ids, fts_ids, k=60, top_k=self.rag.fuse_top_k)

        by_id: dict[str, Hit] = {hit.id: hit for hit in reversed(dense + fts)}
        dense_rank = {hit.id: rank for rank, hit in enumerate(dense, start=1)}
        fts_rank = {hit.id: rank for rank, hit in enumerate(fts, start=1)}

        passages = [
            Passage(
                id=chunk_id,
                text=by_id[chunk_id].text,
                quelle=by_id[chunk_id].quelle,
                seite=by_id[chunk_id].seite,
                rrf_rank=rank,
                rrf_score=score,
                dense_rank=dense_rank.get(chunk_id),
                fts_rank=fts_rank.get(chunk_id),
            )
            for rank, (chunk_id, score) in enumerate(fused, start=1)
            if chunk_id in by_id
        ]
        return SearchResult(question=question, passages=passages)

    def _embed_all(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            vectors.extend(self.embedder.embed(list(texts[start : start + EMBED_BATCH])))
        return vectors
