"""Indexing and search orchestration (spec B, C, H).

Stage 1: PDF/TXT indexing, hybrid search, RRF fusion. HyDE, reranking and the
Laya decision layer are added in later stages and stay behind explicit calls
that are not written yet — asking for them now raises instead of silently
degrading.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from arbiter.chunk import chunk_text
from arbiter.config import Config, Rag
from arbiter.embedding import HttpEmbedder
from arbiter.fusion import rrf_many
from arbiter.hyde import ChatClient, build_prompt, truncate
from arbiter.loaders import OcrHook, OcrPage, load_corpus
from arbiter.ocr import VisionOcr, dots_to_text, render_png, word_overlap
from arbiter.rerank import RerankClient
from arbiter.store import Hit, Record, Store

MAX_FRAGE_ZEICHEN = 2000
EMBED_BATCH = 32
# PaddleOCR-VL is task-prompted; "OCR:" is the task switch for plain text.
PADDLE_PROMPT = "OCR:"


class QuestionTooLong(Exception):
    """Question exceeds the configured limit (spec I3)."""


@dataclass
class IndexReport:
    files: int = 0
    chunks: int = 0
    changed: int = 0
    skipped_files: int = 0
    textless_pages: int = 0
    ocr_pages: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    quelle: str
    seite: int | None
    rrf_rank: int
    rrf_score: float
    rerank_score: float | None = None


@dataclass(frozen=True)
class SearchResult:
    question: str
    passages: list[Passage]
    hyde_text: str | None = None
    rerank: str = "aus"  # "aus" | "aktiv" | "ausgefallen"
    warnungen: list[str] = field(default_factory=list)


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
        # Callable(question, description) -> hypothetical answer document.
        # Left empty here so tests can inject one and the real client stays lazy.
        self.hyde: Callable[[str, str], str] | None = None
        # Callable(query, documents) -> one score per document.
        self.reranker: Callable[[str, list[str]], list[float]] | None = None

    def index(self) -> IndexReport:
        loaded = load_corpus(self.rag.corpus_path, self.rag.extensions, ocr=self._ocr_hook())
        report = IndexReport(
            files=len({doc.quelle for doc in loaded.documents}),
            skipped_files=loaded.skipped_files,
            textless_pages=loaded.textless_pages,
            ocr_pages=sum(1 for doc in loaded.documents if doc.ocr),
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
                        ocr=document.ocr,
                        ocr_paddle=document.ocr_paddle,
                        ocr_ueberlappung=document.ocr_ueberlappung,
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
                ocr=record.ocr,
                ocr_paddle=record.ocr_paddle,
                ocr_ueberlappung=record.ocr_ueberlappung,
            )
            for record, vector in zip(records, vectors, strict=True)
        ]

        report.chunks = len(filled)
        report.changed = self.store.upsert(self.rag.name, filled)
        self.store.ensure_fts(self.rag.name)
        return report

    def search(self, question: str, hyde: bool | None = None) -> SearchResult:
        if len(question) > MAX_FRAGE_ZEICHEN:
            raise QuestionTooLong(
                f"Frage zu lang: {len(question)} Zeichen, erlaubt sind {MAX_FRAGE_ZEICHEN}"
            )

        warnungen: list[str] = []
        use_hyde = self.rag.hyde if hyde is None else hyde

        dense_ids, fts_ids, hits = self._rank_lists(question)
        rankings: list[list[str]] = [dense_ids, fts_ids]
        hyde_text: str | None = None

        if use_hyde:
            try:
                hyde_text = self._hyde_document(question)
            except Exception as exc:
                warnungen.append(f"HyDE fehlgeschlagen, Suche ohne HyDE: {exc}")
                hyde_text = None
            if hyde_text:
                hyde_dense_ids, hyde_fts_ids, hyde_hits = self._rank_lists(hyde_text)
                rankings.extend([hyde_dense_ids, hyde_fts_ids])
                hits.update(hyde_hits)

        fused = rrf_many(rankings, k=60, top_k=self.rag.fuse_top_k)

        passages = [
            Passage(
                id=chunk_id,
                text=hits[chunk_id].text,
                quelle=hits[chunk_id].quelle,
                seite=hits[chunk_id].seite,
                rrf_rank=rank,
                rrf_score=score,
            )
            for rank, (chunk_id, score) in enumerate(fused, start=1)
            if chunk_id in hits
        ]

        rerank_status = "aus"
        if self.rag.rerank and passages:
            try:
                scores = self._rerank_scores(question, [passage.text for passage in passages])
            except Exception as exc:
                warnungen.append(f"Reranker fehlgeschlagen, RRF-Reihenfolge gilt: {exc}")
                rerank_status = "ausgefallen"
            else:
                rerank_status = "aktiv"
                ranked = sorted(
                    zip(passages, scores, strict=True),
                    key=lambda pair: (-pair[1], pair[0].rrf_rank),
                )
                passages = [
                    replace(passage, rerank_score=score)
                    for passage, score in ranked[: self.rag.rerank_top_n]
                ]

        return SearchResult(
            question=question,
            passages=passages,
            hyde_text=hyde_text,
            rerank=rerank_status,
            warnungen=warnungen,
        )

    def _rank_lists(self, text: str) -> tuple[list[str], list[str], dict[str, Hit]]:
        """Both halves of the hybrid search over one query text."""
        vector = self.embedder.embed([text])[0]
        dense = self.store.search_dense(self.rag.name, vector, self.rag.top_k_dense)
        fts = self.store.search_fts(self.rag.name, text, self.rag.top_k_fts)
        hits = {hit.id: hit for hit in reversed(dense + fts)}
        return [hit.id for hit in dense], [hit.id for hit in fts], hits

    def _rerank_scores(self, question: str, documents: list[str]) -> list[float]:
        if self.reranker is None:
            client = RerankClient(self.config.backends.base_url, self.config.backends.rerank_model)
            self.reranker = client.scores
        return self.reranker(question, documents)

    def _hyde_document(self, question: str) -> str:
        """One hypothetical answer document, cut to ``hyde_max_chars``."""
        if self.hyde is None:
            client = ChatClient(self.config.backends.base_url, self.config.backends.chat_model)
            system, user = build_prompt(question, self.rag.description)
            self.hyde = lambda _q, _d: client.complete(system, user)
        text = self.hyde(question, self.rag.description)
        return truncate(str(text), self.rag.hyde_max_chars).strip()

    def _ocr_hook(self) -> OcrHook | None:
        """Both OCR models run on the same rendering; dots-mocr supplies the text."""
        if not self.rag.ocr:
            return None
        backends = self.config.backends
        dots = VisionOcr(backends.base_url, backends.ocr_dots_model)
        paddle = VisionOcr(backends.base_url, backends.ocr_paddle_model)
        dpi = self.rag.ocr_dpi

        def hook(path: Path, page_number: int) -> OcrPage:
            png = render_png(path, page_number, dpi)
            indexed = dots_to_text(dots.text_from_png(png))
            zweitmeinung = paddle.text_from_png(png, prompt=PADDLE_PROMPT)
            return OcrPage(
                text=indexed,
                vergleich=zweitmeinung,
                ueberlappung=word_overlap(indexed, zweitmeinung),
            )

        return hook

    def _embed_all(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            vectors.extend(self.embedder.embed(list(texts[start : start + EMBED_BATCH])))
        return vectors
