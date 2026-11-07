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
from typing import Any

from arbiter.answer import Quelle, build_answer_prompt
from arbiter.chunk import chunk_text
from arbiter.config import Config, Rag
from arbiter.decision import (
    DecisionError,
    SystemOne,
    build_choice_questions,
    build_deckung_questions,
    build_noul_state,
    build_pair_questions,
    build_questions,
    build_widerspruch_questions,
    choice_probabilities,
    level,
    probability,
)
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
    duplicates: int = 0
    entfernt: int = 0
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
class Zulassung:
    passage: Passage
    p: float
    deckung: float | None = None


@dataclass(frozen=True)
class PaarBefund:
    a: str
    b: str
    aehnlichkeit: float | None
    widerspruch_p: float | None
    widerspruch: bool = False


@dataclass(frozen=True)
class Entscheidung:
    beantwortbar: bool
    grund: str | None
    zulassungen: list[Zulassung]
    auswahl: list[Zulassung]
    paare: list[PaarBefund]
    warnungen: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Antwort:
    text: str | None
    grund: str | None
    quellen: list[Quelle]
    entscheidung: Entscheidung | None = None
    fehler: bool = False
    warnungen: list[str] = field(default_factory=list)


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
        # Callable(state, questions) -> answers map; and Callable(system, user) -> text.
        self.decision: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None
        self.answerer: Callable[[str, str], str] | None = None

    def index(self) -> IndexReport:
        loaded = load_corpus(self.rag.corpus_path, self.rag.extensions, ocr=self._ocr_hook())
        report = IndexReport(
            files=len({doc.quelle for doc in loaded.documents}),
            skipped_files=loaded.skipped_files,
            textless_pages=loaded.textless_pages,
            ocr_pages=sum(1 for doc in loaded.documents if doc.ocr),
            duplicates=loaded.duplicates,
            warnings=list(loaded.warnings),
        )
        if not loaded.documents:
            return report

        records: list[Record] = []
        texts: list[str] = []
        for document in loaded.documents:
            page = document.seite or 0
            kopf = f"[{Path(document.quelle).stem}, S. {page}] "
            for number, text in enumerate(
                chunk_text(document.text, self.rag.chunk_chars, self.rag.chunk_overlap),
                start=1,
            ):
                chunk_id = f"{document.quelle}#{page}#{number}"
                inhalt = kopf + text
                records.append(
                    Record(
                        id=chunk_id,
                        text=inhalt,
                        quelle=document.quelle,
                        seite=document.seite,
                        hash=hashlib.sha256(inhalt.encode("utf-8")).hexdigest(),
                        vector=[],
                        ocr=document.ocr,
                        ocr_paddle=document.ocr_paddle,
                        ocr_ueberlappung=document.ocr_ueberlappung,
                    )
                )
                texts.append(inhalt)

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
        report.entfernt = self.store.purge(
            self.rag.name, {document.quelle for document in loaded.documents}
        )
        self.store.ensure_fts(self.rag.name, stem=self.rag.fts_stem)
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

        fused = rrf_many(
            rankings,
            k=60,
            top_k=self.rag.fuse_top_k,
            weights=[1.0] * (len(rankings) - 2) + [self.rag.hyde_gewicht] * 2
            if len(rankings) > 2
            else None,
        )

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
        if not self.rag.hybrid:
            hits = {hit.id: hit for hit in reversed(dense)}
            return [hit.id for hit in dense], [], hits
        fts = self.store.search_fts(self.rag.name, text, self.rag.top_k_fts)
        hits = {hit.id: hit for hit in reversed(dense + fts)}
        return [hit.id for hit in dense], [hit.id for hit in fts], hits

    def bewerte(self, result: SearchResult) -> Entscheidung:
        """Ask the decision layer which passages may serve as evidence (spec F).

        Measured 2026-11-07: one state holding all passages makes the small
        multilingual model answer about the whole text — every P(true) came back
        between 0.98 and 0.99. One state per passage separates a real hit (0.941)
        from noise (0.001-0.028). So each passage gets its own request.
        """
        passages = result.passages
        warnungen = list(result.warnungen)
        if not passages:
            return Entscheidung(False, "keine Passage gefunden", [], [], [], warnungen)

        if self.rag.gate_modus == "choice":
            return self._bewerte_choice(result, passages, warnungen)
        if self.rag.gate_modus == "aus":
            # No decision call: the reranker order alone decides. This exists to
            # measure what the decision layer adds, not as a production default.
            auswahl = [Zulassung(passage=passage, p=1.0) for passage in passages[:3]]
            return Entscheidung(True, None, auswahl, auswahl, [], warnungen)

        zulassungen: list[Zulassung] = []
        for number, passage in enumerate(passages, start=1):
            prob, note = self._gate(result.question, passage)
            if note:
                warnungen.append(f"Auszug {number}: {note}")
            if prob >= self.rag.laya_schwelle:
                zulassungen.append(Zulassung(passage=passage, p=prob))

        if not zulassungen:
            return Entscheidung(False, "keine belegte Passage", [], [], [], warnungen)

        zulassungen = [
            replace(zulassung, deckung=self._deckung(result.question, zulassung.passage))
            for zulassung in zulassungen
        ]

        paare = self._pairs(result.question, zulassungen)
        auswahl = self._select(zulassungen, paare)

        return Entscheidung(
            beantwortbar=True,
            grund=None,
            zulassungen=zulassungen,
            auswahl=auswahl,
            paare=paare,
            warnungen=warnungen,
        )

    def _bewerte_choice(
        self, result: SearchResult, passages: list[Passage], warnungen: list[str]
    ) -> Entscheidung:
        """Gate as one choice over all candidates plus "keine" (E2)."""
        state = build_noul_state(result.question, [passage.text for passage in passages])
        answers = self._ask_decision(state, build_choice_questions(len(passages)))
        probabilities = choice_probabilities(answers)

        if not probabilities:
            warnungen.append("keine Antwort der Prüfschicht")
            return Entscheidung(False, "keine belegte Passage", [], [], [], warnungen)

        bester = max(probabilities, key=lambda option: probabilities[option])
        keine = probabilities.get("keine", 0.0)
        marge = self.rag.gate_marge
        # An absolute threshold is meaningless with eleven options: the mass is
        # spread over them. What counts is the lead over "keine" (measured
        # 2026-11-07: with laya_schwelle as a hard cut nothing was admitted).
        if bester == "keine" or probabilities[bester] < keine + marge:
            return Entscheidung(False, "keine belegte Passage", [], [], [], warnungen)

        nummer = int(bester.lstrip("p"))
        passage = passages[nummer - 1]
        zulassung = Zulassung(passage=passage, p=probabilities[bester])
        zulassung = replace(zulassung, deckung=self._deckung(result.question, passage))
        return Entscheidung(True, None, [zulassung], [zulassung], [], warnungen)

    def _gate(self, question: str, passage: Passage) -> tuple[float, str | None]:
        """P(true) that this one passage answers the question (spec F1, F2)."""
        state = build_noul_state(question, [passage.text])
        answers = self._ask_decision(state, {"p1": build_questions(1)["p1"]})
        return probability(answers, "p1")

    def _deckung(self, question: str, passage: Passage) -> float | None:
        """Coverage level 1..4 for one admissible passage (spec F4)."""
        state = build_noul_state(question, [passage.text])
        answers = self._ask_decision(state, {"d1": build_deckung_questions(1)["d1"]})
        return level(answers, "d1")

    def _pairs(self, question: str, zulassungen: list[Zulassung]) -> list[PaarBefund]:
        """Compare admissible passages pairwise: similarity, contradiction (F7)."""
        if len(zulassungen) < 2:
            return []
        befunde: list[PaarBefund] = []
        for i in range(len(zulassungen)):
            for j in range(i + 1, len(zulassungen)):
                first = zulassungen[i]
                second = zulassungen[j]
                state = build_noul_state(question, [first.passage.text, second.passage.text])
                questions = build_pair_questions([(1, 2)])
                questions.update(build_widerspruch_questions([(1, 2)]))
                answers = self._ask_decision(state, questions)
                widerspruch_p, _ = probability(answers, "w1_2")
                schwelle = self.rag.widerspruch_niveau
                befunde.append(
                    PaarBefund(
                        a=first.passage.id,
                        b=second.passage.id,
                        aehnlichkeit=level(answers, "v1_2"),
                        widerspruch_p=widerspruch_p,
                        widerspruch=schwelle is not None and widerspruch_p >= schwelle,
                    )
                )
        return befunde

    def _select(self, zulassungen: list[Zulassung], paare: list[PaarBefund]) -> list[Zulassung]:
        """Order by coverage; drop only what a measured threshold permits (spec F7)."""
        ordered = sorted(
            zulassungen,
            key=lambda zulassung: (
                -(zulassung.deckung if zulassung.deckung is not None else -1.0),
                zulassung.passage.rrf_rank,
            ),
        )
        drop: set[str] = set()
        for befund in paare:
            redundant = (
                self.rag.redundanz_niveau is not None
                and befund.aehnlichkeit is not None
                and befund.aehnlichkeit >= self.rag.redundanz_niveau
            )
            if redundant or befund.widerspruch:
                drop.add(self._weaker(befund, ordered))
        return [zulassung for zulassung in ordered if zulassung.passage.id not in drop]

    @staticmethod
    def _weaker(befund: PaarBefund, ordered: list[Zulassung]) -> str:
        """The passage with less coverage loses; on a tie the later one in the order."""
        position = {zulassung.passage.id: index for index, zulassung in enumerate(ordered)}
        return befund.a if position.get(befund.a, 0) > position.get(befund.b, 0) else befund.b

    def ask(
        self,
        question: str,
        hyde: bool | None = None,
        vorgabe: SearchResult | None = None,
    ) -> Antwort:
        """Search, decide, answer — in that order (spec F3, G).

        ``vorgabe`` reuses an already computed search result, so a caller that
        measured the search does not pay for a second HyDE call.
        """
        result = vorgabe if vorgabe is not None else self.search(question, hyde=hyde)
        try:
            entscheidung = self.bewerte(result)
        except DecisionError as exc:
            return Antwort(
                text=None,
                grund="Prüfschicht nicht erreichbar",
                quellen=[],
                entscheidung=None,
                fehler=True,
                warnungen=[str(exc)],
            )

        if not entscheidung.beantwortbar:
            return Antwort(
                text=None,
                grund=entscheidung.grund,
                quellen=[],
                entscheidung=entscheidung,
                fehler=False,
                warnungen=entscheidung.warnungen,
            )

        quellen = [
            Quelle(
                passage_id=zulassung.passage.id,
                quelle=zulassung.passage.quelle,
                seite=zulassung.passage.seite,
                text=zulassung.passage.text,
            )
            for zulassung in entscheidung.auswahl[:3]
        ]
        system, user = build_answer_prompt(question, quellen)
        try:
            text = self._answer_text(system, user).strip()
        except Exception as exc:
            return Antwort(
                text=None,
                grund="Antwortmodell nicht erreichbar",
                quellen=quellen,
                entscheidung=entscheidung,
                fehler=True,
                warnungen=[*entscheidung.warnungen, str(exc)],
            )
        if not text:
            return Antwort(
                text=None,
                grund="Antwortmodell lieferte keinen Text",
                quellen=quellen,
                entscheidung=entscheidung,
                fehler=True,
                warnungen=entscheidung.warnungen,
            )
        return Antwort(
            text=text,
            grund=None,
            quellen=quellen,
            entscheidung=entscheidung,
            fehler=False,
            warnungen=entscheidung.warnungen,
        )

    def _ask_decision(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        if self.decision is None:
            client = SystemOne(self.config.backends.base_url, self.config.backends.laya_model)
            self.decision = client.ask
        return self.decision(state, questions)

    def _answer_text(self, system: str, user: str) -> str:
        if self.answerer is None:
            client = ChatClient(self.config.backends.base_url, self.config.backends.chat_model)
            self.answerer = client.complete
        return str(self.answerer(system, user))

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
