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

from arbiter.answer import Quelle, build_answer_prompt
from arbiter.check import build_claim_prompt, build_repair_prompt, split_claims
from arbiter.chunk import chunk_text
from arbiter.config import Config, Rag
from arbiter.decision import (
    GATE_GRAMMATIK,
    DecisionError,
    build_ja_nein_prompt,
    ja_wahrscheinlichkeit,
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
    unchanged: int = 0
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


@dataclass(frozen=True)
class Entscheidung:
    beantwortbar: bool
    grund: str | None
    auswahl: list[Zulassung]
    warnungen: list[str] = field(default_factory=list)
    sicher: bool = False


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
        self.answerer: Callable[[str, str], str] | None = None
        self.gate_logprobs: Callable[[str, str], dict[str, float]] | None = None

    def index(self) -> IndexReport:
        # The check comes first: an unchanged file is not read again, so OCR and
        # embedding are not paid twice for it. Skipped files keep their rows.
        bekannt = self.store.fingerprints(self.rag.name)
        einstellungen = self._ingest_einstellungen()

        def unveraendert(relative: str, kennung: str) -> bool:
            return bekannt.get(relative) == kennung

        loaded = load_corpus(
            self.rag.corpus_path,
            self.rag.extensions,
            ocr=self._ocr_hook(),
            sprache=self.rag.sprache,
            ocr_alle=self.rag.ocr_alle_seiten,
            unveraendert=unveraendert,
            einstellungen=einstellungen,
        )
        report = IndexReport(
            files=len({doc.quelle for doc in loaded.documents}),
            skipped_files=loaded.skipped_files,
            textless_pages=loaded.textless_pages,
            ocr_pages=sum(1 for doc in loaded.documents if doc.ocr),
            duplicates=loaded.duplicates,
            unchanged=loaded.unchanged,
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
                        eltern=document.text if self.rag.eltern_seite else inhalt,
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
        # Skipped files are still part of the corpus; without this line purge would
        # delete their rows the moment they are skipped.
        vorhanden = {document.quelle for document in loaded.documents} | set(loaded.fingerprints)
        report.entfernt = self.store.purge(self.rag.name, vorhanden)
        for relative, kennung in loaded.fingerprints.items():
            if kennung:
                self.store.set_fingerprint(self.rag.name, relative, kennung)
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
                text=(hits[chunk_id].eltern or hits[chunk_id].text)
                if self.rag.eltern_seite
                else hits[chunk_id].text,
                quelle=hits[chunk_id].quelle,
                seite=hits[chunk_id].seite,
                rrf_rank=rank,
                rrf_score=score,
            )
            for rank, (chunk_id, score) in enumerate(fused, start=1)
            if chunk_id in hits
        ]

        if self.rag.eltern_seite:
            # Several spans of one page share the same parent text; keep the best rank.
            gesehen: set[tuple[str, int | None]] = set()
            eindeutig: list[Passage] = []
            for passage in passages:
                schluessel = (passage.quelle, passage.seite)
                if schluessel in gesehen:
                    continue
                gesehen.add(schluessel)
                eindeutig.append(passage)
            passages = eindeutig

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
        """Ask the gate which of the best passages may serve as evidence (spec F).

        One request per candidate. The model answers with a single token and the
        probability of "ja" decides. Measured 2026-11-07 on the goldset: with the
        prompt variant "teil" unanswerable questions reach 0.09 at most while
        answerable ones sit at 1.00, so a threshold in 0.3..0.9 separates them.
        """
        passages = result.passages[: self.rag.gate_kandidaten]
        warnungen = list(result.warnungen)
        if not passages:
            return Entscheidung(False, "keine Passage gefunden", [], warnungen)

        zulassungen: list[Zulassung] = []
        for passage in passages:
            system, user = build_ja_nein_prompt(result.question, passage.text, self.rag.gate_prompt)
            try:
                verteilung = self._logprobs(system, user)
            except Exception as exc:
                warnungen.append(f"Prüfschicht nicht erreichbar: {exc}")
                continue
            p = ja_wahrscheinlichkeit(verteilung)
            if p is None:
                # The model did not answer with ja/nein (measured: 467 of 1000 on long
                # contexts). Refusing here would answer nothing although the passage may
                # be right, so the passage is admitted and the sentence check decides.
                warnungen.append("Prüfschicht ohne verwertbare Antwort, Passage zugelassen")
                zulassungen.append(Zulassung(passage=passage, p=0.0))
                continue
            if p >= self.rag.gate_schwelle:
                zulassungen.append(Zulassung(passage=passage, p=p))

        if not zulassungen:
            return Entscheidung(False, "keine belegte Passage", [], warnungen)
        auswahl = sorted(zulassungen, key=lambda item: item.p, reverse=True)
        # "sicher" heisst: das Gate hat fuer mindestens eine Passage ja gesagt. Nur dann
        # darf als letzter Ausweg die Belegstelle zitiert werden.
        sicher = any(zulassung.p >= self.rag.gate_schwelle for zulassung in zulassungen)
        return Entscheidung(True, None, auswahl, warnungen, sicher)

    def _logprobs(self, system: str, user: str) -> dict[str, float]:
        """Logprob map of the chat model; injectable for tests."""
        if self.gate_logprobs is not None:
            return self.gate_logprobs(system, user)
        client = ChatClient(self.config.backends.base_url, self.config.backends.chat_model)
        return client.probabilities(system, user, grammar=GATE_GRAMMATIK)

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
            for zulassung in entscheidung.auswahl[: self.rag.antwort_passagen]
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

        zusatz: list[str] = []
        if self.rag.antwort_pruefung:
            behalten, verworfen = self._pruefe_saetze(text, quellen)
            if verworfen:
                zusatz = [f"Gestrichen (nicht belegt): {satz[:120]}" for satz in verworfen]
            if behalten:
                text = " ".join(behalten)
            elif entscheidung.sicher:
                # Nothing survived the check, but the gate was sure: quote the source.
                # A verbatim passage cannot be a hallucination.
                text = self._zitat(quellen)
                zusatz.append("Keine belegte Aussage formuliert, Belegstelle zitiert")
            else:
                # The gate could not decide and no sentence was supported. Answering
                # anyway would mean answering from a passage nobody vouched for.
                return Antwort(
                    text=None,
                    grund="keine belegte Passage",
                    quellen=quellen,
                    entscheidung=entscheidung,
                    fehler=False,
                    warnungen=[*entscheidung.warnungen, *zusatz],
                )

        return Antwort(
            text=text,
            grund=None,
            quellen=quellen,
            entscheidung=entscheidung,
            fehler=False,
            warnungen=[*entscheidung.warnungen, *zusatz],
        )

    def _zitat(self, quellen: list[Quelle]) -> str:
        """Verbatim quotation of the best admitted passage — cannot be invented."""
        quelle = quellen[0]
        text = " ".join(quelle.text.split())
        return f"Belegstelle: „{text}“ [{quelle.quelle}, S. {quelle.seite}]"

    def _pruefe_saetze(self, text: str, quellen: list[Quelle]) -> tuple[list[str], list[str]]:
        """Claims that no admitted passage supports (spec: no hallucinations).

        A sentence that names a source is checked against that source. A sentence
        without a citation is checked against every admitted passage — it may be a
        summary of what the others say. Only a sentence that no passage supports
        makes the answer unusable.
        """
        nach_quelle = {(quelle.quelle, quelle.seite): quelle.text for quelle in quellen}
        alle = [quelle.text for quelle in quellen]
        behalten: list[str] = []
        verworfen: list[str] = []
        for satz, quelle, seite in split_claims(text):
            if quelle is None or seite is None:
                kandidaten = alle
            else:
                kandidaten = [
                    auszug
                    for (datei, nummer), auszug in nach_quelle.items()
                    if quelle in datei and nummer == seite
                ]
                if not kandidaten:
                    verworfen.append(satz)
                    continue
            if self._irgendeine_stuetzt(satz, kandidaten):
                behalten.append(satz)
            else:
                verworfen.append(satz)
        return behalten, verworfen

    def _repariere_antwort(self, text: str, quellen: list[Quelle]) -> str | None:
        """One rewrite attempt: keep only what the passages state. Then re-check."""
        system, user = build_repair_prompt(
            [
                (quelle.quelle, quelle.seite, quelle.text)
                for quelle in quellen
                if quelle.seite is not None
            ],
            text,
        )
        try:
            return self._answer_text(system, user).strip() or None
        except Exception:
            return None

    def _irgendeine_stuetzt(self, satz: str, kandidaten: list[str]) -> bool:
        """True as soon as one passage carries the sentence."""
        for auszug in kandidaten:
            system, user = build_claim_prompt(satz, auszug)
            try:
                p = ja_wahrscheinlichkeit(self._logprobs(system, user))
            except Exception:
                continue
            if p is not None and p >= self.rag.antwort_schwelle:
                return True
        return False

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

    def _ingest_einstellungen(self) -> str:
        """Everything that changes the text of a file, as one string.

        Part of the file fingerprint: if one of these values changes, the file is read
        again instead of being skipped.
        """
        rag = self.rag
        return "|".join(
            str(wert)
            for wert in (
                rag.chunk_chars,
                rag.chunk_overlap,
                rag.ocr,
                rag.ocr_alle_seiten,
                rag.ocr_dpi,
                rag.ocr_dpi_paddle,
                rag.sprache,
                self.config.backends.ocr_dots_model,
                self.config.backends.ocr_paddle_model,
                self.config.backends.embed_model,
            )
        )

    def _ocr_hook(self) -> OcrHook | None:
        """Both OCR models run on the same rendering; dots-mocr supplies the text."""
        if not self.rag.ocr:
            return None
        backends = self.config.backends
        dots = VisionOcr(backends.base_url, backends.ocr_dots_model)
        paddle = VisionOcr(backends.base_url, backends.ocr_paddle_model)
        dpi_dots = self.rag.ocr_dpi
        dpi_paddle = self.rag.ocr_dpi_paddle

        def hook(path: Path, page_number: int) -> OcrPage:
            # Both engines get their own rendering, as in DocPipeline: dots reads the
            # layout at 200 dpi, paddle reads the text at 300 dpi with its own prompt.
            png_dots = render_png(path, page_number, dpi_dots)
            indexed = dots_to_text(dots.text_from_png(png_dots))
            png_paddle = render_png(path, page_number, dpi_paddle)
            zweitmeinung = paddle.text_from_png(png_paddle, prompt=PADDLE_PROMPT)
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
