"""Indexing and search orchestration (spec B, C, H).

Indexing, search, the gate and the answer step live here; each one is a method on
:class:`Pipeline` and every model call goes through llama-swap.
"""

from __future__ import annotations

import hashlib
import io
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from arbiter.answer import KEINE_ANTWORT, Quelle, build_answer_prompt
from arbiter.chat import ChatClient
from arbiter.check import build_claim_prompt, build_repair_prompt, ist_meta, split_claims
from arbiter.chunk import Chunk, ChunkParameter, QePaar, chunk_datei
from arbiter.chunk_structure import Kennzahlen
from arbiter.config import Config, Rag
from arbiter.decision import (
    GATE_GRAMMATIK,
    DecisionError,
    build_ja_nein_prompt,
    ja_wahrscheinlichkeit,
)
from arbiter.embedding import HttpEmbedder
from arbiter.fusion import rrf_many
from arbiter.layout import (
    LayoutDetektor,
    Region,
    eindeutige_tabellen,
    entferne_tabellenzeilen,
    tabellen_regionen,
)
from arbiter.loaders import Document, OcrHook, OcrPage, load_corpus, text_im_kasten
from arbiter.ocr import VisionOcr, dots_to_text, render_png, word_overlap
from arbiter.rerank import RerankClient
from arbiter.store import Hit, Record, Store
from arbiter.tabellen import otsl_zu_tabelle, tabelle_zu_markdown, tabellen_chunks

MAX_FRAGE_ZEICHEN = 2000
EMBED_BATCH = 32


@dataclass(frozen=True)
class Tabellenfund:
    """One table a layout detector found on a page (spec 11.16, W4/W5).

    ``zeilen`` are the lines the text layer holds inside the box; they are removed
    from the page text, so the table does not stand twice in the index (D44).
    """

    kasten: tuple[float, float, float, float]
    markdown: str
    zeilen: tuple[str, ...] = ()
    warnungen: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Tabellenstelle:
    """A table with the page it was found on and its position in the page text."""

    seite_index: int
    position: int
    fund: Tabellenfund


def ocr_lesung(dots_text: str, zweitmeinung: str) -> tuple[str, list[str]]:
    """The text of one OCR page: structured when a table was recognised (spec V1, V9).

    ``paddleocr-vl`` answers the table prompt with OTSL. If that yields a table, the
    Markdown of that table is what goes into the index — that is what produces row
    chunks. Without a table the dots reading stays, exactly as before.
    """
    tabelle = otsl_zu_tabelle(zweitmeinung)
    if tabelle is None:
        return dots_text, []
    return tabelle_zu_markdown(tabelle), list(tabelle.warnungen)


class QuestionTooLong(Exception):
    """Question exceeds the configured limit (spec I3)."""


def verkette_seiten(seiten: Sequence[str]) -> tuple[str, list[int]]:
    """Concatenate the pages of one file and keep the start position of each (spec O1)."""
    text = ""
    anfaenge: list[int] = []
    for seite in seiten:
        anfaenge.append(len(text))
        text += seite + "\n"
    return text.rstrip("\n"), anfaenge


def seite_bei(anfaenge: Sequence[int], position: int) -> int:
    """1-based page number of a character position (spec O2)."""
    return bisect_right(list(anfaenge), position)


def _nach_datei(dokumente: Sequence[Document]) -> dict[str, list[Document]]:
    """Documents grouped by file, pages in reading order."""
    nach_datei: dict[str, list[Document]] = {}
    for dokument in dokumente:
        nach_datei.setdefault(dokument.quelle, []).append(dokument)
    return {
        quelle: sorted(seiten, key=lambda teil: teil.seite or 0)
        for quelle, seiten in nach_datei.items()
    }


def _kopf(quelle: str, seite: int | None, pfad: tuple[str, ...]) -> str:
    """Chunk header: file, page and — for structure chunks — the heading path (B6, L5)."""
    kennung = f"[{Path(quelle).stem}, S. {seite or 0}"
    if pfad:
        kennung += " > " + " > ".join(pfad)
    return kennung + "] "


@dataclass(frozen=True)
class DateiBericht:
    """What the decider chose for one file (spec N4)."""

    quelle: str
    strategie: str
    uebersteuert: bool
    wert: float | None
    kennzahlen: Kennzahlen | None
    chunks: int
    begruendung: str = ""
    tabellen: int = 0
    zeilen_chunks: int = 0
    tabellen_seiten: int = 0


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
    tabellen: int = 0
    zeilen_chunks: int = 0
    tabellen_seiten: int = 0
    warnings: list[str] = field(default_factory=list)
    dateien: list[DateiBericht] = field(default_factory=list)


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
    rerank: str = "aus"  # "aus" | "aktiv" | "ausgefallen"
    warnungen: list[str] = field(default_factory=list)


def neuer_pipeline(config: Config, rag: Rag, embedder: HttpEmbedder | None = None) -> Pipeline:
    """A pipeline that can do everything, including question-evidence generation.

    Unit tests build ``Pipeline`` directly and get no generator, so indexing never
    calls a model there (N5). The three front ends use this constructor instead.
    """
    from arbiter.chunk_qe import mache_qe_quelle

    pipeline = Pipeline(config, rag, embedder=embedder)
    pipeline.qe_quelle = mache_qe_quelle(
        base_url=config.backends.base_url,
        model=rag.qe_model or config.backends.chat_model,
        max_fragen=rag.qe_fragen_max,
        fenster=rag.qe_fenster,
        fenster_max=rag.qe_fenster_max,
    )
    return pipeline


class Pipeline:
    """One RAG corpus, one store table, one embedder."""

    def __init__(
        self,
        config: Config,
        rag: Rag,
        embedder: HttpEmbedder | None = None,
        qe_quelle: Callable[[str], list[QePaar]] | None = None,
    ) -> None:
        self.config = config
        self.rag = rag
        self.embedder = embedder or HttpEmbedder(
            base_url=config.backends.base_url, model=config.backends.embed_model
        )
        self.store = Store(config.db_path)
        # Callable(question, description) -> hypothetical answer document.
        # Left empty here so tests can inject one and the real client stays lazy.
        # Callable(query, documents) -> one score per document.
        self.reranker: Callable[[str, list[str]], list[float]] | None = None
        # Callable(state, questions) -> answers map; and Callable(system, user) -> text.
        self.answerer: Callable[[str, str], str] | None = None
        self.gate_logprobs: Callable[[str, str], dict[str, float]] | None = None
        # Question-evidence generation for the QES strategy (spec M3). Left empty here so
        # indexing never calls a model unless someone asked for it; ``neuer_pipeline``
        # wires the real generator for CLI, TUI and API.
        self.qe_quelle = qe_quelle
        # Layout detection (spec 11.16). ``layout_quelle`` is injectable, like the gate:
        # tests replace it, the real pipeline builds it from the configuration.
        self.layout_quelle: Callable[[str, int], list[Tabellenfund]] | None = None
        self._layout_aktiv = False
        self._detektor: LayoutDetektor | None = None
        self._paddle: VisionOcr | None = None

    def index(self) -> IndexReport:
        # The check comes first: an unchanged file is not read again, so OCR and
        # embedding are not paid twice for it. Skipped files keep their rows.
        bekannt = self.store.fingerprints(self.rag.name)
        einstellungen = self._ingest_einstellungen()
        layout_quelle, layout_warnungen = self._layout_quelle()
        self._layout_aktiv = layout_quelle is not None

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
            warnings=[*loaded.warnings, *layout_warnungen],
        )
        if not loaded.documents:
            return report

        records: list[Record] = []
        texts: list[str] = []
        for quelle, seiten in _nach_datei(loaded.documents).items():
            texte, stellen, warnungen = self._layout_anwenden(quelle, seiten, layout_quelle)
            report.warnings.extend(warnungen)
            text, anfaenge = verkette_seiten(texte)
            ergebnis = chunk_datei(
                text,
                quelle,
                self._chunk_parameter(),
                qe_quelle=self._qe_quelle(),
                embedder=self.embedder,
            )
            report.warnings.extend(ergebnis.warnungen)
            layout_chunks, layout_tabellen, layout_zeilen, layout_seiten, layout_hinweise = (
                self._layout_chunks(stellen, anfaenge, quelle, seiten)
            )
            report.warnings.extend(layout_hinweise)
            chunks = sorted(
                [*ergebnis.chunks, *layout_chunks],
                key=lambda chunk: (chunk.start, chunk.ende),
            )
            datei = DateiBericht(
                quelle=quelle,
                strategie=ergebnis.strategie,
                uebersteuert=ergebnis.auswahl is None,
                wert=ergebnis.auswahl.wert if ergebnis.auswahl else None,
                kennzahlen=ergebnis.auswahl.kennzahlen if ergebnis.auswahl else None,
                chunks=len(chunks),
                begruendung=ergebnis.auswahl.begruendung if ergebnis.auswahl else "",
                tabellen=ergebnis.tabellen + layout_tabellen,
                zeilen_chunks=ergebnis.zeilen_chunks + layout_zeilen,
                tabellen_seiten=layout_seiten,
            )
            report.dateien.append(datei)
            report.tabellen += datei.tabellen
            report.zeilen_chunks += datei.zeilen_chunks
            report.tabellen_seiten += layout_seiten
            for number, chunk in enumerate(chunks, start=1):
                seite = seite_bei(anfaenge, chunk.start)
                seite_ende = seite_bei(anfaenge, max(chunk.ende - 1, chunk.start))
                quelle_seite = seiten[seite - 1]
                kopf = _kopf(quelle, quelle_seite.seite, chunk.pfad)
                inhalt = kopf + chunk.text
                eltern = (
                    " ".join(teil.text for teil in seiten[seite - 1 : seite_ende])
                    if self.rag.eltern_seite
                    else inhalt
                )
                chunk_id = f"{quelle}#{quelle_seite.seite or 0}#{number}"
                records.append(
                    Record(
                        id=chunk_id,
                        text=inhalt,
                        quelle=quelle,
                        seite=quelle_seite.seite,
                        seite_ende=seite_ende if quelle_seite.seite else None,
                        hash=hashlib.sha256(inhalt.encode("utf-8")).hexdigest(),
                        vector=[],
                        ocr=quelle_seite.ocr,
                        ocr_paddle=quelle_seite.ocr_paddle,
                        ocr_ueberlappung=quelle_seite.ocr_ueberlappung,
                        eltern=eltern,
                        strategie=ergebnis.strategie,
                    )
                )
                texts.append(inhalt)

        vectors = self._embed_all(texts)
        filled = [
            replace(record, vector=vector) for record, vector in zip(records, vectors, strict=True)
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

    def _layout_quelle(self) -> tuple[Callable[[str, int], list[Tabellenfund]] | None, list[str]]:
        """The table hook, or ``None`` with a warning (spec 11.16, W10, D47)."""
        if not self.rag.layout:
            return None, []
        if self.layout_quelle is not None:
            return self.layout_quelle, []
        if not self.rag.layout_modell:
            return None, ["layout: true ohne layout_modell, Layout ist aus"]
        modell = Path(self.rag.layout_modell).expanduser()
        if not modell.is_file():
            return None, [f"Layout-Modell fehlt: {modell}, Layout ist aus"]
        try:
            detektor = self._layout_detektor()
        except Exception as exc:
            return None, [f"Layout-Modell nicht ladbar ({exc}), Layout ist aus"]
        return self._layout_haken(detektor), []

    def _layout_detektor(self) -> LayoutDetektor:
        if self._detektor is None:
            self._detektor = LayoutDetektor(
                Path(self.rag.layout_modell).expanduser(),
                schwelle=self.rag.layout_schwelle,
                min_flaeche=self.rag.layout_min_flaeche,
            )
        return self._detektor

    def _paddle_client(self) -> VisionOcr:
        if self._paddle is None:
            self._paddle = VisionOcr(
                self.config.backends.base_url, self.config.backends.ocr_paddle_model
            )
        return self._paddle

    def _layout_haken(
        self, detektor: LayoutDetektor
    ) -> Callable[[str, int], list[Tabellenfund]]:
        """Per page: detect, crop, read the crop, keep the table text lines."""

        def haken(quelle: str, seite: int) -> list[Tabellenfund]:
            pfad = self.rag.corpus_path / quelle
            regionen = eindeutige_tabellen(
                tabellen_regionen(
                    detektor.erkenne(render_png(pfad, seite, self.rag.layout_dpi))
                )
            )
            if not regionen:
                return []
            png = render_png(pfad, seite, self.rag.layout_dpi_ocr)
            bild = Image.open(io.BytesIO(png)).convert("RGB")
            pdf = pdfium.PdfDocument(str(pfad))
            try:
                funde: list[Tabellenfund] = []
                for region in regionen:
                    fund = self._tabelle_aus_region(pdf, bild, seite, region)
                    if fund is not None:
                        funde.append(fund)
                return funde
            finally:
                pdf.close()

        return haken

    def _tabelle_aus_region(
        self,
        pdf: pdfium.PdfDocument,
        bild: Image.Image,
        seite: int,
        region: Region,
    ) -> Tabellenfund | None:
        """Crop, read with the table prompt, and keep the text lines (W4, W5, W6)."""
        x1, y1, x2, y2 = region.kasten
        rand = self.rag.layout_rand
        kasten = (
            max(0.0, x1 - rand),
            max(0.0, y1 - rand),
            min(1.0, x2 + rand),
            min(1.0, y2 + rand),
        )
        ausschnitt = bild.crop(
            (
                int(kasten[0] * bild.width),
                int(kasten[1] * bild.height),
                int(kasten[2] * bild.width),
                int(kasten[3] * bild.height),
            )
        )
        puffer = io.BytesIO()
        ausschnitt.save(puffer, format="PNG")
        antwort = self._paddle_client().text_from_png(
            puffer.getvalue(), prompt=self.rag.ocr_paddle_prompt
        )
        tabelle = otsl_zu_tabelle(antwort)
        if tabelle is None:
            return Tabellenfund(
                kasten=region.kasten,
                markdown="",
                warnungen=("Tabellenausschnitt nicht als OTSL gelesen",),
            )
        if tabelle.breite < 2 or not tabelle.zeilen:
            return Tabellenfund(
                kasten=region.kasten,
                markdown="",
                warnungen=(
                    "Tabellenausschnitt ohne nutzbare Zeilen "
                    f"({tabelle.breite} Spalten, {len(tabelle.zeilen)} Zeilen)",
                ),
            )
        zeilen = tuple(
            zeile.strip()
            for zeile in text_im_kasten(pdf, seite, region.kasten).splitlines()
            if zeile.strip()
        )
        return Tabellenfund(
            kasten=region.kasten,
            markdown=tabelle_zu_markdown(tabelle),
            zeilen=zeilen,
            warnungen=tuple(tabelle.warnungen),
        )

    def _layout_anwenden(
        self,
        quelle: str,
        seiten: Sequence[Document],
        layout_quelle: Callable[[str, int], list[Tabellenfund]] | None,
    ) -> tuple[list[str], list[_Tabellenstelle], list[str]]:
        """Page texts after table removal, the tables found, and warnings (W6)."""
        texte = [seite.text for seite in seiten]
        stellen: list[_Tabellenstelle] = []
        warnungen: list[str] = []
        if layout_quelle is None:
            return texte, stellen, warnungen
        for index, seite in enumerate(seiten):
            if seite.seite is None:
                continue
            try:
                befunde = layout_quelle(quelle, seite.seite)
            except Exception as exc:
                warnungen.append(
                    f"{quelle}, Seite {seite.seite}: Layout-Erkennung fehlgeschlagen ({exc})"
                )
                continue
            text = texte[index]
            for befund in sorted(befunde, key=lambda fund: fund.kasten[1]):
                warnungen.extend(
                    f"{quelle}, Seite {seite.seite}: {warnung}" for warnung in befund.warnungen
                )
                ergebnis = entferne_tabellenzeilen(text, befund.zeilen)
                if befund.zeilen and ergebnis.entfernt == 0:
                    warnungen.append(
                        f"{quelle}, Seite {seite.seite}: Tabellentext nicht aus dem "
                        "Seitentext entfernt"
                    )
                elif ergebnis.fehlend:
                    warnungen.append(
                        f"{quelle}, Seite {seite.seite}: {ergebnis.fehlend} Tabellenzeile(n) "
                        "nicht im Seitentext gefunden"
                    )
                text = ergebnis.text
                stellen.append(_Tabellenstelle(index, max(ergebnis.position, 0), befund))
            texte[index] = text
        return texte, stellen, warnungen

    def _layout_chunks(
        self,
        stellen: Sequence[_Tabellenstelle],
        anfaenge: Sequence[int],
        quelle: str,
        seiten: Sequence[Document],
    ) -> tuple[list[Chunk], int, int, int, list[str]]:
        """Table chunks with the position of their table in the file (W5, W7)."""
        chunks: list[Chunk] = []
        hinweise: list[str] = []
        tabellen = 0
        zeilen_chunks = 0
        betroffene_seiten: set[int] = set()
        parameter = self._chunk_parameter()
        for stelle in stellen:
            seite = seiten[stelle.seite_index].seite or 0
            ergebnis = tabellen_chunks(
                stelle.fund.markdown,
                max_chars=parameter.max_chars,
                kopf_zeilen=parameter.tabellen_kopf_zeilen,
                max_zeilen=parameter.tabellen_max_zeilen,
            )
            hinweise.extend(
                f"{quelle}, Seite {seite}: {warnung}" for warnung in ergebnis.warnungen
            )
            start = anfaenge[stelle.seite_index] + stelle.position
            for chunk in ergebnis.chunks:
                chunks.append(replace(chunk, start=start, ende=start + 1))
            tabellen += ergebnis.tabellen
            zeilen_chunks += ergebnis.zeilen_chunks
            if ergebnis.tabellen:
                betroffene_seiten.add(stelle.seite_index)
        return chunks, tabellen, zeilen_chunks, len(betroffene_seiten), hinweise

    def search(self, question: str) -> SearchResult:
        if len(question) > MAX_FRAGE_ZEICHEN:
            raise QuestionTooLong(
                f"Frage zu lang: {len(question)} Zeichen, erlaubt sind {MAX_FRAGE_ZEICHEN}"
            )

        warnungen: list[str] = []

        dense_ids, fts_ids, hits = self._rank_lists(question)
        rankings: list[list[str]] = [dense_ids, fts_ids]

        fused = rrf_many(rankings, k=60, top_k=self.rag.fuse_top_k)

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

    def ask(self, question: str, vorgabe: SearchResult | None = None) -> Antwort:
        """Search, decide, answer — in that order (spec F3, G).

        ``vorgabe`` reuses an already computed search result, so a caller that has
        measured the search does not pay for it twice.
        """
        result = vorgabe if vorgabe is not None else self.search(question)
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

        if KEINE_ANTWORT.casefold() in text.casefold():
            return self._ohne_antwort(
                "Die Auszüge beantworten die Frage nicht", quellen, entscheidung
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
                return self._ohne_antwort("keine belegte Passage", quellen, entscheidung, zusatz)

        return Antwort(
            text=text,
            grund=None,
            quellen=quellen,
            entscheidung=entscheidung,
            fehler=False,
            warnungen=[*entscheidung.warnungen, *zusatz],
        )

    def _ohne_antwort(
        self,
        grund: str,
        quellen: list[Quelle],
        entscheidung: Entscheidung,
        zusatz: list[str] | None = None,
    ) -> Antwort:
        """A refusal carries no sources.

        Listing the checked pages next to "no answer" reads like an answer with its
        sources. The pages are named in the warnings instead, as what they are: checked
        and found insufficient.
        """
        geprueft = ", ".join(f"{quelle.quelle}, S. {quelle.seite}" for quelle in quellen[:5])
        warnungen = list(entscheidung.warnungen)
        if zusatz:
            warnungen.extend(zusatz)
        if geprueft:
            warnungen.append(f"geprüft, nicht ausreichend: {geprueft}")
        return Antwort(
            text=None,
            grund=grund,
            quellen=[],
            entscheidung=entscheidung,
            fehler=False,
            warnungen=warnungen,
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
        for beleg in quellen:
            if beleg.seite is None:
                continue
            volltext = self.store.page_text(self.rag.name, beleg.quelle, beleg.seite)
            if volltext and volltext not in alle:
                alle.append(volltext)
        behalten: list[str] = []
        verworfen: list[str] = []
        for satz, quelle, seite in split_claims(text):
            if ist_meta(satz):
                verworfen.append(satz)
                continue
            if quelle is None or seite is None:
                kandidaten = alle
            else:
                kandidaten = [
                    auszug
                    for (datei, nummer), auszug in nach_quelle.items()
                    if quelle in datei and nummer == seite
                ]
                ganze_seite = self.store.page_text(self.rag.name, quelle, seite)
                if ganze_seite:
                    kandidaten.append(ganze_seite)
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

    def _chunk_parameter(self) -> ChunkParameter:
        """The chunking settings of this corpus (spec 11.4)."""
        rag = self.rag
        return ChunkParameter(
            strategie=rag.chunk_strategie,
            max_chars=rag.chunk_chars,
            overlap=rag.chunk_overlap,
            ebene=rag.struktur_ebene,
            xref_max=rag.xref_max,
            xref_chars=rag.xref_chars,
            qes_budget_faktor=rag.qes_budget_faktor,
            qe_fragen_max=rag.qe_fragen_max,
            qe_hits=rag.qe_hits,
            qes_lambda=rag.qes_lambda,
            tabellen_kopf_zeilen=rag.tabellen_kopf_zeilen,
            tabellen_max_zeilen=rag.tabellen_max_zeilen,
        )

    def _qe_quelle(self) -> Callable[[str], list[QePaar]] | None:
        """Question-evidence generation for the QES strategy (spec M3)."""
        return self.qe_quelle

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
                rag.chunk_strategie,
                rag.struktur_ebene,
                rag.xref_max,
                rag.xref_chars,
                rag.qes_budget_faktor,
                rag.qe_fragen_max,
                rag.qe_hits,
                rag.qes_lambda,
                rag.qe_model,
                rag.ocr,
                rag.ocr_alle_seiten,
                rag.ocr_dpi,
                rag.ocr_dpi_paddle,
                rag.ocr_paddle_prompt,
                rag.tabellen_kopf_zeilen,
                rag.tabellen_max_zeilen,
                rag.layout,
                rag.layout_modell,
                rag.layout_schwelle,
                rag.layout_min_flaeche,
                rag.layout_dpi,
                rag.layout_dpi_ocr,
                rag.layout_rand,
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
            # layout at 200 dpi, paddle reads the table structure at 300 dpi.
            png_dots = render_png(path, page_number, dpi_dots)
            dots_text = dots_to_text(dots.text_from_png(png_dots))
            if self._layout_aktiv:
                # The whole page prompt read the page columns as a table. With layout
                # detection only the detected tables go to paddle (spec W8).
                return OcrPage(text=dots_text, vergleich="", ueberlappung=0.0)
            png_paddle = render_png(path, page_number, dpi_paddle)
            zweitmeinung = paddle.text_from_png(png_paddle, prompt=self.rag.ocr_paddle_prompt)
            # The structured reading is what goes into the index; the dots reading stays
            # beside it as the comparison (spec V1).
            indexed, warnungen = ocr_lesung(dots_text, zweitmeinung)
            return OcrPage(
                text=indexed,
                vergleich=dots_text,
                ueberlappung=word_overlap(indexed, dots_text),
                warnungen=tuple(warnungen),
            )

        return hook

    def _embed_all(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            vectors.extend(self.embedder.embed(list(texts[start : start + EMBED_BATCH])))
        return vectors
