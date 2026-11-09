"""Text chunking (spec B6, K, M8, N5).

``chunk_text`` is the sentence-blind window splitter of spec B6. ``chunk_datei``
is the per-file switch of stage 7: it asks the decider which strategy fits the
document, or follows an override, and returns chunks that carry their character
span in the document text.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from arbiter.chunk_choice import Auswahl

# A sentence ends at . ! or ? followed by whitespace. "1.2" and "1.2-metre" stay
# whole, because the dot is not followed by whitespace.
_SATZENDE = re.compile(r"(?<=[.!?])\s+")

STRATEGIEN = ("auto", "struktur", "qes", "fenster")


@dataclass(frozen=True)
class Chunk:
    """A chunk before the file header is prepended (spec K1, K2).

    ``start`` and ``ende`` are character positions of the chunk's own text in the
    document text; a resolved cross-reference is appended after ``text``.
    """

    text: str
    start: int
    ende: int
    strategie: str
    pfad: tuple[str, ...] = ()


@dataclass(frozen=True)
class QePaar:
    """One generated question with the evidence sentences it needs (spec M2)."""

    frage: str
    belege: tuple[str, ...]


QeQuelle = Callable[[str], list[QePaar]]


class EmbedderLike(Protocol):
    """The only part of the embedder the chunkers use."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class ChunkParameter:
    strategie: str = "auto"
    max_chars: int = 600
    overlap: int = 100
    ebene: int = 2
    xref_max: int = 3
    xref_chars: int = 200
    qes_budget_faktor: float = 3.0
    qe_fragen_max: int = 60
    qe_hits: int = 5
    qes_lambda: float = 0.0001
    tabellen_kopf_zeilen: int = 3
    tabellen_max_zeilen: int = 200


@dataclass(frozen=True)
class Ergebnis:
    chunks: list[Chunk]
    strategie: str
    auswahl: Auswahl | None = None
    warnungen: list[str] = field(default_factory=list)
    tabellen: int = 0
    zeilen_chunks: int = 0


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 100) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars`` characters."""
    return [chunk.text for chunk in chunk_fenster(text, max_chars, overlap)]


def satz_grenzen(text: str) -> list[tuple[int, int]]:
    """Character spans of the sentences of ``text`` (spec M1)."""
    grenzen = [0]
    grenzen.extend(match.start() for match in _SATZENDE.finditer(text))
    grenzen.append(len(text))
    saetze: list[tuple[int, int]] = []
    for links, rechts in itertools.pairwise(grenzen):
        while links < rechts and text[links].isspace():
            links += 1
        while rechts > links and text[rechts - 1].isspace():
            rechts -= 1
        if links < rechts:
            saetze.append((links, rechts))
    return saetze


def _trimmen(text: str, start: int, ende: int) -> tuple[int, int]:
    while start < ende and text[start].isspace():
        start += 1
    while ende > start and text[ende - 1].isspace():
        ende -= 1
    return start, ende


def chunk_fenster(
    text: str, max_chars: int = 1200, overlap: int = 100, strategie: str = "fenster"
) -> list[Chunk]:
    """Spec B6 as chunks with positions: paragraph boundaries first, then a window."""
    if max_chars <= 0:
        raise ValueError("max_chars muss positiv sein")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap muss kleiner als max_chars sein")

    if not text.strip():
        return []

    min_snap = max_chars // 4
    chunks: list[Chunk] = []
    start = 0
    length = len(text)

    while start < length:
        ende = min(start + max_chars, length)
        if ende < length:
            snap = text.rfind("\n\n", start, ende)
            if snap != -1 and snap - start >= min_snap:
                ende = snap
        links, rechts = _trimmen(text, start, ende)
        if links < rechts:
            chunks.append(
                Chunk(
                    text=text[links:rechts],
                    start=links,
                    ende=rechts,
                    strategie=strategie,
                )
            )
        if ende >= length:
            break
        start = max(ende - overlap, start + 1)

    return chunks


def chunk_datei(
    text: str,
    dateiname: str,
    parameter: ChunkParameter,
    *,
    qe_quelle: QeQuelle | None = None,
    embedder: EmbedderLike | None = None,
) -> Ergebnis:
    """Chunk one whole document (spec K, N5)."""
    if parameter.strategie not in STRATEGIEN:
        raise ValueError(
            f"Unbekannte Strategie '{parameter.strategie}', erlaubt: {', '.join(STRATEGIEN)}"
        )

    auswahl: Auswahl | None = None
    strategie = parameter.strategie
    if strategie == "auto":
        # Imported here: chunk_choice reads the structure measures, and
        # chunk_structure reads Chunk from this module.
        from arbiter.chunk_choice import waehle

        gewaehlt = waehle(text, ebene=parameter.ebene, max_chars=parameter.max_chars)
        auswahl = gewaehlt
        strategie = gewaehlt.strategie

    if not text.strip():
        return Ergebnis(chunks=[], strategie=strategie, auswahl=auswahl)

    warnungen: list[str] = []
    # Tables are taken out before any prose chunker runs: no window and no section may
    # cut into a table, and a row must not stand twice in the index (spec V6).
    # Imported here: arbiter.tabellen imports Chunk from this module.
    from arbiter.tabellen import TabellenErgebnis, maskiere_tabellen, tabellen_chunks

    tabellen = TabellenErgebnis()
    if strategie in ("struktur", "fenster"):
        tabellen = tabellen_chunks(
            text,
            max_chars=parameter.max_chars,
            kopf_zeilen=parameter.tabellen_kopf_zeilen,
            max_zeilen=parameter.tabellen_max_zeilen,
        )
        warnungen.extend(f"{dateiname}: {warnung}" for warnung in tabellen.warnungen)
    quelltext = maskiere_tabellen(text) if tabellen.tabellen else text

    if strategie == "struktur":
        chunks = _struktur_chunks(quelltext, parameter)
    elif strategie == "qes":
        qes_chunks = _qes_chunks(quelltext, dateiname, parameter, qe_quelle, embedder, warnungen)
        if qes_chunks is None:
            strategie = "fenster"
            chunks = chunk_fenster(quelltext, parameter.max_chars, parameter.overlap)
        else:
            chunks = qes_chunks
    else:
        chunks = chunk_fenster(quelltext, parameter.max_chars, parameter.overlap)

    if not chunks:
        # Markers the splitter drops must not make a document disappear from the
        # index; the window splitter keeps it searchable.
        chunks = chunk_fenster(quelltext, parameter.max_chars, parameter.overlap)
    if tabellen.tabellen:
        chunks = [replace(chunk, text=_entferne_masken(chunk.text)) for chunk in chunks]
    chunks = sorted([*chunks, *tabellen.chunks], key=lambda chunk: (chunk.start, chunk.ende))
    return Ergebnis(
        chunks=chunks,
        strategie=strategie,
        auswahl=auswahl,
        warnungen=warnungen,
        tabellen=tabellen.tabellen,
        zeilen_chunks=tabellen.zeilen_chunks,
    )


def _entferne_masken(text: str) -> str:
    """Collapse the whitespace a masked table leaves behind (spec V6).

    The table text itself lives in its row chunks. A paragraph chunk that ran over the
    table must not carry thousands of spaces into the index and into the embedding.
    """
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _struktur_chunks(text: str, parameter: ChunkParameter) -> list[Chunk]:
    from arbiter.chunk_structure import chunk_struktur

    return chunk_struktur(
        text,
        ebene=parameter.ebene,
        max_chars=parameter.max_chars,
        overlap=parameter.overlap,
        xref_max=parameter.xref_max,
        xref_chars=parameter.xref_chars,
    )


def _qes_chunks(
    text: str,
    dateiname: str,
    parameter: ChunkParameter,
    qe_quelle: QeQuelle | None,
    embedder: EmbedderLike | None,
    warnungen: list[str],
) -> list[Chunk] | None:
    """QES chunks, or ``None`` when the supervision is missing (spec M8)."""
    paare = qe_quelle(text) if qe_quelle is not None else []
    if not paare:
        warnungen.append(f"{dateiname}: QE-Erzeugung ohne Ergebnis, Rückfall auf Fenster")
        return None
    from arbiter.chunk_qes import chunke_qes

    ergebnis: list[Chunk] = chunke_qes(text, paare, parameter=parameter, embedder=embedder)
    return ergebnis
