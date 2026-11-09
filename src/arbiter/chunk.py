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
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class Ergebnis:
    chunks: list[Chunk]
    strategie: str
    auswahl: Auswahl | None = None
    warnungen: list[str] = field(default_factory=list)


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
    if strategie == "struktur":
        chunks = _struktur_chunks(text, parameter)
    elif strategie == "qes":
        qes_chunks = _qes_chunks(text, dateiname, parameter, qe_quelle, embedder, warnungen)
        if qes_chunks is None:
            strategie = "fenster"
            chunks = chunk_fenster(text, parameter.max_chars, parameter.overlap)
        else:
            chunks = qes_chunks
    else:
        chunks = chunk_fenster(text, parameter.max_chars, parameter.overlap)

    if not chunks:
        # Markers the splitter drops must not make a document disappear from the
        # index; the window splitter keeps it searchable.
        chunks = chunk_fenster(text, parameter.max_chars, parameter.overlap)
    return Ergebnis(chunks=chunks, strategie=strategie, auswahl=auswahl, warnungen=warnungen)


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
