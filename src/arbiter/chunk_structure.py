"""Structure-aware chunking (spec L, StrucChunk style).

A document is read as a sequence of headings and the text under them. Every
heading starts an own chunk, so a subsection is its own retrievable unit and no
text stands twice in the index. Each chunk carries the path of its headings as a
breadcrumb, and an explicit cross-reference pulls the referenced provision's text
into the chunk (spec L5, L6).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from arbiter.chunk import Chunk, satz_grenzen

_MARKDOWN = re.compile(r"^(#{1,6})\s+(\S.*)$")
_NUMMER = re.compile(r"^(\d+(?:\.\d+)*)\s+(\S.*)$")
_PARAGRAPH = re.compile(r"^§\s*(\d+[a-z]?)\s*(.*)$")
_WORTKOPF = re.compile(r"^(Kapitel|Abschnitt|Anhang|Teil)\s+([0-9IVXLC]+)\b\s*(.*)$")
# "nach Abschnitt 1.2", "siehe Abschnitt 1.3", "gemäß § 5", "vgl. 2.1", "Section 4.2"
_XREF = re.compile(
    r"(?:Abschnitt|Ziffer|Kapitel|Section)\s*(\d+(?:\.\d+)+)"
    r"|\b(?:siehe|nach|gemäß|gemäss|vgl\.?)\s+(\d+(?:\.\d+)+)",
    re.IGNORECASE,
)
_SATZENDE = (".", "!", "?")


@dataclass(frozen=True)
class Kopfzeile:
    """One recognised heading line."""

    ebene: int
    nummer: str
    titel: str


@dataclass(frozen=True)
class Abschnitt:
    """One heading with the text up to the next heading of any level."""

    start: int
    ende: int
    ebene: int
    nummer: str
    pfad: tuple[str, ...]


@dataclass(frozen=True)
class Kennzahlen:
    """Structure measures of one document (spec N2)."""

    kopf_anteil: float
    koepfe_pro_1000: float
    xrefs_pro_1000: float


def kopfzeile(zeile: str) -> Kopfzeile | None:
    """The heading a line represents, or ``None`` (spec L1)."""
    roh = zeile.strip()
    if not roh or len(roh) > 120:
        return None
    if roh.endswith((".", ",", ";", ":")):
        # A heading never ends like a sentence; this keeps "2026 war ein gutes Jahr."
        # and footnote lines out of the heading set.
        return None

    markdown = _MARKDOWN.match(roh)
    if markdown is not None:
        titel = markdown.group(2).strip()
        return Kopfzeile(ebene=len(markdown.group(1)), nummer=_nummer_in(titel), titel=titel)

    nummeriert = _NUMMER.match(roh)
    if nummeriert is not None:
        nummer = nummeriert.group(1)
        return Kopfzeile(ebene=nummer.count(".") + 1, nummer=nummer, titel=roh)

    paragraph = _PARAGRAPH.match(roh)
    if paragraph is not None:
        return Kopfzeile(ebene=2, nummer=paragraph.group(1), titel=roh)

    wortkopf = _WORTKOPF.match(roh)
    if wortkopf is not None:
        return Kopfzeile(ebene=2, nummer=wortkopf.group(2), titel=roh)

    if _ist_grosszeile(roh):
        return Kopfzeile(ebene=2, nummer="", titel=roh)
    return None


def _nummer_in(text: str) -> str:
    treffer = re.match(r"(\d+(?:\.\d+)*)\s+\S", text)
    return treffer.group(1) if treffer is not None else ""


def _ist_grosszeile(roh: str) -> bool:
    if not 3 <= len(roh) <= 80:
        return False
    if any(zeichen in roh for zeichen in ".!?:"):
        return False
    if len(roh.split()) < 2:
        return False
    buchstaben = [zeichen for zeichen in roh if zeichen.isalpha()]
    return bool(buchstaben) and all(zeichen.isupper() for zeichen in buchstaben)


def abschnitte(text: str) -> list[Abschnitt]:
    """All sections of ``text`` in reading order, without gaps (spec L2)."""
    koepfe: list[tuple[int, Kopfzeile]] = []
    offset = 0
    for zeile in text.splitlines(keepends=True):
        kopf = kopfzeile(zeile.rstrip("\r\n"))
        if kopf is not None:
            koepfe.append((offset, kopf))
        offset += len(zeile)

    teile: list[Abschnitt] = []
    if koepfe and koepfe[0][0] > 0 and text[: koepfe[0][0]].strip():
        teile.append(Abschnitt(0, koepfe[0][0], 0, "", ()))

    stapel: list[tuple[int, str]] = []
    for index, (start, kopf) in enumerate(koepfe):
        while stapel and stapel[-1][0] >= kopf.ebene:
            stapel.pop()
        pfad = (*tuple(titel for _, titel in stapel), kopf.titel)
        stapel.append((kopf.ebene, kopf.titel))
        ende = koepfe[index + 1][0] if index + 1 < len(koepfe) else len(text)
        teile.append(Abschnitt(start, ende, kopf.ebene, kopf.nummer, pfad))
    return teile


def kennzahlen(text: str, ebene: int, max_chars: int = 600) -> Kennzahlen:
    """Structure measures that decide between the strategies (spec N2, N3).

    Only sections that fit a chunk count as structure: a single huge chapter heading
    does not make a document provision-shaped. Measured 2026-11-08 on a blog PDF: one
    heading over most of the text scored 0.60 coverage and sent a flowing text into the
    structure strategy.
    """
    teile = abschnitte(text)
    zeichen = sum(1 for zeichen in text if not zeichen.isspace())
    if zeichen == 0:
        return Kennzahlen(0.0, 0.0, 0.0)
    kopf_zeichen = sum(
        sum(1 for zeichen in text[teil.start : teil.ende] if not zeichen.isspace())
        for teil in teile
        if teil.ebene >= ebene and teil.ende - teil.start <= max_chars * 1.5
    )
    koepfe = sum(1 for teil in teile if teil.ebene > 0)
    xrefs = sum(1 for _ in _XREF.finditer(text))
    laenge = max(len(text), 1)
    return Kennzahlen(
        kopf_anteil=kopf_zeichen / zeichen,
        koepfe_pro_1000=koepfe * 1000 / laenge,
        xrefs_pro_1000=xrefs * 1000 / laenge,
    )


def median_abschnitt(text: str) -> float:
    """Median length of the sections that have a heading (spec N, revision 11.12)."""
    laengen = sorted(teil.ende - teil.start for teil in abschnitte(text) if teil.ebene > 0)
    if not laengen:
        return 0.0
    mitte = len(laengen) // 2
    if len(laengen) % 2:
        return float(laengen[mitte])
    return (laengen[mitte - 1] + laengen[mitte]) / 2


def chunk_struktur(
    text: str,
    *,
    ebene: int,
    max_chars: int,
    overlap: int,
    xref_max: int,
    xref_chars: int,
) -> list[Chunk]:
    """One chunk per section, with breadcrumb and resolved references (spec L3-L7)."""
    if max_chars <= 0:
        raise ValueError("max_chars muss positiv sein")
    teile = abschnitte(text)
    nach_nummer = {teil.nummer: teil for teil in teile if teil.nummer}
    chunks: list[Chunk] = []
    for teil in teile:
        if not text[teil.start : teil.ende].strip():
            continue
        for start, ende in _packe(text, teil.start, teil.ende, max_chars, overlap):
            anhang = _verweise(text, text[start:ende], teil, nach_nummer, xref_max, xref_chars)
            chunks.append(
                Chunk(
                    text=text[start:ende] + anhang,
                    start=start,
                    ende=ende,
                    strategie="struktur",
                    pfad=teil.pfad,
                )
            )
    return chunks


def _packe(text: str, start: int, ende: int, max_chars: int, overlap: int) -> list[tuple[int, int]]:
    """Section spans: whole up to 1.5 x max_chars, else packed at sentence ends."""
    if ende - start <= max_chars * 1.5:
        return [(start, ende)]

    saetze = satz_grenzen(text[start:ende])
    saetze = [(start + links, start + rechts) for links, rechts in saetze]
    stuecke: list[tuple[int, int]] = []
    index = 0
    while index < len(saetze):
        block_start = saetze[index][0]
        block_ende = block_start
        naechster = index
        while naechster < len(saetze):
            satz_ende = saetze[naechster][1]
            if satz_ende - block_start > max_chars:
                break
            block_ende = satz_ende
            naechster += 1
        if block_ende == block_start:
            # One sentence longer than the window: cut it hard, twice at most.
            block_ende = min(block_start + max_chars, ende)
            naechster = index + 1
        stueck_start, stueck_ende = _trimmen(text, block_start, block_ende)
        if stueck_start < stueck_ende:
            stuecke.append((stueck_start, stueck_ende))
        index = _naechster_start(saetze, index, naechster, block_ende, overlap)
    return stuecke


def _naechster_start(
    saetze: list[tuple[int, int]],
    index: int,
    naechster: int,
    block_ende: int,
    overlap: int,
) -> int:
    """First sentence of the next window: the overlap is carried as whole sentences."""
    if overlap <= 0 or naechster >= len(saetze):
        return naechster
    kandidat = naechster
    # At least one new sentence per window, otherwise the loop never ends.
    while kandidat - 1 > max(0, index) and block_ende - saetze[kandidat - 1][0] <= overlap:
        kandidat -= 1
    return kandidat


def _trimmen(text: str, start: int, ende: int) -> tuple[int, int]:
    while start < ende and text[start].isspace():
        start += 1
    while ende > start and text[ende - 1].isspace():
        ende -= 1
    return start, ende


def _verweise(
    text: str,
    rumpf: str,
    teil: Abschnitt,
    nach_nummer: dict[str, Abschnitt],
    xref_max: int,
    xref_chars: int,
) -> str:
    """Resolved cross-references as an appendix (spec L6, L7)."""
    if xref_max <= 0 or xref_chars <= 0:
        return ""
    gesehen: set[str] = set()
    anhaenge: list[str] = []
    for treffer in _XREF.finditer(rumpf):
        nummer = treffer.group(1) or treffer.group(2)
        if not nummer or nummer == teil.nummer or nummer in gesehen:
            continue
        ziel = nach_nummer.get(nummer)
        if ziel is None:
            continue
        gesehen.add(nummer)
        anhaenge.append(f"↳ {nummer} {ziel.pfad[-1]}: {_rumpf(text, ziel)[:xref_chars].strip()}")
        if len(anhaenge) >= xref_max:
            break
    return "\n" + "\n".join(anhaenge) if anhaenge else ""


def _rumpf(text: str, ziel: Abschnitt) -> str:
    """The section's body without its heading line."""
    eigener = text[ziel.start : ziel.ende]
    grenze = eigener.find("\n")
    return "" if grenze == -1 else eigener[grenze + 1 :]
