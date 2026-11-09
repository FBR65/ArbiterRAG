"""Tables as first-class data (spec V, SPEC 11.14).

PaddleOCR-VL answers a table prompt with OTSL (Open Table Structure Language):
``<fcel>`` carries a cell, ``<ecel>`` an empty one, ``<lcel>`` joins the cell to the left,
``<ucel>`` to the one above, ``<xcel>`` to both, ``<nl>`` ends a row.

The rules here are deliberately conservative: a joined cell repeats the value it joins
to — that is what makes a single row readable on its own — and where nothing can be
joined the cell stays empty. No value is invented, and an unknown token never vanishes
silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from arbiter.chunk import Chunk

_TOKEN = re.compile(r"<(fcel|ecel|lcel|ucel|xcel|nl|b|srow)>")
_ZELLE = ("fcel", "lcel", "ucel", "xcel")
_BEKANNT = {"fcel", "ecel", "lcel", "ucel", "xcel", "nl", "b", "srow"}
_UNBEKANNT = re.compile(r"<([A-Za-z_]+)>")
_MARKDOWN_TRENNER = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


@dataclass(frozen=True)
class Tabelle:
    """A table as the chunker needs it: header and rows of equal width."""

    kopf: tuple[str, ...]
    zeilen: list[tuple[str, ...]] = field(default_factory=list)
    warnungen: list[str] = field(default_factory=list)

    @property
    def breite(self) -> int:
        return len(self.kopf) or (len(self.zeilen[0]) if self.zeilen else 0)


def otsl_zu_tabelle(text: str) -> Tabelle | None:
    """Build a table from OTSL; ``None`` when the text carries no table marks."""
    if not _TOKEN.search(text):
        return None
    warnungen = [
        f"unbekanntes Tabellen-Token <{treffer.group(1)}> als Zellgrenze gelesen"
        for treffer in _UNBEKANNT.finditer(text)
        if treffer.group(1) not in _BEKANNT
    ]
    roh: list[list[tuple[str, str]]] = []
    zeile: list[tuple[str, str]] = []
    letzte = 0
    for treffer in _TOKEN.finditer(text):
        tag = treffer.group(1)
        if tag in ("nl", "srow"):
            if zeile:
                roh.append(zeile)
                zeile = []
            letzte = treffer.end()
            continue
        if tag == "b":
            continue
        if tag == "ecel" and letzte == treffer.start():
            # <ecel> steht ohne Inhalt direkt hinter dem Tag: Zelle bleibt leer.
            zeile.append(("ecel", ""))
            letzte = treffer.end()
            continue
        naechste = _TOKEN.search(text, treffer.end())
        ende = naechste.start() if naechste is not None else len(text)
        inhalt = text[treffer.end() : ende].strip()
        zeile.append((tag, inhalt))
        letzte = treffer.end()
    if zeile:
        roh.append(zeile)
    if not roh:
        return None

    breite = max(len(zeile) for zeile in roh)
    gitter: list[list[str]] = [["" for _ in range(breite)] for _ in roh]
    for zeilen_nummer, zellen in enumerate(roh):
        for spalte, (tag, inhalt) in enumerate(zellen):
            if tag in ("fcel", "ecel"):
                gitter[zeilen_nummer][spalte] = inhalt
            elif tag == "lcel":
                gitter[zeilen_nummer][spalte] = _links(gitter, zeilen_nummer, spalte)
            elif tag == "ucel":
                gitter[zeilen_nummer][spalte] = _oben(gitter, zeilen_nummer, spalte)
            else:  # xcel
                wert = _oben(gitter, zeilen_nummer, spalte) or _links(gitter, zeilen_nummer, spalte)
                gitter[zeilen_nummer][spalte] = wert

    kopf_index = 0 if len(gitter) > 1 else -1
    kopf = tuple(gitter[0]) if kopf_index == 0 else ()
    zeilen = (
        [tuple(zeile) for zeile in gitter[1:]]
        if kopf_index == 0
        else [tuple(zeile) for zeile in gitter]
    )
    return Tabelle(kopf=kopf, zeilen=zeilen, warnungen=warnungen)


def _links(gitter: list[list[str]], zeile: int, spalte: int) -> str:
    for kandidat in range(spalte - 1, -1, -1):
        if gitter[zeile][kandidat]:
            return gitter[zeile][kandidat]
    return ""


def _oben(gitter: list[list[str]], zeile: int, spalte: int) -> str:
    for kandidat in range(zeile - 1, -1, -1):
        if gitter[kandidat][spalte]:
            return gitter[kandidat][spalte]
    return ""


def tabelle_zu_markdown(tabelle: Tabelle) -> str:
    """The table as Markdown — the representation the indexer and the chunker share."""
    breite = tabelle.breite
    kopf = tabelle.kopf or tuple(f"Spalte {nummer + 1}" for nummer in range(breite))
    zeilen = [_zeile_md(kopf), _zeile_md(tuple("---" for _ in kopf))]
    zeilen.extend(_zeile_md(_auffuellen(zeile, breite)) for zeile in tabelle.zeilen)
    return "\n".join(zeilen)


def _zeile_md(zellen: tuple[str, ...]) -> str:
    return "| " + " | ".join(zelle.replace("|", "/") for zelle in zellen) + " |"


def _auffuellen(zeile: tuple[str, ...], breite: int) -> tuple[str, ...]:
    return zeile + tuple("" for _ in range(breite - len(zeile)))


def markdown_zu_tabellen(text: str) -> list[Tabelle]:
    """Every Markdown table of a text, in reading order."""
    tabellen: list[Tabelle] = []
    block: list[str] = []
    for zeile in text.splitlines():
        if zeile.strip().startswith("|"):
            block.append(zeile)
        elif block:
            tabellen.append(_block_zu_tabelle(block))
            block = []
    if block:
        tabellen.append(_block_zu_tabelle(block))
    return [tabelle for tabelle in tabellen if tabelle.zeilen or tabelle.kopf]


def _block_zu_tabelle(block: list[str]) -> Tabelle:
    zellen = [_zellen(zeile) for zeile in block]
    if len(zellen) >= 2 and _MARKDOWN_TRENNER.match(
        block[1].split("|")[1] if "|" in block[1] else ""
    ):
        kopf = zellen[0]
        zeilen = zellen[2:]
        warnungen: list[str] = []
    else:
        # Ohne Trenner ist die erste Zeile keine erkennbare Kopfzeile. Sie als Kopf
        # auszugeben waere geraten; der Chunk sagt stattdessen "Kopf: unbekannt" (V4).
        kopf = ()
        zeilen = zellen
        warnungen = ["Tabelle ohne Kopfzeilentrenner, Kopf: unbekannt"]
    return Tabelle(kopf=kopf, zeilen=zeilen, warnungen=warnungen)


def _zellen(zeile: str) -> tuple[str, ...]:
    roh = zeile.strip()
    if roh.startswith("|"):
        roh = roh[1:]
    if roh.endswith("|"):
        roh = roh[:-1]
    return tuple(teil.strip() for teil in roh.split("|"))


def tabelle_zu_chunks(
    tabelle: Tabelle,
    *,
    pfad: tuple[str, ...],
    max_chars: int,
    kopf_zeilen: int,
) -> list[Chunk]:
    """One chunk for the table head, one per row — each row carries the header (spec V3)."""
    breite = tabelle.breite
    if breite == 0:
        return []
    # Ohne erkennbare Kopfzeile wird das gesagt, nicht erfunden (spec V4).
    kopfzeile = "Kopf: " + (", ".join(tabelle.kopf) if tabelle.kopf else "unbekannt")
    praefix = f"[{' > '.join(pfad)}] " if pfad else ""
    chunks: list[Chunk] = []

    if kopf_zeilen > 0 and tabelle.zeilen:
        kopf_teil = tabelle.zeilen[:kopf_zeilen]
        text = praefix + kopfzeile
        for zeile in kopf_teil:
            text += "\n" + " | ".join(_auffuellen(zeile, breite))
        chunks.append(Chunk(text=text, start=0, ende=0, strategie="tabelle", pfad=pfad))

    for zeile in tabelle.zeilen[kopf_zeilen:]:
        werte = _auffuellen(zeile, breite)
        if not any(werte):
            continue
        for teil in _zeile_teilen(werte, kopfzeile, praefix, max_chars):
            chunks.append(Chunk(text=teil, start=0, ende=0, strategie="tabelle", pfad=pfad))
    return chunks


def _zeile_teilen(
    werte: tuple[str, ...], kopfzeile: str, praefix: str, max_chars: int
) -> list[str]:
    """Split a wide row at cell borders, repeating the header on every part (spec V5)."""
    teile: list[str] = []
    aktuell = ""
    for wert in werte:
        kandidat = f"{aktuell} | {wert}" if aktuell else wert
        if aktuell and len(praefix + kopfzeile + " Zeile: " + kandidat) > max_chars:
            teile.append(praefix + kopfzeile + " Zeile: " + aktuell)
            aktuell = wert
        else:
            aktuell = kandidat
    if aktuell:
        teile.append(praefix + kopfzeile + " Zeile: " + aktuell)
    return teile


@dataclass(frozen=True)
class TabellenBlock:
    """One Markdown table with the span it occupies in the document text."""

    start: int
    ende: int
    tabelle: Tabelle


@dataclass(frozen=True)
class TabellenErgebnis:
    """The table chunks of a text plus what the report needs to say about them."""

    chunks: list[Chunk] = field(default_factory=list)
    warnungen: list[str] = field(default_factory=list)
    tabellen: int = 0
    zeilen_chunks: int = 0


def tabellen_bloecke(text: str) -> list[TabellenBlock]:
    """Every Markdown table block with its character span (spec V3, V6).

    The span is what makes a table chunk belong to its page. Without it every row
    chunk would start at position 0 and be filed under page 1.
    """
    bloecke: list[TabellenBlock] = []
    start = -1
    zeilen: list[str] = []
    offset = 0
    for roh in text.splitlines(keepends=True):
        inhalt = roh.rstrip("\r\n")
        if inhalt.strip().startswith("|"):
            if start < 0:
                start = offset
            zeilen.append(inhalt)
        elif start >= 0:
            bloecke.append(TabellenBlock(start, offset, _block_zu_tabelle(zeilen)))
            start = -1
            zeilen = []
        offset += len(roh)
    if start >= 0:
        bloecke.append(TabellenBlock(start, offset, _block_zu_tabelle(zeilen)))
    return [block for block in bloecke if block.tabelle.zeilen or block.tabelle.kopf]


def tabellen_chunks(
    text: str,
    *,
    max_chars: int,
    kopf_zeilen: int,
    max_zeilen: int = 200,
    pfad: tuple[str, ...] = (),
) -> TabellenErgebnis:
    """One head chunk and one chunk per row, for every table in ``text`` (spec V3)."""
    chunks: list[Chunk] = []
    warnungen: list[str] = []
    tabellen = 0
    zeilen_chunks = 0
    for block in tabellen_bloecke(text):
        tabellen += 1
        tabelle = block.tabelle
        warnungen.extend(tabelle.warnungen)
        if max_zeilen > 0 and len(tabelle.zeilen) > max_zeilen:
            warnungen.append(
                f"Tabelle mit {len(tabelle.zeilen)} Zeilen über der Obergrenze "
                f"{max_zeilen}, Rest nicht indiziert"
            )
            tabelle = replace(tabelle, zeilen=tabelle.zeilen[:max_zeilen])
        for chunk in tabelle_zu_chunks(
            tabelle, pfad=pfad, max_chars=max_chars, kopf_zeilen=kopf_zeilen
        ):
            chunks.append(replace(chunk, start=block.start, ende=block.ende))
            if "Zeile:" in chunk.text:
                zeilen_chunks += 1
    return TabellenErgebnis(
        chunks=chunks, warnungen=warnungen, tabellen=tabellen, zeilen_chunks=zeilen_chunks
    )



def maskiere_tabellen(text: str) -> str:
    """Replace every table block with whitespace of the same length (spec V6).

    The chunker then never cuts into a table, and no text stands twice in the index.
    Length and line breaks stay, so every character position keeps its meaning.
    """
    zeichen = list(text)
    for block in tabellen_bloecke(text):
        for index in range(block.start, block.ende):
            if zeichen[index] != "\n":
                zeichen[index] = " "
    return "".join(zeichen)
