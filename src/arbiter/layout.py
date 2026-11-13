"""Layout detection with DocLayout-YOLO over ONNX (SPEC 11.16).

The model answers with boxes: x1, y1, x2, y2, confidence, class. This module turns
those into page fractions, keeps the classes it knows, and — for the text layer —
removes the table lines that a box covers. Nothing here talks to a network; the
ONNX session reads one file.

The code is deliberately small: the published DocLayout-YOLO repository is AGPL-3.0,
so no line of it is copied. Only the Apache-2.0 weights are used, through this
decoder.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Class order of DocLayout-YOLO-DocStructBench (metadata of the ONNX file).
KLASSEN: dict[int, str] = {
    0: "title",
    1: "plain text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}
TABELLE = "table"
_LEER = re.compile(r"\s+")


@dataclass(frozen=True)
class Region:
    """One detected region: class, confidence and box as page fractions."""

    klasse: str
    konfidenz: float
    kasten: tuple[float, float, float, float]


@dataclass(frozen=True)
class Letterbox:
    """What the resize did, so the boxes can be mapped back to the page."""

    groesse: int
    gain: float
    rand_x: int
    rand_y: int


@dataclass(frozen=True)
class Textbefund:
    """The page text after removal, and what removal did."""

    text: str
    entfernt: int
    fehlend: int
    position: int


def letterbox(bild: Image.Image, groesse: int) -> tuple[np.ndarray, Letterbox]:
    """Resize with unchanged aspect ratio, pad with gray, return BGR NCHW in [0, 1]."""
    breite, hoehe = bild.size
    gain = min(groesse / hoehe, groesse / breite)
    neu_w, neu_h = round(breite * gain), round(hoehe * gain)
    flaeche = Image.new("RGB", (groesse, groesse), (114, 114, 114))
    rand_x, rand_y = (groesse - neu_w) // 2, (groesse - neu_h) // 2
    flaeche.paste(bild.resize((neu_w, neu_h), Image.Resampling.BILINEAR), (rand_x, rand_y))
    array = np.asarray(flaeche, dtype=np.float32)[:, :, ::-1].transpose(2, 0, 1)[None]
    return array / 255.0, Letterbox(groesse, gain, rand_x, rand_y)


def dekodiere(
    vorhersage: np.ndarray,
    seite: tuple[int, int],
    box: Letterbox,
    schwelle: float,
    min_flaeche: float,
) -> list[Region]:
    """Boxes of the model as page fractions, highest confidence first.

    The model returns one row per box: ``x1, y1, x2, y2, confidence, class``.
    Rows below the confidence threshold, boxes smaller than ``min_flaeche`` and
    unknown classes are dropped. Boxes are clamped to the page.
    """
    breite, hoehe = seite
    if vorhersage.size == 0:
        return []
    regionen: list[Region] = []
    for x1, y1, x2, y2, konfidenz, klasse in vorhersage.reshape(-1, 6):
        if float(konfidenz) < schwelle:
            continue
        name = KLASSEN.get(int(klasse))
        if name is None:
            continue
        links = min(max((float(x1) - box.rand_x) / box.gain / breite, 0.0), 1.0)
        oben = min(max((float(y1) - box.rand_y) / box.gain / hoehe, 0.0), 1.0)
        rechts = min(max((float(x2) - box.rand_x) / box.gain / breite, 0.0), 1.0)
        unten = min(max((float(y2) - box.rand_y) / box.gain / hoehe, 0.0), 1.0)
        if rechts <= links or unten <= oben:
            continue
        if (rechts - links) * (unten - oben) < min_flaeche:
            continue
        regionen.append(
            Region(klasse=name, konfidenz=float(konfidenz), kasten=(links, oben, rechts, unten))
        )
    return sorted(regionen, key=lambda region: region.konfidenz, reverse=True)


def tabellen_regionen(regionen: Sequence[Region]) -> list[Region]:
    """Only the regions a table reader can use (spec W3, decision D43)."""
    return [region for region in regionen if region.klasse == TABELLE]


def eindeutige_tabellen(regionen: Sequence[Region], schwelle: float = 0.5) -> list[Region]:
    """Drop table boxes that overlap a better one (spec W3).

    The detector sometimes finds the same table twice, for example the table and its
    heading. Two chunks from one table would stand twice in the index, which is what
    D44 forbids. Overlap is measured against the smaller box, so a box inside another
    counts as the same table.
    """
    behalten: list[Region] = []
    for region in sorted(regionen, key=lambda kandidat: kandidat.konfidenz, reverse=True):
        if all(_ueberlappung(region.kasten, andere.kasten) < schwelle for andere in behalten):
            behalten.append(region)
    return behalten


def _ueberlappung(
    links: tuple[float, float, float, float], rechts: tuple[float, float, float, float]
) -> float:
    breite = min(links[2], rechts[2]) - max(links[0], rechts[0])
    hoehe = min(links[3], rechts[3]) - max(links[1], rechts[1])
    if breite <= 0 or hoehe <= 0:
        return 0.0
    schnitt = breite * hoehe
    kleiner = min(
        (links[2] - links[0]) * (links[3] - links[1]),
        (rechts[2] - rechts[0]) * (rechts[3] - rechts[1]),
    )
    return schnitt / kleiner if kleiner > 0 else 0.0


def entferne_tabellenzeilen(text: str, zeilen: Sequence[str]) -> Textbefund:
    """Remove the given lines from ``text`` (spec W6, decision D44).

    A line is removed only when it stands in the text word for word (whitespace and
    case ignored). Anything not found stays; ``fehlend`` counts it, so the caller can
    warn instead of losing text or silently keeping a duplicate.
    """
    gesucht = Counter(_normal(zeile) for zeile in zeilen if _normal(zeile))
    behalten: list[str] = []
    laenge = 0
    position = -1
    entfernt = 0
    for zeile in text.splitlines(keepends=True):
        schluessel = _normal(zeile)
        if schluessel and gesucht.get(schluessel, 0) > 0:
            if position < 0:
                position = laenge
            gesucht[schluessel] -= 1
            entfernt += 1
            continue
        behalten.append(zeile)
        laenge += len(zeile)
    return Textbefund(
        text="".join(behalten),
        entfernt=entfernt,
        fehlend=sum(gesucht.values()),
        position=position,
    )


def _normal(zeile: str) -> str:
    return _LEER.sub(" ", zeile).strip().casefold()


class LayoutDetektor:
    """One ONNX session; the model file stays outside the repository (spec W2)."""

    def __init__(
        self,
        modell: Path,
        *,
        schwelle: float = 0.25,
        min_flaeche: float = 0.01,
        groesse: int = 1024,
    ) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(modell), providers=["CPUExecutionProvider"])
        self._eingang = self._session.get_inputs()[0].name
        self._schwelle = schwelle
        self._min_flaeche = min_flaeche
        self._groesse = groesse

    def erkenne(self, png: bytes) -> list[Region]:
        """All regions of one rendered page, highest confidence first."""
        bild = Image.open(io.BytesIO(png)).convert("RGB")
        stapel, box = letterbox(bild, self._groesse)
        vorhersage = np.asarray(
            self._session.run(None, {self._eingang: stapel})[0], dtype=np.float32
        )
        return dekodiere(vorhersage, bild.size, box, self._schwelle, self._min_flaeche)
