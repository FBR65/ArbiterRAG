"""Paket 1 (SPEC 11.16): Welche Textkoordinaten liefert pypdfium2?

Prüft an einer Seite des Korpus, ob der Text in einem Kasten über
``get_text_bounded`` lesbar ist und ob seine Zeilen im Seitentext stehen.
Ausgabe: nur Zahlen und Längen, keine Inhalte.

    uv run python scripts/pruefe_zeilenkaesten.py DATEI SEITE X1 Y1 X2 Y2

Die Zahlen sind Seitenanteile mit Ursprung links oben (wie die Layout-Boxen).
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path

import pypdfium2 as pdfium

_LEER = re.compile(r"\s+")


def normal(zeile: str) -> str:
    return _LEER.sub(" ", zeile).strip().casefold()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Textkoordinaten einer PDF-Seite prüfen")
    parser.add_argument("datei", type=Path)
    parser.add_argument("seite", type=int)
    parser.add_argument("kasten", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    args = parser.parse_args(argv)

    pdf = pdfium.PdfDocument(str(args.datei))
    page = pdf[args.seite - 1]
    breite, hoehe = page.get_size()
    textpage = page.get_textpage()
    volltext = textpage.get_text_range()

    x1, y1, x2, y2 = args.kasten
    links = x1 * breite
    rechts = x2 * breite
    unten = (1 - y2) * hoehe
    oben = (1 - y1) * hoehe
    im_kasten = textpage.get_text_bounded(left=links, bottom=unten, right=rechts, top=oben)

    seiten_zeilen = [normal(zeile) for zeile in volltext.splitlines() if normal(zeile)]
    kasten_zeilen = [zeile.strip() for zeile in im_kasten.splitlines() if zeile.strip()]
    gefunden = sum(1 for zeile in kasten_zeilen if normal(zeile) in seiten_zeilen)

    rects = textpage.count_rects()
    chars = textpage.count_chars()
    y_baender = sorted({round(textpage.get_rect(index)[1]) for index in range(rects)})
    zeilen_aus_rects = 1 + sum(
        1 for links_, rechts_ in itertools.pairwise(y_baender) if rechts_ - links_ > 3
    )

    print(f"Seite {args.seite}: {breite:.0f} x {hoehe:.0f} pt")
    print(f"Zeichen: {chars}, Rechtecke: {rects}, Y-Bänder: {zeilen_aus_rects}")
    print(f"Kasten: {links:.0f},{unten:.0f} bis {rechts:.0f},{oben:.0f} pt")
    print(f"Text im Kasten: {len(im_kasten)} Zeichen, {len(kasten_zeilen)} Zeilen")
    print(f"davon im Seitentext gefunden: {gefunden}")
    print(f"Seitentext: {len(volltext)} Zeichen, {len(seiten_zeilen)} nichtleere Zeilen")
    for nummer, zeile in enumerate(kasten_zeilen[:5], start=1):
        print(f"  Zeile {nummer}: {len(zeile)} Zeichen, gefunden={normal(zeile) in seiten_zeilen}")
    pdf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
