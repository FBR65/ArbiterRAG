"""Prüfen, ob pypdfium2 nach Entfernen des Umbruchzeichens saubere Wörter liefert."""

from __future__ import annotations

import sys
from pathlib import Path

import pypdfium2 as pdfium

pfad = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "Downloads" / "ix.26.13.pdf")
seite = int(sys.argv[2]) if len(sys.argv) > 2 else 74

pdf = pdfium.PdfDocument(str(pfad))
roh = pdf[seite - 1].get_textpage().get_text_range()
pdf.close()
sauber = roh.replace("\ufffe", "").replace("\u00ad", "")
print(" ".join(sauber.split())[:430])
print()
for wort in ("Button", "Barrierefreiheit", "gefunden", "Eingabefeld", "Button", "ge"):
    print(f"  {wort!r}: {sauber.count(wort)}")
print(f"Zeichen roh={len(roh)} sauber={len(sauber)}")
