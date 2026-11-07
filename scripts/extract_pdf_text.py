"""Volltext eines PDFs mit Seitenmarken in eine Textdatei schreiben."""

from __future__ import annotations

import sys
from pathlib import Path

from pypdf import PdfReader

pfad = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "ix.26.13.pdf"
ziel = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/ix-voltext.txt")

reader = PdfReader(str(pfad))
teile: list[str] = []
for nummer, seite in enumerate(reader.pages, start=1):
    text = (seite.extract_text() or "").strip()
    teile.append(f"\n<<<S.{nummer}>>>\n{text}")
ziel.write_text("".join(teile), encoding="utf-8")
print("geschrieben:", ziel, ziel.stat().st_size, "Bytes,", len(reader.pages), "Seiten")
