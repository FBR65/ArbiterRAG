"""Überblick über ein einzelnes PDF: Seiten, Anfangstexte, Stichworte."""

from __future__ import annotations

import sys
from pathlib import Path

from pypdf import PdfReader

pfad = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "ix.26.13.pdf"
reader = PdfReader(str(pfad))
print("Datei:", pfad.name, "| Seiten:", len(reader.pages))

for index in range(min(3, len(reader.pages))):
    text = (reader.pages[index].extract_text() or "").strip()
    print(f"--- S.{index + 1}\n{text[:600]}")
