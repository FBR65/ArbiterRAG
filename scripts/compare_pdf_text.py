"""Vergleich der Textgewinnung: pypdf gegen pypdfium2 auf einer Seite."""

from __future__ import annotations

import sys
from pathlib import Path

import pypdfium2 as pdfium
from pypdf import PdfReader

pfad = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "Downloads" / "ix.26.13.pdf")
seite = int(sys.argv[2]) if len(sys.argv) > 2 else 74

pypdf_text = (PdfReader(str(pfad)).pages[seite - 1].extract_text() or "").strip()

pdf = pdfium.PdfDocument(str(pfad))
page = pdf[seite - 1]
pdfium_text = page.get_textpage().get_text_range()
pdf.close()

print(f"=== pypdf, S.{seite}\n{' '.join(pypdf_text.split())[:400]}\n")
print(f"=== pypdfium2, S.{seite}\n{' '.join(pdfium_text.split())[:400]}\n")

for name, text in (("pypdf", pypdf_text), ("pypdfium2", pdfium_text)):
    zerhackt = text.count(" -\n") + text.count("-\n")
    print(f"{name:<10} Zeichen={len(text):<6} Trennstriche am Zeilenende={zerhackt}")
