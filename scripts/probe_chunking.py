"""Probe: what does the decider say about one PDF, and what do the chunks look like?

Run:  uv run python scripts/probe_chunking.py <pdf> [--strategie struktur|qes|fenster]

Reads the text layer with pypdfium2, the same way the loader does. No index write,
no embedding, no OCR — this is a look at the chunking alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pypdfium2 as pdfium

from arbiter.chunk import ChunkParameter, chunk_datei
from arbiter.chunk_choice import waehle


def seiten_text(path: Path) -> str:
    pdf = pdfium.PdfDocument(str(path))
    try:
        seiten = [
            pdf[index].get_textpage().get_text_range().replace("\ufffe", "").replace("\u00ad", "")
            for index in range(len(pdf))
        ]
    finally:
        pdf.close()
    return "\n".join(seiten)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Chunking an einem PDF ansehen")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--strategie", default="auto")
    parser.add_argument("--chunk-chars", type=int, default=600)
    parser.add_argument("--zeigen", type=int, default=3, help="so viele Abschnitte ausgeben")
    args = parser.parse_args(argv)

    text = seiten_text(args.pdf)
    if not text.strip():
        print(f"{args.pdf.name}: keine Textebene")
        return 1

    auswahl = waehle(text, ebene=2, schwelle=0.60)
    print(f"{args.pdf.name}: {len(text)} Zeichen")
    print(
        f"  Entscheider: {auswahl.strategie} (Wert {auswahl.wert:.2f}, "
        f"Kopfanteil {auswahl.kennzahlen.kopf_anteil:.2f}, "
        f"Köpfe {auswahl.kennzahlen.koepfe_pro_1000:.1f}/1000, "
        f"Verweise {auswahl.kennzahlen.xrefs_pro_1000:.1f}/1000)"
    )

    parameter = ChunkParameter(strategie=args.strategie, max_chars=args.chunk_chars)
    ergebnis = chunk_datei(text, args.pdf.name, parameter)
    laengen = [len(chunk.text) for chunk in ergebnis.chunks]
    print(
        f"  Strategie {ergebnis.strategie}: {len(ergebnis.chunks)} Abschnitte, "
        f"Länge min {min(laengen) if laengen else 0} / "
        f"median {sorted(laengen)[len(laengen) // 2] if laengen else 0} / "
        f"max {max(laengen) if laengen else 0}"
    )
    for warnung in ergebnis.warnungen:
        print(f"  Hinweis: {warnung}")
    for chunk in ergebnis.chunks[: args.zeigen]:
        kopf = " > ".join(chunk.pfad)
        print(f"\n  [{chunk.start}:{chunk.ende}] {kopf}")
        print("  " + " ".join(chunk.text.split())[:300])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
