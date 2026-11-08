"""Probe: QES-Chunking an einem echten PDF mit echtem Modell und echtem Embedder.

Run:  uv run python scripts/probe_qes_echt.py <pdf> [--fragen 12]

Zeigt erzeugte Frage-Beleg-Paare, die Fenster und die Indexgroesse. Kein Schreiben in
den Store: der Lauf kostet nur Modellzeit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pypdfium2 as pdfium

from arbiter.chat import ChatClient
from arbiter.chunk import ChunkParameter
from arbiter.chunk_qe import erzeuge_qe
from arbiter.chunk_qes import chunke_qes
from arbiter.config import load_config
from arbiter.embedding import HttpEmbedder
from arbiter.runtime import lade_umgebung


def seiten_text(path: Path) -> str:
    pdf = pdfium.PdfDocument(str(path))
    try:
        return "\n".join(
            pdf[index].get_textpage().get_text_range().replace("\ufffe", "").replace("\u00ad", "")
            for index in range(len(pdf))
        )
    finally:
        pdf.close()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="QES an einem echten PDF")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--fragen", type=int, default=12)
    args = parser.parse_args(argv)

    umgebung = lade_umgebung()
    config = load_config(Path(umgebung.get("ARBITER_CONFIG", "arbiter.yaml")), umgebung)
    rag = next(iter(config.rags.values()))

    text = seiten_text(args.pdf)
    chat = ChatClient(config.backends.base_url, rag.qe_model or config.backends.chat_model)
    embedder = HttpEmbedder(
        base_url=config.backends.base_url, model=config.backends.embed_model
    )

    paare = erzeuge_qe(
        text,
        chat=chat.complete,
        max_fragen=args.fragen,
        fenster=rag.qe_fenster,
        fenster_max=rag.qe_fenster_max,
    )
    print(f"{args.pdf.name}: {len(text)} Zeichen, {len(paare)} Frage-Beleg-Paare")
    for paar in paare[:3]:
        print(f"  FRAGE: {paar.frage}")
        for beleg in paar.belege:
            print(f"    BELEG: {beleg[:90]}")

    chunks = chunke_qes(
        text,
        paare,
        parameter=ChunkParameter(
            strategie="qes", max_chars=rag.chunk_chars, qe_hits=rag.qe_hits
        ),
        embedder=embedder,
    )
    if not chunks:
        print("  keine QES-Abschnitte (Rueckfall auf Fenster waere faellig)")
        return 1
    laengen = [chunk.ende - chunk.start for chunk in chunks]
    abdeckung = sum(laengen)
    print(
        f"  {len(chunks)} Abschnitte, Laenge min {min(laengen)} / median "
        f"{sorted(laengen)[len(laengen) // 2]} / max {max(laengen)}, "
        f"Index {abdeckung} Zeichen = {abdeckung / len(text):.2f} x Dokument"
    )
    for chunk in chunks[:2]:
        print(f"\n  [{chunk.start}:{chunk.ende}] " + " ".join(chunk.text.split())[:240])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
