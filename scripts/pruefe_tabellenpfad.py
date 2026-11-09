"""Paket 5: der Tabellenpfad lief mit den echten Modellen (SPEC 11.15).

Der Lauf zeigt die Kette Prompt -> OTSL -> Tabelle -> Zeilen-Chunks. Ausgegeben
werden nur Zahlen und Längen, keine Zellinhalte: das Bild kann Beträge, Namen und
Steuernummern tragen.

    set -a && . ./arbiter.env && set +a
    uv run python scripts/pruefe_tabellenpfad.py /pfad/zu/seite.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from arbiter.config import load_config
from arbiter.ocr import VisionOcr
from arbiter.pipeline import ocr_lesung
from arbiter.runtime import lade_umgebung
from arbiter.tabellen import otsl_zu_tabelle, tabellen_chunks


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Tabellenpfad mit echten Modellen prüfen")
    parser.add_argument("bild", type=Path)
    parser.add_argument("--rag", default="")
    parser.add_argument("--prompt", default="Table Recognition:")
    args = parser.parse_args(argv)

    if not args.bild.is_file():
        print(f"nicht gefunden: {args.bild}", file=sys.stderr)
        return 1

    umgebung = lade_umgebung()
    config = load_config(Path(umgebung["ARBITER_CONFIG"]), umgebung)
    rag = config.rags[args.rag] if args.rag else next(iter(config.rags.values()))
    png = args.bild.read_bytes()

    backends = config.backends
    paddle = VisionOcr(backends.base_url, backends.ocr_paddle_model)
    dots = VisionOcr(backends.base_url, backends.ocr_dots_model)
    try:
        antwort = paddle.text_from_png(png, prompt=args.prompt)
        dots_text = dots.text_from_png(png)
    finally:
        paddle.close()
        dots.close()

    tabelle = otsl_zu_tabelle(antwort)
    if tabelle is None:
        print("kein OTSL in der Antwort — der Tabellenpfad greift nicht")
        return 2

    markdown, warnungen = ocr_lesung(dots_text, antwort)
    ergebnis = tabellen_chunks(
        markdown,
        max_chars=rag.chunk_chars,
        kopf_zeilen=rag.tabellen_kopf_zeilen,
        max_zeilen=rag.tabellen_max_zeilen,
    )
    kopf_chunks = [chunk for chunk in ergebnis.chunks if "Zeile:" not in chunk.text]
    laengen = [len(chunk.text) for chunk in ergebnis.chunks if "Zeile:" in chunk.text]

    print(f"OTSL-Zeichen: {len(antwort)}")
    print(f"Spalten: {tabelle.breite}, Zeilen: {len(tabelle.zeilen)}")
    print(f"indizierte Lesung: Markdown, {len(markdown)} Zeichen")
    print(f"Vergleich mit der dots-Lesung: {len(dots_text)} Zeichen")
    print(f"Tabellen: {ergebnis.tabellen}")
    print(f"Kopfzeilen im Kopf-Chunk: {rag.tabellen_kopf_zeilen}")
    print(f"Kopf-Chunks: {len(kopf_chunks)}, Zeilen-Chunks: {ergebnis.zeilen_chunks}")
    if laengen:
        print(f"Zeilen-Chunk-Längen: min {min(laengen)}, max {max(laengen)}")
        zu_lang = sum(1 for nummer in laengen if nummer > rag.chunk_chars)
        print(f"über chunk_chars ({rag.chunk_chars}): {zu_lang}")
    print(f"Warnungen: {len([*warnungen, *ergebnis.warnungen])}")
    for warnung in [*warnungen, *ergebnis.warnungen]:
        print(f"  Hinweis: {warnung}")
    return 0 if laengen else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
