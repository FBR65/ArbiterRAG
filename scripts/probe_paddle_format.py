"""Probelauf: welches Ausgabeformat liefert paddleocr-vl bei welchem Prompt? (SPEC 11.14, Paket 1)

Run:
    set -a && . ./arbiter.env && set +a
    uv run python scripts/probe_paddle_format.py [--seite N] [pdf]

Rendert eine Seite und stellt dem Modell drei Aufgaben. Gibt je Prompt die Dauer, die
Länge und den Anfang der Antwort aus, dazu die erkannten OTSL-Token.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pypdfium2 as pdfium

from arbiter.config import load_config
from arbiter.ocr import VisionOcr, render_png
from arbiter.runtime import lade_umgebung

PROMPTS = {
    "OCR:": "reine Textlesung",
    "Table Recognition:": "Tabellen als Tokenfolge oder HTML",
    "Convert this page to Markdown. Keep every table as a Markdown table with header row.": (
        "Markdown mit Tabellen"
    ),
}

OTSL = re.compile(r"<(fcel|ecel|lcel|ucel|xcel|ched|rhed|srow|nl|b)>")


def zahlen_reichste_seite(pdf: pdfium.PdfDocument) -> int:
    beste, beste_seite = 0, 1
    for index in range(len(pdf)):
        text = pdf[index].get_textpage().get_text_range()
        anzahl = sum(1 for zeile in text.splitlines() if len(re.findall(r"\d[\d.,]*", zeile)) >= 3)
        if anzahl > beste:
            beste, beste_seite = anzahl, index + 1
    return beste_seite


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Ausgabeformat von paddleocr-vl")
    parser.add_argument(
        "pdf",
        type=Path,
        nargs="?",
        default=Path("/home/speedy/Downloads/ix.26.13.pdf"),
    )
    parser.add_argument("--seite", type=int, default=0, help="0 = zahlreichste Seite")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    umgebung = lade_umgebung()
    config = load_config(Path(umgebung["ARBITER_CONFIG"]), umgebung)
    pdf = pdfium.PdfDocument(str(args.pdf))
    try:
        seite = args.seite or zahlen_reichste_seite(pdf)
        gesamt = len(pdf)
    finally:
        pdf.close()

    print(f"{args.pdf.name}: {gesamt} Seiten, gewählt Seite {seite}, {args.dpi} dpi", flush=True)
    png = render_png(args.pdf, seite, args.dpi)
    print(f"Rendering: {len(png)} Bytes PNG", flush=True)
    paddle = VisionOcr(config.backends.base_url, config.backends.ocr_paddle_model)

    for prompt, zweck in PROMPTS.items():
        print("\n" + "=" * 78, flush=True)
        print(f"PROMPT: {prompt!r}  ({zweck})", flush=True)
        start = time.monotonic()
        try:
            antwort = paddle.text_from_png(png, prompt=prompt)
        except Exception as exc:
            print(f"  FEHLER nach {time.monotonic() - start:.1f}s: {exc}", flush=True)
            continue
        dauer = time.monotonic() - start
        token = sorted({treffer.group(1) for treffer in OTSL.finditer(antwort)})
        print(f"  Dauer {dauer:.1f}s, {len(antwort)} Zeichen", flush=True)
        print(f"  OTSL-Token: {token or 'keine'}", flush=True)
        print(f"  HTML-Tabellen: {antwort.count('<table')}", flush=True)
        print(f"  Markdown-Pipes: {antwort.count('|')}", flush=True)
        print("  --- Anfang ---", flush=True)
        for zeile in antwort.splitlines()[:25]:
            print("  " + zeile[:160], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
