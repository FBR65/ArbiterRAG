"""Steuerbescheid durch paddleocr-vl über llama-swap (SPEC 11.14, Paket 1).

Privat: Es werden nur Strukturmaße ausgegeben. Zellinhalte werden durch ihre Länge
ersetzt, damit weder Beträge noch Namen noch Nummern im Protokoll landen.

    uv run python scripts/probe_bescheid.py /home/speedy/Downloads/steuerbescheid1.png
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from arbiter.config import load_config
from arbiter.ocr import VisionOcr
from arbiter.runtime import lade_umgebung

PROMPTS = {
    "OCR:": "reine Textlesung",
    "Table Recognition:": "OTSL",
}

OTSL = re.compile(r"<(fcel|ecel|lcel|ucel|xcel|nl)>")
ZELLE = re.compile(r"<(fcel|lcel|ucel|xcel)>(.*?)(?=<(?:fcel|ecel|lcel|ucel|xcel|nl)>)", re.DOTALL)


def struktur(antwort: str) -> dict[str, object]:
    """Tokenfolge zählen, ohne Inhalte auszugeben."""
    tags = [treffer.group(1) for treffer in OTSL.finditer(antwort)]
    zeilen: list[int] = []
    zellen = 0
    for tag in tags:
        if tag == "nl":
            zeilen.append(zellen)
            zellen = 0
        else:
            zellen += 1
    if zellen:
        zeilen.append(zellen)
    laengen = [len(match.group(2).strip()) for match in ZELLE.finditer(antwort)]
    return {
        "tokens": len(tags),
        "zeilen": len(zeilen),
        "zellen_je_zeile": zeilen[:12],
        "leere_zellen": tags.count("ecel"),
        "verbunden_l": tags.count("lcel"),
        "verbunden_u": tags.count("ucel"),
        "verbunden_x": tags.count("xcel"),
        "zelllaengen": laengen[:12],
    }


def main(argv: list[str]) -> int:
    bild = Path(argv[0] if argv else "/home/speedy/Downloads/steuerbescheid1.png")
    if not bild.is_file():
        print(f"nicht gefunden: {bild}", file=sys.stderr)
        return 1

    umgebung = lade_umgebung()
    config = load_config(Path(umgebung["ARBITER_CONFIG"]), umgebung)
    paddle = VisionOcr(config.backends.base_url, config.backends.ocr_paddle_model)

    for prompt, zweck in PROMPTS.items():
        print("\n" + "=" * 78)
        print(f"PROMPT: {prompt!r} ({zweck})")
        start = time.monotonic()
        try:
            antwort = paddle.text_from_png(bild.read_bytes(), prompt=prompt)
        except Exception as exc:
            print(f"  FEHLER nach {time.monotonic() - start:.1f}s: {exc}")
            continue
        print(f"  Dauer {time.monotonic() - start:.1f}s, {len(antwort)} Zeichen")
        if prompt.startswith("Table"):
            for schluessel, wert in struktur(antwort).items():
                print(f"  {schluessel}: {wert}")
            print("  Tokenfolge (erste 60):")
            token = [treffer.group(1) for treffer in OTSL.finditer(antwort)][:60]
            print("  " + " ".join(f"<{tag}>" for tag in token))
        else:
            zeilen = antwort.splitlines()
            print(f"  Zeilen: {len(zeilen)}, leer: {sum(1 for z in zeilen if not z.strip())}")
            print("  Form (Inhalte geschwärzt):")
            for zeile in zeilen[:12]:
                form = "".join("·" if zeichen.isdigit() else "_" for zeichen in zeile[:60])
                print(f"    {len(zeile):>4} Zeichen | {form}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
