"""Holt die ONNX-Gewichte für die Layout-Erkennung (SPEC 11.16, W2).

Die Datei liegt außerhalb des Repositories. Ihre Herkunft ist Apache-2.0:
Gewichte ``juliozhao/DocLayout-YOLO-DocStructBench``, ONNX-Export
``wybxc/DocLayout-YOLO-DocStructBench-onnx``. Der Code des Original-Projekts ist
AGPL-3.0 und wird hier nicht verwendet.

    uv run python scripts/hole_layout_modell.py [ZIEL]
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://huggingface.co/wybxc/DocLayout-YOLO-DocStructBench-onnx/resolve/main/"
    "doclayout_yolo_docstructbench_imgsz1024.onnx"
)
STANDARD = Path.home() / ".cache/arbiterrag/models/doclayout_yolo_docstructbench_imgsz1024.onnx"


def sha256(pfad: Path) -> str:
    digest = hashlib.sha256()
    with pfad.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str]) -> int:
    ziel = Path(argv[0]).expanduser() if argv else STANDARD
    if ziel.is_file():
        print(f"vorhanden: {ziel}")
        print(f"Größe: {ziel.stat().st_size} Bytes")
        print(f"SHA-256: {sha256(ziel)}")
        return 0

    ziel.parent.mkdir(parents=True, exist_ok=True)
    zwischenstand = ziel.with_suffix(".teil")
    print(f"lade {URL}")
    try:
        with urllib.request.urlopen(URL, timeout=120) as antwort, zwischenstand.open("wb") as datei:
            while block := antwort.read(1 << 20):
                datei.write(block)
    except Exception as exc:
        zwischenstand.unlink(missing_ok=True)
        print(f"Download fehlgeschlagen: {exc}", file=sys.stderr)
        return 1
    zwischenstand.replace(ziel)
    print(f"geschrieben: {ziel}")
    print(f"Größe: {ziel.stat().st_size} Bytes")
    print(f"SHA-256: {sha256(ziel)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
