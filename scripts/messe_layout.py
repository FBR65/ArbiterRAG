"""Messung der Layout-Erkennung auf dem Korpus (SPEC 11.16, Paket 5).

Zensus: wie viele Seiten haben Tabellenregionen, wie viele Regionen je Seite.
Recall: wahlweise zwei Indizes (Layout aus/ein) und Recall@10 auf dem Goldset.
Ausgegeben werden nur Zahlen, Kästen und Laufzeiten, keine Inhalte.

    set -a && . ./arbiter.env && set +a
    uv run python scripts/messe_layout.py --seiten 20
    uv run python scripts/messe_layout.py --recall
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import pypdfium2 as pdfium

from arbiter.config import Config, Rag, load_config
from arbiter.eval import load_goldset, recall_at
from arbiter.layout import LayoutDetektor, eindeutige_tabellen, tabellen_regionen
from arbiter.ocr import render_png
from arbiter.pipeline import Pipeline
from arbiter.runtime import lade_umgebung
from arbiter.store import Store

STANDARD_MODELL = (
    Path.home() / ".cache/arbiterrag/models/doclayout_yolo_docstructbench_imgsz1024.onnx"
)


def _seitenzahl(pfad: Path) -> int:
    pdf = pdfium.PdfDocument(str(pfad))
    try:
        return len(pdf)
    finally:
        pdf.close()


def zensus(rag: Rag, detektor: LayoutDetektor, grenze: int, bekannt: set[str]) -> int:
    """Zählt Seiten und Tabellenregionen je Datei (keine Inhalte).

    ``bekannt`` sind die Dateien, die im Index stehen. Fremdsprachige PDFs im
    Verzeichnis gehören nicht zum Korpus und werden nicht gezählt.
    """
    gesamt_seiten = gesamt_tabellen = 0
    print(f"{'Datei':<50} {'Seiten':>7} {'mit Tabelle':>12} {'Tabellen':>9} {'ms/Seite':>9}")
    for pfad in sorted(rag.corpus_path.rglob("*")):
        if not pfad.is_file() or pfad.suffix.lower() != ".pdf":
            continue
        relativ = pfad.relative_to(rag.corpus_path).as_posix()
        if bekannt and relativ not in bekannt:
            continue
        seiten = _seitenzahl(pfad)
        if grenze:
            seiten = min(seiten, grenze)
        mit = tabellen = 0
        laufzeit = 0.0
        for nummer in range(1, seiten + 1):
            start = time.perf_counter()
            regionen = eindeutige_tabellen(
                tabellen_regionen(detektor.erkenne(render_png(pfad, nummer, rag.layout_dpi)))
            )
            laufzeit += time.perf_counter() - start
            if regionen:
                mit += 1
                tabellen += len(regionen)
                for region in regionen:
                    x1, y1, x2, y2 = region.kasten
                    print(
                        f"    S.{nummer}: conf {region.konfidenz:.2f} "
                        f"x {x1:.2f}-{x2:.2f} y {y1:.2f}-{y2:.2f}"
                    )
        relativ = pfad.relative_to(rag.corpus_path)
        pro_seite = laufzeit * 1000 / max(seiten, 1)
        print(f"{relativ.as_posix():<50.50} {seiten:>7} {mit:>12} {tabellen:>9} {pro_seite:>9.0f}")
        gesamt_seiten += seiten
        gesamt_tabellen += tabellen
    print(f"\nSeiten geprüft: {gesamt_seiten}, Tabellenregionen: {gesamt_tabellen}")
    return gesamt_tabellen


def recall(config: Config, rag: Rag, goldset: list, db: Path, modell: Path) -> None:
    """Recall@10 mit Layout aus und an, je eigener Store."""
    print(f"\n{'Layout':<8} {'Abschnitte':>10} {'Index':>9} {'Recall@10':>10}")
    for an in (False, True):
        ziel = db / ("an" if an else "aus")
        if ziel.exists():
            shutil.rmtree(ziel)
        pipeline = Pipeline(
            replace(config, db_path=ziel),
            replace(rag, layout=an, layout_modell=str(modell), rerank=False),
        )
        report = pipeline.index()
        zeichen = sum(
            len(str(row["text"])) for row in pipeline.store._table(rag.name).to_arrow().to_pylist()
        )
        treffer = 0.0
        for gold in goldset:
            result = pipeline.search(gold.frage)  # type: ignore[attr-defined]
            treffer += recall_at(result.passages, gold, 10)  # type: ignore[arg-type]
        print(
            f"{'an' if an else 'aus':<8} {report.chunks:>10} {zeichen:>9} "
            f"{treffer / len(goldset):>10.3f}"
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Layout-Erkennung messen")
    parser.add_argument("--rag", default="", help="Name des Korpus")
    parser.add_argument("--modell", default="", help="Pfad zur ONNX-Datei")
    parser.add_argument("--seiten", type=int, default=0, help="je Datei höchstens N Seiten")
    parser.add_argument("--recall", action="store_true", help="Recall@10 mit Layout aus/an")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("/home/speedy/.hermes/cache/scratch/arbiter-layout"),
        help="Zielverzeichnis der Mess-Stores",
    )
    parser.add_argument("--goldset", type=Path, default=Path("goldset/downloads.yaml"))
    args = parser.parse_args(argv)

    umgebung = lade_umgebung()
    config = load_config(Path(umgebung["ARBITER_CONFIG"]), umgebung)
    rag = config.rags[args.rag] if args.rag else next(iter(config.rags.values()))
    if args.modell:
        rag = replace(rag, layout_modell=args.modell)
    modell = Path(rag.layout_modell).expanduser() if rag.layout_modell else STANDARD_MODELL
    if not modell.is_file():
        print(f"Layout-Modell fehlt: {modell}", file=sys.stderr)
        print("Zuerst: uv run python scripts/hole_layout_modell.py", file=sys.stderr)
        return 1

    detektor = LayoutDetektor(
        modell, schwelle=rag.layout_schwelle, min_flaeche=rag.layout_min_flaeche
    )
    bekannt = {
        str(row["quelle"])
        for row in Store(config.db_path)._table(rag.name).to_arrow().to_pylist()
    }
    if zensus(rag, detektor, args.seiten, bekannt) == 0:
        print("Keine Tabellenregion im Korpus gefunden.")
    if args.recall:
        goldset = [gold for gold in load_goldset(args.goldset) if gold.erwartet]
        if not goldset:
            print(f"Goldset ohne Belegseiten: {args.goldset}", file=sys.stderr)
            return 1
        recall(config, rag, goldset, args.db, modell)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
