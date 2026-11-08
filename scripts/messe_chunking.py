"""Messung Stufe 7.3: Recall@10 je Chunk-Strategie auf dem echten Korpus.

Run:
    set -a && . ./arbiter.env && set +a
    uv run python scripts/messe_chunking.py --db /tmp/arbiter-messung

Jede Strategie bekommt einen eigenen Store unter ``--db``. Der laufende Index aus
``ARBITER_DB`` bleibt unangetastet. Gemessen wird die Fusion ohne Reranker, weil die
Frage hier dem Chunking gilt und nicht dem Reranker.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from arbiter.config import load_config
from arbiter.eval import load_goldset, recall_at
from arbiter.pipeline import neuer_pipeline
from arbiter.runtime import lade_umgebung

STRATEGIEN = ("fenster", "struktur", "qes", "auto")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Recall@10 je Chunk-Strategie")
    parser.add_argument("--db", type=Path, default=Path("/tmp/arbiter-messung"))
    parser.add_argument("--goldset", type=Path, default=Path("goldset/downloads.yaml"))
    parser.add_argument("--rag", default="")
    parser.add_argument("--strategien", default=",".join(STRATEGIEN))
    args = parser.parse_args(argv)

    umgebung = lade_umgebung()
    config = load_config(Path(umgebung["ARBITER_CONFIG"]), umgebung)
    name = args.rag or next(iter(config.rags))
    rag = replace(config.rags[name], rerank=False)
    goldset = [gold for gold in load_goldset(args.goldset) if gold.erwartet]
    if not goldset:
        print(f"Goldset ohne Belegseiten: {args.goldset}", file=sys.stderr)
        return 1

    print(f"{'Strategie':<10} {'Dateien':<28} {'Abschnitte':>10} {'Index':>9} {'Recall@10':>10}")
    for strategie in args.strategien.split(","):
        ziel = args.db / strategie
        if ziel.exists():
            shutil.rmtree(ziel)
        config_db = replace(config, db_path=ziel)
        pipeline = neuer_pipeline(config_db, replace(rag, chunk_strategie=strategie))
        report = pipeline.index()
        if report.files == 0 and report.unchanged == 0 and report.skipped_files == 0:
            print(f"{strategie}: Korpus leer ({rag.corpus_path})", file=sys.stderr)
            return 1

        zeichen = sum(
            len(str(row["text"])) for row in pipeline.store._table(name).to_arrow().to_pylist()
        )
        treffer = 0.0
        for gold in goldset:
            result = pipeline.search(gold.frage)
            treffer += recall_at(result.passages, gold, 10)

        verteilung: dict[str, int] = {}
        for datei in report.dateien:
            verteilung[datei.strategie] = verteilung.get(datei.strategie, 0) + 1
        dateien = ", ".join(f"{wahl}:{zahl}" for wahl, zahl in sorted(verteilung.items()))
        print(
            f"{strategie:<10} {dateien:<28} {report.chunks:>10} {zeichen:>9} "
            f"{treffer / len(goldset):>10.3f}"
        )
        for datei in report.dateien:
            wert = "übersteuert" if datei.wert is None else f"{datei.wert:.2f}"
            print(f"    {datei.quelle:<45.45} {datei.strategie:<9} {wert:>11} {datei.chunks:>5}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
