"""Messung je Datei: welche Strategie holt für DIESE Datei die Belegseite? (Stufe 7.3b)

Run:
    set -a && . ./arbiter.env && set +a
    uv run python scripts/messe_pro_datei.py

Sucht in-memory und dicht (die Konfiguration fährt `hybrid: false`), damit keine zwei
Stores geschrieben werden müssen. Für jede Datei entstehen `struktur`- und
`fenster`-Chunks; geprüft werden nur die Goldfragen dieser Datei.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from arbiter.chunk import Chunk, ChunkParameter, chunk_datei
from arbiter.chunk_choice import waehle
from arbiter.config import load_config
from arbiter.embedding import HttpEmbedder
from arbiter.eval import GoldFrage, load_goldset
from arbiter.loaders import load_corpus
from arbiter.runtime import lade_umgebung

TOP_K = 10
BATCH = 32


def einbetten(embedder: HttpEmbedder, texte: list[str]) -> np.ndarray:
    vektoren: list[list[float]] = []
    for start in range(0, len(texte), BATCH):
        vektoren.extend(embedder.embed(texte[start : start + BATCH]))
    matrix = np.array(vektoren, dtype=np.float32)
    normen = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(normen, 1e-9)


def recall(
    chunks: list[Chunk],
    seiten: list[int],
    quelle: str,
    fragen: list[GoldFrage],
    embedder: HttpEmbedder,
) -> float:
    """Complete-evidence recall@10 der Belegseite, dicht gemessen."""
    if not fragen or not chunks:
        return 0.0
    matrix = einbetten(embedder, [chunk.text for chunk in chunks])
    frage_matrix = einbetten(embedder, [frage.frage for frage in fragen])
    treffer = 0.0
    for zeile, frage in zip(frage_matrix, fragen, strict=True):
        for index in np.argsort(-(matrix @ zeile))[:TOP_K]:
            if _passt(seiten[index], quelle, frage):
                treffer += 1.0
                break
    return treffer / len(fragen)


def _passt(seite: int, quelle: str, frage: GoldFrage) -> bool:
    return any(name in quelle and nummer == seite for name, nummer in frage.erwartet)


def seite_zu(seiten: dict[int, str], position: int) -> int:
    grenze = 0
    for nummer in sorted(seiten):
        grenze += len(seiten[nummer]) + 1
        if position < grenze:
            return nummer
    return sorted(seiten)[-1]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Recall je Datei und Strategie")
    parser.add_argument("--goldset", type=Path, default=Path("goldset/downloads.yaml"))
    parser.add_argument("--min-fragen", type=int, default=3)
    args = parser.parse_args(argv)

    umgebung = lade_umgebung()
    config = load_config(Path(umgebung["ARBITER_CONFIG"]), umgebung)
    rag = next(iter(config.rags.values()))
    embedder = HttpEmbedder(base_url=config.backends.base_url, model=config.backends.embed_model)
    geladen = load_corpus(rag.corpus_path, rag.extensions, sprache=rag.sprache)
    goldset = [gold for gold in load_goldset(args.goldset) if gold.erwartet]

    texte: dict[str, str] = {}
    seiten: dict[str, dict[int, str]] = {}
    for dokument in geladen.documents:
        texte[dokument.quelle] = texte.get(dokument.quelle, "") + dokument.text + "\n"
        seiten.setdefault(dokument.quelle, {})[dokument.seite or 0] = dokument.text

    print(
        f"{'Datei':<44} {'Entscheider':<10} {'Median':>7} "
        f"{'struktur':>9} {'fenster':>8} {'Fragen':>7}"
    )
    for quelle, text in texte.items():
        eigene = [gold for gold in goldset if any(q in quelle for q, _ in gold.erwartet)]
        if len(eigene) < args.min_fragen:
            print(f"{quelle[:44]:<44} {len(eigene)} Fragen — zu wenige, uebersprungen")
            continue
        auswahl = waehle(text, ebene=rag.struktur_ebene, max_chars=rag.chunk_chars)
        werte: dict[str, float] = {}
        for strategie in ("struktur", "fenster"):
            ergebnis = chunk_datei(
                text,
                quelle,
                ChunkParameter(strategie=strategie, max_chars=rag.chunk_chars),
            )
            werte[strategie] = recall(
                ergebnis.chunks,
                [seite_zu(seiten[quelle], chunk.start) for chunk in ergebnis.chunks],
                quelle,
                eigene,
                embedder,
            )
        print(
            f"{quelle[:44]:<44} {auswahl.strategie:<10} {auswahl.wert:>7.0f} "
            f"{werte['struktur']:>9.2f} {werte['fenster']:>8.2f} {len(eigene):>7}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
