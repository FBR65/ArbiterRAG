"""Why does a question come back without the labelled source? Check the index.

Reads the store only, no model calls: does a chunk for the labelled page exist at
all, and does it contain the words of the question and of the expected answer?

Run: uv run python scripts/diagnose_gold.py
"""

from __future__ import annotations

import os
from pathlib import Path

import lancedb

from arbiter.config import load_config
from arbiter.eval import load_goldset

cfg = load_config(Path("arbiter.yaml"), os.environ)
rag = cfg.rags["downloads"]
goldset = load_goldset(Path("goldset/downloads.yaml"))

db = lancedb.connect(str(cfg.db_path))
tabelle = db.open_table(rag.name)
spalten = tabelle.search().select(["quelle", "seite", "text"]).limit(100000).to_list()
print(f"Zeilen im Index: {len(spalten)}")

vorhandene = {(row["quelle"], row["seite"]) for row in spalten}
seiten_je_datei: dict[str, set[int]] = {}
for quelle, seite in vorhandene:
    seiten_je_datei.setdefault(quelle, set()).add(seite)

for gold in goldset:
    if not gold.erwartet:
        continue
    for datei, seite in gold.erwartet:
        treffer = [q for q in seiten_je_datei if datei in q]
        if not treffer:
            print(f"FEHLT (Datei unbekannt): {datei}")
            continue
        quelle = treffer[0]
        hat = seite in seiten_je_datei[quelle]
        seiten = sorted(seiten_je_datei[quelle])
        inhalt = " ".join(
            row["text"] for row in spalten if row["quelle"] == quelle and row["seite"] == seite
        ).casefold()
        stichworte = [wort for wort in gold.frage.casefold().split() if len(wort) > 5]
        drin = sum(1 for wort in stichworte if wort.strip("?.,") in inhalt)
        print(
            f"{'ok  ' if hat else 'FEHLT'} {Path(quelle).name[:34]:<36} S.{seite:<3} "
            f"Index-Seiten {seiten[0]}-{seiten[-1]} ({len(seiten)}) "
            f"Stichworte {drin}/{len(stichworte)}"
        )
