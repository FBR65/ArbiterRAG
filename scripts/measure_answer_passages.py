"""Does it help to give the answer more than three admitted passages?

Prints how many passages the gate admits per question, then runs the chain with
three and with five cited passages.
Run: uv run python scripts/measure_answer_passages.py
"""

from __future__ import annotations

import collections
import os
from dataclasses import replace
from pathlib import Path

from arbiter.config import load_config
from arbiter.eval import load_goldset
from arbiter.pipeline import Pipeline

cfg = load_config(Path("arbiter.yaml"), os.environ)
rag = cfg.rags["downloads"]
goldset = load_goldset(Path("goldset/downloads.yaml"))

zaehler: collections.Counter[int] = collections.Counter()
for gold in goldset:
    entscheidung = Pipeline(cfg, rag).bewerte(Pipeline(cfg, rag).search(gold.frage))
    zaehler[len(entscheidung.auswahl)] += 1
print("Zugelassene Passagen je Frage:", dict(sorted(zaehler.items())), flush=True)
print(
    f"Fragen mit mehr als drei zugelassenen Passagen: "
    f"{sum(n for k, n in zaehler.items() if k > 3)} von {len(goldset)}",
    flush=True,
)
print(f"\n{'zitiert':<9} {'Antworten':>10} {'Beleg ok':>9} {'falsch':>7}", flush=True)

for anzahl in (3, 5):
    pipeline = Pipeline(cfg, replace(rag, antwort_passagen=anzahl))
    antworten = richtig = falsch = 0
    for gold in goldset:
        antwort = pipeline.ask(gold.frage)
        if antwort.text is None:
            continue
        antworten += 1
        treffer = any(
            quelle in q.quelle and seite == q.seite
            for q in antwort.quellen
            for quelle, seite in gold.erwartet
        )
        richtig += 1 if treffer else 0
        falsch += 0 if treffer else 1
    print(f"{anzahl:<9} {antworten:>10} {richtig:>9} {falsch:>7}", flush=True)
