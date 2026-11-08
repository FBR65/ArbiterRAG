"""Measure the full chain (search + gate + answer) for one configuration.

One full run over the goldset: search, gate, answer.
Run: ARBITER_DB=... ARBITER_EMBED_MODEL=... uv run python scripts/measure_chain.py
"""

from __future__ import annotations

import os
from pathlib import Path

from arbiter.config import load_config
from arbiter.eval import load_goldset
from arbiter.pipeline import Pipeline

cfg = load_config(Path("arbiter.yaml"), os.environ)
rag = cfg.rags["downloads"]
goldset = load_goldset(Path("goldset/downloads.yaml"))
print(f"Embedder: {cfg.backends.embed_model}  DB: {cfg.db_path}", flush=True)
print(
    f"{'Lauf':<6} {'Antworten':>10} {'Beleg ok':>9} {'ohne Beleg':>11} "
    f"{'Zitate':>7} {'verworfen':>10}",
    flush=True,
)

for lauf in (1,):
    pipeline = Pipeline(cfg, rag)
    antworten = richtig = falsch = zitate = 0
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
        if treffer:
            richtig += 1
        else:
            falsch += 1
        if (antwort.text or "").startswith("Belegstelle:"):
            zitate += 1
    print(
        f"{lauf!s:<6} {antworten:>10} {richtig:>9} {falsch:>7} {len(goldset) - antworten:>10}",
        flush=True,
    )
