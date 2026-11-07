"""Measure the full chain (search + gate + answer) for one configuration.

Writes one row per HyDE setting, so the same run yields "with" and "without".
Run: ARBITER_DB=... ARBITER_EMBED_MODEL=... uv run python scripts/measure_chain.py
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from arbiter.config import load_config
from arbiter.eval import load_goldset
from arbiter.pipeline import Pipeline

cfg = load_config(Path("arbiter.yaml"), os.environ)
rag = cfg.rags["downloads"]
goldset = load_goldset(Path("goldset/downloads.yaml"))
print(f"Embedder: {cfg.backends.embed_model}  DB: {cfg.db_path}", flush=True)
print(
    f"{'HyDE':<6} {'Antworten':>10} {'Beleg ok':>9} {'ohne Beleg':>11} "
    f"{'Zitate':>7} {'verworfen':>10}",
    flush=True,
)

for hyde in (False, True):
    pipeline = Pipeline(cfg, replace(rag, hyde=hyde))
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
        f"{hyde!s:<6} {antworten:>10} {richtig:>9} {falsch:>7} {len(goldset) - antworten:>10}",
        flush=True,
    )
