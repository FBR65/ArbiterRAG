"""Calibration of the logprob gate: P(ja) on the gold passage against the rest.

Run: uv run python scripts/measure_logprob_gate.py
"""

from __future__ import annotations

import os
from pathlib import Path

from arbiter.config import load_config
from arbiter.decision import build_ja_nein_prompt, ja_wahrscheinlichkeit
from arbiter.eval import load_goldset
from arbiter.hyde import ChatClient
from arbiter.pipeline import Pipeline

cfg = load_config(Path("arbiter.yaml"), os.environ)
rag = cfg.rags["downloads"]
pipeline = Pipeline(cfg, rag)
client = ChatClient(cfg.backends.base_url, cfg.backends.chat_model)
goldset = load_goldset(Path("goldset/downloads.yaml"))


def stats(werte: list[float], name: str) -> None:
    if not werte:
        print(f"  {name}: keine Werte")
        return
    geordnet = sorted(werte)
    print(
        f"  {name}: n={len(werte)} min={geordnet[0]:.3f} "
        f"median={geordnet[len(geordnet) // 2]:.3f} max={geordnet[-1]:.3f}"
    )


for variante in ("streng", "teil", "beleg"):
    werte_ja: list[float] = []
    werte_leer: list[float] = []
    for gold in goldset:
        bester = 0.0
        for passage in pipeline.search(gold.frage).passages:
            system, user = build_ja_nein_prompt(gold.frage, passage.text, variante)
            p = ja_wahrscheinlichkeit(client.probabilities(system, user))
            if p is not None and p > bester:
                bester = p
        (werte_ja if gold.erwartet else werte_leer).append(bester)

    print(f"=== Variante {variante}")
    stats(werte_ja, "beantwortbar, beste Passage")
    stats(werte_leer, "nicht beantwortbar, beste Passage")
    for schwelle in (0.3, 0.9, 0.99):
        beantwortet = sum(1 for wert in werte_ja if wert >= schwelle)
        leer = sum(1 for wert in werte_leer if wert >= schwelle)
        print(
            f"  Schwelle {schwelle:.2f}: beantwortbar {beantwortet}/{len(werte_ja)}, "
            f"nicht beantwortbar beantwortet {leer}/{len(werte_leer)}"
        )
    print(flush=True)
