"""Calibrate the Laya gate on the gold set (spec F2, stage 5).

For every question the top-k passages are gated one by one. Reported:
- how often the passage with the gold evidence is admitted,
- how often a question without any answer in the corpus is nevertheless admitted.
Run: uv run python scripts/calibrate_gate.py
"""

from __future__ import annotations

import os
from pathlib import Path

from arbiter.config import load_config
from arbiter.decision import SystemOne, build_noul_state, build_questions
from arbiter.eval import ist_beleg, load_goldset
from arbiter.pipeline import Pipeline

cfg = load_config(Path("arbiter.yaml"), os.environ)
rag = cfg.rags["downloads"]
pipeline = Pipeline(cfg, rag)
client = SystemOne(cfg.backends.base_url, cfg.backends.laya_model)
goldset = load_goldset(Path("goldset/downloads.yaml"))

treffer: list[float] = []
kein_treffer: list[float] = []
leer: list[float] = []

for gold in goldset:
    result = pipeline.search(gold.frage)
    beste_gold = 0.0
    beste_sonst = 0.0
    for passage in result.passages:
        answers = client.ask(
            build_noul_state(gold.frage, [passage.text]),
            {"p1": build_questions(1)["p1"]},
        )
        p = float(answers.get("p1", {}).get("noul", 0.0))
        if gold.erwartet and ist_beleg(passage, gold.erwartet):
            beste_gold = max(beste_gold, p)
        else:
            beste_sonst = max(beste_sonst, p)
    if gold.erwartet:
        treffer.append(beste_gold)
        kein_treffer.append(beste_sonst)
    else:
        leer.append(beste_sonst)
    print(f"{gold.frage[:52]:<54} gold={beste_gold:.3f} sonst={beste_sonst:.3f}", flush=True)

print("\nSchwellen-Sweep")
print(f"{'Schwelle':>8} {'belegbar erkannt':>18} {'nicht beantwortbar abgelehnt':>30}")
for schwelle in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    erkannt = sum(1 for p in treffer if p >= schwelle)
    abgelehnt = sum(1 for p in leer if p < schwelle)
    print(
        f"{schwelle:>8.1f} {erkannt:>10}/{len(treffer):<7} {abgelehnt:>16}/{len(leer):<13}"
    )

print(f"\nGold-Passage nie im Pool: {sum(1 for p in treffer if p == 0.0)} von {len(treffer)}")
