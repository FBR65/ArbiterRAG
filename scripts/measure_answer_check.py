"""Measure Laya as answer checker: is the support score different for good and bad answers?

Run: uv run python scripts/measure_answer_check.py
"""

from __future__ import annotations

import os
from pathlib import Path

from arbiter.config import load_config
from arbiter.decision import SystemOne
from arbiter.eval import load_goldset
from arbiter.pipeline import Pipeline

cfg = load_config(Path("arbiter.yaml"), os.environ)
rag = cfg.rags["downloads"]
pipeline = Pipeline(cfg, rag)
goldset = load_goldset(Path("goldset/downloads.yaml"))

gut: list[float] = []
schlecht: list[float] = []
for gold in goldset:
    antwort = pipeline.ask(gold.frage)
    if antwort.text is None or antwort.entscheidung is None:
        continue
    stuetzung = pipeline._pruefe_antwort(  # noqa: SLF001 - measurement only
        antwort.text,
        [
            type(antwort.quellen[0])(
                passage_id=q.passage_id, quelle=q.quelle, seite=q.seite, text=q.text
            )
            for q in antwort.quellen
        ],
    )
    treffer = bool(gold.erwartet) and any(
        quelle in q.quelle and q.seite == seite
        for q in antwort.quellen
        for quelle, seite in gold.erwartet
    )
    ziel = gut if treffer else schlecht
    ziel.append(stuetzung)
    print(f"{gold.frage[:50]:<52} stuetze={stuetzung:.3f} treffer={treffer}", flush=True)


def stats(werte: list[float], name: str) -> None:
    if not werte:
        print(f"{name}: keine Werte")
        return
    print(
        f"{name}: n={len(werte)} min={min(werte):.3f} median={sorted(werte)[len(werte) // 2]:.3f} "
        f"max={max(werte):.3f}"
    )


print()
stats(gut, "richtige Antwort")
stats(schlecht, "falsche Antwort")
print(f"\n{'Schwelle':>8} {'richtig behalten':>18} {'falsch abgelehnt':>18}")
for schwelle in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99):
    behalten = sum(1 for wert in gut if wert >= schwelle)
    abgelehnt = sum(1 for wert in schlecht if wert < schwelle)
    print(f"{schwelle:>8.2f} {behalten:>10}/{len(gut):<7} {abgelehnt:>10}/{len(schlecht):<7}")
