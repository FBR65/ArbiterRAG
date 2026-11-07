"""Abschlusstest gegen das Heft iX Developer 2026 - Testing (ix.26.13.pdf).

Die erwarteten Antworten sind vorher am Volltext geprüft worden, nicht aus dem
Gedächtnis gesetzt. Erwartete Seite = Seite, auf der die Aussage steht.

Run: uv run python scripts/abschlusstest_ix.py
"""

from __future__ import annotations

import os
from pathlib import Path

from arbiter.config import load_config
from arbiter.pipeline import Pipeline

FAELLE = [
    (
        "Welche drei Schritte nennt der Artikel für die Modernisierung mit Golden Master?",
        112,
        "Einstiegspunkte festlegen, automatisierte Testsuite bauen, Testsuite ausweiten",
    ),
    (
        "Wozu dient Spec-driven Development bei KI-Agenten?",
        101,
        "Es erzwingt den Test vor dem Code (Given-When-Then), damit der Agent nicht "
        "den Test an den Code anpasst",
    ),
    (
        "Woran erkennt der Test im Vitest Browser Mode den Button?",
        74,
        "an seiner Aria-Rolle über getByRole",
    ),
    (
        "Was regelt die WCAG-Erfolgsbedingung 2.4.3?",
        39,
        "die Fokus-Reihenfolge",
    ),
    (
        "Was macht ein KI-Agent, wenn ein Test rot ist?",
        100,
        "Er löscht den Test oder schwächt die Assertion ab",
    ),
    (
        "Wie lautet die offizielle Berufsbezeichnung der Tester im Interview?",
        128,
        "Quality Engineer",
    ),
    (
        "Wie hoch ist die Teilnahmegebühr für die iX-Konferenz 2026?",
        None,
        "nicht im Heft — es darf keine Antwort geben",
    ),
    (
        "Welche Playwright-Mindestversion nennt der Artikel?",
        None,
        "nicht im Heft — es darf keine Antwort geben",
    ),
]

cfg = load_config(Path("arbiter.yaml"), os.environ)
pipeline = Pipeline(cfg, cfg.rags["downloads"])

for frage, seite, erwartet in FAELLE:
    antwort = pipeline.ask(frage)
    quellen = sorted({q.seite for q in antwort.quellen if q.seite is not None})
    print(f"\n=== {frage}")
    print(f"ERWARTET (S. {seite}): {erwartet}")
    print(f"ANTWORT: {antwort.text or 'KEINE (' + str(antwort.grund) + ')'}")
    print(f"QUELLEN Seiten: {quellen}")
    if antwort.warnungen:
        print(f"WARNUNGEN: {antwort.warnungen[:2]}")
