"""Calibrate the logprob gate on Ger-RAG-eval task 4 (1000 labelled pairs).

Task 4 of deutsche-telekom/Ger-RAG-eval asks exactly what our gate asks: can this
context answer this question, yes or no. 500 yes, 500 no. That replaces the eight
negative cases of our own goldset.

Run: uv run python scripts/calibrate_gate_ger_rag_eval.py
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pyarrow.parquet as pq

from arbiter.config import load_config
from arbiter.decision import build_ja_nein_prompt, ja_wahrscheinlichkeit
from arbiter.hyde import ChatClient

QUELLE = Path.home() / ".cache/arbiterrag/ger-rag-eval/task4_test.parquet"
ZIEL = Path.home() / ".cache/arbiterrag/ger-rag-eval/gate-messung.csv"

cfg = load_config(Path("arbiter.yaml"), os.environ)
client = ChatClient(cfg.backends.base_url, cfg.backends.chat_model)

zeilen = pq.read_table(QUELLE).to_pylist()
ergebnisse: list[tuple[str, float]] = []
ohne = 0

for index, zeile in enumerate(zeilen, start=1):
    system, user = build_ja_nein_prompt(
        zeile["question"], zeile["context"], cfg.rags["downloads"].gate_prompt
    )
    try:
        p = ja_wahrscheinlichkeit(client.probabilities(system, user))
    except Exception as exc:
        print(f"Ausfall bei Zeile {index}: {exc}", flush=True)
        continue
    if p is None:
        # "no usable answer" counts as not admitted: for a system that must never
        # invent, a missing decision is a no. Kept in the table, not silently dropped.
        ohne += 1
        p = 0.0
    ergebnisse.append((zeile["target"], p))
    if index % 100 == 0:
        print(f"{index}/1000", flush=True)

ZIEL.parent.mkdir(parents=True, exist_ok=True)
with ZIEL.open("w", newline="", encoding="utf-8") as handle:
    schreiber = csv.writer(handle)
    schreiber.writerow(["target", "p_ja"])
    schreiber.writerows(ergebnisse)

ja = [p for ziel, p in ergebnisse if ziel == "J"]
nein = [p for ziel, p in ergebnisse if ziel == "N"]
print(f"\nGemessen: {len(ergebnisse)} von {len(zeilen)}; ohne verwertbare Antwort: {ohne}")
print(f"J: n={len(ja)} median={sorted(ja)[len(ja) // 2]:.3f}")
print(f"N: n={len(nein)} median={sorted(nein)[len(nein) // 2]:.3f}")
print(f"\n{'Schwelle':>8} {'J erkannt':>12} {'N abgelehnt':>13} {'Genauigkeit':>12}")
for schwelle in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
    j_ok = sum(1 for p in ja if p >= schwelle)
    n_ok = sum(1 for p in nein if p < schwelle)
    genau = (j_ok + n_ok) / len(ergebnisse)
    print(f"{schwelle:>8.2f} {j_ok:>7}/{len(ja):<4} {n_ok:>8}/{len(nein):<4} {genau:>11.1%}")
