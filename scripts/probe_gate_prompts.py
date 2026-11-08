"""Which prompt makes qwen answer with one usable token on long contexts?

Run: uv run python scripts/probe_gate_prompts.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow.parquet as pq

from arbiter.chat import ChatClient
from arbiter.config import load_config
from arbiter.decision import ja_wahrscheinlichkeit

QUELLE = Path.home() / ".cache/arbiterrag/ger-rag-eval/task4_test.parquet"

SYSTEME = {
    "teil (aktuell)": (
        "Du prüfst, ob ein Text zur Beantwortung einer Frage beiträgt. "
        "Antworte mit genau einem Wort: ja oder nein. Antworte mit ja, wenn "
        "der Text Angaben enthält, die für die Antwort nötig sind — auch wenn "
        "sie allein nicht ausreichen."
    ),
    "buchstabe": (
        "Du prüfst, ob ein Text zur Beantwortung einer Frage beiträgt. "
        "Antworte mit genau einem Buchstaben: J für ja, N für nein. "
        "Kein weiteres Wort."
    ),
    "knapp": ("Trägt der Text zur Beantwortung der Frage bei? Antworte nur mit: ja oder nein."),
}

cfg = load_config(Path("arbiter.yaml"), os.environ)
client = ChatClient(cfg.backends.base_url, cfg.backends.chat_model)
zeilen = pq.read_table(QUELLE).to_pylist()[:100]

for name, system in SYSTEME.items():
    ja_werte: list[float] = []
    nein_werte: list[float] = []
    ohne = 0
    for zeile in zeilen:
        user = f"Frage: {zeile['question']}\n\nText: {zeile['context']}"
        verteilung = client.probabilities(system, user)
        p = ja_wahrscheinlichkeit(verteilung)
        if p is None:
            p = (
                0.0
                if any(t.strip().casefold().startswith(("n", "nein")) for t in verteilung)
                else None
            )
        if p is None:
            ohne += 1
            continue
        (ja_werte if zeile["target"] == "J" else nein_werte).append(p)
    ohne_ja = sum(1 for p in ja_werte if p >= 0.5)
    ohne_nein = sum(1 for p in nein_werte if p < 0.5)
    print(
        f"{name:<16} ohne_antwort={ohne:>3} "
        f"J erkannt={ohne_ja}/{len(ja_werte) + ohne} "
        f"N abgelehnt={ohne_nein}/{len(nein_werte) + ohne}",
        flush=True,
    )
