"""Compare the two gate shapes: one state with all passages vs one state per passage."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from arbiter.config import load_config
from arbiter.decision import SystemOne, build_noul_state, build_questions
from arbiter.pipeline import Pipeline

cfg = load_config(Path("arbiter.yaml"), os.environ)
rag = cfg.rags["downloads"]
pipeline = Pipeline(cfg, rag)
client = SystemOne(cfg.backends.base_url, cfg.backends.laya_model)

for frage in sys.argv[1:]:
    result = pipeline.search(frage)
    print(f"\n=== {frage}")
    for passage in result.passages:
        print(f"  [{passage.rrf_rank}] {passage.quelle[:40]} S{passage.seite}")

    gesamt = client.ask(
        build_noul_state(frage, [p.text for p in result.passages]),
        build_questions(len(result.passages)),
    )
    print("  alle in einem State :", [round(v["noul"], 3) for v in gesamt.values()])

    einzel = []
    for passage in result.passages:
        answers = client.ask(
            build_noul_state(frage, [passage.text]),
            {"p1": build_questions(1)["p1"]},
        )
        einzel.append(round(answers["p1"]["noul"], 3))
    print("  je Passage einzeln  :", einzel)
