"""Probe the Laya gate on one question (debug, not part of the suite)."""

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
frage = sys.argv[1]

result = pipeline.search(frage)
print(f"Passagen: {len(result.passages)}, rerank={result.rerank}, hyde={bool(result.hyde_text)}")
for passage in result.passages:
    print(
        f"  [{passage.rrf_rank}] {passage.quelle[:50]} S{passage.seite} "
        f"rerank={passage.rerank_score}"
    )

state = build_noul_state(frage, [p.text for p in result.passages])
print(f"\nState: {len(state)} Zeichen")
client = SystemOne(cfg.backends.base_url, cfg.backends.laya_model)
answers = client.ask(state, build_questions(len(result.passages)))
for key, value in answers.items():
    print(f"  {key}: {value}")
